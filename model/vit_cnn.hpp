#pragma once
#include <vector>

#include "model.hpp"

#include "../layers/layer.hpp"
#include "../layers/conv2d_layer.hpp"
#include "../layers/batch_normalization_layer.hpp"
#include "../layers/visual_transformer.hpp"
#include "../layers/dense_layer.hpp"
#include "../layers/pooling_layer.hpp"
#include "../layers/flatten_layer.hpp"

class VTCNN : public Model
{
private:
  Conv2DLayer *conv_layer;
  BatchNorm2DLayer *bn;
  std::vector<VisualTransformer *> vt_layers;
  MaxPool2DLayer *avgpool;
  FlattenLayer *flatten;
  DenseLayer *fc;
  SoftmaxLayer *softmax;
  Tensor input_deltas;

  int in_channels;
  int vt_channels;
  int tokens;
  int vt_layers_num;
  int vt_layer_res;

public:
  VTCNN(
      int image_channels,
      int input_dim,
      int vt_layers_num,
      int tokens,
      int token_channels,
      int vt_channels,
      int transformer_enc_layers,
      int transformer_heads,
      int transformer_fc_dim = 1024,
      int transformer_dropout = 0.5,
      int num_classes = 1000) : in_channels(vt_channels / 2),
                                vt_channels(vt_channels),
                                tokens(tokens),
                                vt_layers_num(vt_layers_num),
                                vt_layer_res(input_dim / 14)
  {

    conv_layer = new Conv2DLayer(image_channels, in_channels, 3, 3, 1, 1);
    bn = new BatchNorm2DLayer(in_channels);
    // First VT layer (filter based)
    vt_layers.push_back(new VisualTransformer(
        in_channels,
        vt_channels,
        token_channels,
        tokens,
        "filter",
        token_channels,
        transformer_enc_layers,
        transformer_heads,
        transformer_fc_dim,
        transformer_dropout,
        true));

    // Subsequent VT layers (recurrent)
    for (int i = 1; i < vt_layers_num; ++i)
    {
      vt_layers.push_back(new VisualTransformer(
          vt_channels,
          vt_channels,
          token_channels,
          tokens,
          "recurrent",
          token_channels,
          transformer_enc_layers,
          transformer_heads,
          transformer_fc_dim,
          transformer_dropout,
          true));
    }

    avgpool = new MaxPool2DLayer(vt_layer_res, vt_layer_res);
    flatten = new FlattenLayer();
    fc = new DenseLayer(8 * 14 * 14, num_classes);
    softmax = new SoftmaxLayer();

    if (conv_layer->has_weights())
      conv_layer->set_optimizer(new SGD(0.001f));
    if (bn->has_weights())
      bn->set_optimizer(new SGD(0.001f));
    for (auto *vt : vt_layers)
      if (vt->has_weights())
        vt->set_optimizer(new SGD(0.001f));
    if (fc->has_weights())
      fc->set_optimizer(new SGD(0.001f));
    // Si el softmax también tiene pesos (usualmente no)
    if (softmax->has_weights())
      softmax->set_optimizer(new SGD(0.001f));
  }

  Tensor forward(const Tensor &input) override
  {
    Tensor x = conv_layer->forward({input})[0]; // [32, 4, 28, 28] ← Conv2D: 1 canal → 4
    x = bn->forward({x})[0];                    // [32, 4, 28, 28] ← BatchNorm2D

    int N = x.shape[0]; // N = 32
    int C = x.shape[1]; // C = 4
    int H = x.shape[2]; // H = 28
    int W = x.shape[3]; // W = 28

    x = x.flatten(2);           // [32, 4, 784] ← HW = 28×28 = 784
    x = x.transpose({0, 2, 1}); // [32, 784, 4] ← (N, HW, C)

    Tensor t;
    std::vector<Tensor> output_vt = vt_layers[0]->forward({x});
    x = output_vt[0]; // [32, 784, 8] ← Primer VT layer: out_channels = vt_channels = 8
    t = output_vt[1]; // [32, tokens, token_channels] ← ej: [32, 16, 16]

    for (int i = 1; i < vt_layers_num; ++i)
    {
      std::vector<Tensor> output_vt = vt_layers[i]->forward({x, t});
      x = output_vt[0]; // [32, 784, 8]
      t = output_vt[1]; // [32, 16, 16]
    }

    x = x.reshape({N, vt_channels, H, W}); // [32, 8, 28, 28] ← reshape desde (32, 784, 8)
                                           // porque 784 = 28×28, se puede reestructurar así

    x = avgpool->forward({x})[0]; // [32, 8, 14, 14] ← MaxPool2D(2, 2)
    x = flatten->forward({x})[0]; // [32, 8x14x14] ← aplana todo excepto batch
    x = fc->forward({x})[0];      // [32, num_classes] ← DenseLayer(8 → num_classes)
    x = softmax->forward({x})[0];
    output = x;
    return output;
  }

  void backward(const Tensor &targets) override
  {
    // Última capa: DenseLayer (fc)
    softmax->backward(&targets, nullptr);
    fc->backward(nullptr, softmax);
    flatten->backward(nullptr, fc);
    // avgpool recibe como siguiente capa a fc
    avgpool->backward(nullptr, flatten);

    // Backward de VisualTransformers
    // for (int i = vt_layers_num - 1; i >= 0; --i)
    // {
    //   VisualTransformer *next = (i + 1 < vt_layers_num) ? vt_layers[i + 1] : nullptr;
    //   vt_layers[i]->backward(nullptr, next);
    // }
    vt_layers[0]->backward(nullptr, avgpool);

    // bn recibe como siguiente capa al primer VisualTransformer
    Tensor vt_deltas = vt_layers[0]->get_input_deltas(); // [128, 784, 4]

    // Reshape a formato BatchNorm2D [N, C, H, W]
    Tensor bn_deltas = vt_deltas.transpose({0, 2, 1})  // [128, 4, 784]
                           .reshape({128, 4, 28, 28}); // [128, 4, 28, 28]
    bn->backward(&bn_deltas, nullptr);

    // conv_layer recibe como siguiente capa a bn
    conv_layer->backward(nullptr, bn);
  }

  void update_weights(float batch_size) override
  {
    conv_layer->update_weights(batch_size);
    bn->update_weights(batch_size);
    fc->update_weights(batch_size);
    for (auto *vt : vt_layers)
      vt->update_weights(batch_size);
  }

  void zero_grad() override
  {
    conv_layer->zero_grad();
    bn->zero_grad();
    fc->zero_grad();
    for (auto *vt : vt_layers)
      vt->zero_grad();
  }

  void set_training(bool is_training) override
  {
    conv_layer->set_training(is_training);
    bn->set_training(is_training);
    fc->set_training(is_training);
    for (auto *vt : vt_layers)
      vt->set_training(is_training);
  }

  const Tensor &get_output() const
  {
    return output;
  }

  ~VTCNN()
  {
    delete conv_layer;
    delete bn;
    delete avgpool;
    delete fc;
    for (auto *vt : vt_layers)
      delete vt;
  }
};
