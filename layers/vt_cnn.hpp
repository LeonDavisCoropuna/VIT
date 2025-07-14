#pragma once
#include "all_layers.hpp"

class VTCNN : public Layer
{
private:
  Conv2DLayer *conv_layer;
  BatchNorm2DLayer *bn;
  std::vector<VisualTransformer *> vt_layers;
  MaxPool2DLayer *avgpool;
  FlattenLayer *flatten;
  DenseLayer *fc;
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
                                vt_layer_res(input_dim / 16)
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
    fc = new DenseLayer(vt_channels, num_classes);
  }

  std::vector<Tensor> forward(const std::vector<Tensor> &inputs) override
  {
    Tensor x = conv_layer->forward(inputs)[0];
    x = bn->forward({x})[0];

    int N = x.shape[0];
    int C = x.shape[1];
    int H = x.shape[2];
    int W = x.shape[3];

    x = x.flatten(2);           // [N, C, HW]
    x = x.transpose({0, 2, 1}); // [N, HW, C]

    Tensor t;
    std::vector<Tensor> output_vt = vt_layers[0]->forward({x});
    x = output_vt[0];
    t = output_vt[1];

    for (int i = 1; i < vt_layers_num; ++i)
    {
      std::vector<Tensor> output_vt = vt_layers[i]->forward({x, t});
      x = output_vt[0];
      t = output_vt[1];
    }

    x = x.transpose({0, 2, 1}); // [N, C, HW]
    x = x.reshape({N, vt_channels, vt_layer_res, vt_layer_res});

    x = avgpool->forward({x})[0]; // [N, C, 1, 1]
    x = flatten->forward({x})[0]; // [N, C]

    x = fc->forward({x})[0];
    return {x};
  }

  void backward(const Tensor *targets = nullptr, const Layer *next_layer = nullptr) override {}
  void update_weights(float batch_size) override {}
  void zero_grad() override {}
  void set_training(bool is_trainig) override {}
  const Tensor &get_input_deltas() const override { return input_deltas; }
};
