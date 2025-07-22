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

  int N;
  int C;
  int H;
  int W;

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
    fc = new DenseLayer(vt_channels * 14 * 14, num_classes);
    softmax = new SoftmaxLayer();

    conv_layer->set_optimizer(new SGD(0.001f));
    bn->set_optimizer(new SGD(0.001f));
    for (auto *vt : vt_layers)
      if (vt->has_weights())
        vt->set_optimizer(new SGD(0.001f));
    fc->set_optimizer(new SGD(0.001f));
    softmax->set_optimizer(new SGD(0.001f));
  }

  Tensor forward(const Tensor &input) override
  {
    // input shape [128, 1, 28, 28] ← MNIST: 128 imágenes de 1 canal (grises) de 28x28
    // capacidad de la red: [N, C, H, W] = [128, 4, 28, 28]
    // vector length data 100352

    // forward pass
    Tensor x = conv_layer->forward({input})[0];
    // x shape after forward: [128, 8, 28, 28] ← 128 imágenes de 8 canales (después de la convolución) de 28x28
    // x data length: 802816

    x = bn->forward({x})[0];
    // x shape after forward: [128, 8, 28, 28] ← BatchNorm2D mantiene el tamaño
    // x data length: 802816
    // N = batch size, C = channels, H = height, W = width

    N = x.shape[0]; // N = 128
    C = x.shape[1]; // C = 8
    H = x.shape[2]; // H = 28
    W = x.shape[3]; // W = 28

    // Reshape para VisualTransformer
    // x shape before reshape: [128, 8, 28, 28] ← [N, C, H, W]
    // x data length before reshape: 802816
    x = x.flatten(2);
    // x shape after flatten: [128, 8, 784] ← aplana H y W
    // x data length after flatten: 802816

    // Transpose para VisualTransformer
    // x shape before transpose: [128, 8, 784] ← [N, C, HW]
    // x data length before transpose: 802816
    x = x.transpose({0, 2, 1});
    // x shape after transpose: [128, 784, 8] ← transponemos para que sea (N, HW, C)
    // x data length after transpose: 802816
    // Ahora x tiene la forma [N, HW, C] = [128, 784, 8]

    Tensor t;

    // Inicializar tokens visuales (VT)
    // t shape: [128, tokens, token_channels] ← [N, tokens, token_channels]
    std::vector<Tensor> output_vt = vt_layers[0]->forward({x});
    // output_vt es un vector de tensores, donde:
    // output_vt[0] es la salida de la capa VisualTransformer (VT)
    // output_vt[1] son los tokens visuales generados por la capa VT
    // output_vt[0] tiene la forma [N, HW, vt_channels] = [128, 784, 16]
    // output_vt[1] tiene la forma [N, tokens, token_channels] = [128, 16, 32]
    // output_vt[0] es la salida de la capa VisualTransformer (VT)
    // output_vt[1] son los tokens visuales generados por la capa VT

    // output_vt[1] tiene la forma [N, tokens, token_channels] = [128, 784, 16]
    x = output_vt[0];
    // x es la salida de la capa VisualTransformer (VT)
    // x tiene la forma [N, HW, vt_channels] = [128, 784, 16]

    t = output_vt[1];
    // t es el tensor de tokens visuales generados por la capa VT
    // t tiene la forma [N, tokens, token_channels] = [128, 16, 32]
    // t data length: 8192

    // Iterar sobre las capas VisualTransformer restantes
    for (int i = 1; i < vt_layers_num; ++i)
    {
      std::vector<Tensor> output_vt = vt_layers[i]->forward({x, t});

      x = output_vt[0];
      // x es la salida de la capa VisualTransformer (VT)
      // x tiene la forma [N, HW, vt_channels] = [32, 784, 8]
      // x data length: 25088

      t = output_vt[1];
      // t es el tensor de tokens visuales generados por la capa VT
      // t tiene la forma [N, tokens, token_channels] = [32, 16, 16]
      // t data length: 8192
    }

    // x tiene la forma [N, HW, vt_channels] = [128, 784, 16]
    x = x.reshape({N, vt_channels, H, W});
    // x shape after reshape: [128, 16, 28, 28] ← reshape a (N, C, H, W)
    // x data length after reshape: 12544

    // Aplicar MaxPool2D
    // x tiene la forma [N, C, H, W] = [128, 16, 28, 28]
    x = avgpool->forward({x})[0];
    // x shape after MaxPool2D: [128, 16, 14, 14] ← reduce H y W a la mitad
    // x data length after MaxPool2D: 3136

    // Aplanar para la capa densa
    // x tiene la forma [N, C, H, W] = [128, 16, 14, 14]
    x = flatten->forward({x})[0]; //
    // x shape after flatten: [128, 3136] ← aplana a (N, C*H*W) = (128, 16*14*14)

    // Aplicar capa densa
    // x tiene la forma [N, C*H*W] = [128, 3136]
    // fc layer: DenseLayer(16*14*14 → num_classes)
    x = fc->forward({x})[0];
    // x shape after DenseLayer: [128, 10] ← aplana a (N, num_classes)

    // Ahora x tiene la forma [N, num_classes] = [128, 10]

    // Aplicar softmax
    x = softmax->forward({x})[0];
    // x shape after SoftmaxLayer: [128, 10] ← salida de softmax
    // Ahora x tiene la forma [N, num_classes] = [128, 10]

    // Guardar la salida final
    // output es la salida final de la red
    output = x;
    // output tiene la forma [N, num_classes] = [128, 10]

    return output;
  }

  void backward(const Tensor &targets) override
  {
    // Última capa: DenseLayer (fc)  targets shape: [128, 10] ← targets de la red
    softmax->backward(&targets, nullptr); // softmax after backward: [128, 10] all input output and outputdeltas
    fc->backward(nullptr, softmax);       // fc after backward:  weight [3136 10] bias [10] input : [128 3136] output [128 10] and grad weight [3136 10] grad bias[10] input deltas [128, 3136]  output deltas [128, 10]
    flatten->backward(nullptr, fc);       // input deltas [128, 16, 14, 14] output  [128, 3136] and input [128, 16, 14, 14]
    // avgpool recibe como siguiente capa a fc
    avgpool->backward(nullptr, flatten); // input deltas [128, 16, 28, 28] output  [128, 16, 14, 14] and input [128, 16, 28, 28]

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
    Tensor bn_deltas = vt_deltas.transpose({0, 2, 1}) // [128, 4, 784]
                           .reshape({N, C, H, W});    // [128, 4, 28, 28]
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

/*
  void save(const std::string &filename) const 
  {
    std::ofstream out(filename, std::ios::binary);
    if (!out)
      throw std::runtime_error("No se pudo abrir el archivo para guardar el modelo: " + filename);

    conv_layer->save(out);
    bn->save(out);
    for (auto *vt : vt_layers)
      vt->save(out);
    fc->save(out);
    // No es necesario guardar flatten, maxpool o softmax si no tienen pesos
    out.close();
  }

  void load(const std::string &filename) 
  {
    std::ifstream in(filename, std::ios::binary);
    if (!in)
      throw std::runtime_error("No se pudo abrir el archivo para cargar el modelo: " + filename);

    conv_layer->load(in);
    bn->load(in);
    for (auto *vt : vt_layers)
      vt->load(in);
    fc->load(in);
    in.close();
  }
*/
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
