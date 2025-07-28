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

  void save(const std::string &filename) const
  {
    std::ofstream out(filename, std::ios::binary);
    if (!out)
      throw std::runtime_error("No se pudo abrir el archivo para guardar el modelo: " + filename);

    int num_layers = 2 + vt_layers_num + 1;
    std::cout << "[SAVE] Number of layers: " << num_layers << "\n";
    out.write(reinterpret_cast<const char *>(&num_layers), sizeof(int));

    // Conv2DLayer
    std::string conv_type = conv_layer->get_type();
    int conv_type_len = conv_type.size();
    std::cout << "[SAVE] Conv2DLayer type: " << conv_type << ", length: " << conv_type_len << "\n";
    out.write(reinterpret_cast<const char *>(&conv_type_len), sizeof(int));
    out.write(conv_type.c_str(), conv_type_len);
    if (conv_layer->has_weights())
    {
      Tensor weights, bias;
      conv_layer->get_parameters(weights, bias);
      int w_dim = weights.shape.size();
      int w_size = weights.data.size();
      std::cout << "[SAVE] Conv2DLayer weights shape: [";
      for (int s : weights.shape)
        std::cout << s << " ";
      std::cout << "], size: " << w_size << "\n";
      out.write(reinterpret_cast<const char *>(&w_dim), sizeof(int));
      out.write(reinterpret_cast<const char *>(weights.shape.data()), w_dim * sizeof(int));
      out.write(reinterpret_cast<const char *>(&w_size), sizeof(int));
      out.write(reinterpret_cast<const char *>(weights.data.data()), w_size * sizeof(float));
      int b_dim = bias.shape.size();
      int b_size = bias.data.size();
      std::cout << "[SAVE] Conv2DLayer bias shape: [";
      for (int s : bias.shape)
        std::cout << s << " ";
      std::cout << "], size: " << b_size << "\n";
      out.write(reinterpret_cast<const char *>(&b_dim), sizeof(int));
      out.write(reinterpret_cast<const char *>(bias.shape.data()), b_dim * sizeof(int));
      out.write(reinterpret_cast<const char *>(&b_size), sizeof(int));
      out.write(reinterpret_cast<const char *>(bias.data.data()), b_size * sizeof(float));
    }

    // BatchNorm2DLayer
    std::string bn_type = bn->get_type();
    int bn_type_len = bn_type.size();
    std::cout << "[SAVE] BatchNorm2DLayer type: " << bn_type << ", length: " << bn_type_len << "\n";
    out.write(reinterpret_cast<const char *>(&bn_type_len), sizeof(int));
    out.write(bn_type.c_str(), bn_type_len);
    if (bn->has_weights())
    {
      bn->save(out);
    }

    // VisualTransformer layers
    for (const auto *vt : vt_layers)
    {
      std::string vt_type = vt->get_type();
      int vt_type_len = vt_type.size();
      std::cout << "[SAVE] VisualTransformer type: " << vt_type << ", length: " << vt_type_len << "\n";
      out.write(reinterpret_cast<const char *>(&vt_type_len), sizeof(int));
      out.write(vt_type.c_str(), vt_type_len);
      if (vt->has_weights())
      {
        std::vector<Tensor> vt_params = vt->get_parameters();
        int num_params = vt_params.size();
        std::cout << "[SAVE] VisualTransformer num_params: " << num_params << "\n";
        out.write(reinterpret_cast<const char *>(&num_params), sizeof(int));
        for (int i = 0; i < num_params; ++i)
        {
          const auto &param = vt_params[i];
          int p_dim = param.shape.size();
          int p_size = param.data.size();
          std::cout << "[SAVE] VisualTransformer param " << i << " shape: [";
          for (int s : param.shape)
            std::cout << s << " ";
          std::cout << "], size: " << p_size << "\n";
          out.write(reinterpret_cast<const char *>(&p_dim), sizeof(int));
          out.write(reinterpret_cast<const char *>(param.shape.data()), p_dim * sizeof(int));
          out.write(reinterpret_cast<const char *>(&p_size), sizeof(int));
          out.write(reinterpret_cast<const char *>(param.data.data()), p_size * sizeof(float));
        }
      }
    }

    // DenseLayer
    std::string fc_type = fc->get_type();
    int fc_type_len = fc_type.size();
    std::cout << "[SAVE] DenseLayer type: " << fc_type << ", length: " << fc_type_len << "\n";
    out.write(reinterpret_cast<const char *>(&fc_type_len), sizeof(int));
    out.write(fc_type.c_str(), fc_type_len);
    if (fc->has_weights())
    {
      Tensor weights, bias;
      fc->get_parameters(weights, bias);
      int w_dim = weights.shape.size();
      int w_size = weights.data.size();
      std::cout << "[SAVE] DenseLayer weights shape: [";
      for (int s : weights.shape)
        std::cout << s << " ";
      std::cout << "], size: " << w_size << "\n";
      out.write(reinterpret_cast<const char *>(&w_dim), sizeof(int));
      out.write(reinterpret_cast<const char *>(weights.shape.data()), w_dim * sizeof(int));
      out.write(reinterpret_cast<const char *>(&w_size), sizeof(int));
      out.write(reinterpret_cast<const char *>(weights.data.data()), w_size * sizeof(float));
      int b_dim = bias.shape.size();
      int b_size = bias.data.size();
      std::cout << "[SAVE] DenseLayer bias shape: [";
      for (int s : bias.shape)
        std::cout << s << " ";
      std::cout << "], size: " << b_size << "\n";
      out.write(reinterpret_cast<const char *>(&b_dim), sizeof(int));
      out.write(reinterpret_cast<const char *>(bias.shape.data()), b_dim * sizeof(int));
      out.write(reinterpret_cast<const char *>(&b_size), sizeof(int));
      out.write(reinterpret_cast<const char *>(bias.data.data()), b_size * sizeof(float));
    }

    out.close();
    std::cout << "[SAVE] Model saved to " << filename << "\n";
  }

  void load(const std::string &filename)
  {
    std::ifstream in(filename, std::ios::binary);
    if (!in)
      throw std::runtime_error("No se pudo abrir el archivo para cargar el modelo: " + filename);

    int num_layers;
    in.read(reinterpret_cast<char *>(&num_layers), sizeof(int));
    std::cout << "[LOAD] Number of layers: " << num_layers << "\n";
    if (num_layers != (2 + vt_layers_num + 1))
      throw std::runtime_error("Cantidad de capas no coincide con el modelo actual");

    // Conv2DLayer
    int conv_type_len;
    in.read(reinterpret_cast<char *>(&conv_type_len), sizeof(int));
    std::string conv_type(conv_type_len, ' ');
    in.read(&conv_type[0], conv_type_len);
    std::cout << "[LOAD] Conv2DLayer type: " << conv_type << ", length: " << conv_type_len << "\n";
    if (conv_type != conv_layer->get_type())
      throw std::runtime_error("Tipo de capa Conv2D no coincide");
    if (conv_layer->has_weights())
    {
      int w_dim, w_size;
      in.read(reinterpret_cast<char *>(&w_dim), sizeof(int));
      std::vector<int> w_shape(w_dim);
      in.read(reinterpret_cast<char *>(w_shape.data()), w_dim * sizeof(int));
      in.read(reinterpret_cast<char *>(&w_size), sizeof(int));
      std::vector<float> w_data(w_size);
      in.read(reinterpret_cast<char *>(w_data.data()), w_size * sizeof(float));
      std::cout << "[LOAD] Conv2DLayer weights shape: [";
      for (int s : w_shape)
        std::cout << s << " ";
      std::cout << "], size: " << w_size << "\n";
      int b_dim, b_size;
      in.read(reinterpret_cast<char *>(&b_dim), sizeof(int));
      std::vector<int> b_shape(b_dim);
      in.read(reinterpret_cast<char *>(b_shape.data()), b_dim * sizeof(int));
      in.read(reinterpret_cast<char *>(&b_size), sizeof(int));
      std::vector<float> b_data(b_size);
      in.read(reinterpret_cast<char *>(b_data.data()), b_size * sizeof(float));
      std::cout << "[LOAD] Conv2DLayer bias shape: [";
      for (int s : b_shape)
        std::cout << s << " ";
      std::cout << "], size: " << b_size << "\n";

      // Verificación de formas
      Tensor expected_weights, expected_bias;
      conv_layer->get_parameters(expected_weights, expected_bias);
      if (w_shape != expected_weights.shape || b_shape != expected_bias.shape)
        throw std::runtime_error("Conv2DLayer: Shape mismatch during load");

      Tensor w_tensor, b_tensor;
      w_tensor.shape = w_shape;
      w_tensor.data = w_data;
      b_tensor.shape = b_shape;
      b_tensor.data = b_data;
      conv_layer->set_parameters(w_tensor, b_tensor);
    }

    // BatchNorm2DLayer
    int bn_type_len;
    in.read(reinterpret_cast<char *>(&bn_type_len), sizeof(int));
    std::string bn_type(bn_type_len, ' ');
    in.read(&bn_type[0], bn_type_len);
    std::cout << "[LOAD] BatchNorm2DLayer type: " << bn_type << ", length: " << bn_type_len << "\n";
    if (bn_type != bn->get_type())
      throw std::runtime_error("Tipo de capa BatchNorm2D no coincide");
    if (bn->has_weights())
    {
      bn->load(in);
    }

    // VisualTransformer layers
    for (auto *vt : vt_layers)
    {
      int vt_type_len;
      in.read(reinterpret_cast<char *>(&vt_type_len), sizeof(int));
      std::string vt_type(vt_type_len, ' ');
      in.read(&vt_type[0], vt_type_len);
      std::cout << "[LOAD] VisualTransformer type: " << vt_type << ", length: " << vt_type_len << "\n";
      if (vt_type != vt->get_type())
        throw std::runtime_error("Tipo de capa VisualTransformer no coincide");
      if (vt->has_weights())
      {
        int num_params;
        in.read(reinterpret_cast<char *>(&num_params), sizeof(int));
        std::cout << "[LOAD] VisualTransformer num_params: " << num_params << "\n";
        std::vector<Tensor> vt_params(num_params);
        for (int i = 0; i < num_params; ++i)
        {
          int p_dim, p_size;
          in.read(reinterpret_cast<char *>(&p_dim), sizeof(int));
          std::vector<int> p_shape(p_dim);
          in.read(reinterpret_cast<char *>(p_shape.data()), p_dim * sizeof(int));
          in.read(reinterpret_cast<char *>(&p_size), sizeof(int));
          std::vector<float> p_data(p_size);
          in.read(reinterpret_cast<char *>(p_data.data()), p_size * sizeof(float));
          std::cout << "[LOAD] VisualTransformer param " << i << " shape: [";
          for (int s : p_shape)
            std::cout << s << " ";
          std::cout << "], size: " << p_size << "\n";
          vt_params[i].shape = p_shape;
          vt_params[i].data = p_data;
        }
        vt->set_parameters(vt_params);
      }
    }

    // DenseLayer
    int fc_type_len;
    in.read(reinterpret_cast<char *>(&fc_type_len), sizeof(int));
    std::string fc_type(fc_type_len, ' ');
    in.read(&fc_type[0], fc_type_len);
    std::cout << "[LOAD] DenseLayer type: " << fc_type << ", length: " << fc_type_len << "\n";
    if (fc_type != fc->get_type())
      throw std::runtime_error("Tipo de capa Dense no coincide");
    if (fc->has_weights())
    {
      int w_dim, w_size;
      in.read(reinterpret_cast<char *>(&w_dim), sizeof(int));
      std::vector<int> w_shape(w_dim);
      in.read(reinterpret_cast<char *>(w_shape.data()), w_dim * sizeof(int));
      in.read(reinterpret_cast<char *>(&w_size), sizeof(int));
      std::vector<float> w_data(w_size);
      in.read(reinterpret_cast<char *>(w_data.data()), w_size * sizeof(float));
      std::cout << "[LOAD] DenseLayer weights shape: [";
      for (int s : w_shape)
        std::cout << s << " ";
      std::cout << "], size: " << w_size << "\n";
      int b_dim, b_size;
      in.read(reinterpret_cast<char *>(&b_dim), sizeof(int));
      std::vector<int> b_shape(b_dim);
      in.read(reinterpret_cast<char *>(b_shape.data()), b_dim * sizeof(int));
      in.read(reinterpret_cast<char *>(&b_size), sizeof(int));
      std::vector<float> b_data(b_size);
      in.read(reinterpret_cast<char *>(b_data.data()), b_size * sizeof(float));
      std::cout << "[LOAD] DenseLayer bias shape: [";
      for (int s : b_shape)
        std::cout << s << " ";
      std::cout << "], size: " << b_size << "\n";
      Tensor w_tensor, b_tensor;
      w_tensor.shape = w_shape;
      w_tensor.data = w_data;
      b_tensor.shape = b_shape;
      b_tensor.data = b_data;
      fc->set_parameters(w_tensor, b_tensor);
    }

    in.close();
    std::cout << "[LOAD] Model loaded from " << filename << "\n";
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
