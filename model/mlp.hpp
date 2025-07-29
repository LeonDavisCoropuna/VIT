#pragma once
#include "../layers/layer.hpp"
#include "../layers/dense_layer.hpp"
#include "../layers/relu_layer.hpp"
#include "../layers/softmax_layer.hpp"
#include "../layers/dropout_layer.hpp"
#include "model.hpp"
#include <fstream>

class MLP : public Model
{
private:
  std::vector<Layer *> layers;
  Tensor output;

public:
  MLP()
  {
    layers.push_back(new DenseLayer(784, 128));
    // layers.push_back(new BatchNormLayer1D(128));
    layers.push_back(new ReLULayer());
    layers.push_back(new DropoutLayer(0.25f));
    layers.push_back(new DenseLayer(128, 64));
    // layers.push_back(new BatchNormLayer1D(64));
    layers.push_back(new ReLULayer());
    layers.push_back(new DenseLayer(64, 8));
    layers.push_back(new SoftmaxLayer());

    // Asignar optimizador a las capas con pesos
    for (auto *layer : layers)
    {
      if (layer->has_weights())
        layer->set_optimizer(new SGD(0.01f)); // comparte el mismo SGD
    }
  }

  void save(const std::string &filename)
  {
    std::ofstream out(filename, std::ios::binary);
    if (!out)
      throw std::runtime_error("No se pudo abrir el archivo para guardar el modelo");

    int num_layers = layers.size();
    out.write(reinterpret_cast<char *>(&num_layers), sizeof(int));

    for (Layer *layer : layers)
    {
      std::string type = layer->get_type();
      int type_len = type.size();
      out.write(reinterpret_cast<char *>(&type_len), sizeof(int));
      out.write(type.c_str(), type_len);

      if (layer->has_weights())
      {
        Tensor w, b;
        static_cast<DenseLayer *>(layer)->get_parameters(w, b);

        int w_size = w.data.size();
        int b_size = b.data.size();
        int w_dim = w.shape.size();
        int b_dim = b.shape.size();

        out.write(reinterpret_cast<char *>(&w_dim), sizeof(int));
        out.write(reinterpret_cast<char *>(w.shape.data()), w_dim * sizeof(int));
        out.write(reinterpret_cast<char *>(&w_size), sizeof(int));
        out.write(reinterpret_cast<char *>(w.data.data()), w_size * sizeof(float));

        out.write(reinterpret_cast<char *>(&b_dim), sizeof(int));
        out.write(reinterpret_cast<char *>(b.shape.data()), b_dim * sizeof(int));
        out.write(reinterpret_cast<char *>(&b_size), sizeof(int));
        out.write(reinterpret_cast<char *>(b.data.data()), b_size * sizeof(float));
      }
    }

    out.close();
  }

  void load(const std::string &filename)
  {
    std::ifstream in(filename, std::ios::binary);
    if (!in)
      throw std::runtime_error("No se pudo abrir el archivo para cargar el modelo");

    int num_layers_file = 0;
    in.read(reinterpret_cast<char *>(&num_layers_file), sizeof(int));
    if (num_layers_file != layers.size())
      throw std::runtime_error("Cantidad de capas no coincide con el modelo actual");

    for (Layer *layer : layers)
    {
      int type_len = 0;
      in.read(reinterpret_cast<char *>(&type_len), sizeof(int));
      std::string type(type_len, ' ');
      in.read(&type[0], type_len);

      if (type != layer->get_type())
        throw std::runtime_error("Tipo de capa no coincide");

      if (layer->has_weights())
      {
        int w_dim, b_dim, w_size, b_size;

        in.read(reinterpret_cast<char *>(&w_dim), sizeof(int));
        std::vector<int> w_shape(w_dim);
        in.read(reinterpret_cast<char *>(w_shape.data()), w_dim * sizeof(int));
        in.read(reinterpret_cast<char *>(&w_size), sizeof(int));
        std::vector<float> w_data(w_size);
        in.read(reinterpret_cast<char *>(w_data.data()), w_size * sizeof(float));

        in.read(reinterpret_cast<char *>(&b_dim), sizeof(int));
        std::vector<int> b_shape(b_dim);
        in.read(reinterpret_cast<char *>(b_shape.data()), b_dim * sizeof(int));
        in.read(reinterpret_cast<char *>(&b_size), sizeof(int));
        std::vector<float> b_data(b_size);
        in.read(reinterpret_cast<char *>(b_data.data()), b_size * sizeof(float));

        Tensor w_tensor, b_tensor;
        w_tensor.shape = w_shape;
        w_tensor.data = w_data;
        b_tensor.shape = b_shape;
        b_tensor.data = b_data;

        static_cast<DenseLayer *>(layer)->set_parameters(w_tensor, b_tensor);
      }
    }

    in.close();
  }

  Tensor forward(const Tensor &input)
  {
    Tensor x = input.reshape({input.shape[0], 784}); // [batch, 784]
    std::vector<Tensor> x_vec = {x};
    for (auto *layer : layers)
      x_vec[0] = layer->forward(x_vec)[0];
    output = x_vec[0];
    return output;
  }

  void backward(const Tensor &targets)
  {
    const Tensor *target_ptr = &targets;
    layers.back()->backward(&targets, nullptr);
    for (int i = layers.size() - 2; i >= 0; --i)
      layers[i]->backward(nullptr, i + 1 < layers.size() ? layers[i + 1] : nullptr);
  }

  void update_weights(float batch_size)
  {
    for (auto *layer : layers)
      if (layer->has_weights())
        layer->update_weights(batch_size);
  }

  void zero_grad()
  {
    for (auto *layer : layers)
      layer->zero_grad();
  }

  void set_training(bool is_training)
  {
    for (auto *layer : layers)
      layer->set_training(is_training);
  }

  ~MLP()
  {
    for (auto *layer : layers)
      delete layer;
  }

  const Tensor &get_output() const
  {
    return output;
  }
};
