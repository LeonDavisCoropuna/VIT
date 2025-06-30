#pragma once
#include "../layers/layer.hpp"
#include "../layers/dense_layer.hpp"
#include "../layers/relu_layer.hpp"
#include "../layers/softmax_layer.hpp"
#include "../layers/dropout_layer.hpp"
#include "model.hpp"

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
    layers.push_back(new DenseLayer(64, 10));
    layers.push_back(new SoftmaxLayer());

    // Asignar optimizador a las capas con pesos
    for (auto *layer : layers)
    {
      if (layer->has_weights())
        layer->set_optimizer(new SGD(0.01f)); // comparte el mismo SGD
    }
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
