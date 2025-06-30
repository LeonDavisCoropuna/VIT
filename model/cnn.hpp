#pragma once
#include "../layers/layer.hpp"
#include "../layers/conv2d_layer.hpp"
#include "../layers/pooling_layer.hpp"
#include "../layers/flatten_layer.hpp"
#include "../layers/dense_layer.hpp"
#include "../layers/relu_layer.hpp"
#include "../layers/softmax_layer.hpp"
#include "../layers/dropout_layer.hpp"
#include "../layers/batch_normalization_layer.hpp"

#include "model.hpp"

class CNN : public Model
{
private:
  std::vector<Layer *> layers;
  Tensor output;

public:
  CNN()
  {
    // Entrada esperada: [N, 1, 28, 28]

    // Conv1: 1 -> 8 canales, kernel 3x3
    layers.push_back(new Conv2DLayer(1, 4, 3, 3, 1, 1)); // stride 1, padding 1
    layers.push_back(new BatchNorm2DLayer(4));
    layers.push_back(new ReLULayer());

    // MaxPool: reduce de 28x28 -> 4x4
    layers.push_back(new MaxPool2DLayer(7, 7));

    // Flatten: 8x4x4
    layers.push_back(new FlattenLayer());

    // Dense final: 4x4x4 -> 10
    layers.push_back(new DenseLayer(4 * 4 * 4, 10));
    layers.push_back(new SoftmaxLayer());

    // Asignar optimizador a las capas con pesos
    for (auto *layer : layers)
    {
      if (layer->has_weights())
        layer->set_optimizer(new SGD(0.01f)); // comparte el mismo SGD
    }
  }

  Tensor forward(const Tensor &x)
  {
    Tensor out = x;
    for (auto layer : layers)
    {
      out = layer->forward({out})[0];
    }
    output = out;
    return out;
  }

  void backward(const Tensor &target)
  {
    layers.back()->backward(&target, nullptr);

    for (int i = layers.size() - 2; i >= 0; --i)
    {
      layers[i]->backward(nullptr, layers[i + 1]);
    }
  }

  void update_weights(float batch_size)
  {
    for (auto layer : layers)
    {
      if (layer->has_weights())
        layer->update_weights(batch_size);
    }
  }

  void zero_grad()
  {
    for (auto layer : layers)
    {
      layer->zero_grad();
    }
  }

  void set_training(bool is_train)
  {
    for (auto layer : layers)
    {
      layer->set_training(is_train);
    }
  }

  void set_optimizer(Optimizer *opt)
  {
    for (auto layer : layers)
    {
      layer->set_optimizer(opt);
    }
  }

  const Tensor &get_output() const
  {
    return output;
  }
};
