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

    // Entrada: 28x28x3
    layers.push_back(new Conv2DLayer(1, 16, 3, 3, 1, 1)); // in_channels=3, out=16, kernel=3x3, pad=1, stride=1
    layers.push_back(new BatchNorm2DLayer(16));
    layers.push_back(new ReLULayer());

    // MaxPool 2x2 -> 14x14x16
    layers.push_back(new MaxPool2DLayer(2, 2)); // kernel=2x2, stride=2

    // Conv2D: 14x14x16 -> 14x14x4
    layers.push_back(new Conv2DLayer(16, 4, 3, 3, 1, 1)); // in=16, out=4
    layers.push_back(new BatchNorm2DLayer(4));
    layers.push_back(new ReLULayer());

    // MaxPool 2x2 -> 7x7x4
    layers.push_back(new MaxPool2DLayer(2, 2));

    // Flatten: 7x7x4 = 196
    layers.push_back(new FlattenLayer());

    // Dense(196 -> 16)
    layers.push_back(new DenseLayer(196, 16));
    layers.push_back(new ReLULayer());

    // Dense(16 -> 10)
    layers.push_back(new DenseLayer(16, 10));
    layers.push_back(new SoftmaxLayer());

    // Asignar optimizador a las capas con pesos
    for (auto *layer : layers)
    {
      if (layer->has_weights())
        layer->set_optimizer(new SGD(0.002f)); // comparte el mismo SGD
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
