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
    // Entrada esperada: [N, 3, 28, 28]

    // Primera capa convolucional: Conv2D(3, 16, 3x3, padding=1, stride=1)
    layers.push_back(new Conv2DLayer(1, 16, 3, 3, 1, 1));
    layers.push_back(new ReLULayer());
    layers.push_back(new MaxPool2DLayer(2, 2)); // [N, 16, 14, 14]

    // Segunda capa convolucional: Conv2D(16, 4, 3x3, padding=1, stride=1)
    layers.push_back(new Conv2DLayer(16, 4, 3, 3, 1, 1));
    layers.push_back(new ReLULayer());
    layers.push_back(new MaxPool2DLayer(2, 2)); // [N, 4, 7, 7]

    // Flatten: 4*7*7 = 196
    layers.push_back(new FlattenLayer());

    // Capa totalmente conectada: 196 -> 16
    layers.push_back(new DenseLayer(196, 16));
    layers.push_back(new ReLULayer());

    // Capa de salida: 16 -> 10 (clases)
    layers.push_back(new DenseLayer(16, 10));
    layers.push_back(new SoftmaxLayer());

    // Asignar optimizador SGD con lr=0.002 a las capas con pesos
    for (auto *layer : layers)
    {
      if (layer->has_weights())
        layer->set_optimizer(new SGD(0.01f));
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
