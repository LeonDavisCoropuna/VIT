#pragma once
#include "layer.hpp"
#include <cmath>

class SoftmaxLayer : public Layer
{
private:
  Tensor input;
  Tensor output;
  Tensor input_deltas;
  bool training;

public:
  SoftmaxLayer() : training(true) {}

  std::vector<Tensor> forward(const std::vector<Tensor> &inputs) override
  {
    input = inputs[0];         // shape: (batch_size, num_classes)
    output = input.softmax(1); // softmax en la dimensión de clases (dim = 1)
    return {output};
  }

  void backward(const Tensor *targets = nullptr,
                const Layer *next_layer = nullptr) override
  {
    if (targets)
    {
      // Caso común con CrossEntropy: dL/dz = softmax - y
      input_deltas = output - *targets;
    }
    else if (next_layer)
    {
      // Backprop general: usar derivada del softmax
      const Tensor &delta = next_layer->get_input_deltas();   // [B, C]
      input_deltas = Tensor::softmax_backward(output, delta); // [B, C]
    }
    else
    {
      throw std::runtime_error("SoftmaxLayer::backward: targets or next_layer must be provided.");
    }
  }

  void update_weights(float batch_size) override {} // No tiene pesos

  void zero_grad() override {} // Nada que hacer

  void set_training(bool is_training) override
  {
    training = is_training;
  }

  const Tensor &get_input_deltas() const override
  {
    return input_deltas;
  }

  bool has_weights() const override { return false; }
};
