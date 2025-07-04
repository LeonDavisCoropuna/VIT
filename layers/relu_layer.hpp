#pragma once
#include "layer.hpp"

class ReLULayer : public Layer
{
private:
  Tensor input;
  Tensor input_deltas;
  Tensor output;
  bool training;

public:
  std::vector<Tensor> forward(const std::vector<Tensor> &inputs) override
  {
    input = inputs[0];
    output = input.relu();
    return {output};
  }

  void backward(const Tensor *targets = nullptr, const Layer *next_layer = nullptr) override
  {
    const Tensor &delta = next_layer->get_input_deltas();
    input_deltas = delta * input.relu_derivative();
  }

  void update_weights(float) override {}
  void zero_grad() override {}
  void set_training(bool is_training) override { training = is_training; }
  const Tensor &get_input_deltas() const override { return input_deltas; }
};