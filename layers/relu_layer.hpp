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
    output = input;
    for (size_t i = 0; i < output.data.size(); ++i)
      output.data[i] = std::max(0.0f, input.data[i]);
    return {output};
  }

  void backward(const Tensor *targets = nullptr, const Layer *next_layer = nullptr) override
  {
    const Tensor &delta = next_layer->get_input_deltas();
    input_deltas = delta;
    for (size_t i = 0; i < input.data.size(); ++i)
      input_deltas.data[i] *= (input.data[i] > 0.0f ? 1.0f : 0.0f);
  }

  void update_weights(float) override {}
  void zero_grad() override {}
  void set_training(bool is_training) override { training = is_training; }
  const Tensor &get_input_deltas() const override { return input_deltas; }
};
