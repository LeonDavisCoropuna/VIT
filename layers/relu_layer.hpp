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
    std::string re = delta.printsummary("ReLU Layer Backward Delta");

    const Tensor &tem_raw = input.relu_derivative();
    std::string re1 = tem_raw.printsummary("ReLU Layer Backward Derivative");

    // 👉 Asegurar que `tem` tenga el mismo shape que `delta`
    Tensor tem = tem_raw.reshape(delta.shape);
    std::string re2 = tem.printsummary("ReLU Layer Backward Derivative Reshaped");

    input_deltas = delta * tem;

    std::string re3 = input_deltas.printsummary("ReLU Layer Backward Input Deltas *");

  }

  void update_weights(float) override {}
  void zero_grad() override {}
  void set_training(bool is_training) override { training = is_training; }
  const Tensor &get_input_deltas() const override { return input_deltas; }
};