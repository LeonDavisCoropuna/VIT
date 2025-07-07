#pragma once
#include "layer.hpp"

class DropoutLayer : public Layer
{
private:
  float dropout_prob;
  Tensor mask;
  Tensor input_deltas;

public:
  DropoutLayer(float prob = 0.5f) : dropout_prob(prob) {}

  std::vector<Tensor> forward(const std::vector<Tensor> &inputs) override
  {
    Tensor x = inputs[0];

    if (is_training)
    {
      mask = Tensor::bernoulli(x.shape, 1.0f - dropout_prob); // 1 con prob 1-p
      return {(x * mask) / (1.0f - dropout_prob)};
    }
    else
    {
      return {x};
    }
  }

  void backward(const Tensor *targets = nullptr,
                const Layer *next_layer = nullptr) override
  {
    const Tensor &next_delta = next_layer->get_input_deltas();
    input_deltas = next_delta * mask; // Solo pasan las que estaban activas
  }

  void update_weights(float) override {}
  void zero_grad() override {}

  void set_training(bool training) override
  {
    is_training = training;
  }

  const Tensor &get_input_deltas() const override
  {
    return input_deltas;
  }
};
