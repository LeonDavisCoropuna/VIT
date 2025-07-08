#pragma once
#include "layer.hpp"

class DropoutLayer : public Layer
{
private:
  float dropoutProb;
  Tensor dropoutMask;
  Tensor inputDeltas;

public:
  DropoutLayer(float prob = 0.5f) : dropoutProb(prob) {}

  std::vector<Tensor> forward(const std::vector<Tensor> &inputs) override
  {
    Tensor input = inputs[0];

    if (is_training)
    {
      dropoutMask = Tensor::bernoulli(input.shape, 1.0f - dropoutProb); // 1 con prob 1-p
      return {(input * dropoutMask) / (1.0f - dropoutProb)};
    }
    else
    {
      return {input};
    }
  }

  void backward(const Tensor *targets = nullptr,
                const Layer *next_layer = nullptr) override
  {
    const Tensor &nextDelta = next_layer->get_input_deltas();
    inputDeltas = nextDelta * dropoutMask; // Solo pasan las que estaban activas
  }

  void update_weights(float) override {}
  void zero_grad() override {}

  void set_training(bool training) override
  {
    is_training = training;
  }

  const Tensor &get_input_deltas() const override
  {
    return inputDeltas;
  }
};
