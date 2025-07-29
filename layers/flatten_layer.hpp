#pragma once
#include "layer.hpp"

class FlattenLayer : public Layer
{
private:
  Tensor input;
  Tensor output;
  Tensor inputDeltas;

public:
  std::vector<Tensor> forward(const std::vector<Tensor> &inputs) override
  {
    input = inputs[0]; // [N, C, H, W]
    int batchSize = input.shape[0];

    int flattenedSize = 1;
    for (size_t i = 1; i < input.shape.size(); ++i)
      flattenedSize *= input.shape[i];

    std::vector<int> newShape = {batchSize, flattenedSize};
    output = input.reshape(newShape);
    return {output};
  }

  void backward(const Tensor *targets = nullptr,
                const Layer *next_layer = nullptr) override
  {
    const Tensor &delta = next_layer->get_input_deltas(); // [N, C×H×W]
    inputDeltas = delta.reshape(input.shape);             // volver a [N, C, H, W]
  }

  void update_weights(float /*batchSize*/) override {}
  void zero_grad() override {}
  void set_training(bool isTraining) override { is_training = isTraining; }
  const Tensor &get_input_deltas() const override { return inputDeltas; }
  bool has_weights() const override { return false; }
};
