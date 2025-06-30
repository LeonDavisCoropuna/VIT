#pragma once
#include "layer.hpp"

class FlattenLayer : public Layer
{
private:
  Tensor input;
  Tensor output;
  Tensor input_deltas;

public:
  std::vector<Tensor> forward(const std::vector<Tensor> &inputs) override
  {
    input = inputs[0]; // [N, C, H, W]
    int N = input.shape[0];

    int flattened_size = 1;
    for (size_t i = 1; i < input.shape.size(); ++i)
      flattened_size *= input.shape[i];

    std::vector<int> new_shape = {N, flattened_size};
    output = input.reshape(new_shape); // Usa reshape sin -1
    return {output};
  }

  void backward(const Tensor *targets = nullptr,
                const Layer *next_layer = nullptr) override
  {
    const Tensor &delta = next_layer->get_input_deltas(); // [N, C×H×W]
    input_deltas = delta.reshape(input.shape);            // volver a [N, C, H, W]
  }

  void update_weights(float batch_size) override {}
  void zero_grad() override {}
  void set_training(bool is_train) override { is_training = is_train; }
  const Tensor &get_input_deltas() const override { return input_deltas; }
  bool has_weights() const override { return false; }
};
