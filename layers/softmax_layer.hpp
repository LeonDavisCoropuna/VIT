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
    input = inputs[0]; // shape: (batch_size, num_classes)
    int batch_size = input.shape[0];
    int num_classes = input.shape[1];

    output = Tensor(input.shape);

    for (int b = 0; b < batch_size; ++b)
    {
      // Obtener el puntero a la fila actual
      float max_val = -std::numeric_limits<float>::infinity();
      for (int c = 0; c < num_classes; ++c)
      {
        float val = input.data[b * num_classes + c];
        if (val > max_val)
          max_val = val;
      }

      // Exponentes y suma
      float sum = 0.0f;
      for (int c = 0; c < num_classes; ++c)
      {
        output.data[b * num_classes + c] = std::exp(input.data[b * num_classes + c] - max_val);
        sum += output.data[b * num_classes + c];
      }

      // División normalizada
      for (int c = 0; c < num_classes; ++c)
      {
        output.data[b * num_classes + c] /= sum;
      }
    }

    return {output};
  }

  void backward(const Tensor *targets = nullptr,
                const Layer *next_layer = nullptr) override
  {
    if (targets)
    {
      // Cuando se usa con entropía cruzada: delta = softmax - y
      input_deltas = output - *targets;
    }
    else if (next_layer)
    {
      // Se propaga desde la siguiente capa
      const Tensor &delta = next_layer->get_input_deltas(); // (B, C)
      int B = delta.shape[0];
      int C = delta.shape[1];
      input_deltas = Tensor({B, C});

      for (int b = 0; b < B; ++b)
      {
        for (int i = 0; i < C; ++i)
        {
          float grad = 0.0f;
          for (int j = 0; j < C; ++j)
          {
            float s_i = output.data[b * C + i];
            float s_j = output.data[b * C + j];
            float delta_val = delta.data[b * C + j];
            grad += ((i == j) ? s_i * (1 - s_i) : -s_i * s_j) * delta_val;
          }
          input_deltas.data[b * C + i] = grad;
        }
      }
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
