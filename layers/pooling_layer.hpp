#pragma once
#include "../utils/tensor.hpp"
#include "layer.hpp"

class MaxPool2DLayer : public Layer
{
private:
  int kernel_size;
  int stride;
  Tensor input;
  Tensor output;
  Tensor input_deltas;
  std::vector<int> max_indices; // Para el backward

public:
  MaxPool2DLayer(int kernel_size, int stride)
      : kernel_size(kernel_size), stride(stride) {}

  std::vector<Tensor> forward(const std::vector<Tensor> &inputs) override
  {
    input = inputs[0]; // [N, C, H, W]
    int N = input.shape[0];
    int C = input.shape[1];
    int H = input.shape[2];
    int W = input.shape[3];

    int H_out = (H - kernel_size) / stride + 1;
    int W_out = (W - kernel_size) / stride + 1;

    output = Tensor({N, C, H_out, W_out});
    max_indices.resize(N * C * H_out * W_out);

    for (int n = 0; n < N; ++n)
    {
      for (int c = 0; c < C; ++c)
      {
        for (int i = 0; i < H_out; ++i)
        {
          for (int j = 0; j < W_out; ++j)
          {
            float max_val = -std::numeric_limits<float>::infinity();
            int max_idx = -1;

            for (int ki = 0; ki < kernel_size; ++ki)
            {
              for (int kj = 0; kj < kernel_size; ++kj)
              {
                int h = i * stride + ki;
                int w = j * stride + kj;
                float val = input.at4d(n, c, h, w);
                if (val > max_val)
                {
                  max_val = val;
                  max_idx = h * W + w;
                }
              }
            }

            output.at4d(n, c, i, j) = max_val;
            max_indices[((n * C + c) * H_out + i) * W_out + j] = max_idx;
          }
        }
      }
    }

    return {output};
  }

  void backward(const Tensor *targets = nullptr,
                const Layer *next_layer = nullptr) override
  {
    const Tensor &delta = next_layer->get_input_deltas(); // [N, C, H_out, W_out]
    input_deltas = Tensor::zeros(input.shape);

    int N = input.shape[0];
    int C = input.shape[1];
    int H_out = delta.shape[2];
    int W_out = delta.shape[3];
    int W = input.shape[3];

    for (int n = 0; n < N; ++n)
    {
      for (int c = 0; c < C; ++c)
      {
        for (int i = 0; i < H_out; ++i)
        {
          for (int j = 0; j < W_out; ++j)
          {
            int idx = ((n * C + c) * H_out + i) * W_out + j;
            int max_pos = max_indices[idx];
            int h = max_pos / W;
            int w = max_pos % W;
            input_deltas.at4d(n, c, h, w) = delta.at4d(n, c, i, j);
          }
        }
      }
    }
  }

  void update_weights(float batch_size) override {}
  void zero_grad() override {}
  const Tensor &get_input_deltas() const override { return input_deltas; }
  void set_training(bool is_train) override { is_training = is_train; }
  bool has_weights() const override { return false; }
};
