// maxpool2d_layer.hpp
#pragma once
#include "layer.hpp"


class MaxPool2DLayerr : public Layer
{
private:
  int kernel_size;
  int stride;
  Tensor input;
  Tensor output;
  Tensor input_deltas;
  Tensor max_indices; // ahora es un Tensor en GPU

public:
  MaxPool2DLayerr(int kernel_size, int stride)
      : kernel_size(kernel_size), stride(stride) {}

  std::vector<Tensor> forward(const std::vector<Tensor> &inputs) override
  {
    std::cout << "[DEBUG] MaxPool2DLayer forward\n";
    input = inputs[0]; // [N, C, H, W]
    int N = input.shape[0];
    int C = input.shape[1];
    int H = input.shape[2];
    int W = input.shape[3];

    int H_out = (H - kernel_size) / stride + 1;
    int W_out = (W - kernel_size) / stride + 1;

    output = Tensor({N, C, H_out, W_out}, input.is_cuda);
    max_indices = Tensor({N, C, H_out, W_out}, input.is_cuda); // nuevo: Tensor INT

    if (input.is_cuda)
    {
      std::cout << "[INFO] MaxPool2DLayer running on GPU\n";
      output = maxpool2d_cuda_forwardd(input, max_indices.data, kernel_size, stride);

      std::cout << "[DEBUG] ✅ Output shape: [" << output.shape[0] << ", " << output.shape[1] << ", " << output.shape[2] << ", " << output.shape[3] << "] | is_cuda: " << output.is_cuda << "\n";
      return {output};
    }

    std::cout << "[INFO] MaxPool2DLayer running on CPU\n";
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
                float val = input.at({n, c, h, w});
                if (val > max_val)
                {
                  max_val = val;
                  max_idx = h * W + w;
                }
              }
            }

            output.at({n, c, i, j}) = max_val;
            reinterpret_cast<int *>(max_indices.data)[((n * C + c) * H_out + i) * W_out + j] = max_idx;
          }
        }
      }
    }

    return {output};
  }

  void backward(const Tensor *targets = nullptr, const Layer *next_layer = nullptr) override
  {
    std::cout << "[BACKWARD] MaxPool2DLayer::backward - inicio\n";
    const Tensor &delta = next_layer->get_input_deltas(); // [N, C, H_out, W_out]

    input_deltas = Tensor::zeros(input.shape, input.is_cuda);

    if (input.is_cuda)
    {
      std::cout << "[INFO] MaxPool2DLayer backward running on GPU\n";
      input_deltas = maxpool2d_cuda_backward(delta, input, max_indices, kernel_size, stride);
      std::cout << "[DEBUG] ✅ MaxPool2DLayer backward completed on GPU\n";
    }
    else
    {
      std::cout << "[INFO] MaxPool2DLayer backward running on CPU\n";
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
              int max_pos = reinterpret_cast<int *>(max_indices.data)[idx];
              int h = max_pos / W;
              int w = max_pos % W;
              input_deltas.at({n, c, h, w}) = delta.at({n, c, i, j});
            }
          }
        }
      }
    }

    std::cout << "[BACKWARD] MaxPool2DLayer::backward - fin\n";
  }

  void update_weights(float batch_size) override {}
  void zero_grad() override {}
  const Tensor &get_input_deltas() const override { return input_deltas; }
  void set_training(bool is_train) override { is_training = is_train; }
  bool has_weights() const override { return false; }
};
