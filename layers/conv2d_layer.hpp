#pragma once
#include "layer.hpp"

class Conv2DLayer : public Layer
{
private:
  int in_channels, out_channels;
  int kernel_h, kernel_w;
  int stride, padding;

  Tensor weights; // [out_channels, in_channels, kernel_h, kernel_w]
  Tensor biases;  // [out_channels]
  Tensor grad_weights;
  Tensor grad_biases;
  Tensor input;
  Tensor input_deltas;
  Tensor output;

public:
  Conv2DLayer(int in_channels, int out_channels, int kernel_h, int kernel_w,
              int stride = 1, int padding = 0)
      : in_channels(in_channels), out_channels(out_channels),
        kernel_h(kernel_h), kernel_w(kernel_w),
        stride(stride), padding(padding)
  {

    int fan_in = in_channels * kernel_h * kernel_w;
    // Para convoluciones con ReLU, usar Kaiming normal
    weights = Tensor::kaiming_normal({out_channels, in_channels, kernel_h, kernel_w}, fan_in);
    biases = Tensor::zeros({out_channels});
    grad_weights = Tensor::zeros(weights.shape);
    grad_biases = Tensor::zeros(biases.shape);
  }

  std::vector<Tensor> forward(const std::vector<Tensor> &inputs) override
  {
    input = inputs[0]; // [N, C_in, H_in, W_in]
    const int N = input.shape[0];
    const int C_in = input.shape[1];
    const int H_in = input.shape[2];
    const int W_in = input.shape[3];

    const int C_out = weights.shape[0];
    const int K_h = weights.shape[2];
    const int K_w = weights.shape[3];

    const int H_out = (H_in + 2 * padding - K_h) / stride + 1;
    const int W_out = (W_in + 2 * padding - K_w) / stride + 1;

    output = Tensor({N, C_out, H_out, W_out});
    output.fill(0.0f);

    // Elimina el padding manual y maneja los bordes directamente
    for (int n = 0; n < N; ++n)
    {
      for (int co = 0; co < C_out; ++co)
      {
        for (int ho = 0; ho < H_out; ++ho)
        {
          for (int wo = 0; wo < W_out; ++wo)
          {
            float sum = biases.at1d(co);

            for (int ci = 0; ci < C_in; ++ci)
            {
              for (int kh = 0; kh < K_h; ++kh)
              {
                for (int kw = 0; kw < K_w; ++kw)
                {
                  int h_in = ho * stride + kh - padding;
                  int w_in = wo * stride + kw - padding;

                  if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in)
                  {
                    sum += input.at4d(n, ci, h_in, w_in) *
                           weights.at4d(co, ci, kh, kw);
                  }
                }
              }
            }
            output.at4d(n, co, ho, wo) = sum;
          }
        }
      }
    }
    return {output};
  }

  void backward(const Tensor *targets = nullptr, const Layer *next_layer = nullptr) override
  {
    const Tensor &delta = next_layer->get_input_deltas(); // [N, C_out, H_out, W_out]

    const int N = input.shape[0];
    const int C_in = input.shape[1];
    const int H_in = input.shape[2];
    const int W_in = input.shape[3];

    const int C_out = weights.shape[0];
    const int K_h = weights.shape[2];
    const int K_w = weights.shape[3];

    const int H_out = delta.shape[2];
    const int W_out = delta.shape[3];

    // Padding input para grad_weights
    Tensor padded_input = input;
    if (padding > 0)
      padded_input = input.pad({0, 0, 0, 0, padding, padding, padding, padding});

    grad_weights = Tensor::zeros(weights.shape);
    grad_biases = Tensor::zeros({C_out});
    input_deltas = Tensor::zeros(input.shape);

    // grad_bias
    for (int n = 0; n < N; ++n)
      for (int co = 0; co < C_out; ++co)
        for (int ho = 0; ho < H_out; ++ho)
          for (int wo = 0; wo < W_out; ++wo)
            grad_biases.at1d(co) += delta.at4d(n, co, ho, wo);

    // grad_weights
    for (int co = 0; co < C_out; ++co)
    {
      for (int ci = 0; ci < C_in; ++ci)
      {
        for (int kh = 0; kh < K_h; ++kh)
        {
          for (int kw = 0; kw < K_w; ++kw)
          {
            float sum = 0.0f;
            for (int n = 0; n < N; ++n)
            {
              for (int ho = 0; ho < H_out; ++ho)
              {
                for (int wo = 0; wo < W_out; ++wo)
                {
                  int h_in = ho * stride + kh;
                  int w_in = wo * stride + kw;
                  sum += padded_input.at4d(n, ci, h_in, w_in) * delta.at4d(n, co, ho, wo);
                }
              }
            }
            grad_weights.at4d(co, ci, kh, kw) = sum;
          }
        }
      }
    }

    // input_deltas
    Tensor padded_deltas = Tensor::zeros({N, C_in, H_in + 2 * padding, W_in + 2 * padding});

    for (int n = 0; n < N; ++n)
    {
      for (int co = 0; co < C_out; ++co)
      {
        for (int ho = 0; ho < H_out; ++ho)
        {
          for (int wo = 0; wo < W_out; ++wo)
          {
            for (int ci = 0; ci < C_in; ++ci)
            {
              for (int kh = 0; kh < K_h; ++kh)
              {
                for (int kw = 0; kw < K_w; ++kw)
                {
                  int h_in = ho * stride + kh;
                  int w_in = wo * stride + kw;
                  if (h_in >= 0 && h_in < padded_deltas.shape[2] &&
                      w_in >= 0 && w_in < padded_deltas.shape[3])
                  {
                    float flipped_weight = weights.at4d(co, ci, K_h - 1 - kh, K_w - 1 - kw);
                    padded_deltas.at4d(n, ci, h_in, w_in) += delta.at4d(n, co, ho, wo) * flipped_weight;
                  }
                }
              }
            }
          }
        }
      }
    }

    // Quitar padding si es necesario
    if (padding > 0)
    {
      input_deltas = padded_deltas.slice(2, padding, padding + H_in).slice(3, padding, padding + W_in);
    }
    else
    {
      input_deltas = padded_deltas;
    }
  }

  void update_weights(float batch_size) override
  {
    if (optimizer)
    {
      grad_weights = grad_weights / batch_size;
      grad_biases = grad_biases / batch_size;
      optimizer->update(weights, grad_weights,
                        biases, grad_biases);
    }
  }

  void zero_grad() override
  {
    grad_weights.fill(0.0f);
    grad_biases.fill(0.0f);
  }

  void set_training(bool training) override
  {
    is_training = training;
  }

  const Tensor &get_input_deltas() const override
  {
    return input_deltas;
  }

  bool has_weights() const override
  {
    return true;
  }
};
