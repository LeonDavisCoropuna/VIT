#pragma once
#include "layer.hpp"

class DenseLayer : public Layer
{
public:
  Tensor weights;
  Tensor bias;
  Tensor input;
  Tensor output;
  Tensor grad_weights;
  Tensor grad_bias;
  Tensor input_deltas;
  Tensor output_deltas;
  bool use_bias;
  bool training;

  DenseLayer(int in_features, int out_features, bool use_bias = true)
      : use_bias(use_bias), training(true)
  {
    // Xavier initialization
    float limit = std::sqrt(6.0f / (in_features + out_features));
    weights = Tensor::rand_uniform({in_features, out_features}, -limit, limit);
    grad_weights = Tensor::zeros({in_features, out_features});

    if (use_bias)
    {
      bias = Tensor::zeros({out_features});
      grad_bias = Tensor::zeros({out_features});
    }
  }

  void get_parameters(Tensor &out_weights, Tensor &out_bias) const
  {
    out_weights.shape = weights.shape;
    out_weights.data = weights.data;

    if (use_bias)
    {
      out_bias.shape = bias.shape;
      out_bias.data = bias.data;
    }
    else
    {
      out_bias.shape = {};
      out_bias.data.clear();
    }
  }

  void set_parameters(const Tensor &new_weights, const Tensor &new_bias)
  {
    weights.shape = new_weights.shape;
    weights.data = new_weights.data;

    if (use_bias && !new_bias.data.empty())
    {
      bias.shape = new_bias.shape;
      bias.data = new_bias.data;
    }
    else
    {
      bias.shape = {};
      bias.data.clear();
    }
  }

  std::vector<Tensor> forward(const std::vector<Tensor> &inputs) override
  {
    input = inputs[0];              // (N, in_features)
    output = input.matmul(weights); // (N, out_features)

    if (use_bias)
      output = output + bias.reshape({1, -1}); // broadcasting

    output_deltas = output;
    return {output};
  }

  void backward(const Tensor *targets = nullptr,
                const Layer *next_layer = nullptr) override
  {
    Tensor delta; // (N, out_features)
    if (targets)
    {
      delta = *targets;
    }
    else
    {
      delta = next_layer->get_input_deltas();
    }

    Tensor input_T = input.transpose({1, 0}); // (in_features, N)
    grad_weights = input_T.matmul(delta);     // (in_features, out_features)

    if (use_bias)
      grad_bias = delta.sum(0); // sum across batch (eje 0 = filas)

    Tensor weights_T = weights.transpose({1, 0}); // (out_features, in_features)
    input_deltas = delta.matmul(weights_T);       // (N, in_features)
  }

  void update_weights(float batch_size) override
  {
    if (optimizer)
    {
      grad_weights = grad_weights / batch_size;
      grad_bias = grad_bias / batch_size;

      optimizer->update(weights, grad_weights,
                        bias, grad_bias);
    }
  }

  void zero_grad() override
  {
    grad_weights.fill(0.0f);
    if (use_bias)
      grad_bias.fill(0.0f);
  }

  void set_training(bool is_training) override
  {
    training = is_training;
  }

  const Tensor &get_output() const
  {
    return output;
  }

  const Tensor &get_input_deltas() const override
  {
    return input_deltas;
  }

  const Tensor &get_last_input() const
  {
    return input;
  }

  std::string get_type() const override
  {
    return "Dense";
  }
  
  bool has_weights() const override
  {
    return true;
  }
};
