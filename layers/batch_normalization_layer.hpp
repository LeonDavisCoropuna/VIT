#pragma once
#include "layer.hpp"
class BatchNormLayer1D : public Layer
{
private:
  Tensor gamma, beta;
  Tensor running_mean, running_var;
  Tensor mean, var, std_inv, x_norm;
  Tensor input, input_deltas;
  Tensor grad_gamma, grad_beta;
  float momentum;
  float eps;
  int num_features;

public:
  BatchNormLayer1D(int features, float eps = 1e-5f, float momentum = 0.1f)
      : num_features(features), eps(eps), momentum(momentum)
  {
    gamma = Tensor({features});
    gamma.fill(1);
    beta = Tensor::zeros({features});
    running_mean = Tensor::zeros({features});
    running_var = Tensor({features});
    running_var.fill(1);
    grad_gamma = Tensor::zeros({features});
    grad_beta = Tensor::zeros({features});
  }

  std::vector<Tensor> forward(const std::vector<Tensor> &inputs) override
  {
    input = inputs[0]; // [N, C]
    int N = input.shape[0];
    int C = input.shape[1];

    if (is_training)
    {
      mean = input.mean(0, true);                // [1, C]
      var = (input - mean).pow(2).mean(0, true); // [1, C]
      std_inv = (var + eps).sqrt().reciprocal(); // [1, C]
      x_norm = (input - mean) * std_inv;         // [N, C]

      // Actualiza estadísticas acumuladas
      running_mean = mean.reshape({C}) * momentum + running_mean * (1 - momentum);
      running_var = var.reshape({C}) * momentum + running_var * (1 - momentum);
    }
    else
    {
      std_inv = (running_var + eps).sqrt().reciprocal().reshape({1, C});
      x_norm = (input - running_mean.reshape({1, C})) * std_inv;
    }

    Tensor out = x_norm * gamma.reshape({1, C}) + beta.reshape({1, C}); // [N, C]
    return {out};
  }

  void backward(const Tensor *targets, const Layer *next_layer) override
  {
    Tensor delta;
    if (targets)
    {
      delta = *targets;
    }
    else
    {
      delta = next_layer->get_input_deltas(); // [N, C]
    }

    int N = input.shape[0];
    int C = input.shape[1];

    grad_gamma = (delta * x_norm).sum(0); // [C]
    grad_beta = delta.sum(0);             // [C]

    Tensor dx_norm = delta * gamma.reshape({1, C});                                                  // [N, C]
    Tensor dvar = ((dx_norm * (input - mean)) * -0.5f * std_inv.pow(3)).sum(0);                      // [C]
    Tensor dmean = (dx_norm * (std_inv * (-1))).sum(0) + dvar * (-2.0f / N) * (input - mean).sum(0); // [C]

    Tensor dvar_flat = dvar.reshape({C});
    Tensor dmean_flat = dmean.reshape({C});

    input_deltas = dx_norm * std_inv +
                   (dvar_flat.reshape({1, C}) * 2.0f / N) * (input - mean) +
                   dmean_flat.reshape({1, C}) / N;
  }

  void update_weights(float batch_size) override
  {
    if (optimizer)
    {
      grad_gamma = grad_gamma / batch_size;
      grad_beta = grad_beta / batch_size;
      Tensor dummy;
      optimizer->update(gamma, grad_gamma, dummy, dummy);
      optimizer->update(beta, grad_beta, dummy, dummy);
    }
  }

  void zero_grad() override
  {
    grad_gamma.fill(0.0f);
    grad_beta.fill(0.0f);
  }

  void set_training(bool train) override
  {
    is_training = train;
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

class BatchNorm2DLayer : public Layer
{
private:
  Tensor gamma, beta;
  Tensor running_mean, running_var;
  Tensor mean, var, std_inv, x_norm;
  Tensor input, input_deltas;
  Tensor grad_gamma, grad_beta;
  float momentum;
  float eps;
  int num_channels;

public:
  BatchNorm2DLayer(int channels, float eps = 1e-5f, float momentum = 0.1f)
      : num_channels(channels), eps(eps), momentum(momentum)
  {
    gamma = Tensor({channels});
    gamma.fill(1);
    beta = Tensor::zeros({channels});
    running_mean = Tensor::zeros({channels});
    running_var = Tensor({channels});
    running_var.fill(1);
    grad_gamma = Tensor::zeros({channels});
    grad_beta = Tensor::zeros({channels});
  }

  std::vector<Tensor> forward(const std::vector<Tensor> &inputs) override
  {
    input = inputs[0]; // [N, C, H, W]
    int N = input.shape[0];
    int C = input.shape[1];
    int H = input.shape[2];
    int W = input.shape[3];

    Tensor reshaped = input.reshape({N, C, H * W}).transpose({2, 0, 1}); // [H*W, N, C]

    if (is_training)
    {
      mean = input.mean({0, 2, 3}, true); // [1, C, 1, 1]

      var = (input - mean).pow(2).mean({0, 2, 3}, true); // [1, C, 1, 1]
      std_inv = (var + eps).sqrt().reciprocal();         // [1, C, 1, 1]

      x_norm = (input - mean) * std_inv; // [N, C, H, W]

      // Asegúrate de que mean y var estén colapsados a [C] para actualizar los acumuladores
      running_mean = mean.reshape({C}) * momentum + running_mean * (1 - momentum);
      running_var = var.reshape({C}) * momentum + running_var * (1 - momentum);
    }
    else
    {
      Tensor running_mean_broadcasted = running_mean.reshape({1, C, 1, 1});
      Tensor running_var_broadcasted = running_var.reshape({1, C, 1, 1});

      std_inv = (running_var_broadcasted + eps).sqrt().reciprocal();
      x_norm = (input - running_mean_broadcasted) * std_inv;
    }

    Tensor gamma_broadcasted = gamma.reshape({1, C, 1, 1});
    Tensor beta_broadcasted = beta.reshape({1, C, 1, 1});
    Tensor output = x_norm * gamma_broadcasted + beta_broadcasted;
    return {output};
  }

  void backward(const Tensor *targets, const Layer *next_layer) override
  {
    if (targets == nullptr && next_layer == nullptr)
    {
      throw std::runtime_error("BatchNorm2D: No hay fuente de gradiente (ni targets ni next_layer)");
    }

    // Obtener el delta adecuado según lo disponible
    const Tensor &delta = (targets != nullptr) ? *targets : next_layer->get_input_deltas();
    
    int N = input.shape[0];
    int C = input.shape[1];
    int H = input.shape[2];
    int W = input.shape[3];
    int HW = H * W;

    grad_gamma = (delta * x_norm).sum({0, 2, 3}); // [C]
    grad_beta = delta.sum({0, 2, 3});             // [C]

    Tensor dx_norm = delta * gamma.reshape({1, C, 1, 1});                               // [N, C, H, W]
    Tensor dvar = ((dx_norm * (input - mean)) * -0.5f * std_inv.pow(3)).sum({0, 2, 3}); // [C]
    Tensor dmean = (dx_norm * std_inv * -1.0f).sum({0, 2, 3}) +
                   dvar * ((input - mean).sum({0, 2, 3}) * (-2.0f / (N * HW)));

    input_deltas = dx_norm * std_inv +
                   (dvar.reshape({1, C, 1, 1}) * 2.0f / (N * HW)) * (input - mean) +
                   dmean.reshape({1, C, 1, 1}) / (N * HW);
  }

  void update_weights(float batch_size) override
  {
    if (optimizer)
    {
      grad_gamma = grad_gamma / batch_size;
      grad_beta = grad_beta / batch_size;
      Tensor dummy;
      optimizer->update(gamma, grad_gamma, dummy, dummy);
      optimizer->update(beta, grad_beta, dummy, dummy);
    }
  }

  void zero_grad() override
  {
    grad_gamma.fill(0.0f);
    grad_beta.fill(0.0f);
  }

  void set_training(bool train) override
  {
    is_training = train;
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