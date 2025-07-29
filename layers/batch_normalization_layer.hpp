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

  std::string get_type() const override
  {
    return "BatchNorm1D";
  }
  void get_parameters(Tensor &out_gamma, Tensor &out_beta) const
  {
    out_gamma.shape = gamma.shape;
    out_gamma.data = gamma.data;
    out_beta.shape = beta.shape;
    out_beta.data = beta.data;
  }

  void set_parameters(const Tensor &new_gamma, const Tensor &new_beta)
  {
    gamma.shape = new_gamma.shape;
    gamma.data = new_gamma.data;
    beta.shape = new_beta.shape;
    beta.data = new_beta.data;
    grad_gamma = Tensor::zeros({num_features});
    grad_beta = Tensor::zeros({num_features});
    // No reinicializar running_mean ni running_var; se cargan en load()
  }

  void save(std::ostream &out) const
  {
    Tensor out_gamma, out_beta;
    get_parameters(out_gamma, out_beta);

    int g_dim = out_gamma.shape.size();
    int g_size = out_gamma.data.size();
    out.write(reinterpret_cast<const char *>(&g_dim), sizeof(int));
    out.write(reinterpret_cast<const char *>(out_gamma.shape.data()), g_dim * sizeof(int));
    out.write(reinterpret_cast<const char *>(&g_size), sizeof(int));
    out.write(reinterpret_cast<const char *>(out_gamma.data.data()), g_size * sizeof(float));

    int b_dim = out_beta.shape.size();
    int b_size = out_beta.data.size();
    out.write(reinterpret_cast<const char *>(&b_dim), sizeof(int));
    out.write(reinterpret_cast<const char *>(out_beta.shape.data()), b_dim * sizeof(int));
    out.write(reinterpret_cast<const char *>(&b_size), sizeof(int));
    out.write(reinterpret_cast<const char *>(out_beta.data.data()), b_size * sizeof(float));

    int rm_dim = running_mean.shape.size();
    int rm_size = running_mean.data.size();
    out.write(reinterpret_cast<const char *>(&rm_dim), sizeof(int));
    out.write(reinterpret_cast<const char *>(running_mean.shape.data()), rm_dim * sizeof(int));
    out.write(reinterpret_cast<const char *>(&rm_size), sizeof(int));
    out.write(reinterpret_cast<const char *>(running_mean.data.data()), rm_size * sizeof(float));

    int rv_dim = running_var.shape.size();
    int rv_size = running_var.data.size();
    out.write(reinterpret_cast<const char *>(&rv_dim), sizeof(int));
    out.write(reinterpret_cast<const char *>(running_var.shape.data()), rv_dim * sizeof(int));
    out.write(reinterpret_cast<const char *>(&rv_size), sizeof(int));
    out.write(reinterpret_cast<const char *>(running_var.data.data()), rv_size * sizeof(float));
  }

  void load(std::istream &in)
  {
    int g_dim, g_size;
    in.read(reinterpret_cast<char *>(&g_dim), sizeof(int));
    std::vector<int> g_shape(g_dim);
    in.read(reinterpret_cast<char *>(g_shape.data()), g_dim * sizeof(int));
    in.read(reinterpret_cast<char *>(&g_size), sizeof(int));
    std::vector<float> g_data(g_size);
    in.read(reinterpret_cast<char *>(g_data.data()), g_size * sizeof(float));

    int b_dim, b_size;
    in.read(reinterpret_cast<char *>(&b_dim), sizeof(int));
    std::vector<int> b_shape(b_dim);
    in.read(reinterpret_cast<char *>(b_shape.data()), b_dim * sizeof(int));
    in.read(reinterpret_cast<char *>(&b_size), sizeof(int));
    std::vector<float> b_data(b_size);
    in.read(reinterpret_cast<char *>(b_data.data()), b_size * sizeof(float));

    int rm_dim, rm_size;
    in.read(reinterpret_cast<char *>(&rm_dim), sizeof(int));
    std::vector<int> rm_shape(rm_dim);
    in.read(reinterpret_cast<char *>(rm_shape.data()), rm_dim * sizeof(int));
    in.read(reinterpret_cast<char *>(&rm_size), sizeof(int));
    std::vector<float> rm_data(rm_size);
    in.read(reinterpret_cast<char *>(rm_data.data()), rm_size * sizeof(float));

    int rv_dim, rv_size;
    in.read(reinterpret_cast<char *>(&rv_dim), sizeof(int));
    std::vector<int> rv_shape(rv_dim);
    in.read(reinterpret_cast<char *>(rv_shape.data()), rv_dim * sizeof(int));
    in.read(reinterpret_cast<char *>(&rv_size), sizeof(int));
    std::vector<float> rv_data(rv_size);
    in.read(reinterpret_cast<char *>(rv_data.data()), rv_size * sizeof(float));

    // Verificación de formas
    if (g_shape[0] != num_features || b_shape[0] != num_features ||
        rm_shape[0] != num_features || rv_shape[0] != num_features)
      throw std::runtime_error("BatchNormLayer1D: Shape mismatch during load");

    Tensor g_tensor, b_tensor, rm_tensor, rv_tensor;
    g_tensor.shape = g_shape;
    g_tensor.data = g_data;
    b_tensor.shape = b_shape;
    b_tensor.data = b_data;
    rm_tensor.shape = rm_shape;
    rm_tensor.data = rm_data;
    rv_tensor.shape = rv_shape;
    rv_tensor.data = rv_data;

    set_parameters(g_tensor, b_tensor);
    running_mean.shape = rm_tensor.shape;
    running_mean.data = rm_tensor.data;
    running_var.shape = rv_tensor.shape;
    running_var.data = rv_tensor.data;
  }

  std::vector<Tensor> forward(const std::vector<Tensor> &inputs) override
  {
    input = inputs[0]; // [N, C]
    int N = input.shape[0];
    int C = input.shape[1];

    if (is_training)
    {
      // input tiene la forma [N, C] = [2048, 32]
      // std::cout << "BatchNorm1D: Usando estadísticas de batch" << std::endl;
      mean = input.mean(0, true);                // [1, 32]
      var = (input - mean).pow(2).mean(0, true); // [1, 32]
      std_inv = (var + eps).sqrt().reciprocal(); // [1, 32]
      x_norm = (input - mean) * std_inv;         // [2048, 32]

      // Actualiza estadísticas acumuladas
      running_mean = mean.reshape({C}) * momentum + running_mean * (1 - momentum); // 32
      running_var = var.reshape({C}) * momentum + running_var * (1 - momentum);    // 32
    }
    else
    {
      std::cout << "BatchNorm1D: Usando estadísticas acumuladas" << std::endl;
      std_inv = (running_var + eps).sqrt().reciprocal().reshape({1, C});
      x_norm = (input - running_mean.reshape({1, C})) * std_inv;
    }

    Tensor out = x_norm * gamma.reshape({1, C}) + beta.reshape({1, C}); // [N, C] [2048, 32]
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
  std::string get_type() const override
  {
    return "BatchNorm2D";
  }

  void get_parameters(Tensor &out_gamma, Tensor &out_beta) const
  {
    out_gamma.shape = gamma.shape;
    out_gamma.data = gamma.data;
    out_beta.shape = beta.shape;
    out_beta.data = beta.data;
  }

  void set_parameters(const Tensor &new_gamma, const Tensor &new_beta)
  {
    gamma.shape = new_gamma.shape;
    gamma.data = new_gamma.data;
    beta.shape = new_beta.shape;
    beta.data = new_beta.data;
    grad_gamma = Tensor::zeros({num_channels});
    grad_beta = Tensor::zeros({num_channels});
  }

  void save(std::ostream &out) const
  {
    Tensor out_gamma, out_beta;
    get_parameters(out_gamma, out_beta);

    int g_dim = out_gamma.shape.size();
    int g_size = out_gamma.data.size();
    //std::cout << "[SAVE] BatchNorm2DLayer gamma shape: [";
    //for (int s : out_gamma.shape)
     // std::cout << s << " ";
    //std::cout << "], size: " << g_size << "\n";
    out.write(reinterpret_cast<const char *>(&g_dim), sizeof(int));
    out.write(reinterpret_cast<const char *>(out_gamma.shape.data()), g_dim * sizeof(int));
    out.write(reinterpret_cast<const char *>(&g_size), sizeof(int));
    out.write(reinterpret_cast<const char *>(out_gamma.data.data()), g_size * sizeof(float));

    int b_dim = out_beta.shape.size();
    int b_size = out_beta.data.size();
    //std::cout << "[SAVE] BatchNorm2DLayer beta shape: [";
    //for (int s : out_beta.shape)
    //  std::cout << s << " ";
   // std::cout << "], size: " << b_size << "\n";
    out.write(reinterpret_cast<const char *>(&b_dim), sizeof(int));
    out.write(reinterpret_cast<const char *>(out_beta.shape.data()), b_dim * sizeof(int));
    out.write(reinterpret_cast<const char *>(&b_size), sizeof(int));
    out.write(reinterpret_cast<const char *>(out_beta.data.data()), b_size * sizeof(float));

    int rm_dim = running_mean.shape.size();
    int rm_size = running_mean.data.size();
    //std::cout << "[SAVE] BatchNorm2DLayer running_mean shape: [";
    //for (int s : running_mean.shape)
     // std::cout << s << " ";
    //std::cout << "], size: " << rm_size << "\n";
    out.write(reinterpret_cast<const char *>(&rm_dim), sizeof(int));
    out.write(reinterpret_cast<const char *>(running_mean.shape.data()), rm_dim * sizeof(int));
    out.write(reinterpret_cast<const char *>(&rm_size), sizeof(int));
    out.write(reinterpret_cast<const char *>(running_mean.data.data()), rm_size * sizeof(float));

    int rv_dim = running_var.shape.size();
    int rv_size = running_var.data.size();
    //std::cout << "[SAVE] BatchNorm2DLayer running_var shape: [";
    //for (int s : running_var.shape)
     // std::cout << s << " ";
    //std::cout << "], size: " << rv_size << "\n";
    out.write(reinterpret_cast<const char *>(&rv_dim), sizeof(int));
    out.write(reinterpret_cast<const char *>(running_var.shape.data()), rv_dim * sizeof(int));
    out.write(reinterpret_cast<const char *>(&rv_size), sizeof(int));
    out.write(reinterpret_cast<const char *>(running_var.data.data()), rv_size * sizeof(float));
  }

  void load(std::istream &in)
  {
    int g_dim, g_size;
    in.read(reinterpret_cast<char *>(&g_dim), sizeof(int));
    std::vector<int> g_shape(g_dim);
    in.read(reinterpret_cast<char *>(g_shape.data()), g_dim * sizeof(int));
    in.read(reinterpret_cast<char *>(&g_size), sizeof(int));
    std::vector<float> g_data(g_size);
    in.read(reinterpret_cast<char *>(g_data.data()), g_size * sizeof(float));
    std::cout << "[LOAD] BatchNorm2DLayer gamma shape: [";
    //for (int s : g_shape)
      //std::cout << s << " ";
    //std::cout << "], size: " << g_size << "\n";

    int b_dim, b_size;
    in.read(reinterpret_cast<char *>(&b_dim), sizeof(int));
    std::vector<int> b_shape(b_dim);
    in.read(reinterpret_cast<char *>(b_shape.data()), b_dim * sizeof(int));
    in.read(reinterpret_cast<char *>(&b_size), sizeof(int));
    std::vector<float> b_data(b_size);
    in.read(reinterpret_cast<char *>(b_data.data()), b_size * sizeof(float));
    //std::cout << "[LOAD] BatchNorm2DLayer beta shape: [";
    //for (int s : b_shape)
      //std::cout << s << " ";
    //std::cout << "], size: " << b_size << "\n";

    int rm_dim, rm_size;
    in.read(reinterpret_cast<char *>(&rm_dim), sizeof(int));
    std::vector<int> rm_shape(rm_dim);
    in.read(reinterpret_cast<char *>(rm_shape.data()), rm_dim * sizeof(int));
    in.read(reinterpret_cast<char *>(&rm_size), sizeof(int));
    std::vector<float> rm_data(rm_size);
    in.read(reinterpret_cast<char *>(rm_data.data()), rm_size * sizeof(float));
    //std::cout << "[LOAD] BatchNorm2DLayer running_mean shape: [";
    //for (int s : rm_shape)
      //std::cout << s << " ";
    //std::cout << "], size: " << rm_size << "\n";

    int rv_dim, rv_size;
    in.read(reinterpret_cast<char *>(&rv_dim), sizeof(int));
    std::vector<int> rv_shape(rv_dim);
    in.read(reinterpret_cast<char *>(rv_shape.data()), rv_dim * sizeof(int));
    in.read(reinterpret_cast<char *>(&rv_size), sizeof(int));
    std::vector<float> rv_data(rv_size);
    in.read(reinterpret_cast<char *>(rv_data.data()), rv_size * sizeof(float));
   // std::cout << "[LOAD] BatchNorm2DLayer running_var shape: [";
   // for (int s : rv_shape)
   //   std::cout << s << " ";
   // std::cout << "], size: " << rv_size << "\n";

    Tensor g_tensor, b_tensor, rm_tensor, rv_tensor;
    g_tensor.shape = g_shape;
    g_tensor.data = g_data;
    b_tensor.shape = b_shape;
    b_tensor.data = b_data;
    rm_tensor.shape = rm_shape;
    rm_tensor.data = rm_data;
    rv_tensor.shape = rv_shape;
    rv_tensor.data = rv_data;

    gamma.shape = g_tensor.shape;
    gamma.data = g_tensor.data;
    beta.shape = b_tensor.shape;
    beta.data = b_tensor.data;
    running_mean.shape = rm_tensor.shape;
    running_mean.data = rm_tensor.data;
    running_var.shape = rv_tensor.shape;
    running_var.data = rv_tensor.data;

    grad_gamma = Tensor::zeros({num_channels});
    grad_beta = Tensor::zeros({num_channels});
  }
};