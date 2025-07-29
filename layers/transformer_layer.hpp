#pragma once
#include "all_layers.hpp"

class TransformerLayer : public Layer
{
private:
  int token_channels;
  int attn_dim;
  int num_heads;
  MultiHeadAttention *attention;
  DenseLayer *linear1;
  DenseLayer *linear2;
  BatchNormLayer1D *norm1;
  BatchNormLayer1D *norm2;
  Tensor last_output;

public:
  TransformerLayer(int token_channels, int attn_dim, int num_heads, float dropout = 0.5)
      : token_channels(token_channels), attn_dim(attn_dim), num_heads(num_heads)
  {
    attention = new MultiHeadAttention(token_channels, attn_dim, num_heads); // Actualizado
    linear1 = new DenseLayer(token_channels, token_channels);
    linear2 = new DenseLayer(token_channels, token_channels);
    norm1 = new BatchNormLayer1D(token_channels);
    norm2 = new BatchNormLayer1D(token_channels);

    attention->set_optimizer(new SGD(0.001f));
    linear1->set_optimizer(new SGD(0.001f));
    linear2->set_optimizer(new SGD(0.001f));
    norm1->set_optimizer(new SGD(0.001f));
    norm2->set_optimizer(new SGD(0.001f));
  }

  ~TransformerLayer()
  {
    delete attention;
    delete linear1;
    delete linear2;
    delete norm1;
    delete norm2;
  }

  std::vector<Tensor> forward(const std::vector<Tensor> &inputs) override
  {
    Tensor x = inputs[0]; // [L, N, C] = [16, 128, 32]
    // x tiene la forma (L, N, C) = (tokens, batch_size, token_channels)

    // x beforeward: [L, N, C] = [16, 128, 32]
    Tensor attn_out = attention->forward({x})[0]; // [L, N, C]
    // atten_out after forward: [L, N, C] = [16, 128, 32]
    // attn_out tiene la forma (L, N, C) = (tokens, batch_size, attn_dim)

    // -------------------------------
    // 🔁 Residual Connection
    // x before residual: [L, N, C] = [16, 128, 32]
    // attn_out before residual: [L, N, C] = [16, 128, 32]
    Tensor residual = x + attn_out; // [L, N, C]
    // residual after residual: [L, N, C] = [16, 128, 32]

    // -------------------------------
    // 🔁 Normalización 1: BatchNorm1D
    // -------------------------------

    // Reshape residual to [L*N, C] for BatchNorm1D
    // residual before reshape: [L, N, C] = [16, 128, 32]
    // residual.shape[0] = L = 16
    // residual.shape[1] = N = 128
    // residual.shape[2] = C = 32
    // norm_input before reshape: [L*N, C] = [2048, 32]

    Tensor norm_input = residual.reshape({residual.shape[0] * residual.shape[1], residual.shape[2]}); // [L*N, C] = [2048, 16]
    // norm_input after reshape: [L*N, C] = [2048, 32]
    // norm_input tiene la forma (L*N, C) = (2048, 32)

    // Forward BatchNorm1D
    // norm_input before forward: [L*N, C] = [2048, 32]
    Tensor norm_output = norm1->forward({norm_input})[0]; // [2048, C]
    // norm_output after forward: [L*N, C] = [2048, 32]

    // norm_output tiene la forma (L*N, C) = (2048, 32)
    // x.shape[0] = L = 16
    // x.shape[1] = N = 128
    // x.shape[2] = C = 32
    // Reshape norm_output back to [L, N, C]
    Tensor a = norm_output.reshape({x.shape[0], x.shape[1], x.shape[2]}); // [L, N, C]
    // a after reshape: [L, N, C] = [16, 128, 32]

    // -------------------------------
    // 🔁 Feedforward MLP
    // -------------------------------
    int L = a.shape[0];
    int N = a.shape[1];
    int C = a.shape[2];

    // a before reshape: [L, N, C] = [16, 128, 32]
    Tensor a_flat = a.reshape({L * N, C}); // [2048, C]
    // a_flat after reshape: [L*N, C] = [2048, 32]

    Tensor b_flat = linear1->forward({a_flat})[0].relu(); // [2048, D]
    // b_flat tiene la forma (L*N, D) = (2048, 32)

    Tensor b_out = linear2->forward({b_flat})[0]; // [2048, C] ← igual que `a_flat` (residual)
    // b_out tiene la forma (L*N, C) = (2048, 32)

    // Reshape b_out back to [L, N, C]
    Tensor b = b_out.reshape({L, N, C}); // [L, N, C]
    // b after reshape: [L, N, C] = [16, 128, 32]

    // -------------------------------
    // 🔁 Normalización 2: BatchNorm1D
    // -------------------------------
    // b tiene la forma (L, N, C) = (16, 128, 32)
    // a tiene la forma (L, N, C) = (16, 128, 32)

    Tensor b_residual = a + b; // [L, N, C]
    // b_residual tiene la forma (L, N, C) = (16, 128, 32)

    // Reshape b_residual to [L*N, C] for BatchNorm1D
    Tensor b_norm_input = b_residual.reshape({L * N, C}); // [2048, C]
    // b_norm_input after reshape: [L*N, C] = [2048, 32]

    // Forward BatchNorm1D
    // norm2 recibe el input reshaped [L*N, C] = [2048, 32]
    Tensor b_norm_output = norm2->forward({b_norm_input})[0]; // [2048, C]
    // b_norm_output after forward: [L*N, C] = [2048, 32]

    last_output = b_norm_output.reshape({L, N, C}); // [L, N, C]
    // last_output after reshape: [L, N, C] = [16, 128, 32]

    return {last_output};
  }

  void backward(const Tensor *targets = nullptr, const Layer *next_layer = nullptr) override
  {
    // Obtener el delta de la siguiente capa o de los targets
    Tensor delta;
    if (next_layer != nullptr)
    {
      // Si tenemos la siguiente capa, pedimos sus deltas de entrada
      delta = next_layer->get_input_deltas();
    }
    else if (targets != nullptr)
    {
      // Si tenemos targets, calculamos el delta inicial
      delta = last_output - *targets;
    }
    else
    {
      throw std::runtime_error("TransformerLayer backward: either next_layer or targets must be provided");
    }

    // Reshape delta a la forma esperada [L*N, C]
    int L = last_output.shape[0];
    int N = last_output.shape[1];
    int C = last_output.shape[2];
    delta = delta.reshape({L * N, C});

    // -------------------------------
    // 🔁 Backward Normalización 2
    // -------------------------------
    norm2->backward(&delta, nullptr);
    Tensor b_norm_delta = norm2->get_input_deltas(); // [L*N, C]
    b_norm_delta = b_norm_delta.reshape({L, N, C});  // [L, N, C]

    // Derivada de la suma residual (a + b)
    Tensor a_delta = b_norm_delta;
    Tensor b_delta = b_norm_delta;

    // -------------------------------
    // 🔁 Backward Feedforward MLP
    // -------------------------------
    Tensor b = b_delta.reshape({L * N, C}); // [L*N, C]

    // Backward linear2
    linear2->backward(&b, nullptr);
    Tensor b_flat_delta = linear2->get_input_deltas(); // [L*N, D]

    // Backward ReLU y linear1
    Tensor a_flat = a_delta.reshape({L * N, C}); // [L*N, C]
    linear1->backward(&b_flat_delta, nullptr);
    Tensor a_flat_delta = linear1->get_input_deltas(); // [L*N, C]

    // Sumar deltas del camino de atención y feedforward
    Tensor norm1_delta = a_flat_delta + a_flat.reshape({L * N, C}); // [L*N, C]

    // -------------------------------
    // 🔁 Backward Normalización 1
    // -------------------------------
    norm1->backward(&norm1_delta, nullptr);
    Tensor residual_delta = norm1->get_input_deltas();  // [L*N, C]
    residual_delta = residual_delta.reshape({L, N, C}); // [L, N, C]

    // Derivada de la suma residual (x + attn_out)
    Tensor x_delta = residual_delta;
    Tensor attn_out_delta = residual_delta;

    // -------------------------------
    // 🔁 Backward Self-Attention
    // -------------------------------
    attention->backward(&attn_out_delta, nullptr);

    // Sumar delta del camino directo y de atención
    Tensor input_delta = x_delta + attention->get_input_deltas();

    // Guardar los deltas de entrada para ser usados por capas anteriores
    last_output = input_delta;
  }
  void update_weights(float batch_size) override
  {
    attention->update_weights(batch_size);
    linear1->update_weights(batch_size);
    linear2->update_weights(batch_size);
    norm1->update_weights(batch_size);
    norm2->update_weights(batch_size);
  }
  void zero_grad() override
  {
    attention->zero_grad();
    linear1->zero_grad();
    linear2->zero_grad();
    norm1->zero_grad();
    norm2->zero_grad();
  }
  void set_training(bool is_training) override {}
  /*
  void set_training(bool is_training) override
  {
    attention->set_training(is_training);
    linear1->set_training(is_training);
    linear2->set_training(is_training);
    norm1->set_training(is_training);
    norm2->set_training(is_training);
  }*/
  const Tensor &get_input_deltas() const override { return last_output; }

  std::string get_type() const override
  {
    return "TransformerLayer";
  }

  std::vector<Tensor> get_parameters() const
  {
    std::vector<Tensor> params;
    if (attention->has_weights())
    {
      std::vector<Tensor> attn_params = attention->get_parameters();
      params.insert(params.end(), attn_params.begin(), attn_params.end());
    }
    Tensor weights1, bias1, weights2, bias2;
    linear1->get_parameters(weights1, bias1);
    linear2->get_parameters(weights2, bias2);
    params.push_back(weights1);
    params.push_back(bias1);
    params.push_back(weights2);
    params.push_back(bias2);
    Tensor gamma1, beta1, gamma2, beta2;
    norm1->get_parameters(gamma1, beta1);
    norm2->get_parameters(gamma2, beta2);
    params.push_back(gamma1);
    params.push_back(beta1);
    params.push_back(gamma2);
    params.push_back(beta2);
    return params;
  }

  void set_parameters(const std::vector<Tensor> &new_params)
  {
    size_t param_index = 0;
    if (attention->has_weights())
    {
      std::vector<Tensor> attn_params = attention->get_parameters();
      std::vector<Tensor> new_attn_params(
          new_params.begin() + param_index,
          new_params.begin() + param_index + attn_params.size());
      attention->set_parameters(new_attn_params);
      param_index += attn_params.size();
    }
    linear1->set_parameters(new_params[param_index], new_params[param_index + 1]);
    param_index += 2;
    linear2->set_parameters(new_params[param_index], new_params[param_index + 1]);
    param_index += 2;
    norm1->set_parameters(new_params[param_index], new_params[param_index + 1]);
    param_index += 2;
    norm2->set_parameters(new_params[param_index], new_params[param_index + 1]);
  }

  void save(std::ostream &out) const
  {
    std::string attn_type = attention->get_type();
    int attn_type_len = attn_type.size();
    out.write(reinterpret_cast<const char *>(&attn_type_len), sizeof(int));
    out.write(attn_type.c_str(), attn_type_len);
    attention->save(out);

    std::vector<Tensor> params = get_parameters();
    int num_params = params.size();
    out.write(reinterpret_cast<const char *>(&num_params), sizeof(int));
    for (const auto &param : params)
    {
      int p_dim = param.shape.size();
      int p_size = param.data.size();
      out.write(reinterpret_cast<const char *>(&p_dim), sizeof(int));
      out.write(reinterpret_cast<const char *>(param.shape.data()), p_dim * sizeof(int));
      out.write(reinterpret_cast<const char *>(&p_size), sizeof(int));
      out.write(reinterpret_cast<const char *>(param.data.data()), p_size * sizeof(float));
    }
  }

  void load(std::istream &in)
  {
    int attn_type_len;
    in.read(reinterpret_cast<char *>(&attn_type_len), sizeof(int));
    std::string attn_type(attn_type_len, ' ');
    in.read(&attn_type[0], attn_type_len);
    if (attn_type != attention->get_type())
      throw std::runtime_error("Tipo de MultiHeadAttention no coincide: esperado " +
                               attention->get_type() + ", encontrado " + attn_type);
    attention->load(in);

    int num_params;
    in.read(reinterpret_cast<char *>(&num_params), sizeof(int));
    size_t expected_params = attention->get_parameters().size() + 8; // MultiHeadAttention + 2 (linear1) + 2 (linear2) + 2 (norm1) + 2 (norm2)
    if (num_params != static_cast<int>(expected_params))
      throw std::runtime_error("TransformerLayer: Expected " + std::to_string(expected_params) +
                               " parameter tensors, got " + std::to_string(num_params));

    std::vector<Tensor> params(num_params);
    for (int i = 0; i < num_params; ++i)
    {
      int p_dim, p_size;
      in.read(reinterpret_cast<char *>(&p_dim), sizeof(int));
      std::vector<int> p_shape(p_dim);
      in.read(reinterpret_cast<char *>(p_shape.data()), p_dim * sizeof(int));
      in.read(reinterpret_cast<char *>(&p_size), sizeof(int));
      std::vector<float> p_data(p_size);
      in.read(reinterpret_cast<char *>(p_data.data()), p_size * sizeof(float));
      params[i].shape = p_shape;
      params[i].data = p_data;
    }
    set_parameters(params);
  }
};
