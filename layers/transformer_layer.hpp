#pragma once
#include "layer.hpp"
#include "dense_layer.hpp"
#include "batch_normalization_layer.hpp"
#include "../utils/tensor.hpp"

class SelfAttention : public Layer
{
private:
  DenseLayer *query_linear;
  DenseLayer *key_linear;
  Tensor last_input;
  Tensor input_deltas;

  Tensor delta_query_flat; // Almacena deltas para actualización
  Tensor delta_key_flat;   // Almacena deltas para actualización

public:
  SelfAttention(int channels, int attn_dim)
  {
    query_linear = new DenseLayer(channels, attn_dim, false);
    key_linear = new DenseLayer(channels, attn_dim, false);

    key_linear->set_optimizer(new SGD(0.001f));
    query_linear->set_optimizer(new SGD(0.001f));
  }

  ~SelfAttention()
  {
    delete query_linear;
    delete key_linear;
  }

  std::vector<Tensor> forward(const std::vector<Tensor> &inputs) override
  {
    const Tensor &x = inputs[0]; // [B, T, D]
    last_input = x;

    int B = x.shape[0];
    int T = x.shape[1];
    int D = x.shape[2];

    Tensor x_flat = x.reshape({B * T, D});

    Tensor key_flat = key_linear->forward({x_flat})[0];     // [B*T, Dk]
    Tensor query_flat = query_linear->forward({x_flat})[0]; // [B*T, Dk]

    int Dk = key_flat.shape[1];

    Tensor key = key_flat.reshape({B, T, Dk});     // [B, T, Dk]
    Tensor query = query_flat.reshape({B, T, Dk}); // [B, T, Dk]
    Tensor query_T = query.transpose({0, 2, 1});   // [B, Dk, T]

    Tensor attn = key.matmul(query_T).softmax(2); // [B, T, T]
    Tensor output = attn.matmul(x);               // [B, T, D]

    return {output};
  }

  void backward(const Tensor *targets = nullptr, const Layer *next_layer = nullptr) override
  {
    // Obtener el delta de la capa siguiente
    Tensor delta;
    if (targets != nullptr)
    {
      delta = *targets; // Shape (16, 128, 16) [B, T, D]
    }
    else if (next_layer != nullptr)
    {
      delta = next_layer->get_input_deltas();
    }
    else
    {
      throw std::runtime_error("SelfAttention: No hay fuente de delta para backward.");
    }

    // Recuperar los valores del forward
    const Tensor &x = last_input; // [B, T, D]
    int B = x.shape[0];
    int T = x.shape[1];
    int D = x.shape[2];

    // Recalcular valores necesarios del forward
    Tensor x_flat = x.reshape({B * T, D});
    Tensor key_flat = key_linear->forward({x_flat})[0];     // [B*T, Dk]
    Tensor query_flat = query_linear->forward({x_flat})[0]; // [B*T, Dk]
    int Dk = key_flat.shape[1];

    Tensor key = key_flat.reshape({B, T, Dk});     // [B, T, Dk]
    Tensor query = query_flat.reshape({B, T, Dk}); // [B, T, Dk]
    Tensor query_T = query.transpose({0, 2, 1});   // [B, Dk, T]
    Tensor attn = key.matmul(query_T).softmax(2);  // [B, T, T]

    // ----------------------------------------------------------
    // Backward completo
    // ----------------------------------------------------------

    // 1. Backward de attn.matmul(x) [output = attn @ x]
    // delta viene de la capa siguiente con shape [B, T, D]

    // Derivada respecto a attn: delta @ x.T
    Tensor x_T = x.transpose({0, 2, 1});   // [B, D, T]
    Tensor delta_attn = delta.matmul(x_T); // [B, T, T]

    // Derivada respecto a x: attn.T @ delta
    Tensor attn_T = attn.transpose({0, 2, 1}); // [B, T, T]
    Tensor delta_x = attn_T.matmul(delta);     // [B, T, D]

    // 2. Backward del softmax
    Tensor delta_before_softmax = Tensor::softmax_backward(attn, delta_attn);

    // 3. Backward de key.matmul(query_T)
    // delta_before_softmax es [B, T, T]

    // Derivada respecto a key: delta_before_softmax @ query
    Tensor delta_key = delta_before_softmax.matmul(query); // [B, T, Dk]

    // Derivada respecto a query_T: key.T @ delta_before_softmax
    // Pero como query_T es [B, Dk, T], necesitamos transpuesta
    Tensor key_T = key.transpose({0, 2, 1});                   // [B, Dk, T]
    Tensor delta_query_T = key_T.matmul(delta_before_softmax); // [B, Dk, T]
    Tensor delta_query = delta_query_T.transpose({0, 2, 1});   // [B, T, Dk]

    // 4. Backward de las capas lineales
    delta_key_flat = delta_key.reshape({B * T, Dk});
    key_linear->backward(&delta_key_flat);

    delta_query_flat = delta_query.reshape({B * T, Dk});
    query_linear->backward(&delta_query_flat);

    // 5. Combinar los deltas
    Tensor delta_key_linear = key_linear->get_input_deltas().reshape({B, T, D});
    Tensor delta_query_linear = query_linear->get_input_deltas().reshape({B, T, D});

    // El delta total es la suma de:
    // 1. El delta que viene de x a través del residual (delta_x)
    // 2. El delta que viene de las capas lineales
    input_deltas = delta_x + delta_key_linear + delta_query_linear;
  }

  void update_weights(float batch_size) override
  {
    if (optimizer)
    {
      query_linear->update_weights(batch_size);
      key_linear->update_weights(batch_size);
    }
  }
  void zero_grad() override
  {
    delta_query_flat.fill(0.0f); // Almacena deltas para actualización
    delta_key_flat.fill(0.0f);   // Almacena deltas para actualización
    query_linear->zero_grad();
    key_linear->zero_grad();
  }
  void set_training(bool is_training) override {}
  const Tensor &get_input_deltas() const override { return input_deltas; }
};

class TransformerLayer : public Layer
{
private:
  int token_channels;
  int attn_dim;
  SelfAttention *attention;
  DenseLayer *linear1;
  DenseLayer *linear2;
  BatchNormLayer1D *norm1;
  BatchNormLayer1D *norm2;
  Tensor last_output;

public:
  TransformerLayer(int token_channels, int attn_dim, float dropout = 0.5)
      : token_channels(token_channels), attn_dim(attn_dim)
  {
    attention = new SelfAttention(token_channels, attn_dim);
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
    Tensor x = inputs[0];                         // [L, N, C] = [16, 128, 16]
    Tensor attn_out = attention->forward({x})[0]; // [L, N, C]
    Tensor residual = x + attn_out;               // [L, N, C]

    // -------------------------------
    // 🔁 Normalización 1: BatchNorm1D
    // -------------------------------
    Tensor norm_input = residual.reshape({residual.shape[0] * residual.shape[1], residual.shape[2]}); // [L*N, C] = [2048, 16]
    Tensor norm_output = norm1->forward({norm_input})[0];                                             // [2048, C]
    Tensor a = norm_output.reshape({x.shape[0], x.shape[1], x.shape[2]});                             // [L, N, C]

    // -------------------------------
    // 🔁 Feedforward MLP
    // -------------------------------
    int L = a.shape[0];
    int N = a.shape[1];
    int C = a.shape[2];

    Tensor a_flat = a.reshape({L * N, C});                // [2048, C]
    Tensor b_flat = linear1->forward({a_flat})[0].relu(); // [2048, D]
    Tensor b_out = linear2->forward({b_flat})[0];         // [2048, C] ← igual que `a_flat` (residual)

    Tensor b = b_out.reshape({L, N, C}); // [L, N, C]

    // -------------------------------
    // 🔁 Normalización 2: BatchNorm1D
    // -------------------------------
    Tensor b_residual = a + b;                                // [L, N, C]
    Tensor b_norm_input = b_residual.reshape({L * N, C});     // [2048, C]
    Tensor b_norm_output = norm2->forward({b_norm_input})[0]; // [2048, C]
    last_output = b_norm_output.reshape({L, N, C});           // [L, N, C]

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
  const Tensor &get_input_deltas() const override { return last_output; }
};
