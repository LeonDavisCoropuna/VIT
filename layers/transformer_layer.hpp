#pragma once
#include "layer.hpp"
#include "dense_layer.hpp"
#include "batch_normalization_layer.hpp"
#include "../utils/tensor.hpp"
#include "multi_head_attention.hpp"

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
