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