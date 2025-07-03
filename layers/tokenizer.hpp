#pragma once
#include "all_layers.hpp"

// --- FilterTokenizer ---
class FilterTokenizer : public Layer
{
private:
  int tokens, in_channels, token_channels;
  DenseLayer *linear1, *linear2;
  Tensor cache1, cache2, token_cache, input_deltas;
  SoftmaxLayer softmax;
  Tensor input_cache;

public:
  FilterTokenizer(int in_channels, int token_channels, int tokens)
      : tokens(tokens), in_channels(in_channels), token_channels(token_channels)
  {
    linear1 = new DenseLayer(in_channels, tokens);
    linear2 = new DenseLayer(in_channels, token_channels);
  }

  std::vector<Tensor> forward(const std::vector<Tensor> &inputs) override
  {
    Tensor x = inputs[0]; // [N, HW, C]
    input_cache = x;

    int N = x.shape[0];
    int HW = x.shape[1];
    int C = x.shape[2];

    Tensor x_flat = x.reshape({N * HW, C});
    Tensor a_flat = linear1->forward({x_flat})[0];       // [N*HW, L]
    Tensor a = a_flat.reshape({N, HW, a_flat.shape[1]}); // [N, HW, L]
    cache1 = a;

    a = a.softmax(1); // softmax sobre HW
    cache2 = a;

    a = a.transpose({0, 2, 1}); // [N, L, HW]
    a = a.matmul(x);            // [N, L, C]

    Tensor a_flat2 = a.reshape({N * a.shape[1], C});                   // [N*L, C]
    Tensor out_flat = linear2->forward({a_flat2})[0];                  // [N*L, D]
    Tensor out = out_flat.reshape({N, a.shape[1], out_flat.shape[1]}); // [N, L, D]

    token_cache = out;
    return {out};
  }

  void backward2(const Tensor &delta_tokens, const Tensor &delta_x)
  {
    // Retrieve cached values from forward pass
    Tensor x = input_cache;        // [N, HW, C]
    Tensor a_pre_softmax = cache1; // [N, HW, L]
    Tensor a_weights = cache2;     // [N, HW, L] (after softmax)
    Tensor out = token_cache;      // [N, L, D]

    const int N = x.shape[0];             // batch size (128)
    const int HW = x.shape[1];            // spatial dim (784)
    const int C = x.shape[2];             // input channels (4)
    const int L = a_pre_softmax.shape[2]; // num tokens (16)
    const int D = out.shape[2];           // token channels (16)

    // Initialize input deltas
    input_deltas = Tensor({N, HW, C});
    input_deltas.fill(0.0f); // Initialize to zero

    // ----------------------------------------------------------
    // 1. Backprop through linear2 (token projection)
    // ----------------------------------------------------------
    if (!delta_tokens.data.empty())
    {
      Tensor delta_tokens_flat = delta_tokens.reshape({N * L, D}); // [2048, 16]
      linear2->backward(&delta_tokens_flat);
      Tensor delta_a_flat = linear2->get_input_deltas(); // [2048, 4]
      Tensor delta_a = delta_a_flat.reshape({N, L, C});  // [128, 16, 4]

      // Backprop through attention matmul: a.matmul(x)
      // a_weights shape: [N, HW, L]
      // x shape: [N, HW, C]

      // Gradient for attention weights
      Tensor delta_weights = delta_a.matmul(x.transpose({0, 2, 1})); // [N, L, HW]
      delta_weights = delta_weights.transpose({0, 2, 1});            // [N, HW, L]

      // Gradient for x (from attention path)
      Tensor delta_x_att = a_weights.matmul(delta_a); // [N, HW, C]
      input_deltas = input_deltas + delta_x_att;

      // Backprop through softmax
      Tensor delta_pre_softmax = Tensor::softmax_backward(a_weights, delta_weights); // [N, HW, L]

      // Backprop through linear1
      Tensor delta_pre_softmax_flat = delta_pre_softmax.reshape({N * HW, L});
      linear1->backward(&delta_pre_softmax_flat);
      Tensor delta_x_linear1 = linear1->get_input_deltas().reshape({N, HW, C});
      input_deltas = input_deltas + delta_x_linear1;
    }

    // ----------------------------------------------------------
    // 2. Backprop through direct input path (if delta_x provided)
    // ----------------------------------------------------------
    if (!delta_x.data.empty())
    {
      input_deltas = input_deltas + delta_x;
    }
  }

  void backward(const Tensor *targets, const Layer *next_layer) override
  {
    // Obtener el delta de entrada (gradiente respecto a x)
    Tensor delta_x;
    if (targets)
    {
      delta_x = *targets; // Shape [N, HW, C] (128, 784, 4)
    }
    else if (next_layer)
    {
      delta_x = next_layer->get_input_deltas();
    }
    else
    {
      throw std::runtime_error("FilterTokenizer: No gradient source");
    }

    // Recuperar valores guardados durante el forward
    Tensor x = input_cache;        // [N, HW, C] (128, 784, 4)
    Tensor a_pre_softmax = cache1; // [N, HW, L]
    Tensor a_weights = cache2;     // [N, HW, L] (post softmax)
    Tensor tokens = token_cache;   // [N, L, D]

    const int N = x.shape[0];
    const int HW = x.shape[1];
    const int C = x.shape[2];
    const int L = a_pre_softmax.shape[2];
    const int D = tokens.shape[2];

    // ----------------------------------------------------------
    // 1. Backward a través de la segunda linear (linear2)
    // ----------------------------------------------------------
    // En forward: linear2 transformó [N*L, C] -> [N*L, D]
    // Necesitamos calcular el gradiente respecto a sus inputs

    // Como estamos recibiendo delta_x, necesitamos reconstruir el gradiente
    // que debería llegar a linear2 basado en cómo x afecta a los tokens

    // Esto requiere calcular cómo cambian los tokens cuando cambia x
    Tensor delta_a = a_weights.transpose({0, 2, 1}).matmul(delta_x); // [N, L, C]

    Tensor delta_a_flat = delta_a.reshape({N * L, C});
    linear2->backward(&delta_a_flat);
    Tensor delta_linear2_input = linear2->get_input_deltas(); // [N*L, C]

    // ----------------------------------------------------------
    // 2. Backward de la multiplicación atención (a.matmul(x))
    // ----------------------------------------------------------
    // a_weights: [N, HW, L] (post softmax)
    // En forward: tokens = a_weights^T * x

    // Gradiente respecto a los pesos de atención
    Tensor delta_weights = delta_x.matmul(tokens.transpose({0, 2, 1})); // [N, HW, L]

    // Gradiente respecto a x (contribución desde atención)
    Tensor delta_x_att = a_weights.matmul(tokens); // [N, HW, C]

    // ----------------------------------------------------------
    // 3. Backward del softmax sobre HW
    // ----------------------------------------------------------
    Tensor delta_pre_softmax = a_weights * (delta_weights -
                                            (a_weights * delta_weights).sum(1, true)); // [N, HW, L]

    // ----------------------------------------------------------
    // 4. Backward a través de linear1
    // ----------------------------------------------------------
    Tensor delta_pre_softmax_flat = delta_pre_softmax.reshape({N * HW, L});
    linear1->backward(&delta_pre_softmax_flat);
    Tensor delta_x_linear1 = linear1->get_input_deltas().reshape({N, HW, C});

    // ----------------------------------------------------------
    // 5. Combinar todos los gradientes respecto a x
    // ----------------------------------------------------------
    input_deltas = delta_x_att + delta_x_linear1;
  }

  const Tensor &get_input_deltas() const override
  {
    return input_deltas;
  }

  void update_weights(float batch_size) override
  {
    linear1->update_weights(batch_size);
    linear2->update_weights(batch_size);
  }

  void zero_grad() override
  {
    linear1->zero_grad();
    linear2->zero_grad();
  }

  bool has_weights() const override { return true; }

  void set_training(bool train) override
  {
    is_training = train;
    linear1->set_training(train);
    linear2->set_training(train);
  }

  ~FilterTokenizer()
  {
    delete linear1;
    delete linear2;
  }
};

// --- RecurrentTokenizer ---
class RecurrentTokenizer : public Layer
{
private:
  int token_channels;
  DenseLayer *linear1, *linear2;
  Tensor cache1, cache2, token_cache, input_deltas;

public:
  RecurrentTokenizer(int in_channels, int token_channels)
      : token_channels(token_channels)
  {
    linear1 = new DenseLayer(token_channels, token_channels);
    linear2 = new DenseLayer(in_channels, token_channels);
  }

  std::vector<Tensor> forward(const std::vector<Tensor> &inputs) override
  {
    Tensor x = inputs[0]; // [N, HW, C]
    Tensor t = inputs[1]; // [N, L, D]

    Tensor a = linear1->forward({t})[0]; // [N, L, D]
    Tensor b = linear2->forward({x})[0]; // [N, HW, D]

    a = a.transpose({0, 2, 1}); // [N, D, L]
    a = b.matmul(a);            // [N, HW, L]
    cache1 = a;

    a = a.softmax(1); // [N, HW, L]
    cache2 = a;

    a = a.transpose({0, 2, 1}); // [N, L, HW]
    Tensor out = a.matmul(b);   // [N, L, D]
    token_cache = out;
    return {out};
  }

  void backward(const Tensor *targets, const Layer *next_layer) override
  {
    const Tensor &delta = next_layer->get_input_deltas(); // [N, L, D]

    Tensor b = linear2->get_last_input();      // [N, HW, D]
    Tensor attn = cache2;                      // [N, HW, L]
    Tensor attn_T = attn.transpose({0, 2, 1}); // [N, L, HW]

    // Paso 1: backward del último matmul: attn_T.matmul(b)
    Tensor db_from_matmul = attn_T.transpose({0, 2, 1}).matmul(delta); // [N, HW, D]
    Tensor d_attn_T = delta.matmul(b.transpose({0, 2, 1}));                 // [N, L, HW]
    Tensor d_attn = d_attn_T.transpose({0, 2, 1});                     // [N, HW, L]

    // Paso 2: softmax backward
    Tensor d_softmax_input = Tensor::softmax_backward(cache1, d_attn); // [N, HW, L]

    // Paso 3: backward del matmul: b.matmul(a)
    Tensor a_transposed = linear1->get_last_input().transpose({0, 2, 1});       // [N, D, L]
    Tensor db_from_attn = d_softmax_input.matmul(a_transposed.transpose({0, 2, 1})); // [N, HW, D]
    Tensor da = b.transpose({0, 2, 1}).matmul(d_softmax_input);                 // [N, D, L]
    Tensor da_T = da.transpose({0, 2, 1});                                      // [N, L, D]

    // Paso 4: backward sobre linear1 (de t)
    linear1->backward(&da_T, nullptr);
    Tensor dt = linear1->get_input_deltas(); // [N, L, D]

    // Paso 5: backward sobre linear2 (de x)
    Tensor db_total = db_from_matmul + db_from_attn;
    linear2->backward(&db_total, nullptr);
    Tensor dx = linear2->get_input_deltas(); // [N, HW, C]

    input_deltas = dx; // solo se retorna el gradiente con respecto a x
  }

  void update_weights(float batch_size) override
  {
    linear1->update_weights(batch_size);
    linear2->update_weights(batch_size);
  }

  void zero_grad() override
  {
    linear1->zero_grad();
    linear2->zero_grad();
  }

  bool has_weights() const override { return true; }

  void set_training(bool train) override
  {
    is_training = train;
    linear1->set_training(train);
    linear2->set_training(train);
  }

  const Tensor &get_input_deltas() const override
  {
    return input_deltas;
  }

  ~RecurrentTokenizer()
  {
    delete linear1;
    delete linear2;
  }
};