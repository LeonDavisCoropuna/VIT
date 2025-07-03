#pragma once
#include "layer.hpp"

class ProjectorLayer : public Layer
{
private:
  int in_channels, out_channels, token_channels;
  DenseLayer *linear1, *linear2, *linear3;
  BatchNormLayer1D *norm;
  ReLULayer *relu;
  DenseLayer *downsample;
  Tensor cache, input_deltas;
  Tensor x_cache;
  Tensor token_cache;

  Tensor token_deltas; // Nuevo miembro para almacenar los deltas de los tokens

public:
  ProjectorLayer(int in_channels, int out_channels, int token_channels)
      : in_channels(in_channels), out_channels(out_channels), token_channels(token_channels)
  {

    linear1 = new DenseLayer(in_channels, token_channels, false);    // feature map to query
    linear2 = new DenseLayer(token_channels, token_channels, false); // tokens to key
    linear3 = new DenseLayer(token_channels, out_channels);          // tokens to value

    norm = new BatchNormLayer1D(out_channels);
    relu = new ReLULayer();
    downsample = nullptr;

    if (in_channels != out_channels)
    {
      downsample = new DenseLayer(in_channels, out_channels);
    }

    linear1->set_optimizer(new SGD(0.001f));
    linear2->set_optimizer(new SGD(0.001f));
    linear3->set_optimizer(new SGD(0.001f));
    norm->set_optimizer(new SGD(0.001f));
    downsample->set_optimizer(new SGD(0.001f));
  }

  std::vector<Tensor> forward(const std::vector<Tensor> &inputs) override
  {
    Tensor x = inputs[0]; // (N, HW, C_in) = (128, 784, 4)
    Tensor t = inputs[1]; // (N, L, D)     = (128, 16, 16)

    // Guardar para backward
    x_cache = x;
    token_cache = t;

    // -----------------------
    // 1. Shapes e info
    // -----------------------
    int N = x.shape[0];
    int HW = x.shape[1];
    int C_in = x.shape[2];

    int L = t.shape[1];
    int D = t.shape[2];

    // -----------------------
    // 2. Linear(x): (N*HW, C_in) -> (N, HW, C_out)
    // -----------------------
    Tensor x_flat = x.reshape({N * HW, C_in});                 // [N*HW, C_in]
    Tensor x_q_flat = linear1->forward({x_flat})[0];           // [N*HW, C_out]
    Tensor x_q = x_q_flat.reshape({N, HW, x_q_flat.shape[1]}); // [N, HW, C_out]

    // -----------------------
    // 3. Linear(t): (N*L, D) -> (N, L, C_out)
    // -----------------------
    Tensor t_flat = t.reshape({N * L, D});                    // [N*L, D]
    Tensor t_q_flat = linear2->forward({t_flat})[0];          // [N*L, C_out]
    Tensor t_q = t_q_flat.reshape({N, L, t_q_flat.shape[1]}); // [N, L, C_out]

    // -----------------------
    // 4. Attention: (N, HW, C_out) × (N, C_out, L) -> (N, HW, L)
    // -----------------------
    t_q = t_q.transpose({0, 2, 1}); // [N, C_out, L]
    Tensor a = x_q.matmul(t_q);     // [N, HW, L]
    a = a.softmax(2);               // [N, HW, L]
    cache = a;

    // -----------------------
    // 5. Re-proyectar t: (N, L, D) -> (N, L, C_out)
    // -----------------------
    Tensor t_proj_flat = linear3->forward({t_flat})[0];                // [N*L, C_out]
    Tensor t_proj = t_proj_flat.reshape({N, L, t_proj_flat.shape[1]}); // [N, L, C_out]

    // -----------------------
    // 6. Atención aplicada: (N, HW, L) × (N, L, C_out) -> (N, HW, C_out)
    // -----------------------
    Tensor a_out = a.matmul(t_proj); // [N, HW, C_out]

    // -----------------------
    // 7. Residual + norm + activación
    // -----------------------
    if (downsample != nullptr)
    {
      Tensor x_ds_flat = downsample->forward({x_flat})[0]; // [N*HW, C_out]
      x = x_ds_flat.reshape({N, HW, x_ds_flat.shape[1]});  // [N, HW, C_out]
    }

    x = x + a_out; // Residual connection

    Tensor norm_input = x.reshape({N * HW, x.shape[2]}); // [N * HW, C_out]

    // Aplicar BatchNorm1D
    Tensor norm_output = norm->forward({norm_input})[0]; // [N * HW, C_out]

    // Restaurar a [N, HW, C_out]
    x = norm_output.reshape({N, HW, x.shape[2]}); // [N, HW, C_out]
    x = relu->forward({x})[0];

    return {x};
  }

  void backward(const Tensor *targets = nullptr, const Layer *next_layer = nullptr) override
  {
    // Paso 1: Backward ReLU (ya implementado en el código proporcionado)
    relu->backward(nullptr, next_layer);
    Tensor delta = relu->get_input_deltas(); // [N, C, H, W] (128, 8, 28, 28)

    // Reshape delta a la forma esperada [N, HW, C_out]
    int N = delta.shape[0];
    int C_out = delta.shape[1];
    int H = delta.shape[2];
    int W = delta.shape[3];
    int HW = H * W;

    delta = delta.reshape({N, HW, C_out});

    // Paso 2: Backward BatchNorm
    Tensor norm_delta = delta.reshape({N * HW, C_out});
    norm->backward(&norm_delta, nullptr);
    Tensor norm_input_delta = norm->get_input_deltas();

    // Paso 3: Backward residual connection
    Tensor residual_delta = norm_input_delta.reshape({N, HW, C_out});

    // Derivada de la suma es 1 para ambos caminos
    Tensor a_out_delta = residual_delta;
    Tensor x_delta = residual_delta;

    // Paso 4: Backward downsample (si existe)
    if (downsample != nullptr)
    {
      Tensor x_flat = x_cache.reshape({N * HW, in_channels});
      Tensor x_ds_delta_flat = x_delta.reshape({N * HW, out_channels});
      downsample->backward(&x_ds_delta_flat);
      Tensor x_ds_input_delta = downsample->get_input_deltas();
      input_deltas = x_ds_input_delta.reshape({N, HW, in_channels});
    }
    else
    {
      input_deltas = x_delta;
    }

    // Paso 5: Backward atención aplicada (a.matmul(t_proj))
    Tensor a = cache; // [N, HW, L]
    Tensor t_proj_flat = linear3->forward({token_cache.reshape({N * token_cache.shape[1], token_channels})})[0];
    Tensor t_proj = t_proj_flat.reshape({N, token_cache.shape[1], out_channels}); // [N, L, C_out]

    // Derivada respecto a 'a'
    Tensor a_delta = a_out_delta.matmul(t_proj.transpose({0, 2, 1})); // [N, HW, L]

    // Derivada respecto a 't_proj'
    Tensor t_proj_delta = a.transpose({0, 2, 1}).matmul(a_out_delta); // [N, L, C_out]

    // Paso 6: Backward softmax y atención
    Tensor softmax_delta = Tensor::softmax_backward(a, a_delta);

    // Paso 7: Backward de x_q.matmul(t_q)
    Tensor x_q = linear1->forward({x_cache.reshape({N * HW, in_channels})})[0].reshape({N, HW, token_channels});
    Tensor t_q = linear2->forward({token_cache.reshape({N * token_cache.shape[1], token_channels})})[0]
                     .reshape({N, token_cache.shape[1], token_channels})
                     .transpose({0, 2, 1}); // [N, C_out, L]

    // Derivada respecto a x_q
    Tensor x_q_delta = softmax_delta.matmul(t_q.transpose({0, 2, 1})); // [N, HW, C_out]

    // Derivada respecto a t_q
    Tensor t_q_delta = x_q.transpose({0, 2, 1}).matmul(softmax_delta); // [N, C_out, L]
    t_q_delta = t_q_delta.transpose({0, 2, 1});                        // [N, L, C_out]

    // Paso 8: Backward linear3 (t_proj)
    Tensor t_proj_delta_flat = t_proj_delta.reshape({N * token_cache.shape[1], out_channels});
    linear3->backward(&t_proj_delta_flat);

    // Obtener el delta de los tokens desde linear3
    Tensor t_proj_input_delta = linear3->get_input_deltas(); // [N*L, token_channels]

    // Paso 9: Backward linear2 (t_q)
    Tensor t_q_delta_flat = t_q_delta.reshape({N * token_cache.shape[1], token_channels});
    linear2->backward(&t_q_delta_flat);

    // Obtener el delta de los tokens desde linear2
    Tensor t_q_input_delta = linear2->get_input_deltas(); // [N*L, token_channels]

    // Combinar los deltas de ambas rutas (linear2 y linear3)
    token_deltas = t_proj_input_delta + t_q_input_delta;
    token_deltas = token_deltas.reshape({N, token_cache.shape[1], token_channels}); // [N, L, D]

    // Paso 10: Backward linear1 (x_q)
    Tensor x_q_delta_flat = x_q_delta.reshape({N * HW, token_channels});
    linear1->backward(&x_q_delta_flat);

    // Sumar delta del camino directo (x) y del camino de atención (x_q)
    if (downsample != nullptr)
    {
      Tensor x_q_input_delta = linear1->get_input_deltas().reshape({N, HW, in_channels});
      input_deltas = input_deltas + x_q_input_delta;
    }
  }

  void update_weights(float batch_size) override
  {
    linear1->update_weights(batch_size);
    linear2->update_weights(batch_size);
    linear3->update_weights(batch_size);
    norm->update_weights(batch_size);
    if (downsample)
      downsample->update_weights(batch_size);
  }

  void zero_grad() override
  {
    linear1->zero_grad();
    linear2->zero_grad();
    linear3->zero_grad();
    norm->zero_grad();
    if (downsample)
      downsample->zero_grad();
  }

  void set_training(bool training) override
  {
    is_training = training;
    linear1->set_training(training);
    linear2->set_training(training);
    linear3->set_training(training);
    norm->set_training(training);
    if (downsample)
      downsample->set_training(training);
  }

  const Tensor &get_input_deltas() const override
  {
    return input_deltas;
  }
  Tensor get_token_deltas() const
  {
    return token_deltas;
  }

  bool has_weights() const override { return true; }

  ~ProjectorLayer()
  {
    delete linear1;
    delete linear2;
    delete linear3;
    delete norm;
    if (downsample)
      delete downsample;
  }
};
