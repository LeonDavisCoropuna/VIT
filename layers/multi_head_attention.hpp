#pragma once
#include "layer.hpp"

class MultiHeadAttention : public Layer
{
private:
  int num_heads; // Número de cabezas de atención
  int channels;  // Dimensión de entrada (D)
  int head_dim;  // Dimensión por cabeza (attn_dim / num_heads)

  std::vector<DenseLayer *> query_linears; // Una capa por cabeza para query
  std::vector<DenseLayer *> key_linears;   // Una capa por cabeza para key
  std::vector<DenseLayer *> value_linears; // Una capa por cabeza para value
  DenseLayer *output_linear;               // Capa lineal para proyección final (W^O)

  Tensor last_input;                    // Almacena la entrada para backward
  Tensor input_deltas;                  // Almacena los deltas de entrada
  std::vector<Tensor> delta_query_flat; // Deltas para query por cabeza
  std::vector<Tensor> delta_key_flat;   // Deltas para key por cabeza
  std::vector<Tensor> delta_value_flat; // Deltas para value por cabeza
  Tensor delta_output_flat;             // Deltas para la proyección final

public:
  MultiHeadAttention(int channels, int attn_dim, int num_heads)
      : channels(channels), num_heads(num_heads), head_dim(attn_dim / num_heads)
  {
    if (attn_dim % num_heads != 0)
    {
      throw std::runtime_error("attn_dim debe ser divisible por num_heads");
    }

    // Inicializar capas lineales para cada cabeza
    for (int i = 0; i < num_heads; ++i)
    {
      query_linears.push_back(new DenseLayer(channels, head_dim, false));
      key_linears.push_back(new DenseLayer(channels, head_dim, false));
      value_linears.push_back(new DenseLayer(channels, head_dim, false));

      query_linears[i]->set_optimizer(new SGD(0.001f));
      key_linears[i]->set_optimizer(new SGD(0.001f));
      value_linears[i]->set_optimizer(new SGD(0.001f));
    }

    // Capa lineal para la proyección final
    output_linear = new DenseLayer(attn_dim, channels, false);
    output_linear->set_optimizer(new SGD(0.001f));

    delta_query_flat.resize(num_heads);
    delta_key_flat.resize(num_heads);
    delta_value_flat.resize(num_heads);
  }

  ~MultiHeadAttention()
  {
    for (int i = 0; i < num_heads; ++i)
    {
      delete query_linears[i];
      delete key_linears[i];
      delete value_linears[i];
    }
    delete output_linear;
  }

  std::vector<Tensor> forward(const std::vector<Tensor> &inputs) override
  {
    const Tensor &x = inputs[0]; // [B, T, D]
    last_input = x;

    int B = x.shape[0];
    int T = x.shape[1];
    int D = x.shape[2];

    Tensor x_flat = x.reshape({B * T, D}); // [B*T, D]
    std::vector<Tensor> head_outputs(num_heads);

    for (int h = 0; h < num_heads; ++h)
    {
      // Proyecciones lineales para query, key y value
      Tensor query_flat = query_linears[h]->forward({x_flat})[0]; // [B*T, head_dim]
      Tensor key_flat = key_linears[h]->forward({x_flat})[0];     // [B*T, head_dim]
      Tensor value_flat = value_linears[h]->forward({x_flat})[0]; // [B*T, head_dim]

      // Reshape a [B, T, head_dim]
      Tensor query = query_flat.reshape({B, T, head_dim});
      Tensor key = key_flat.reshape({B, T, head_dim});
      Tensor value = value_flat.reshape({B, T, head_dim});

      // Calcular atención para la cabeza h
      Tensor query_T = query.transpose({0, 2, 1});                        // [B, head_dim, T]
      Tensor scores = key.matmul(query_T);                                // [B, T, T]
      scores = scores * (1.0f / std::sqrt(static_cast<float>(head_dim))); // Escala
      Tensor attn = scores.softmax(2);                                    // [B, T, T]
      Tensor head_out = attn.matmul(value);                               // [B, T, head_dim]

      head_outputs[h] = head_out;
    }

    // Concatenar las salidas de todas las cabezas

    Tensor concat_output = Tensor::concat(head_outputs, 2); // Concatenar en la dimensión 2 (canales)
    // Proyección final
    Tensor concat_flat = concat_output.reshape({B * T, head_dim * num_heads});
    Tensor output_flat = output_linear->forward({concat_flat})[0]; // [B*T, D]
    Tensor output = output_flat.reshape({B, T, D});                // [B, T, D]

    return {output};
  }

  void backward(const Tensor *targets = nullptr, const Layer *next_layer = nullptr) override
  {
    // Obtener el delta de la capa siguiente
    Tensor delta;
    if (targets != nullptr)
    {
      delta = *targets; // [B, T, D]
    }
    else if (next_layer != nullptr)
    {
      delta = next_layer->get_input_deltas();
    }
    else
    {
      throw std::runtime_error("MultiHeadAttention: No hay fuente de delta para backward.");
    }

    // Dimensiones
    int B = delta.shape[0];
    int T = delta.shape[1];
    int D = delta.shape[2];

    // Backward de la proyección final
    Tensor delta_flat = delta.reshape({B * T, D});
    output_linear->backward(&delta_flat);
    Tensor delta_concat_flat = output_linear->get_input_deltas(); // [B*T, head_dim * num_heads]
    Tensor delta_concat = delta_concat_flat.reshape({B, T, head_dim * num_heads});

    // Backward por cada cabeza
    std::vector<Tensor> delta_heads(num_heads);
    for (int h = 0; h < num_heads; ++h)
    {
      // Extraer el delta correspondiente a la cabeza h
      Tensor delta_head = delta_concat.slice({0, 0, h * head_dim}, {B, T, (h + 1) * head_dim}); // [B, T, head_dim]

      // Recalcular valores del forward para esta cabeza
      Tensor x_flat = last_input.reshape({B * T, D});
      Tensor query_flat = query_linears[h]->forward({x_flat})[0]; // [B*T, head_dim]
      Tensor key_flat = key_linears[h]->forward({x_flat})[0];     // [B*T, head_dim]
      Tensor value_flat = value_linears[h]->forward({x_flat})[0]; // [B*T, head_dim]

      Tensor query = query_flat.reshape({B, T, head_dim});
      Tensor key = key_flat.reshape({B, T, head_dim});
      Tensor value = value_flat.reshape({B, T, head_dim});

      Tensor query_T = query.transpose({0, 2, 1});                        // [B, head_dim, T]
      Tensor scores = key.matmul(query_T);                                // [B, T, T]
      scores = scores * (1.0f / std::sqrt(static_cast<float>(head_dim))); // Escala
      Tensor attn = scores.softmax(2);                                    // [B, T, T]

      // Backward de attn.matmul(value)
      Tensor value_T = value.transpose({0, 2, 1});                       // [B, head_dim, T]
      Tensor delta_attn = delta_head.matmul(value_T);                    // [B, T, T]
      Tensor delta_value = attn.transpose({0, 2, 1}).matmul(delta_head); // [B, T, head_dim]

      // Backward del softmax
      Tensor delta_scores = Tensor::softmax_backward(attn, delta_attn); // [B, T, T]

      // Backward de key.matmul(query_T)
      Tensor delta_key = delta_scores.matmul(query);                        // [B, T, head_dim]
      Tensor delta_query_T = key.transpose({0, 2, 1}).matmul(delta_scores); // [B, head_dim, T]
      Tensor delta_query = delta_query_T.transpose({0, 2, 1});              // [B, T, head_dim]

      // Backward de las capas lineales
      delta_key_flat[h] = delta_key.reshape({B * T, head_dim});
      key_linears[h]->backward(&delta_key_flat[h]);

      delta_query_flat[h] = delta_query.reshape({B * T, head_dim});
      query_linears[h]->backward(&delta_query_flat[h]);

      delta_value_flat[h] = delta_value.reshape({B * T, head_dim});
      value_linears[h]->backward(&delta_value_flat[h]);
    }

    // Combinar los deltas de las capas lineales
    Tensor delta_x = Tensor({B, T, D});
    for (int h = 0; h < num_heads; ++h)
    {
      delta_x = delta_x + query_linears[h]->get_input_deltas().reshape({B, T, D});
      delta_x = delta_x + key_linears[h]->get_input_deltas().reshape({B, T, D});
      delta_x = delta_x + value_linears[h]->get_input_deltas().reshape({B, T, D});
    }

    input_deltas = delta_x;
  }

  void update_weights(float batch_size) override
  {
    for (int h = 0; h < num_heads; ++h)
    {
      query_linears[h]->update_weights(batch_size);
      key_linears[h]->update_weights(batch_size);
      value_linears[h]->update_weights(batch_size);
    }
    output_linear->update_weights(batch_size);
  }

  void zero_grad() override
  {
    for (int h = 0; h < num_heads; ++h)
    {
      delta_query_flat[h].fill(0.0f);
      delta_key_flat[h].fill(0.0f);
      delta_value_flat[h].fill(0.0f);
      query_linears[h]->zero_grad();
      key_linears[h]->zero_grad();
      value_linears[h]->zero_grad();
    }
    output_linear->zero_grad();
  }

  void set_training(bool is_training) override
  {
    for (int h = 0; h < num_heads; ++h)
    {
      query_linears[h]->set_training(is_training);
      key_linears[h]->set_training(is_training);
      value_linears[h]->set_training(is_training);
    }
    output_linear->set_training(is_training);
  }

  const Tensor &get_input_deltas() const override { return input_deltas; }
};