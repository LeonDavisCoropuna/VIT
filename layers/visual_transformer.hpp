#pragma once
#include "layer.hpp"
#include "projector_layer.hpp"
#include "tokenizer.hpp"
#include "transformer_layer.hpp"

#include "../layers/conv2d_layer.hpp"
#include "../layers/batch_normalization_layer.hpp"
#include "../layers/visual_transformer.hpp"
#include "../layers/dense_layer.hpp"
#include "../layers/pooling_layer.hpp"
#include "../layers/flatten_layer.hpp"

class VisualTransformer : public Layer
{
private:
  int in_channels, out_channels, token_channels, tokens, attn_dim;
  bool is_projected;
  std::string tokenizer_type;

  FilterTokenizer *tokenizer;
  TransformerLayer *transformer;
  ProjectorLayer *projector;

  Tensor input_cache;
  Tensor token_cache;

  Tensor input_deltas;

public:
  VisualTransformer(int in_channels, int out_channels, int token_channels, int tokens,
                    const std::string &tokenizer_type, int attn_dim, int transformer_enc_layers,
                    int transformer_heads, int transformer_fc_dim, float transformer_dropout,
                    bool is_projected = true)
      : in_channels(in_channels), out_channels(out_channels), token_channels(token_channels),
        tokens(tokens), attn_dim(attn_dim), is_projected(is_projected), tokenizer_type(tokenizer_type)
  {

    if (tokenizer_type == "filter")
    {
      tokenizer = new FilterTokenizer(in_channels, token_channels, tokens);
    }
    else if (tokenizer_type == "recurrent")
    {
      // tokenizer = new RecurrentTokenizer(in_channels, token_channels);
    }
    else
    {
      throw std::runtime_error("Unknown tokenizer type: " + tokenizer_type);
    }

    transformer = new TransformerLayer(token_channels, attn_dim, transformer_heads);

    if (is_projected)
    {
      projector = new ProjectorLayer(in_channels, out_channels, token_channels);
    }
    else
    {
      projector = nullptr;
    }

    tokenizer->set_optimizer(new SGD(0.001f));
    transformer->set_optimizer(new SGD(0.001f));
    projector->set_optimizer(new SGD(0.001f));
  }

  ~VisualTransformer() override
  {
    delete tokenizer;
    delete transformer;
    if (projector)
      delete projector;
  }

  std::vector<Tensor> forward(const std::vector<Tensor> &inputs) override
  {
    Tensor x = inputs[0];
    Tensor t;

    input_cache = x;

    if (tokenizer_type == "filter")
    {
      t = tokenizer->forward({x})[0];
    }
    else
    {
      t = tokenizer->forward({x, inputs[1]})[0];
    }

    // (N, L, C) -> (L, N, C)
    t = t.permute({1, 0, 2});

    // Transformer
    t = transformer->forward({t})[0];

    // (L, N, C) -> (N, L, C)
    t = t.permute({1, 0, 2});

    token_cache = t;
    Tensor out;
    if (is_projected && projector)
    {
      out = projector->forward({x, t})[0];
    }
    return {out, t};
  }

  void backward(const Tensor *targets = nullptr, const Layer *next_layer = nullptr) override
  {
    // Paso 1: Backward del projector (si existe)
    Tensor delta_x; // Gradiente respecto a la imagen de entrada (x)
    Tensor delta_t; // Gradiente respecto a los tokens

    if (is_projected && projector)
    {
      // El projector recibe {x, t} y produce out
      // Necesitamos obtener los deltas para ambos inputs
      projector->backward(nullptr, next_layer);

      // El projector debería devolver dos deltas (uno para x y otro para t)
      // Asumiendo que get_input_deltas() devuelve un vector con ambos
      delta_x = projector->get_input_deltas();
      delta_t = projector->get_token_deltas();
    }
    else
    {
      // Si no hay projector, el delta viene directamente de la siguiente capa
      delta_t = next_layer->get_input_deltas();
      delta_x = Tensor(); // Inicializar como tensor vacío
    }

    // Paso 2: Backward del transformer
    // Preparamos los tokens para el transformer (L, N, C)
    Tensor t_permuted = token_cache.permute({1, 0, 2});

    // El transformer espera el delta en su formato de salida (L, N, C)
    Tensor delta_t_permuted = delta_t.permute({1, 0, 2});
    transformer->backward(&delta_t_permuted);

    // Obtenemos el delta de salida del transformer
    Tensor delta_transformer_out = transformer->get_input_deltas();

    // Paso 3: Backward del tokenizer
    // Sumamos los deltas que vienen del transformer y los que vienen del projector (si aplica)
    if (!delta_x.data.empty())
    {
      // Si tenemos delta_x del projector, lo pasamos al tokenizer
      // Paso 3: Backward del tokenizer
      if (tokenizer_type == "filter")
      {
        // FilterTokenizer solo necesita delta_x (respecto a la imagen)
        if (!delta_x.data.empty())
        {
          tokenizer->backward2(delta_t, delta_x);
        }
        else
        {
          // Si no hay delta_x (ruta sin projector)
          // Solo pasamos el gradiente que viene del transformer
          Tensor delta_transformer_final = delta_transformer_out.permute({1, 0, 2});
          tokenizer->backward(&delta_transformer_final, nullptr);
        }
      }
      // else
      // {
      //   // Para otros tokenizers que procesan múltiples inputs
      //   tokenizer->backward({delta_x, delta_t});
      // }
    }
    else
    {
      // Si no hay delta_x, solo pasamos los deltas del transformer
      // (dependiendo de cómo funciona tu tokenizer)
      Tensor delta_transformer_final = delta_transformer_out.permute({1, 0, 2});
      tokenizer->backward(&delta_transformer_final, nullptr);
    }

    // Guardar los deltas de entrada si es necesario
    input_deltas = tokenizer->get_input_deltas();
  }

  void update_weights(float batch_size) override
  {
    // Actualizar pesos de todos los componentes
    tokenizer->update_weights(batch_size);
    transformer->update_weights(batch_size);
    if (is_projected && projector)
    {
      projector->update_weights(batch_size);
    }
  }
  void zero_grad() override
  {
    // Reiniciar gradientes de todos los componentes
    tokenizer->zero_grad();
    transformer->zero_grad();
    if (is_projected && projector)
    {
      projector->zero_grad();
    }
  }
  void set_training(bool is_training_) override { is_training = is_training_; }
  const Tensor &get_input_deltas() const override { return input_deltas; }
};
