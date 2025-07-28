#pragma once
#include "all_layers.hpp"

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
    // inputs should contain at least one tensor (the image)
    // inputs size: 1 or 2

    Tensor x = inputs[0];
    Tensor t;

    // x shape (N, C, H, W) : [N, in_channels, H, W]
    // x data length: N * C * H * W

    input_cache = x;

    if (tokenizer_type == "filter")
    {
      // FilterTokenizer solo necesita la imagen de entrada

      t = tokenizer->forward({x})[0];
      // t tiene la forma (N, tokens, token_channels) = [128, 16, 32]
      // t data length: N * tokens * token_channels
    }
    else
    {
      t = tokenizer->forward({x, inputs[1]})[0];
    }

    // (N, L, C) -> (L, N, C)
    // t before permute:  [128, 16, 32]
    // t data length before permute: N * tokens * token_channels
    t = t.permute({1, 0, 2});
    // t after permute:  [16, 128, 32]
    // t data length after permute: tokens * N * token_channels

    // Transformer
    // t before forward: [16, 128, 32] = (tokens, N, token_channels)
    t = transformer->forward({t})[0];
    // t after forward: [16, 128, 32] = (tokens, N, attn_dim)
    // t data length after forward: tokens * N * attn_dim

    // (L, N, C) -> (N, L, C)
    // t before permute: [16, 128, 32]
    t = t.permute({1, 0, 2});
    // t after permute: [128, 16, 32]
    // t data length after permute: N * tokens * attn_dim

    token_cache = t;
    Tensor out;
    // Si hay un projector, lo usamos para proyectar la imagen de entrada y los tokens

    if (is_projected && projector)
    {
      out = projector->forward({x, t})[0];
      // out tiene la forma (N, L, C) = [128, 784, 16]
      // out data length: N * tokens * out_channels
    }

    // out shape: [128, 784, 16]
    // t shape: [128, 16, 32]
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
    if (!delta_x.empty())
    {
      // Si tenemos delta_x del projector, lo pasamos al tokenizer
      // Paso 3: Backward del tokenizer
      if (tokenizer_type == "filter")
      {
        // FilterTokenizer solo necesita delta_x (respecto a la imagen)
        if (!delta_x.empty())
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
  /*
  void set_training(bool is_training_) override
  {
    is_training = is_training_;
    tokenizer->set_training(is_training);
    transformer->set_training(is_training);
    if (is_projected && projector)
      projector->set_training(is_training);
  }
  */
  const Tensor &get_input_deltas() const override { return input_deltas; }

  bool has_weights() const override
  {
    return true;
  }

  std::string get_type() const override
  {
    return "VisualTransformer";
  }

  std::vector<Tensor> get_parameters() const
  {
    std::vector<Tensor> params;

    // Obtener parámetros del tokenizer
    if (tokenizer->has_weights())
    {
      std::vector<Tensor> tokenizer_params = tokenizer->get_parameters();
      params.insert(params.end(), tokenizer_params.begin(), tokenizer_params.end());
    }

    // Obtener parámetros del transformer
    if (transformer->has_weights())
    {
      std::vector<Tensor> transformer_params = transformer->get_parameters();
      params.insert(params.end(), transformer_params.begin(), transformer_params.end());
    }

    // Obtener parámetros del projector (si existe)
    if (is_projected && projector && projector->has_weights())
    {
      std::vector<Tensor> projector_params = projector->get_parameters();
      params.insert(params.end(), projector_params.begin(), projector_params.end());
    }

    return params;
  }

  void set_parameters(const std::vector<Tensor> &new_params)
  {
    size_t param_index = 0;

    // Establecer parámetros del tokenizer
    if (tokenizer->has_weights())
    {
      std::vector<Tensor> tokenizer_params = tokenizer->get_parameters();
      std::vector<Tensor> new_tokenizer_params(
          new_params.begin() + param_index,
          new_params.begin() + param_index + tokenizer_params.size());
      tokenizer->set_parameters(new_tokenizer_params);
      param_index += tokenizer_params.size();
    }

    // Establecer parámetros del transformer
    if (transformer->has_weights())
    {
      std::vector<Tensor> transformer_params = transformer->get_parameters();
      std::vector<Tensor> new_transformer_params(
          new_params.begin() + param_index,
          new_params.begin() + param_index + transformer_params.size());
      transformer->set_parameters(new_transformer_params);
      param_index += transformer_params.size();
    }

    // Establecer parámetros del projector (si existe)
    if (is_projected && projector && projector->has_weights())
    {
      std::vector<Tensor> projector_params = projector->get_parameters();
      std::vector<Tensor> new_projector_params(
          new_params.begin() + param_index,
          new_params.begin() + param_index + projector_params.size());
      projector->set_parameters(new_projector_params);
    }
  }

  void save(std::ostream &out) const
  {
    // Guardar parámetros del tokenizer
    std::string tokenizer_type_str = tokenizer->get_type();
    int tokenizer_type_len = tokenizer_type_str.size();
    out.write(reinterpret_cast<const char *>(&tokenizer_type_len), sizeof(int));
    out.write(tokenizer_type_str.c_str(), tokenizer_type_len);
    tokenizer->save(out);

    // Guardar parámetros del transformer
    std::string transformer_type = transformer->get_type();
    int transformer_type_len = transformer_type.size();
    out.write(reinterpret_cast<const char *>(&transformer_type_len), sizeof(int));
    out.write(transformer_type.c_str(), transformer_type_len);
    transformer->save(out);

    // Guardar parámetros del projector (si existe)
    int has_projector = is_projected && projector ? 1 : 0;
    out.write(reinterpret_cast<const char *>(&has_projector), sizeof(int));
    if (has_projector)
    {
      std::string projector_type = projector->get_type();
      int projector_type_len = projector_type.size();
      out.write(reinterpret_cast<const char *>(&projector_type_len), sizeof(int));
      out.write(projector_type.c_str(), projector_type_len);
      projector->save(out);
    }
  }

  void load(std::istream &in)
  {
    // Cargar parámetros del tokenizer
    int tokenizer_type_len;
    in.read(reinterpret_cast<char *>(&tokenizer_type_len), sizeof(int));
    std::string tokenizer_type_str(tokenizer_type_len, ' ');
    in.read(&tokenizer_type_str[0], tokenizer_type_len);
    if (tokenizer_type_str != tokenizer->get_type())
      throw std::runtime_error("Tipo de tokenizer no coincide: esperado " + tokenizer->get_type() +
                               ", encontrado " + tokenizer_type_str);
    tokenizer->load(in);

    // Cargar parámetros del transformer
    int transformer_type_len;
    in.read(reinterpret_cast<char *>(&transformer_type_len), sizeof(int));
    std::string transformer_type(transformer_type_len, ' ');
    in.read(&transformer_type[0], transformer_type_len);
    if (transformer_type != transformer->get_type())
      throw std::runtime_error("Tipo de transformer no coincide: esperado " + transformer->get_type() +
                               ", encontrado " + transformer_type);
    transformer->load(in);

    // Cargar parámetros del projector (si existe)
    int has_projector;
    in.read(reinterpret_cast<char *>(&has_projector), sizeof(int));
    if (has_projector != (is_projected && projector ? 1 : 0))
      throw std::runtime_error("Configuración de projector no coincide");
    if (has_projector)
    {
      int projector_type_len;
      in.read(reinterpret_cast<char *>(&projector_type_len), sizeof(int));
      std::string projector_type(projector_type_len, ' ');
      in.read(&projector_type[0], projector_type_len);
      if (projector_type != projector->get_type())
        throw std::runtime_error("Tipo de projector no coincide: esperado " + projector->get_type() +
                                 ", encontrado " + projector_type);
      projector->load(in);
    }
  }
};
