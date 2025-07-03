#pragma once
#include <vector>
#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <random>

#include <omp.h>

class Tensor
{
public:
  std::vector<int> shape;
  std::vector<float> data;

  static std::mt19937 global_gen; // Generador global

  static void set_seed(unsigned int seed)
  {
    global_gen.seed(seed);
  }

  Tensor() {}

  Tensor(const std::vector<int> &shape) : shape(shape)
  {
    int total = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    data.resize(total, 0.0f);
  }
  Tensor(const std::vector<int> &shape, std::vector<float> data) : shape(shape), data(data)
  {
    int total = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    data.resize(total, 0.0f);
  }

  // -------------------- Inicialización --------------------
  static Tensor zeros(const std::vector<int> &shape)
  {
    return Tensor(shape); // se llena con 0 por defecto
  }

  static Tensor rand_uniform(const std::vector<int> &shape, float low, float high)
  {
    Tensor t(shape);
    std::uniform_real_distribution<float> dist(low, high);
    for (float &x : t.data)
      x = dist(global_gen);
    return t;
  }

  static Tensor xavier_normal(const std::vector<int> &shape, int fan_in, int fan_out)
  {
    Tensor t(shape);
    float scale = std::sqrt(2.0f / (fan_in + fan_out));
    std::normal_distribution<float> dist(0.0f, scale);
    for (float &x : t.data)
      x = dist(global_gen);
    return t;
  }

  static Tensor xavier_uniform(const std::vector<int> &shape, int fan_in, int fan_out)
  {
    Tensor t(shape);
    float scale = std::sqrt(6.0f / (fan_in + fan_out));
    std::uniform_real_distribution<float> dist(-scale, scale);
    for (float &x : t.data)
      x = dist(global_gen);
    return t;
  }

  static Tensor kaiming_normal(const std::vector<int> &shape, int fan_in, bool use_negative = true)
  {
    Tensor t(shape);
    float scale = std::sqrt(2.0f / fan_in);
    std::normal_distribution<float> dist(0.0f, scale);
    for (float &x : t.data)
      x = dist(global_gen);
    return t;
  }

  static Tensor kaiming_uniform(const std::vector<int> &shape, int fan_in, bool use_negative = true)
  {
    Tensor t(shape);
    float scale = std::sqrt(6.0f / fan_in);
    std::uniform_real_distribution<float> dist(-scale, scale);
    for (float &x : t.data)
      x = dist(global_gen);
    return t;
  }

  static Tensor one_hot(const Tensor &labels, int num_classes)
  {
    if (labels.shape.size() != 2 || labels.shape[1] != 1)
      throw std::invalid_argument("Labels tensor debe tener shape [batch_size, 1]");

    int batch_size = labels.shape[0];
    std::vector<float> one_hot_data(batch_size * num_classes, 0.0f);

    for (int i = 0; i < batch_size; ++i)
    {
      int label = static_cast<int>(labels.data[i]);
      if (label < 0 || label >= num_classes)
        throw std::out_of_range("Label fuera de rango en one_hot");

      one_hot_data[i * num_classes + label] = 1.0f;
    }

    return Tensor({batch_size, num_classes}, one_hot_data);
  }

  Tensor slice(const std::vector<int> &starts, const std::vector<int> &ends) const
  {
    assert(starts.size() == shape.size());
    assert(ends.size() == shape.size());

    // Verificar límites y calcular nueva forma
    std::vector<int> new_shape;
    for (size_t i = 0; i < shape.size(); ++i)
    {
      assert(starts[i] >= 0 && ends[i] <= shape[i] && starts[i] < ends[i]);
      new_shape.push_back(ends[i] - starts[i]);
    }

    // Calcular strides para la forma original
    std::vector<int> strides(shape.size(), 1);
    for (int i = shape.size() - 2; i >= 0; --i)
    {
      strides[i] = strides[i + 1] * shape[i + 1];
    }

    // Calcular tamaño del nuevo tensor
    int total_size = 1;
    for (int dim : new_shape)
    {
      total_size *= dim;
    }
    std::vector<float> new_data(total_size);

// Copiar datos
#pragma omp parallel for collapse(3) schedule(static)
    for (int b = starts[0]; b < ends[0]; ++b)
    {
      for (int t = starts[1]; t < ends[1]; ++t)
      {
        for (int d = starts[2]; d < ends[2]; ++d)
        {
          int old_idx = b * strides[0] + t * strides[1] + d * strides[2];
          int new_idx = (b - starts[0]) * (new_shape[1] * new_shape[2]) +
                        (t - starts[1]) * new_shape[2] +
                        (d - starts[2]);
          new_data[new_idx] = data[old_idx];
        }
      }
    }

    return Tensor(new_shape, new_data);
  }

  Tensor slice(int dim, int start, int end) const
  {
    assert(dim >= 0 && dim < shape.size());
    assert(start >= 0 && end <= shape[dim] && start < end);

    std::vector<int> new_shape = shape;
    new_shape[dim] = end - start;

    int outer = 1;
    int inner = 1;

    for (int i = 0; i < dim; ++i)
      outer *= shape[i];

    for (int i = dim + 1; i < shape.size(); ++i)
      inner *= shape[i];

    std::vector<float> new_data;
    new_data.reserve(outer * (end - start) * inner);

    for (int o = 0; o < outer; ++o)
    {
      for (int i = start; i < end; ++i)
      {
        int base_index = (o * shape[dim] + i) * inner;
        new_data.insert(new_data.end(), data.begin() + base_index, data.begin() + base_index + inner);
      }
    }

    return Tensor(new_shape, new_data);
  }

  static Tensor concat(const std::vector<Tensor> &tensors, int dim)
  {
    assert(!tensors.empty());

    // Verificar que todas las formas son compatibles excepto en 'dim'
    std::vector<int> new_shape = tensors[0].shape;
    int concat_dim_size = new_shape[dim];

    for (size_t i = 1; i < tensors.size(); ++i)
    {
      assert(tensors[i].shape.size() == new_shape.size());
      for (size_t j = 0; j < new_shape.size(); ++j)
      {
        if (j != dim)
        {
          assert(tensors[i].shape[j] == new_shape[j]);
        }
      }
      concat_dim_size += tensors[i].shape[dim];
    }

    new_shape[dim] = concat_dim_size;
    Tensor result(new_shape);

    // Copiar datos
    int offset = 0;
    for (const auto &tensor : tensors)
    {
      std::vector<int> starts(result.shape.size(), 0);
      std::vector<int> ends = result.shape;
      starts[dim] = offset;
      ends[dim] = offset + tensor.shape[dim];

      result.slice(starts, ends) = tensor;
      offset += tensor.shape[dim];
    }

    return result;
  }

  Tensor pad(const std::vector<int> &pads) const
  {
    assert(shape.size() == 4); // Suponemos formato [N, C, H, W]
    assert(pads.size() == 8);  // 2 pads por dimensión: [N_pre, N_post, C_pre, C_post, H_pre, H_post, W_pre, W_post]

    int N = shape[0], C = shape[1], H = shape[2], W = shape[3];
    int pad_N0 = pads[0], pad_N1 = pads[1];
    int pad_C0 = pads[2], pad_C1 = pads[3];
    int pad_H0 = pads[4], pad_H1 = pads[5];
    int pad_W0 = pads[6], pad_W1 = pads[7];

    int new_N = N + pad_N0 + pad_N1;
    int new_C = C + pad_C0 + pad_C1;
    int new_H = H + pad_H0 + pad_H1;
    int new_W = W + pad_W0 + pad_W1;

    std::vector<float> padded_data(new_N * new_C * new_H * new_W, 0.0f);

    for (int n = 0; n < N; ++n)
      for (int c = 0; c < C; ++c)
        for (int h = 0; h < H; ++h)
          for (int w = 0; w < W; ++w)
          {
            int old_idx = ((n * C + c) * H + h) * W + w;
            int new_n = n + pad_N0;
            int new_c = c + pad_C0;
            int new_h = h + pad_H0;
            int new_w = w + pad_W0;
            int new_idx = ((new_n * new_C + new_c) * new_H + new_h) * new_W + new_w;
            padded_data[new_idx] = data[old_idx];
          }

    return Tensor({new_N, new_C, new_H, new_W}, padded_data);
  }

  float at(const std::vector<int> &index) const
  {
    int flat_index = 0;
    int multiplier = 1;
    for (int i = shape.size() - 1; i >= 0; --i)
    {
      flat_index += index[i] * multiplier;
      multiplier *= shape[i];
    }
    return data[flat_index];
  }
  inline float &at(std::initializer_list<int> indices)
  {
    int offset = 0;
    int stride = 1;
    int dims = shape.size();

    assert(indices.size() == dims);

    auto idx_it = indices.end();
    auto shape_it = shape.end();

    for (int i = 0; i < dims; ++i)
    {
      --idx_it;
      --shape_it;
      offset += (*idx_it) * stride;
      stride *= *shape_it;
    }

    return data[offset];
  }

  inline const float &at(std::initializer_list<int> indices) const
  {
    int offset = 0;
    int stride = 1;
    int dims = shape.size();

    assert(indices.size() == dims);

    auto idx_it = indices.end();
    auto shape_it = shape.end();

    for (int i = 0; i < dims; ++i)
    {
      --idx_it;
      --shape_it;
      offset += (*idx_it) * stride;
      stride *= *shape_it;
    }

    return data[offset];
  }

  Tensor flatten(int start_dim) const
  {
    int total_dims = shape.size();
    if (start_dim < 0 || start_dim >= total_dims)
      throw std::invalid_argument("Invalid start_dim in flatten.");

    // Calcular la dimensión colapsada
    int flattened_dim = 1;
    for (int i = start_dim; i < total_dims; ++i)
      flattened_dim *= shape[i];

    // Nueva forma: mantiene las primeras `start_dim` dimensiones y aplana el resto
    std::vector<int> new_shape;
    for (int i = 0; i < start_dim; ++i)
      new_shape.push_back(shape[i]);
    new_shape.push_back(flattened_dim);

    // Retorna nuevo tensor con misma data y nueva forma
    Tensor result = *this;
    result.shape = new_shape;
    return result;
  }

  void fill(float value)
  {
    std::fill(data.begin(), data.end(), value);
  }

  Tensor sum(int axis, bool keepdims = false) const
  {
    if (axis < 0)
      axis += shape.size();

    if (shape.size() == 2 && axis == 0)
    {
      int rows = shape[0];
      int cols = shape[1];
      Tensor result({cols});

      // Inicialización a cero en paralelo
#pragma omp parallel for
      for (int j = 0; j < cols; ++j)
        result.at({j}) = 0.0f;

      // Suma por columnas con reducción manual
#pragma omp parallel
      {
        std::vector<float> local_sum(cols, 0.0f);

#pragma omp for nowait
        for (int i = 0; i < rows; ++i)
          for (int j = 0; j < cols; ++j)
            local_sum[j] += this->at({i, j});

#pragma omp critical
        {
          for (int j = 0; j < cols; ++j)
            result.at({j}) += local_sum[j];
        }
      }

      if (keepdims)
        result.shape = {1, cols};

      return result;
    }
    else if (shape.size() == 3 && axis == 2)
    {
      int N = shape[0];
      int HW = shape[1];
      int L = shape[2];
      Tensor result({N, HW});

      // Inicialización a cero
#pragma omp parallel for collapse(2)
      for (int n = 0; n < N; ++n)
        for (int hw = 0; hw < HW; ++hw)
          result.at({n, hw}) = 0.0f;

      // Reducción por el último eje
#pragma omp parallel for collapse(2)
      for (int n = 0; n < N; ++n)
      {
        for (int hw = 0; hw < HW; ++hw)
        {
          float sum = 0.0f;
#pragma omp simd reduction(+ : sum)
          for (int l = 0; l < L; ++l)
          {
            sum += this->at({n, hw, l});
          }
          result.at({n, hw}) = sum;
        }
      }

      if (keepdims)
        result.shape = {N, HW, 1};

      return result;
    }

    throw std::runtime_error("sum: unsupported shape or axis");
  }
  Tensor sum(const std::vector<int> &dims) const
  {
    // Paso 1: Identificar qué ejes reducir
    std::vector<bool> reduce_axes(shape.size(), false);
    for (int d : dims)
      reduce_axes[d] = true;

    // Paso 2: Calcular la nueva forma (sin los ejes a reducir)
    std::vector<int> new_shape;
    for (size_t i = 0; i < shape.size(); ++i)
      if (!reduce_axes[i])
        new_shape.push_back(shape[i]);

    if (new_shape.empty())
      new_shape.push_back(1);

    // Paso 3: Crear el tensor resultado
    Tensor result(new_shape);
    std::fill(result.data.begin(), result.data.end(), 0.0f);

    // Paso 4: Precomputar strides de input y output
    std::vector<int> in_strides(shape.size());
    in_strides[shape.size() - 1] = 1;
    for (int i = shape.size() - 2; i >= 0; --i)
      in_strides[i] = in_strides[i + 1] * shape[i + 1];

    std::vector<int> out_strides(new_shape.size());
    if (!new_shape.empty())
    {
      out_strides[new_shape.size() - 1] = 1;
      for (int i = new_shape.size() - 2; i >= 0; --i)
        out_strides[i] = out_strides[i + 1] * new_shape[i + 1];
    }

    // Paso 5: Mapeo directo con strides (sin unravel/ravel)
    const int total = data.size();
    const int out_ndims = new_shape.size();

#pragma omp parallel
    {
      std::vector<float> local_result(result.data.size(), 0.0f);

#pragma omp for nowait
      for (int flat_idx = 0; flat_idx < total; ++flat_idx)
      {
        int idx = flat_idx;
        int out_index = 0;
        int out_pos = 0;

        for (size_t i = 0, out_i = 0; i < shape.size(); ++i)
        {
          int coord = idx / in_strides[i];
          idx %= in_strides[i];

          if (!reduce_axes[i])
          {
            out_index += coord * out_strides[out_i];
            ++out_i;
          }
        }

        local_result[out_index] += data[flat_idx];
      }

#pragma omp critical
      {
        for (size_t i = 0; i < result.data.size(); ++i)
          result.data[i] += local_result[i];
      }
    }

    return result;
  }

  // Tensor::mean: media sobre una dimensión (dim), opcionalmente manteniendo esa dimensión
  Tensor mean(int dim, bool keepdim) const
  {
    assert(dim >= 0 && dim < shape.size());

    int N = shape[dim];
    std::vector<int> new_shape = shape;
    if (keepdim)
      new_shape[dim] = 1;
    else
      new_shape.erase(new_shape.begin() + dim);

    // Crear tensor resultado
    Tensor result(new_shape);
    std::vector<int> index(shape.size(), 0);

    // Total número de elementos del nuevo tensor
    int result_size = result.data.size();

    for (int i = 0; i < result_size; ++i)
    {
      // Calcular el índice base del nuevo tensor
      int count = 0;
      float sum = 0.0f;

      for (int j = 0; j < N; ++j)
      {
        index[dim] = j;
        sum += this->at(index);
        ++count;
      }

      result.data[i] = sum / count;

      // Avanzar índice (sin contar la dim de reducción)
      for (int k = shape.size() - 1; k >= 0; --k)
      {
        if (k == dim)
          continue;
        index[k]++;
        if (index[k] < shape[k])
          break;
        index[k] = 0;
      }
    }

    return result;
  }

  Tensor mean(const std::vector<int> &dims, bool keepdims) const
  {
    Tensor s = this->sum(dims);

    int reduce_size = 1;
    for (int d : dims)
      reduce_size *= shape[d];

    s = s / static_cast<float>(reduce_size);

    if (keepdims)
    {
      // expand shape con 1s en las posiciones reducidas
      std::vector<int> new_shape;
      size_t s_idx = 0;
      for (size_t i = 0; i < shape.size(); ++i)
      {
        if (std::find(dims.begin(), dims.end(), i) != dims.end())
          new_shape.push_back(1);
        else
          new_shape.push_back(s.shape[s_idx++]);
      }
      return s.reshape(new_shape);
    }

    return s;
  }

  Tensor pow(float exponent) const
  {
    Tensor out = *this; // Copia los metadatos (shape, etc.)
    const size_t n = out.data.size();

#pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < n; ++i)
    {
      out.data[i] = std::pow(out.data[i], exponent);
    }
    return out;
  }

  Tensor sqrt() const
  {
    Tensor out = *this;
    const size_t n = out.data.size();

#pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < n; ++i)
    {
      out.data[i] = std::sqrt(out.data[i]);
    }
    return out;
  }

  Tensor reciprocal() const
  {
    Tensor out = *this;
    const size_t n = out.data.size();

#pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < n; ++i)
    {
      out.data[i] = 1.0f / out.data[i];
    }
    return out;
  }

  Tensor transpose(const std::vector<int> &axes) const
  {
    if (axes.size() != shape.size())
      throw std::runtime_error("Número de ejes no coincide con las dimensiones del tensor");

    // Nueva forma (igual que antes)
    std::vector<int> new_shape(shape.size());
    for (size_t i = 0; i < axes.size(); ++i)
      new_shape[i] = shape[axes[i]];

    // Cálculo de strides (igual que antes)
    std::vector<int> old_strides(shape.size());
    old_strides.back() = 1;
    for (int i = shape.size() - 2; i >= 0; --i)
      old_strides[i] = old_strides[i + 1] * shape[i + 1];

    // Cálculo de strides del tensor transpuesto
    std::vector<int> new_strides(shape.size());
    new_strides.back() = 1;
    for (int i = new_shape.size() - 2; i >= 0; --i)
      new_strides[i] = new_strides[i + 1] * new_shape[i + 1];

    // Transponer datos (paralelizado)
    Tensor out(new_shape);
    const size_t total_elements = data.size();

#pragma omp parallel for schedule(guided)
    for (size_t i = 0; i < total_elements; ++i)
    {
      // Índices originales (igual que antes)
      int idx = i;
      std::vector<int> old_indices(shape.size());
      for (size_t j = 0; j < shape.size(); ++j)
      {
        old_indices[j] = idx / old_strides[j];
        idx %= old_strides[j];
      }

      // Reordenar índices (igual que antes)
      std::vector<int> new_indices(shape.size());
      for (size_t j = 0; j < shape.size(); ++j)
        new_indices[j] = old_indices[axes[j]];

      // Calcular nuevo índice lineal (igual que antes)
      int flat_index = 0;
      for (size_t j = 0; j < shape.size(); ++j)
        flat_index += new_indices[j] * new_strides[j];

      out.data[flat_index] = data[i];
    }

    return out;
  }

  Tensor permute(const std::vector<int> &dims) const
  {
    if (dims.size() != shape.size())
      throw std::runtime_error("La cantidad de dimensiones no coincide con el tamaño del tensor.");

    // Verifica que 'dims' sea una permutación válida
    std::vector<bool> seen(shape.size(), false);
    for (int d : dims)
    {
      if (d < 0 || d >= (int)shape.size() || seen[d])
        throw std::runtime_error("Permutación inválida.");
      seen[d] = true;
    }

    // Calcular nueva forma
    std::vector<int> new_shape(shape.size());
    for (size_t i = 0; i < dims.size(); ++i)
      new_shape[i] = shape[dims[i]];

    // Calcular strides actuales
    std::vector<int> old_strides(shape.size());
    old_strides[shape.size() - 1] = 1;
    for (int i = shape.size() - 2; i >= 0; --i)
      old_strides[i] = old_strides[i + 1] * shape[i + 1];

    // Calcular nuevos strides
    std::vector<int> new_strides(shape.size());
    for (size_t i = 0; i < dims.size(); ++i)
      new_strides[i] = old_strides[dims[i]];

    // Transponer datos
    Tensor out(new_shape);
    for (size_t idx = 0; idx < data.size(); ++idx)
    {
      // Obtener índice multidimensional original
      int offset = idx;
      std::vector<int> indices(shape.size());
      for (size_t i = 0; i < shape.size(); ++i)
      {
        indices[i] = offset / old_strides[i];
        offset %= old_strides[i];
      }

      // Permutar los índices
      std::vector<int> permuted_indices(shape.size());
      for (size_t i = 0; i < shape.size(); ++i)
        permuted_indices[i] = indices[dims[i]];

      // Calcular nuevo índice lineal
      int new_offset = 0;
      for (size_t i = 0; i < shape.size(); ++i)
        new_offset += permuted_indices[i] * new_strides[i];

      out.data[new_offset] = data[idx];
    }

    return out;
  }

  // -------------------- Matmul --------------------
  Tensor matmul(const Tensor &other) const
  {
    if (shape.size() == 2 && other.shape.size() == 2)
    {
      int m = shape[0], n = shape[1];
      int n2 = other.shape[0], p = other.shape[1];
      assert(n == n2);
      Tensor result({m, p});

// Paralelización del bucle externo
#pragma omp parallel for collapse(2) schedule(static)
      for (int i = 0; i < m; ++i)
      {
        for (int j = 0; j < p; ++j)
        {
          float sum = 0.0f;
// Desenrollamos el bucle interno para mejor rendimiento
#pragma omp simd reduction(+ : sum)
          for (int k = 0; k < n; ++k)
          {
            sum += data[i * n + k] * other.data[k * p + j];
          }
          result.data[i * p + j] = sum;
        }
      }
      return result;
    }

    // Batch matmul: A(b, m, n) @ B(b, n, p)
    if (shape.size() == 3 && other.shape.size() == 3)
    {
      int B1 = shape[0], M = shape[1], N = shape[2];
      int B2 = other.shape[0], N2 = other.shape[1], P = other.shape[2];
      assert(B1 == B2 && N == N2);
      Tensor result({B1, M, P});

// Paralelización a nivel de batch y filas
#pragma omp parallel for collapse(2) schedule(dynamic)
      for (int b = 0; b < B1; ++b)
      {
        for (int i = 0; i < M; ++i)
        {
          for (int j = 0; j < P; ++j)
          {
            float sum = 0.0f;
// Vectorización del bucle interno
#pragma omp simd reduction(+ : sum)
            for (int k = 0; k < N; ++k)
            {
              sum += data[b * M * N + i * N + k] *
                     other.data[b * N * P + k * P + j];
            }
            result.data[b * M * P + i * P + j] = sum;
          }
        }
      }
      return result;
    }

    throw std::invalid_argument("matmul only supports 2D or 3D tensors with matching inner dimensions");
  }

  // -------------------- Reshape --------------------
  Tensor reshape(const std::vector<int> &new_shape) const
  {
    int total_old = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());

    std::vector<int> resolved_shape = new_shape;
    int neg_one_index = -1;
    int known_product = 1;

    for (size_t i = 0; i < new_shape.size(); ++i)
    {
      if (new_shape[i] == -1)
      {
        if (neg_one_index != -1)
          throw std::invalid_argument("Solo se permite un -1 en reshape");
        neg_one_index = i;
      }
      else
      {
        known_product *= new_shape[i];
      }
    }

    if (neg_one_index != -1)
    {
      if (total_old % known_product != 0)
        throw std::runtime_error("reshape con -1 no es divisible con las dimensiones existentes");

      resolved_shape[neg_one_index] = total_old / known_product;
    }

    int total_new = std::accumulate(resolved_shape.begin(), resolved_shape.end(), 1, std::multiplies<int>());
    if (total_old != total_new)
      throw std::runtime_error("reshape: el número de elementos no coincide");

    Tensor out = *this;
    out.shape = resolved_shape;
    return out;
  }
  static Tensor bernoulli(const std::vector<int> &shape, float p)
  {
    int total = 1;
    for (int dim : shape)
      total *= dim;

    std::vector<float> result_data(total);

    std::bernoulli_distribution dist(p);

    for (int i = 0; i < total; ++i)
    {
      result_data[i] = dist(global_gen) ? 1.0f : 0.0f;
    }

    return Tensor(shape, result_data);
  }

  // Funciones funcionales

  Tensor relu() const
  {
    Tensor out(shape);
    for (size_t i = 0; i < data.size(); ++i)
    {
      out.data[i] = std::max(0.0f, data[i]);
    }
    return out;
  }
  Tensor relu_derivative() const
  {
    Tensor result(shape);
    for (size_t i = 0; i < data.size(); ++i)
    {
      result.data[i] = data[i] > 0 ? 1.0f : 0.0f;
    }
    return result;
  }

  Tensor softmax(int dim) const
  {
    assert(dim >= 0 && dim < shape.size());

    Tensor out(shape);
    std::vector<int> strides(shape.size());
    strides.back() = 1;
    for (int i = shape.size() - 2; i >= 0; --i)
      strides[i] = strides[i + 1] * shape[i + 1];

    int dim_size = shape[dim];
    int outer = 1, inner = 1;

    for (int i = 0; i < dim; ++i)
      outer *= shape[i];
    for (int i = dim + 1; i < shape.size(); ++i)
      inner *= shape[i];

    for (int o = 0; o < outer; ++o)
    {
      for (int i = 0; i < inner; ++i)
      {
        float max_val = -std::numeric_limits<float>::infinity();

        // Encontrar máximo para estabilidad numérica
        for (int d = 0; d < dim_size; ++d)
        {
          int idx = o * dim_size * inner + d * inner + i;
          max_val = std::max(max_val, data[idx]);
        }

        // Calcular suma de exponentes
        float sum = 0.0f;
        for (int d = 0; d < dim_size; ++d)
        {
          int idx = o * dim_size * inner + d * inner + i;
          out.data[idx] = std::exp(data[idx] - max_val);
          sum += out.data[idx];
        }

        // Normalizar
        for (int d = 0; d < dim_size; ++d)
        {
          int idx = o * dim_size * inner + d * inner + i;
          out.data[idx] /= sum;
        }
      }
    }

    return out;
  }

  static Tensor softmax_backward(const Tensor &softmax, const Tensor &delta)
  {
    Tensor sum = (delta * softmax).sum(-1, true); // [N, HW, 1]
    return softmax * (delta - sum);
  }

  // -------------------- Overloads --------------------

  Tensor operator+(const Tensor &other) const
  {
    // Caso 1: shape igual → suma directa
    if (shape == other.shape)
    {
      Tensor result(shape);
      const size_t total_elements = data.size();

#pragma omp parallel for schedule(static)
      for (size_t i = 0; i < total_elements; ++i)
        result.data[i] = data[i] + other.data[i];

      return result;
    }

    // Caso 2: broadcasting si other.shape == {1, D} y this.shape == {B, D}
    if (shape.size() == 2 && other.shape.size() == 2 &&
        other.shape[0] == 1 && shape[1] == other.shape[1])
    {
      int B = shape[0];
      int D = shape[1];
      Tensor result(shape);

#pragma omp parallel for collapse(2) schedule(static)
      for (int b = 0; b < B; ++b)
        for (int d = 0; d < D; ++d)
          result.data[b * D + d] = data[b * D + d] + other.data[d];

      return result;
    }

    // Caso 3: broadcasting para tensores 4D (ej. en CNNs)
    if (shape.size() == 4 && other.shape.size() == 4 &&
        other.shape[0] == 1 &&
        other.shape[2] == 1 &&
        other.shape[3] == 1 &&
        shape[1] == other.shape[1])
    {
      int B = shape[0];
      int C = shape[1];
      int H = shape[2];
      int W = shape[3];
      Tensor result(shape);

#pragma omp parallel for collapse(3) schedule(static)
      for (int b = 0; b < B; ++b)
        for (int c = 0; c < C; ++c)
          for (int h = 0; h < H; ++h)
            for (int w = 0; w < W; ++w)
            {
              int idx = ((b * C + c) * H + h) * W + w;
              result.data[idx] = data[idx] + other.data[c];
            }

      return result;
    }

    throw std::runtime_error("operator+: Shapes incompatibles para suma con broadcasting");
  }

  Tensor operator-(const Tensor &other) const
  {
    assert(shape.size() == other.shape.size());
    Tensor result(shape);

    // Precalcular strides para this y other
    std::vector<int> self_strides(shape.size(), 1);
    std::vector<int> other_strides(shape.size(), 1);

    for (int i = shape.size() - 2; i >= 0; --i)
    {
      self_strides[i] = self_strides[i + 1] * shape[i + 1];
      other_strides[i] = other_strides[i + 1] * other.shape[i + 1];
    }

#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(data.size()); ++i)
    {
      int index_self = i;
      int index_other = 0;

      int rem = i;
      for (size_t d = 0; d < shape.size(); ++d)
      {
        int coord = rem / self_strides[d];
        rem %= self_strides[d];

        int coord_other = (other.shape[d] == 1) ? 0 : coord;
        index_other += coord_other * other_strides[d];
      }

      result.data[i] = data[i] - other.data[index_other];
    }

    return result;
  }
  Tensor operator*(const Tensor &other) const
  {
    assert(shape.size() == other.shape.size());

    Tensor result(shape);

    // Precalcular strides para `other`
    std::vector<int> other_strides(other.shape.size(), 1);
    for (int i = other.shape.size() - 2; i >= 0; --i)
    {
      other_strides[i] = other_strides[i + 1] * other.shape[i + 1];
    }

#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(data.size()); ++i)
    {
      int flat_other = 0;
      int remainder = i;
      int stride = 1;

      for (int d = shape.size() - 1; d >= 0; --d)
      {
        int coord = (remainder % shape[d]);
        remainder /= shape[d];

        int coord_other = (other.shape[d] == 1) ? 0 : coord;
        flat_other += coord_other * other_strides[d];
      }

      result.data[i] = data[i] * other.data[flat_other];
    }

    return result;
  }

  Tensor operator/(float scalar) const
  {
    Tensor result(shape);
    const size_t total_elements = data.size();
    const float inv_scalar = 1.0f / scalar; // Optimización: división por multiplicación

#pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < total_elements; ++i)
      result.data[i] = data[i] * inv_scalar; // Multiplicación es más rápida que división

    return result;
  }

  Tensor operator+(float scalar) const
  {
    Tensor result(shape);
    const size_t total_elements = data.size();

#pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < total_elements; ++i)
      result.data[i] = data[i] + scalar;

    return result;
  }

  Tensor operator*(float scalar) const
  {
    Tensor result(shape);
    const size_t total_elements = data.size();

#pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < total_elements; ++i)
      result.data[i] = data[i] * scalar;

    return result;
  }

  // In-place
  Tensor &operator=(const Tensor &other)
  {
    shape = other.shape;
    data = other.data;
    return *this;
  }
};
