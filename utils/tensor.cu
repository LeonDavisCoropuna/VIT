#pragma once
#include <vector>
#include <numeric>
#include <random>
#include <stdexcept>
#include <cuda_runtime.h>
#include <string>
#include <algorithm>
#include <curand.h>
#include <curand_kernel.h>

// CUDA error checking macro
#define CUDA_CHECK(call)                                                                                                                            \
  do                                                                                                                                                \
  {                                                                                                                                                 \
    cudaError_t err = call;                                                                                                                         \
    if (err != cudaSuccess)                                                                                                                         \
    {                                                                                                                                               \
      throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err) + " in " + __FILE__ + " at line " + std::to_string(__LINE__)); \
    }                                                                                                                                               \
  } while (0)

// CUDA kernels
__global__ void add_kernel(float *a, float *b, float *result, size_t size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
  {
    result[idx] = a[idx] + b[idx];
  }
}
__global__ void kaiming_normal_kernel(float *data, int size, float std, unsigned long seed)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;

  // Inicializar curand en cada hilo
  curandState state;
  curand_init(seed, idx, 0, &state);

  // Normal distribuido
  float rand_val = curand_normal(&state) * std;
  data[idx] = rand_val;
}
__host__ __device__ inline int get_flat_index(const std::vector<int> &shape,
                                              const std::vector<int> &coords)
{
  int idx = 0;
  int stride = 1;
  for (int i = shape.size() - 1; i >= 0; --i)
  {
    idx += coords[i] * stride;
    stride *= shape[i];
  }
  return idx;
}

__global__ void bernoulli_kernel(float *data, float prob, int size, unsigned long seed)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;

  // CURAND setup
  curandState state;
  curand_init(seed, idx, 0, &state);

  float rand_val = curand_uniform(&state); // genera número en (0,1)
  data[idx] = rand_val < prob ? 1.0f : 0.0f;
}

__global__ void pad4d_kernel(const float *input, float *output,
                             int N, int C, int H, int W,
                             int pad_top, int pad_bottom,
                             int pad_left, int pad_right,
                             int H_out, int W_out)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = N * C * H_out * W_out;
  if (idx >= total)
    return;

  int w_out = idx % W_out;
  int h_out = (idx / W_out) % H_out;
  int c = (idx / (W_out * H_out)) % C;
  int n = idx / (W_out * H_out * C);

  int h_in = h_out - pad_top;
  int w_in = w_out - pad_left;

  int out_index = ((n * C + c) * H_out + h_out) * W_out + w_out;

  if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W)
  {
    int in_index = ((n * C + c) * H + h_in) * W + w_in;
    output[out_index] = input[in_index];
  }
  else
  {
    output[out_index] = 0.0f; // Cero por default
  }
}

__global__ void fill_kernel(float *data, float value, int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
  {
    data[idx] = value;
  }
}
__global__ void conv2d_forward_kernel(
    const float *input, const float *weights, const float *biases,
    float *output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int K_h, int K_w,
    int H_out, int W_out,
    int stride, int padding)
{
  int n = blockIdx.x;
  int co = blockIdx.y;
  int ho = threadIdx.y;
  int wo = threadIdx.x;

  if (ho >= H_out || wo >= W_out)
    return;

  int out_idx = ((n * C_out + co) * H_out + ho) * W_out + wo;
  float sum = biases[co];

  for (int ci = 0; ci < C_in; ++ci)
  {
    for (int kh = 0; kh < K_h; ++kh)
    {
      for (int kw = 0; kw < K_w; ++kw)
      {
        int h_in = ho * stride + kh - padding;
        int w_in = wo * stride + kw - padding;

        if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in)
        {
          int in_idx = ((n * C_in + ci) * H_in + h_in) * W_in + w_in;
          int w_idx = ((co * C_in + ci) * K_h + kh) * K_w + kw;
          sum += input[in_idx] * weights[w_idx];
        }
      }
    }
  }
  output[out_idx] = sum;
}

__global__ void matmul2d_kernel(const float *A, const float *B, float *C,
                                int m, int n, int p)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < m && col < p)
  {
    float sum = 0.0f;
    for (int k = 0; k < n; ++k)
    {
      sum += A[row * n + k] * B[k * p + col];
    }
    C[row * p + col] = sum;
  }
}

__global__ void matmul3d_kernel(const float *A, const float *B, float *C,
                                int b, int m, int n, int p)
{
  int batch = blockIdx.z;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (batch < b && row < m && col < p)
  {
    float sum = 0.0f;
    for (int k = 0; k < n; ++k)
    {
      sum += A[batch * m * n + row * n + k] *
             B[batch * n * p + k * p + col];
    }
    C[batch * m * p + row * p + col] = sum;
  }
}
__global__ void permute_kernel(const float *input, float *output,
                               int *input_strides, int *output_strides,
                               int *perm, int dims, int total_elems)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_elems)
    return;

  int input_idx = 0;
  int remainder = idx;

  // Calcula el índice original en input usando permutación inversa
  for (int i = 0; i < dims; ++i)
  {
    int coord = remainder / output_strides[i];
    remainder = remainder % output_strides[i];
    input_idx += coord * input_strides[perm[i]];
  }

  output[idx] = input[input_idx];
}

__global__ void flatten_kernel(const float *input, float *output, int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
  {
    output[idx] = input[idx];
  }
}

__global__ void add_broadcast_2d_kernel(float *a, float *b, float *result, int B, int D)
{
  int batch = blockIdx.y;
  int d = blockIdx.x * blockDim.x + threadIdx.x;
  if (batch < B && d < D)
  {
    result[batch * D + d] = a[batch * D + d] + b[d];
  }
}
__global__ void add_broadcast_4d_kernel(float *a, float *b, float *result, int B, int C, int H, int W)
{
  int batch = blockIdx.z;
  int channel = blockIdx.y;
  int h = blockIdx.x;
  int w = threadIdx.x;
  if (batch < B && channel < C && h < H && w < W)
  {
    int idx = ((batch * C + channel) * H + h) * W + w;
    result[idx] = a[idx] + b[channel];
  }
}

__global__ void subtract_broadcast_kernel(float *a, float *b, float *result, int *shape_a, int *shape_b, int dims, size_t size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
  {
    int flat_idx_b = 0;
    int temp = idx;
    for (int d = dims - 1; d >= 0; --d)
    {
      int coord = temp % shape_a[d];
      temp /= shape_a[d];
      flat_idx_b += (shape_b[d] == 1 ? 0 : coord) * (d == dims - 1 ? 1 : shape_b[d + 1]);
    }
    result[idx] = a[idx] - b[flat_idx_b];
  }
}

__global__ void transpose_3d_kernel(const float *input, float *output,
                                    int dim0, int dim1, int dim2,
                                    int axis1, int axis2)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = dim0 * dim1 * dim2;
  if (idx >= total)
    return;

  int i = idx / (dim1 * dim2); // batch
  int j = (idx / dim2) % dim1; // eje1
  int k = idx % dim2;          // eje2

  int shape[3] = {dim0, dim1, dim2};
  int in_pos[3] = {i, j, k};
  int out_pos[3] = {i, j, k};

  out_pos[axis1] = in_pos[axis2];
  out_pos[axis2] = in_pos[axis1];

  int out_idx = out_pos[0] * shape[axis1] * shape[axis2] +
                out_pos[1] * shape[axis2] +
                out_pos[2];

  output[out_idx] = input[idx];
}

__global__ void multiply_broadcast_kernel(float *a, float *b, float *result, int *shape_a, int *shape_b, int dims, size_t size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
  {
    int flat_idx_b = 0;
    int temp = idx;
    for (int d = dims - 1; d >= 0; --d)
    {
      int coord = temp % shape_a[d];
      temp /= shape_a[d];
      flat_idx_b += (shape_b[d] == 1 ? 0 : coord) * (d == dims - 1 ? 1 : shape_b[d + 1]);
    }
    result[idx] = a[idx] * b[flat_idx_b];
  }
}

__global__ void scalar_divide_kernel(float *a, float scalar, float *result, size_t size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
  {
    result[idx] = a[idx] / scalar;
  }
}

__global__ void scalar_add_kernel(float *a, float scalar, float *result, size_t size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
  {
    result[idx] = a[idx] + scalar;
  }
}

__global__ void scalar_multiply_kernel(float *a, float scalar, float *result, size_t size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
  {
    result[idx] = a[idx] * scalar;
  }
}

__global__ void relu_kernel(float *a, float *result, size_t size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
  {
    result[idx] = fmaxf(0.0f, a[idx]);
  }
}

__global__ void relu_derivative_kernel(float *a, float *result, size_t size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
  {
    result[idx] = a[idx] > 0.0f ? 1.0f : 0.0f;
  }
}

__global__ void softmax_kernel(float *a, float *result, int outer, int dim_size, int inner, size_t size)
{
  int o = blockIdx.y;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (o < outer && i < inner)
  {
    float max_val = -1e38f;
    for (int d = 0; d < dim_size; ++d)
    {
      int idx = o * dim_size * inner + d * inner + i;
      max_val = fmaxf(max_val, a[idx]);
    }
    float sum = 0.0f;
    for (int d = 0; d < dim_size; ++d)
    {
      int idx = o * dim_size * inner + d * inner + i;
      result[idx] = expf(a[idx] - max_val);
      sum += result[idx];
    }
    for (int d = 0; d < dim_size; ++d)
    {
      int idx = o * dim_size * inner + d * inner + i;
      result[idx] /= sum;
    }
  }
}

class Tensor
{
public:
  std::vector<int> shape;
  float *d_data; // Device pointer for CUDA
  size_t size;   // Total number of elements

  static std::mt19937 global_gen;

  static void set_seed(unsigned int seed)
  {
    global_gen.seed(seed);
  }

  Tensor() : d_data(nullptr), size(0) {}

  Tensor(const std::vector<int> &shape) : shape(shape)
  {
    size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    CUDA_CHECK(cudaMalloc(&d_data, size * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_data, 0, size * sizeof(float)));
  }

  Tensor(const std::vector<int> &shape, const std::vector<float> &data) : shape(shape)
  {
    size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    if (data.size() != size)
    {
      throw std::runtime_error("Data size does not match shape");
    }
    CUDA_CHECK(cudaMalloc(&d_data, size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_data, data.data(), size * sizeof(float), cudaMemcpyHostToDevice));
  }

  ~Tensor()
  {
    if (d_data)
    {
      CUDA_CHECK(cudaFree(d_data));
    }
  }

  Tensor(const Tensor &other) : shape(other.shape), size(other.size)
  {
    CUDA_CHECK(cudaMalloc(&d_data, size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_data, other.d_data, size * sizeof(float), cudaMemcpyDeviceToDevice));
  }

  Tensor &operator=(const Tensor &other)
  {
    if (this != &other)
    {
      if (d_data)
      {
        CUDA_CHECK(cudaFree(d_data));
      }
      shape = other.shape;
      size = other.size;
      CUDA_CHECK(cudaMalloc(&d_data, size * sizeof(float)));
      CUDA_CHECK(cudaMemcpy(d_data, other.d_data, size * sizeof(float), cudaMemcpyDeviceToDevice));
    }
    return *this;
  }

  Tensor(Tensor &&other) noexcept : shape(std::move(other.shape)), d_data(other.d_data), size(other.size)
  {
    other.d_data = nullptr;
    other.size = 0;
  }

  Tensor &operator=(Tensor &&other) noexcept
  {
    if (this != &other)
    {
      if (d_data)
      {
        CUDA_CHECK(cudaFree(d_data));
      }
      shape = std::move(other.shape);
      d_data = other.d_data;
      size = other.size;
      other.d_data = nullptr;
      other.size = 0;
    }
    return *this;
  }

  static Tensor zeros(const std::vector<int> &shape)
  {
    return Tensor(shape);
  }
  static Tensor bernoulli(const std::vector<int> &shape, float prob)
  {
    int size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    Tensor out(shape);

    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    unsigned long seed = static_cast<unsigned long>(std::random_device{}());

    bernoulli_kernel<<<blocks, threads>>>(out.d_data, prob, size, seed);
    cudaDeviceSynchronize();

    return out;
  }

  std::vector<float> to_host() const
  {
    std::vector<float> host_data(size);
    CUDA_CHECK(cudaMemcpy(host_data.data(), d_data, size * sizeof(float), cudaMemcpyDeviceToHost));
    return host_data;
  }

  Tensor flatten(int start_dim) const
  {
    int total_dims = shape.size();
    if (start_dim < 0 || start_dim >= total_dims)
      throw std::invalid_argument("Invalid start_dim in flatten.");

    // Calcular la nueva forma
    int flattened_dim = 1;
    for (int i = start_dim; i < total_dims; ++i)
      flattened_dim *= shape[i];

    std::vector<int> new_shape;
    for (int i = 0; i < start_dim; ++i)
      new_shape.push_back(shape[i]);
    new_shape.push_back(flattened_dim);

    // Crear tensor de salida
    Tensor result(new_shape);

    // Copiar datos (mismo orden lineal)
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    flatten_kernel<<<gridSize, blockSize>>>(d_data, result.d_data, size);
    CUDA_CHECK(cudaDeviceSynchronize());

    return result;
  }
  Tensor matmul(const Tensor &other) const
  {
    if (shape.size() == 2 && other.shape.size() == 2)
    {
      int m = shape[0], n = shape[1];
      int n2 = other.shape[0], p = other.shape[1];
      if (n != n2)
        throw std::invalid_argument("Dimension mismatch in matmul (2D)");

      Tensor result({m, p});

      dim3 block(16, 16);
      dim3 grid((p + 15) / 16, (m + 15) / 16);
      matmul2d_kernel<<<grid, block>>>(d_data, other.d_data, result.d_data, m, n, p);
      CUDA_CHECK(cudaDeviceSynchronize());

      return result;
    }

    if (shape.size() == 3 && other.shape.size() == 3)
    {
      int b = shape[0], m = shape[1], n = shape[2];
      int b2 = other.shape[0], n2 = other.shape[1], p = other.shape[2];
      if (b != b2 || n != n2)
        throw std::invalid_argument("Dimension mismatch in matmul (3D)");

      Tensor result({b, m, p});

      dim3 block(16, 16);
      dim3 grid((p + 15) / 16, (m + 15) / 16, b);
      matmul3d_kernel<<<grid, block>>>(d_data, other.d_data, result.d_data, b, m, n, p);
      CUDA_CHECK(cudaDeviceSynchronize());

      return result;
    }

    throw std::invalid_argument("matmul only supports 2D or 3D tensors with compatible shapes.");
  }

  // Operator overloads
  Tensor operator+(const Tensor &other) const
  {
    if (shape == other.shape)
    {
      Tensor result(shape);
      const int block_size = 256;
      const int grid_size = (size + block_size - 1) / block_size;
      add_kernel<<<grid_size, block_size>>>(d_data, other.d_data, result.d_data, size);
      CUDA_CHECK(cudaDeviceSynchronize());
      return result;
    }
    if (shape.size() == 2 && other.shape.size() == 2 && other.shape[0] == 1 && shape[1] == other.shape[1])
    {
      Tensor result(shape);
      int B = shape[0], D = shape[1];
      dim3 block_size(256, 1);
      dim3 grid_size((D + block_size.x - 1) / block_size.x, B);
      add_broadcast_2d_kernel<<<grid_size, block_size>>>(d_data, other.d_data, result.d_data, B, D);
      CUDA_CHECK(cudaDeviceSynchronize());
      return result;
    }
    if (shape.size() == 4 && other.shape.size() == 4 && other.shape[0] == 1 && other.shape[2] == 1 && other.shape[3] == 1 && shape[1] == other.shape[1])
    {
      Tensor result(shape);
      int B = shape[0], C = shape[1], H = shape[2], W = shape[3];
      dim3 block_size(W, 1);
      dim3 grid_size(H, C, B);
      add_broadcast_4d_kernel<<<grid_size, block_size>>>(d_data, other.d_data, result.d_data, B, C, H, W);
      CUDA_CHECK(cudaDeviceSynchronize());
      return result;
    }
    throw std::runtime_error("operator+: Shapes incompatible for addition with broadcasting");
  }

  Tensor operator-(const Tensor &other) const
  {
    if (shape.size() != other.shape.size())
    {
      throw std::runtime_error("operator-: Incompatible shape dimensions");
    }
    Tensor result(shape);
    int *d_shape_a, *d_shape_b;
    CUDA_CHECK(cudaMalloc(&d_shape_a, shape.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_shape_b, other.shape.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_shape_a, shape.data(), shape.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_shape_b, other.shape.data(), other.shape.size() * sizeof(int), cudaMemcpyHostToDevice));
    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;
    subtract_broadcast_kernel<<<grid_size, block_size>>>(d_data, other.d_data, result.d_data, d_shape_a, d_shape_b, shape.size(), size);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_shape_a));
    CUDA_CHECK(cudaFree(d_shape_b));
    return result;
  }

  Tensor operator*(const Tensor &other) const
  {
    if (shape.size() != other.shape.size())
    {
      throw std::runtime_error("operator*: Incompatible shape dimensions");
    }
    Tensor result(shape);
    int *d_shape_a, *d_shape_b;
    CUDA_CHECK(cudaMalloc(&d_shape_a, shape.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_shape_b, other.shape.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_shape_a, shape.data(), shape.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_shape_b, other.shape.data(), other.shape.size() * sizeof(int), cudaMemcpyHostToDevice));
    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;
    multiply_broadcast_kernel<<<grid_size, block_size>>>(d_data, other.d_data, result.d_data, d_shape_a, d_shape_b, shape.size(), size);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_shape_a));
    CUDA_CHECK(cudaFree(d_shape_b));
    return result;
  }

  Tensor operator/(float scalar) const
  {
    Tensor result(shape);
    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;
    scalar_divide_kernel<<<grid_size, block_size>>>(d_data, scalar, result.d_data, size);
    CUDA_CHECK(cudaDeviceSynchronize());
    return result;
  }

  Tensor operator+(float scalar) const
  {
    Tensor result(shape);
    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;
    scalar_add_kernel<<<grid_size, block_size>>>(d_data, scalar, result.d_data, size);
    CUDA_CHECK(cudaDeviceSynchronize());
    return result;
  }

  Tensor operator*(float scalar) const
  {
    Tensor result(shape);
    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;
    scalar_multiply_kernel<<<grid_size, block_size>>>(d_data, scalar, result.d_data, size);
    CUDA_CHECK(cudaDeviceSynchronize());
    return result;
  }

  Tensor relu() const
  {
    Tensor result(shape);
    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;
    relu_kernel<<<grid_size, block_size>>>(d_data, result.d_data, size);
    CUDA_CHECK(cudaDeviceSynchronize());
    return result;
  }

  Tensor relu_derivative() const
  {
    Tensor result(shape);
    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;
    relu_derivative_kernel<<<grid_size, block_size>>>(d_data, result.d_data, size);
    CUDA_CHECK(cudaDeviceSynchronize());
    return result;
  }

  Tensor softmax(int dim) const
  {
    if (dim < 0 || dim >= shape.size())
    {
      throw std::runtime_error("Invalid dimension for softmax");
    }
    Tensor result(shape);
    int outer = 1, inner = 1, dim_size = shape[dim];
    for (int i = 0; i < dim; ++i)
      outer *= shape[i];
    for (int i = dim + 1; i < shape.size(); ++i)
      inner *= shape[i];
    dim3 block_size(256, 1);
    dim3 grid_size((inner + block_size.x - 1) / block_size.x, outer);
    softmax_kernel<<<grid_size, block_size>>>(d_data, result.d_data, outer, dim_size, inner, size);
    CUDA_CHECK(cudaDeviceSynchronize());
    return result;
  }

  static Tensor softmax_backward(const Tensor &softmax, const Tensor &delta)
  {
    // Note: Implementation requires a sum reduction, which is complex in vanilla CUDA.
    // For simplicity, this would require a separate reduction kernel or thrust.
    // As per your request to avoid external libraries, this is left as a placeholder.
    throw std::runtime_error("softmax_backward not implemented in vanilla CUDA");
  }

  Tensor transpose(const std::vector<int> &axes) const
  {
    if (shape.size() != 3 || axes.size() != 3)
    {
      throw std::runtime_error("Only 3D transpose supported");
    }

    // Validar que axes sea una permutación válida de {0,1,2}
    std::vector<int> ref = {0, 1, 2};
    std::vector<int> check = axes;
    std::sort(check.begin(), check.end());
    if (check != ref)
    {
      throw std::runtime_error("Invalid axes permutation");
    }

    std::vector<int> new_shape = {shape[axes[0]], shape[axes[1]], shape[axes[2]]};
    Tensor result(new_shape);

    int dim0 = shape[0], dim1 = shape[1], dim2 = shape[2];
    int total = dim0 * dim1 * dim2;

    int blockSize = 256;
    int gridSize = (total + blockSize - 1) / blockSize;

    transpose_3d_kernel<<<gridSize, blockSize>>>(
        d_data, result.d_data,
        dim0, dim1, dim2,
        axes[1], axes[2]);

    CUDA_CHECK(cudaDeviceSynchronize());
    return result;
  }
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
        throw std::runtime_error("reshape con -1 no divisible con dimensiones existentes");
      resolved_shape[neg_one_index] = total_old / known_product;
    }

    int total_new = std::accumulate(resolved_shape.begin(), resolved_shape.end(), 1, std::multiplies<int>());
    if (total_old != total_new)
      throw std::runtime_error("reshape: el número de elementos no coincide");

    // No se copia memoria, solo se cambia la forma
    Tensor reshaped = *this;
    reshaped.shape = resolved_shape;
    return reshaped;
  }
  Tensor permute(const std::vector<int> &perm) const
  {
    int dims = shape.size();
    if (perm.size() != dims)
      throw std::invalid_argument("Permute: perm size must match tensor rank");

    // Calcular nueva forma
    std::vector<int> new_shape(dims);
    for (int i = 0; i < dims; ++i)
      new_shape[i] = shape[perm[i]];

    // Calcular strides originales y nuevos
    std::vector<int> input_strides(dims, 1);
    std::vector<int> output_strides(dims, 1);

    for (int i = dims - 2; i >= 0; --i)
    {
      input_strides[i] = input_strides[i + 1] * shape[i + 1];
      output_strides[i] = output_strides[i + 1] * new_shape[i + 1];
    }

    // Reservar espacio para nuevo tensor
    Tensor out(new_shape);

    // Copiar estructuras auxiliares a device
    int *d_input_strides, *d_output_strides, *d_perm;
    CUDA_CHECK(cudaMalloc(&d_input_strides, dims * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_output_strides, dims * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_perm, dims * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_input_strides, input_strides.data(), dims * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_output_strides, output_strides.data(), dims * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_perm, perm.data(), dims * sizeof(int), cudaMemcpyHostToDevice));

    // Ejecutar kernel
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    permute_kernel<<<blocks, threads>>>(
        d_data, out.d_data,
        d_input_strides, d_output_strides,
        d_perm, dims, size);

    CUDA_CHECK(cudaDeviceSynchronize());

    // Liberar memoria auxiliar
    cudaFree(d_input_strides);
    cudaFree(d_output_strides);
    cudaFree(d_perm);

    return out;
  }
  void fill(float value)
  {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    fill_kernel<<<blocks, threads>>>(d_data, value, size);
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  Tensor conv2d(const Tensor &kernel, const Tensor &bias, int stride, int padding) const
  {
    // Input: [N, C_in, H_in, W_in]
    // Kernel: [C_out, C_in, K_h, K_w]
    // Bias: [C_out]

    int N = shape[0], C_in = shape[1], H_in = shape[2], W_in = shape[3];
    int C_out = kernel.shape[0], K_h = kernel.shape[2], K_w = kernel.shape[3];
    int H_out = (H_in + 2 * padding - K_h) / stride + 1;
    int W_out = (W_in + 2 * padding - K_w) / stride + 1;

    Tensor out({N, C_out, H_out, W_out});

    dim3 grid(N, C_out);
    dim3 block(W_out, H_out); // cuidado: si H_out o W_out > 1024 ajusta

    conv2d_forward_kernel<<<grid, block>>>(
        d_data, kernel.d_data, bias.d_data,
        out.d_data,
        N, C_in, H_in, W_in,
        C_out, K_h, K_w,
        H_out, W_out,
        stride, padding);

    CUDA_CHECK(cudaDeviceSynchronize());

    return out;
  }
  static Tensor kaiming_normal(const std::vector<int> &shape, int fan_in)
  {
    int size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    float std = std::sqrt(2.0f / fan_in);

    Tensor out(shape);

    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    unsigned long seed = std::random_device{}();

    kaiming_normal_kernel<<<blocks, threads>>>(out.d_data, size, std, seed);
    CUDA_CHECK(cudaDeviceSynchronize());

    return out;
  }
  Tensor pad(const std::vector<int> &pads) const
  {
    if (shape.size() != 4)
      throw std::invalid_argument("Pad only supports 4D tensors");

    if (pads.size() != 8)
      throw std::invalid_argument("Pad vector must have 8 values");

    // Formato: {N0, N1, C0, C1, H0, H1, W0, W1}
    int pad_top = pads[4], pad_bottom = pads[5];
    int pad_left = pads[6], pad_right = pads[7];

    int N = shape[0], C = shape[1], H = shape[2], W = shape[3];
    int H_out = H + pad_top + pad_bottom;
    int W_out = W + pad_left + pad_right;

    std::vector<int> new_shape = {N, C, H_out, W_out};
    Tensor output(new_shape);

    int threads = 256;
    int total = N * C * H_out * W_out;
    int blocks = (total + threads - 1) / threads;

    pad4d_kernel<<<blocks, threads>>>(
        d_data, output.d_data,
        N, C, H, W,
        pad_top, pad_bottom,
        pad_left, pad_right,
        H_out, W_out);

    CUDA_CHECK(cudaDeviceSynchronize());

    return output;
  }
  inline float &at(const std::vector<int> &coords)
  {
    int idx = get_flat_index(shape, coords);
    return data[idx];
  }

  inline const float &at(const std::vector<int> &coords) const
  {
    int idx = get_flat_index(shape, coords);
    return data[idx];
  }
};