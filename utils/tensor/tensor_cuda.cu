#include <cuda_runtime.h>
#include <random>
#include <vector>
#include <numeric>
#include <stdexcept>

__global__ void fillKernel(float *data, float value, int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
  {
    data[idx] = value;
  }
}

__global__ void scalarAddKernel(float *result, const float *input, float scalar, int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
  {
    result[idx] = input[idx] + scalar;
  }
}

__global__ void scalarMulKernel(float *result, const float *input, float scalar, int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
  {
    result[idx] = input[idx] * scalar;
  }
}
__global__ void addKernel(
    float *result, const float *a, const float *b,
    const int *a_shape, const int *b_shape,
    const int *a_strides, const int *b_strides,
    int dims, int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
  {
    int a_idx = 0, b_idx = 0;
    int temp = idx;
    for (int d = 0; d < dims; ++d)
    {
      int coord = temp / a_strides[d];
      temp %= a_strides[d];
      a_idx += coord * a_strides[d];
      int coord_b = (b_shape[d] == 1) ? 0 : coord;
      b_idx += coord_b * b_strides[d];
    }
    result[idx] = a[a_idx] + b[b_idx];
  }
}

// Kernel for element-wise subtraction
__global__ void elementwiseSubKernel(float *output, const float *a, const float *b, int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
  {
    output[idx] = a[idx] - b[idx];
  }
}

// Kernel for element-wise multiplication
__global__ void elementwiseMulKernel(float *output, const float *a, const float *b, int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
  {
    output[idx] = a[idx] * b[idx];
  }
}
// Kernel for element-wise subtraction with broadcasting
__global__ void subKernel(float *result, const float *a, const float *b,
                          int *a_shape, int *b_shape, int *a_strides, int *b_strides, int dims, int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
  {
    int a_idx = 0, b_idx = 0;
    int temp = idx;
    for (int d = 0; d < dims; ++d)
    {
      int coord = temp / a_strides[d];
      temp %= a_strides[d];
      a_idx += coord * a_strides[d];
      int coord_b = (b_shape[d] == 1) ? 0 : coord;
      b_idx += coord_b * b_strides[d];
    }
    result[idx] = a[a_idx] - b[b_idx];
  }
}
// Kernel for element-wise multiplication with broadcasting
__global__ void mulKernel(float *result, const float *a, const float *b,
                          int *a_shape, int *b_shape, int *a_strides, int *b_strides, int dims, int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
  {
    int a_idx = 0, b_idx = 0;
    int temp = idx;
    for (int d = 0; d < dims; ++d)
    {
      int coord = temp / a_strides[d];
      temp %= a_strides[d];
      a_idx += coord * a_strides[d];
      int coord_b = (b_shape[d] == 1) ? 0 : coord;
      b_idx += coord_b * b_strides[d];
    }
    result[idx] = a[a_idx] * b[b_idx];
  }
}

__global__ void reluKernel(float *output, const float *input, int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
  {
    output[idx] = max(0.0f, input[idx]);
  }
}

__global__ void reluDerivativeKernel(float *output, const float *input, int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
  {
    output[idx] = input[idx] > 0.0f ? 1.0f : 0.0f;
  }
}
__global__ void softmaxMaxKernel(float *max_vals, const float *input, int outer, int dim_size, int inner)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < outer * inner)
  {
    int o = idx / inner;
    int i = idx % inner;
    float max_val = -1.0e30f; // Approximate -infinity for float
    for (int d = 0; d < dim_size; ++d)
    {
      int in_idx = o * dim_size * inner + d * inner + i;
      max_val = fmaxf(max_val, input[in_idx]);
    }
    max_vals[idx] = max_val;
  }
}

// Kernel to compute exp and sum
__global__ void softmaxExpSumKernel(float *output, float *sums, const float *input, const float *max_vals,
                                    int outer, int dim_size, int inner)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < outer * inner)
  {
    int o = idx / inner;
    int i = idx % inner;
    float sum = 0.0f;
    for (int d = 0; d < dim_size; ++d)
    {
      int in_idx = o * dim_size * inner + d * inner + i;
      output[in_idx] = expf(input[in_idx] - max_vals[idx]);
      sum += output[in_idx];
    }
    sums[idx] = sum;
  }
}

// Kernel to normalize
__global__ void softmaxNormalizeKernel(float *output, const float *sums, int outer, int dim_size, int inner)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < outer * inner)
  {
    int o = idx / inner;
    int i = idx % inner;
    for (int d = 0; d < dim_size; ++d)
    {
      int out_idx = o * dim_size * inner + d * inner + i;
      output[out_idx] /= sums[idx];
    }
  }
}

__global__ void bernoulliKernel(float *output, float p, int size, unsigned int seed)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
  {
    // Simple LCG for pseudo-random numbers
    unsigned int x = seed + idx;
    x = x * 1103515245 + 12345; // LCG parameters
    float r = static_cast<float>(x) / static_cast<float>(0xffffffff);
    output[idx] = (r < p) ? 1.0f : 0.0f;
  }
}

__global__ void matmul3DKernel(float *result, const float *a, const float *b, int B, int M, int N, int P)
{
  int batch = blockIdx.z * blockDim.z + threadIdx.z;
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  if (batch < B && i < M && j < P)
  {
    float sum = 0.0f;
    for (int k = 0; k < N; ++k)
    {
      sum += a[batch * M * N + i * N + k] * b[batch * N * P + k * P + j];
    }
    result[batch * M * P + i * P + j] = sum;
  }
}

__global__ void matmul2DKernel(float *result, const float *a, const float *b, int m, int n, int p)
{
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < m && j < p)
  {
    float sum = 0.0f;
    for (int k = 0; k < n; ++k)
    {
      sum += a[i * n + k] * b[k * p + j];
    }
    result[i * p + j] = sum;
  }
}
__global__ void transposeKernel(float *output, const float *input, int *axes, int *old_shape,
                                int *old_strides, int *new_strides, int dims, int total_size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total_size)
  {
    // Compute original multi-dimensional indices
    int temp = idx;
    int old_index = 0;
    for (int j = 0; j < dims; ++j)
    {
      int coord = temp / old_strides[j];
      old_index += coord * old_strides[j];
      temp %= old_strides[j];
    }
    // Compute new indices based on axes permutation
    int new_index = 0;
    temp = idx;
    for (int j = 0; j < dims; ++j)
    {
      int coord = temp / old_strides[j];
      new_index += coord * new_strides[axes[j]];
      temp %= old_strides[j];
    }
    output[new_index] = input[old_index];
  }
}

__global__ void sqrtKernel(float *output, const float *input, int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
  {
    output[idx] = sqrtf(input[idx]);
  }
}
__global__ void reciprocalKernel(float *output, const float *input, int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
  {
    output[idx] = 1.0f / input[idx];
  }
}

__global__ void powKernel(float *output, const float *input, float exponent, int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
  {
    output[idx] = powf(input[idx], exponent);
  }
}

__global__ void scalarDivKernel(float *result, const float *input, float scalar, int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
  {
    result[idx] = input[idx] / scalar;
  }
}

__global__ void meanDimKernel(float *output, const float *input, int outer, int N, int inner, int shape_dim)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < outer * inner)
  {
    int o = idx / inner;
    int i = idx % inner;
    float sum = 0.0f;
    for (int n = 0; n < N; ++n)
    {
      int in_idx = (o * shape_dim + n) * inner + i;
      sum += input[in_idx];
    }
    output[idx] = sum / static_cast<float>(N);
  }
}

__global__ void sumAxis0Kernel(float *output, const float *input, int rows, int cols)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < cols)
  {
    float sum = 0.0f;
    for (int i = 0; i < rows; ++i)
    {
      sum += input[i * cols + j];
    }
    output[j] = sum;
  }
}

__global__ void sumAxis2Kernel(float *output, const float *input, int N, int HW, int L)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N * HW)
  {
    int n = idx / HW;
    int hw = idx % HW;
    float sum = 0.0f;
    for (int l = 0; l < L; ++l)
    {
      sum += input[(n * HW + hw) * L + l];
    }
    output[n * HW + hw] = sum;
  }
}
__global__ void sumMultiDimsKernel(float *output, const float *input, int *in_strides, int *out_strides,
                                   bool *reduce_axes, int total_size, int out_size, int dims)
{
  int flat_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (flat_idx < total_size)
  {
    int out_index = 0;
    int idx = flat_idx;
    for (int i = 0, out_i = 0; i < dims; ++i)
    {
      int coord = idx / in_strides[i];
      idx %= in_strides[i];
      if (!reduce_axes[i])
      {
        out_index += coord * out_strides[out_i];
        ++out_i;
      }
    }
    atomicAdd(&output[out_index], input[flat_idx]);
  }
}

// Kernel CUDA para inicialización uniforme (alternativa para evitar CPU)
__global__ void uniformKernel(float *data, int size, float low, float high, unsigned int seed)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
  {
    // Generador simple en GPU (no tan robusto como mt19937)
    unsigned int x = seed + idx;
    x = x ^ (x >> 3);
    x = x ^ (x << 7);
    x = x ^ (x >> 5);
    float r = static_cast<float>(x) / static_cast<float>(0xffffffff);
    data[idx] = low + r * (high - low);
  }
}

__global__ void sliceDimKernel(float *output, const float *input, int start, int end, int outer, int inner, int shape_dim, int new_dim_size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < outer * new_dim_size * inner)
  {
    int o = idx / (new_dim_size * inner);
    int rest = idx % (new_dim_size * inner);
    int i = rest / inner;
    int j = rest % inner;
    int old_idx = (o * shape_dim + (start + i)) * inner + j;
    int new_idx = (o * new_dim_size + i) * inner + j;
    output[new_idx] = input[old_idx];
  }
}

__global__ void oneHotKernel(float *output, const float *labels, int batch_size, int num_classes)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < batch_size)
  {
    int label = static_cast<int>(labels[idx]);
    if (label >= 0 && label < num_classes)
    {
      output[idx * num_classes + label] = 1.0f;
    }
  }
}

// Kernel for padding operation
__global__ void padKernel(float *output, const float *input, int N, int C, int H, int W,
                          int new_N, int new_C, int new_H, int new_W,
                          int pad_N0, int pad_C0, int pad_H0, int pad_W0)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_size = new_N * new_C * new_H * new_W;
  if (idx < total_size)
  {
    // Compute output indices
    int n = idx / (new_C * new_H * new_W);
    int tmp = idx % (new_C * new_H * new_W);
    int c = tmp / (new_H * new_W);
    tmp = tmp % (new_H * new_W);
    int h = tmp / new_W;
    int w = tmp % new_W;

    // Check if within original tensor bounds
    if (n >= pad_N0 && n < (N + pad_N0) &&
        c >= pad_C0 && c < (C + pad_C0) &&
        h >= pad_H0 && h < (H + pad_H0) &&
        w >= pad_W0 && w < (W + pad_W0))
    {
      // Compute input index
      int in_n = n - pad_N0;
      int in_c = c - pad_C0;
      int in_h = h - pad_H0;
      int in_w = w - pad_W0;
      int in_idx = ((in_n * C + in_c) * H + in_h) * W + in_w;
      output[idx] = input[in_idx];
    }
    else
    {
      output[idx] = 0.0f; // Padding with zeros
    }
  }
}
__global__ void concatKernel(
    float *output,
    float **inputs,
    int *input_sizes,
    int num_tensors,
    int dim,
    int total_size,
    int offset,
    int *strides,
    int *shape,
    int dims)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total_size)
  {
    int temp = idx;
    int tensor_idx = 0;
    int d = dim;
    int coord_d = (temp / strides[d]) % shape[d];

    // Encuentra qué tensor corresponde a esta coordenada en la dimensión de concat
    while (tensor_idx < num_tensors - 1 && coord_d >= input_sizes[tensor_idx])
    {
      coord_d -= input_sizes[tensor_idx];
      offset += input_sizes[tensor_idx] * strides[d];
      tensor_idx++;
    }

    // Calcula el índice del elemento dentro del tensor de origen
    int input_idx = 0;
    for (int i = dims - 1; i >= 0; --i)
    {
      int coord = (i == d) ? coord_d : (temp / strides[i]) % shape[i];
      input_idx += coord * strides[i];
    }

    // Copia el valor al tensor de salida
    output[idx] = inputs[tensor_idx][input_idx];
  }
}
__global__ void sliceKernel(float *output, const float *input, int *starts, int *ends, int *old_shape, int *new_shape, int *strides, int dims, int total_size)
{
  int new_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (new_idx < total_size)
  {
    // Convert new_idx to new multi-dimensional indices
    int temp = new_idx;
    int old_idx = 0;
    for (int d = dims - 1; d >= 0; --d)
    {
      int coord = temp % new_shape[d] + starts[d];
      old_idx += coord * strides[d];
      temp /= new_shape[d];
    }
    output[new_idx] = input[old_idx];
  }
}

class Tensor
{
public:
  std::vector<int> shape;
  float *data; // Puntero a memoria GPU
  size_t size; // Tamaño total del tensor

  static std::mt19937 global_gen; // Generador global (para CPU)

  static void set_seed(unsigned int seed)
  {
    global_gen.seed(seed);
  }

  Tensor() : data(nullptr), size(0) {}

  Tensor(const std::vector<int> &shape) : shape(shape)
  {
    size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    cudaError_t err = cudaMalloc(&data, size * sizeof(float));
    if (err != cudaSuccess)
    {
      throw std::runtime_error("cudaMalloc failed: " + std::string(cudaGetErrorString(err)));
    }
    // Inicializar a cero
    cudaMemset(data, 0, size * sizeof(float));
  }

  Tensor(const std::vector<int> &shape, const std::vector<float> &host_data) : shape(shape)
  {
    size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    if (host_data.size() != size)
    {
      throw std::runtime_error("Data size does not match tensor shape");
    }
    cudaError_t err = cudaMalloc(&data, size * sizeof(float));
    if (err != cudaSuccess)
    {
      throw std::runtime_error("cudaMalloc failed: " + std::string(cudaGetErrorString(err)));
    }
    // Copiar datos desde CPU a GPU
    err = cudaMemcpy(data, host_data.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
      throw std::runtime_error("cudaMemcpy failed: " + std::string(cudaGetErrorString(err)));
    }
  }

  ~Tensor()
  {
    if (data)
    {
      cudaFree(data);
    }
  }
  // Copy assignment operator to handle GPU memory
  Tensor &operator=(const Tensor &other)
  {
    if (this != &other)
    {
      if (data)
      {
        cudaFree(data);
      }
      shape = other.shape;
      size = other.size;
      cudaError_t err = cudaMalloc(&data, size * sizeof(float));
      if (err != cudaSuccess)
      {
        throw std::runtime_error("cudaMalloc failed: " + std::string(cudaGetErrorString(err)));
      }
      err = cudaMemcpy(data, other.data, size * sizeof(float), cudaMemcpyDeviceToDevice);
      if (err != cudaSuccess)
      {
        throw std::runtime_error("cudaMemcpy failed: " + std::string(cudaGetErrorString(err)));
      }
    }
    return *this;
  }

  // -------------------- Inicialización --------------------
  static Tensor zeros(const std::vector<int> &shape)
  {
    return Tensor(shape); // Ya inicializa a cero en el constructor
  }

  static Tensor rand_uniform(const std::vector<int> &shape, float low, float high)
  {
    Tensor t(shape);
    // Generar datos en CPU temporalmente
    std::vector<float> host_data(t.size);
    std::uniform_real_distribution<float> dist(low, high);
    for (float &x : host_data)
    {
      x = dist(global_gen);
    }
    // Copiar a GPU
    cudaError_t err = cudaMemcpy(t.data, host_data.data(), t.size * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
      throw std::runtime_error("cudaMemcpy failed: " + std::string(cudaGetErrorString(err)));
    }
    return t;
  }

  static Tensor rand_uniform_gpu(const std::vector<int> &shape, float low, float high)
  {
    Tensor t(shape);
    int threadsPerBlock = 256;
    int blocks = (t.size + threadsPerBlock - 1) / threadsPerBlock;
    uniformKernel<<<blocks, threadsPerBlock>>>(t.data, t.size, low, high, static_cast<unsigned int>(time(nullptr)));
    cudaDeviceSynchronize(); // global, no uses Tensor::cudaDeviceSynchronize()

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
      throw std::runtime_error("CUDA kernel failed: " + std::string(cudaGetErrorString(err)));
    }
    return t;
  }

  static Tensor xavier_normal(const std::vector<int> &shape, int fan_in, int fan_out)
  {
    Tensor t(shape);
    float scale = std::sqrt(2.0f / (fan_in + fan_out));
    std::vector<float> host_data(t.size);
    std::normal_distribution<float> dist(0.0f, scale);
    for (float &x : host_data)
    {
      x = dist(global_gen);
    }
    cudaError_t err = cudaMemcpy(t.data, host_data.data(), t.size * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
      throw std::runtime_error("cudaMemcpy failed: " + std::string(cudaGetErrorString(err)));
    }
    return t;
  }

  static Tensor xavier_uniform(const std::vector<int> &shape, int fan_in, int fan_out)
  {
    Tensor t(shape);
    float scale = std::sqrt(6.0f / (fan_in + fan_out));
    std::vector<float> host_data(t.size);
    std::uniform_real_distribution<float> dist(-scale, scale);
    for (float &x : host_data)
    {
      x = dist(global_gen);
    }
    cudaError_t err = cudaMemcpy(t.data, host_data.data(), t.size * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
      throw std::runtime_error("cudaMemcpy failed: " + std::string(cudaGetErrorString(err)));
    }
    return t;
  }

  static Tensor kaiming_normal(const std::vector<int> &shape, int fan_in, bool use_negative = true)
  {
    Tensor t(shape);
    float scale = std::sqrt(2.0f / fan_in);
    std::vector<float> host_data(t.size);
    std::normal_distribution<float> dist(0.0f, scale);
    for (float &x : host_data)
    {
      x = dist(global_gen);
    }
    cudaError_t err = cudaMemcpy(t.data, host_data.data(), t.size * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
      throw std::runtime_error("cudaMemcpy failed: " + std::string(cudaGetErrorString(err)));
    }
    return t;
  }

  static Tensor kaiming_uniform(const std::vector<int> &shape, int fan_in, bool use_negative = true)
  {
    Tensor t(shape);
    float scale = std::sqrt(6.0f / fan_in);
    std::vector<float> host_data(t.size);
    std::uniform_real_distribution<float> dist(-scale, scale);
    for (float &x : host_data)
    {
      x = dist(global_gen);
    }
    cudaError_t err = cudaMemcpy(t.data, host_data.data(), t.size * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
      throw std::runtime_error("cudaMemcpy failed: " + std::string(cudaGetErrorString(err)));
    }
    return t;
  }

  static Tensor one_hot(const Tensor &labels, int num_classes)
  {
    if (labels.shape.size() != 2 || labels.shape[1] != 1)
    {
      throw std::invalid_argument("Labels tensor must have shape [batch_size, 1]");
    }

    int batch_size = labels.shape[0];
    std::vector<int> new_shape = {batch_size, num_classes};
    Tensor result(new_shape);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocks = (batch_size + threadsPerBlock - 1) / threadsPerBlock;
    oneHotKernel<<<blocks, threadsPerBlock>>>(result.data, labels.data, batch_size, num_classes);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
      throw std::runtime_error("oneHotKernel failed: " + std::string(cudaGetErrorString(err)));
    }

    return result;
  }

  Tensor slice(const std::vector<int> &starts, const std::vector<int> &ends) const
  {
    if (starts.size() != shape.size() || ends.size() != shape.size())
    {
      throw std::invalid_argument("starts and ends must match tensor dimensions");
    }

    // Verify bounds and compute new shape
    std::vector<int> new_shape;
    for (size_t i = 0; i < shape.size(); ++i)
    {
      if (starts[i] < 0 || ends[i] > shape[i] || starts[i] >= ends[i])
      {
        throw std::out_of_range("Invalid slice bounds");
      }
      new_shape.push_back(ends[i] - starts[i]);
    }

    // Calculate strides for original shape
    std::vector<int> strides(shape.size(), 1);
    for (int i = shape.size() - 2; i >= 0; --i)
    {
      strides[i] = strides[i + 1] * shape[i + 1];
    }

    // Allocate output tensor
    int total_size = std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies<int>());
    Tensor result(new_shape);

    // Copy shape, starts, ends, and strides to GPU
    int *d_starts, *d_ends, *d_old_shape, *d_new_shape, *d_strides;
    cudaMalloc(&d_starts, shape.size() * sizeof(int));
    cudaMalloc(&d_ends, shape.size() * sizeof(int));
    cudaMalloc(&d_old_shape, shape.size() * sizeof(int));
    cudaMalloc(&d_new_shape, shape.size() * sizeof(int));
    cudaMalloc(&d_strides, shape.size() * sizeof(int));

    cudaMemcpy(d_starts, starts.data(), shape.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ends, ends.data(), shape.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_old_shape, shape.data(), shape.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_new_shape, new_shape.data(), shape.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_strides, strides.data(), shape.size() * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocks = (total_size + threadsPerBlock - 1) / threadsPerBlock;
    sliceKernel<<<blocks, threadsPerBlock>>>(result.data, data, d_starts, d_ends, d_old_shape, d_new_shape, d_strides, shape.size(), total_size);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
      cudaFree(d_starts);
      cudaFree(d_ends);
      cudaFree(d_old_shape);
      cudaFree(d_new_shape);
      cudaFree(d_strides);
      throw std::runtime_error("sliceKernel failed: " + std::string(cudaGetErrorString(err)));
    }

    // Free auxiliary GPU memory
    cudaFree(d_starts);
    cudaFree(d_ends);
    cudaFree(d_old_shape);
    cudaFree(d_new_shape);
    cudaFree(d_strides);
    return result;
  }

  Tensor slice(int dim, int start, int end) const
  {
    if (dim < 0 || dim >= shape.size() || start < 0 || end > shape[dim] || start >= end)
    {
      throw std::out_of_range("Invalid slice indices");
    }

    std::vector<int> new_shape = shape;
    new_shape[dim] = end - start;

    int outer = 1;
    int inner = 1;

    for (int i = 0; i < dim; ++i)
      outer *= shape[i];
    for (int i = dim + 1; i < shape.size(); ++i)
      inner *= shape[i];

    int total_size = outer * (end - start) * inner;
    Tensor result(new_shape);

    int threadsPerBlock = 256;
    int blocks = (total_size + threadsPerBlock - 1) / threadsPerBlock;
    sliceDimKernel<<<blocks, threadsPerBlock>>>(
        result.data,
        data,
        start,
        end,
        outer,
        inner,
        shape[dim],
        end - start);

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
      throw std::runtime_error("sliceDimKernel failed: " + std::string(cudaGetErrorString(err)));
    }

    return result;
  }

  static Tensor concat(const std::vector<Tensor> &tensors, int dim)
  {
    if (tensors.empty())
    {
      throw std::invalid_argument("No tensors provided for concat");
    }

    // Verify shapes are compatible
    std::vector<int> new_shape = tensors[0].shape;
    int concat_dim_size = 0;
    for (const auto &t : tensors)
    {
      if (t.shape.size() != new_shape.size())
      {
        throw std::invalid_argument("Tensor dimensions mismatch");
      }
      for (size_t i = 0; i < new_shape.size(); ++i)
      {
        if (i != dim && t.shape[i] != new_shape[i])
        {
          throw std::invalid_argument("Tensor shapes mismatch in dimension " + std::to_string(i));
        }
      }
      concat_dim_size += t.shape[dim];
    }
    new_shape[dim] = concat_dim_size;

    // Allocate output tensor
    Tensor result(new_shape);
    int total_size = result.size;

    // Prepare data for kernel
    float **d_inputs;
    cudaMalloc(&d_inputs, tensors.size() * sizeof(float *));
    std::vector<float *> h_inputs(tensors.size());
    for (size_t i = 0; i < tensors.size(); ++i)
    {
      h_inputs[i] = tensors[i].data;
    }
    cudaMemcpy(d_inputs, h_inputs.data(), tensors.size() * sizeof(float *), cudaMemcpyHostToDevice);

    std::vector<int> input_sizes(tensors.size());
    for (size_t i = 0; i < tensors.size(); ++i)
    {
      input_sizes[i] = tensors[i].shape[dim];
    }
    int *d_input_sizes;
    cudaMalloc(&d_input_sizes, tensors.size() * sizeof(int));
    cudaMemcpy(d_input_sizes, input_sizes.data(), tensors.size() * sizeof(int), cudaMemcpyHostToDevice);

    std::vector<int> strides(new_shape.size(), 1);
    for (int i = new_shape.size() - 2; i >= 0; --i)
    {
      strides[i] = strides[i + 1] * new_shape[i + 1];
    }
    int *d_strides, *d_shape;
    cudaMalloc(&d_strides, new_shape.size() * sizeof(int));
    cudaMalloc(&d_shape, new_shape.size() * sizeof(int));
    cudaMemcpy(d_strides, strides.data(), new_shape.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_shape, new_shape.data(), new_shape.size() * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocks = (total_size + threadsPerBlock - 1) / threadsPerBlock;
    int offset = 0;
    concatKernel<<<blocks, threadsPerBlock>>>(result.data, d_inputs, d_input_sizes, tensors.size(), dim, total_size, offset, d_strides, d_shape, new_shape.size());
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
      cudaFree(d_inputs);
      cudaFree(d_input_sizes);
      cudaFree(d_strides);
      cudaFree(d_shape);
      throw std::runtime_error("concatKernel failed: " + std::string(cudaGetErrorString(err)));
    }

    // Free auxiliary memory
    cudaFree(d_inputs);
    cudaFree(d_input_sizes);
    cudaFree(d_strides);
    cudaFree(d_shape);

    return result;
  }

  Tensor pad(const std::vector<int> &pads) const
  {
    if (shape.size() != 4)
    {
      throw std::invalid_argument("Tensor must have 4 dimensions [N, C, H, W]");
    }
    if (pads.size() != 8)
    {
      throw std::invalid_argument("Pads must have 8 values [N_pre, N_post, C_pre, C_post, H_pre, H_post, W_pre, W_post]");
    }

    int N = shape[0], C = shape[1], H = shape[2], W = shape[3];
    int pad_N0 = pads[0], pad_N1 = pads[1];
    int pad_C0 = pads[2], pad_C1 = pads[3];
    int pad_H0 = pads[4], pad_H1 = pads[5];
    int pad_W0 = pads[6], pad_W1 = pads[7];

    int new_N = N + pad_N0 + pad_N1;
    int new_C = C + pad_C0 + pad_C1;
    int new_H = H + pad_H0 + pad_H1;
    int new_W = W + pad_W0 + pad_W1;

    std::vector<int> new_shape = {new_N, new_C, new_H, new_W};
    Tensor result(new_shape);

    // Launch kernel
    int total_size = new_N * new_C * new_H * new_W;
    int threadsPerBlock = 256;
    int blocks = (total_size + threadsPerBlock - 1) / threadsPerBlock;
    padKernel<<<blocks, threadsPerBlock>>>(result.data, data, N, C, H, W,
                                           new_N, new_C, new_H, new_W,
                                           pad_N0, pad_C0, pad_H0, pad_W0);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
      throw std::runtime_error("padKernel failed: " + std::string(cudaGetErrorString(err)));
    }

    return result;
  }

  float at(const std::vector<int> &index) const
  {
    if (index.size() != shape.size())
    {
      throw std::invalid_argument("Index size must match tensor dimensions");
    }
    int flat_index = 0;
    int multiplier = 1;
    for (int i = shape.size() - 1; i >= 0; --i)
    {
      if (index[i] < 0 || index[i] >= shape[i])
      {
        throw std::out_of_range("Index out of bounds");
      }
      flat_index += index[i] * multiplier;
      multiplier *= shape[i];
    }
    float result;
    cudaError_t err = cudaMemcpy(&result, data + flat_index, sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
      throw std::runtime_error("cudaMemcpy failed: " + std::string(cudaGetErrorString(err)));
    }
    return result;
  }

  float &at(std::initializer_list<int> indices)
  {
    if (indices.size() != shape.size())
    {
      throw std::invalid_argument("Index size must match tensor dimensions");
    }
    int offset = 0;
    int stride = 1;
    auto idx_it = indices.end();
    auto shape_it = shape.end();
    for (int i = 0; i < shape.size(); ++i)
    {
      --idx_it;
      --shape_it;
      if (*idx_it < 0 || *idx_it >= *shape_it)
      {
        throw std::out_of_range("Index out of bounds");
      }
      offset += (*idx_it) * stride;
      stride *= *shape_it;
    }
    // Create a host-managed pointer to GPU memory (not ideal, see note)
    static float *host_ptr = nullptr;
    if (host_ptr)
    {
      cudaFreeHost(host_ptr);
    }
    cudaMallocHost(&host_ptr, sizeof(float));
    cudaError_t err = cudaMemcpy(host_ptr, data + offset, sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
      throw std::runtime_error("cudaMemcpy failed: " + std::string(cudaGetErrorString(err)));
    }
    return *host_ptr;
  }

  const float &at(std::initializer_list<int> indices) const
  {
    if (indices.size() != shape.size())
    {
      throw std::invalid_argument("Index size must match tensor dimensions");
    }
    int offset = 0;
    int stride = 1;
    auto idx_it = indices.end();
    auto shape_it = shape.end();
    for (int i = 0; i < shape.size(); ++i)
    {
      --idx_it;
      --shape_it;
      if (*idx_it < 0 || *idx_it >= *shape_it)
      {
        throw std::out_of_range("Index out of bounds");
      }
      offset += (*idx_it) * stride;
      stride *= *shape_it;
    }
    static float result;
    cudaError_t err = cudaMemcpy(&result, data + offset, sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
      throw std::runtime_error("cudaMemcpy failed: " + std::string(cudaGetErrorString(err)));
    }
    return result;
  }

  Tensor flatten(int start_dim) const
  {
    if (start_dim < 0 || start_dim >= static_cast<int>(shape.size()))
    {
      throw std::invalid_argument("Invalid start_dim in flatten");
    }

    // Calculate flattened dimension
    int flattened_dim = 1;
    for (size_t i = start_dim; i < shape.size(); ++i)
    {
      flattened_dim *= shape[i];
    }

    // Create new shape
    std::vector<int> new_shape;
    for (int i = 0; i < start_dim; ++i)
    {
      new_shape.push_back(shape[i]);
    }
    new_shape.push_back(flattened_dim);

    // Create new tensor with same data
    Tensor result = *this; // Copies shape and allocates new GPU memory
    result.shape = new_shape;
    cudaError_t err = cudaMemcpy(result.data, data, size * sizeof(float), cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess)
    {
      throw std::runtime_error("cudaMemcpy failed: " + std::string(cudaGetErrorString(err)));
    }
    return result;
  }

  void fill(float value)
  {
    int threadsPerBlock = 256;
    int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    fillKernel<<<blocks, threadsPerBlock>>>(data, value, size);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
      throw std::runtime_error("fillKernel failed: " + std::string(cudaGetErrorString(err)));
    }
  }

  Tensor sum(int axis, bool keepdims = false) const
  {
    if (axis < 0)
    {
      axis += shape.size();
    }
    if (axis < 0 || axis >= static_cast<int>(shape.size()))
    {
      throw std::invalid_argument("Invalid axis");
    }

    if (shape.size() == 2 && axis == 0)
    {
      int rows = shape[0];
      int cols = shape[1];
      std::vector<int> new_shape = {cols};
      if (keepdims)
      {
        new_shape = {1, cols};
      }
      Tensor result(new_shape);

      int threadsPerBlock = 256;
      int blocks = (cols + threadsPerBlock - 1) / threadsPerBlock;
      sumAxis0Kernel<<<blocks, threadsPerBlock>>>(result.data, data, rows, cols);
      cudaDeviceSynchronize();
      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess)
      {
        throw std::runtime_error("sumAxis0Kernel failed: " + std::string(cudaGetErrorString(err)));
      }
      return result;
    }
    else if (shape.size() == 3 && axis == 2)
    {
      int N = shape[0];
      int HW = shape[1];
      int L = shape[2];
      std::vector<int> new_shape = {N, HW};
      if (keepdims)
      {
        new_shape = {N, HW, 1};
      }
      Tensor result(new_shape);

      int threadsPerBlock = 256;
      int blocks = (N * HW + threadsPerBlock - 1) / threadsPerBlock;
      sumAxis2Kernel<<<blocks, threadsPerBlock>>>(result.data, data, N, HW, L);
      cudaDeviceSynchronize();
      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess)
      {
        throw std::runtime_error("sumAxis2Kernel failed: " + std::string(cudaGetErrorString(err)));
      }
      return result;
    }

    throw std::runtime_error("sum: unsupported shape or axis");
  }

  Tensor sum(const std::vector<int> &dims) const
  {
    // Identify axes to reduce
    std::vector<int> reduce_axes(shape.size(), 0);
    for (int d : dims)
    {
      if (d < 0 || d >= static_cast<int>(shape.size()))
        throw std::invalid_argument("Invalid dimension in dims");
      reduce_axes[d] = 1;
    }
    // Compute new shape
    std::vector<int> new_shape;
    for (size_t i = 0; i < shape.size(); ++i)
    {
      if (!reduce_axes[i])
      {
        new_shape.push_back(shape[i]);
      }
    }
    if (new_shape.empty())
    {
      new_shape.push_back(1);
    }

    // Create result tensor
    Tensor result(new_shape);

    // Compute strides
    std::vector<int> in_strides(shape.size());
    in_strides[shape.size() - 1] = 1;
    for (int i = shape.size() - 2; i >= 0; --i)
    {
      in_strides[i] = in_strides[i + 1] * shape[i + 1];
    }

    std::vector<int> out_strides(new_shape.size());
    if (!new_shape.empty())
    {
      out_strides[new_shape.size() - 1] = 1;
      for (int i = new_shape.size() - 2; i >= 0; --i)
      {
        out_strides[i] = out_strides[i + 1] * new_shape[i + 1];
      }
    }

    // Allocate GPU memory for strides and reduce_axes
    int *d_in_strides, *d_out_strides;
    bool *d_reduce_axes;
    cudaMalloc(&d_in_strides, shape.size() * sizeof(int));
    cudaMalloc(&d_out_strides, new_shape.size() * sizeof(int));
    cudaMalloc(&d_reduce_axes, shape.size() * sizeof(bool));
    cudaMemcpy(d_in_strides, in_strides.data(), shape.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_strides, out_strides.data(), new_shape.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_reduce_axes, reduce_axes.data(), shape.size() * sizeof(bool), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    sumMultiDimsKernel<<<blocks, threadsPerBlock>>>(result.data, data, d_in_strides, d_out_strides,
                                                    d_reduce_axes, size, result.size, shape.size());
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
      cudaFree(d_in_strides);
      cudaFree(d_out_strides);
      cudaFree(d_reduce_axes);
      throw std::runtime_error("sumMultiDimsKernel failed: " + std::string(cudaGetErrorString(err)));
    }

    // Free GPU memory
    cudaFree(d_in_strides);
    cudaFree(d_out_strides);
    cudaFree(d_reduce_axes);

    return result;
  }

  Tensor mean(int dim, bool keepdim) const
  {
    if (dim < 0 || dim >= static_cast<int>(shape.size()))
    {
      throw std::invalid_argument("Invalid dimension");
    }

    // Compute new shape
    std::vector<int> new_shape = shape;
    if (keepdim)
    {
      new_shape[dim] = 1;
    }
    else
    {
      new_shape.erase(new_shape.begin() + dim);
    }
    if (new_shape.empty())
    {
      new_shape.push_back(1);
    }

    // Create result tensor
    Tensor result(new_shape);

    // Compute outer and inner sizes
    int outer = 1, inner = 1;
    for (int i = 0; i < dim; ++i)
    {
      outer *= shape[i];
    }
    for (size_t i = dim + 1; i < shape.size(); ++i)
    {
      inner *= shape[i];
    }
    int N = shape[dim];

    // Launch kernel
    int threadsPerBlock = 256;
    int blocks = (outer * inner + threadsPerBlock - 1) / threadsPerBlock;
    meanDimKernel<<<blocks, threadsPerBlock>>>(result.data, data, outer, N, inner, shape[dim]);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
      throw std::runtime_error("meanDimKernel failed: " + std::string(cudaGetErrorString(err)));
    }

    return result;
  }

  Tensor operator/(float value) const
  {
    Tensor result(shape);
    int threadsPerBlock = 256;
    int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    scalarDivKernel<<<blocks, threadsPerBlock>>>(result.data, data, value, size);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
      throw std::runtime_error("scalarDivKernel failed: " + std::string(cudaGetErrorString(err)));
    }
    return result;
  }

  Tensor reshape(const std::vector<int> &new_shape) const
  {
    int new_size = std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies<int>());
    if (new_size != static_cast<int>(size))
    {
      throw std::invalid_argument("Reshape size must match original size");
    }
    Tensor result = *this; // Copies data to new GPU memory
    result.shape = new_shape;
    cudaError_t err = cudaMemcpy(result.data, data, size * sizeof(float), cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess)
    {
      throw std::runtime_error("cudaMemcpy failed: " + std::string(cudaGetErrorString(err)));
    }
    return result;
  }

  Tensor pow(float exponent) const
  {
    Tensor out(shape);
    int threadsPerBlock = 256;
    int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    powKernel<<<blocks, threadsPerBlock>>>(out.data, data, exponent, size);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
      throw std::runtime_error("powKernel failed: " + std::string(cudaGetErrorString(err)));
    }
    return out;
  }

  Tensor sqrt() const
  {
    Tensor out(shape);
    int threadsPerBlock = 256;
    int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    sqrtKernel<<<blocks, threadsPerBlock>>>(out.data, data, size);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
      throw std::runtime_error("sqrtKernel failed: " + std::string(cudaGetErrorString(err)));
    }
    return out;
  }

  Tensor reciprocal() const
  {
    Tensor out(shape);
    int threadsPerBlock = 256;
    int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    reciprocalKernel<<<blocks, threadsPerBlock>>>(out.data, data, size);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
      throw std::runtime_error("reciprocalKernel failed: " + std::string(cudaGetErrorString(err)));
    }
    return out;
  }

  Tensor transpose(const std::vector<int> &axes) const
  {
    if (axes.size() != shape.size())
    {
      throw std::runtime_error("Number of axes must match tensor dimensions");
    }

    // Validate axes
    std::vector<bool> seen(shape.size(), false);
    for (int d : axes)
    {
      if (d < 0 || d >= static_cast<int>(shape.size()) || seen[d])
      {
        throw std::runtime_error("Invalid axes permutation");
      }
      seen[d] = true;
    }

    // Compute new shape
    std::vector<int> new_shape(shape.size());
    for (size_t i = 0; i < axes.size(); ++i)
    {
      new_shape[i] = shape[axes[i]];
    }

    // Compute strides
    std::vector<int> old_strides(shape.size());
    old_strides[shape.size() - 1] = 1;
    for (int i = shape.size() - 2; i >= 0; --i)
    {
      old_strides[i] = old_strides[i + 1] * shape[i + 1];
    }

    std::vector<int> new_strides(new_shape.size());
    new_strides[new_shape.size() - 1] = 1;
    for (int i = new_shape.size() - 2; i >= 0; --i)
    {
      new_strides[i] = new_strides[i + 1] * new_shape[i + 1];
    }

    // Allocate result tensor
    Tensor out(new_shape);

    // Copy axes and strides to GPU
    int *d_axes, *d_old_shape, *d_old_strides, *d_new_strides;
    cudaMalloc(&d_axes, shape.size() * sizeof(int));
    cudaMalloc(&d_old_shape, shape.size() * sizeof(int));
    cudaMalloc(&d_old_strides, shape.size() * sizeof(int));
    cudaMalloc(&d_new_strides, shape.size() * sizeof(int));

    cudaMemcpy(d_axes, axes.data(), shape.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_old_shape, shape.data(), shape.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_old_strides, old_strides.data(), shape.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_new_strides, new_strides.data(), shape.size() * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    transposeKernel<<<blocks, threadsPerBlock>>>(out.data, data, d_axes, d_old_shape,
                                                 d_old_strides, d_new_strides, shape.size(), size);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
      cudaFree(d_axes);
      cudaFree(d_old_shape);
      cudaFree(d_old_strides);
      cudaFree(d_new_strides);
      throw std::runtime_error("transposeKernel failed: " + std::string(cudaGetErrorString(err)));
    }

    // Free GPU memory
    cudaFree(d_axes);
    cudaFree(d_old_shape);
    cudaFree(d_old_strides);
    cudaFree(d_new_strides);

    return out;
  }

  Tensor permute(const std::vector<int> &dims) const
  {
    // `permute` is identical to `transpose` in this implementation
    return transpose(dims);
  }

  Tensor matmul(const Tensor &other) const
  {
    if (shape.size() == 2 && other.shape.size() == 2)
    {
      int m = shape[0], n = shape[1];
      int n2 = other.shape[0], p = other.shape[1];
      if (n != n2)
      {
        throw std::invalid_argument("Matrix dimensions must match for matmul");
      }
      Tensor result({m, p});

      dim3 threadsPerBlock(16, 16);
      dim3 blocks((p + threadsPerBlock.x - 1) / threadsPerBlock.x,
                  (m + threadsPerBlock.y - 1) / threadsPerBlock.y);
      matmul2DKernel<<<blocks, threadsPerBlock>>>(result.data, data, other.data, m, n, p);
      cudaDeviceSynchronize();
      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess)
      {
        throw std::runtime_error("matmul2DKernel failed: " + std::string(cudaGetErrorString(err)));
      }
      return result;
    }
    else if (shape.size() == 3 && other.shape.size() == 3)
    {
      int B1 = shape[0], M = shape[1], N = 2;
      int B2 = other.shape[0], N2 = other.shape[1], P = other.shape[2];
      if (B1 != B2 || N != N2)
      {
        throw std::invalid_argument("Batch matrix dimensions must match for matmul");
      }
      Tensor result({B1, M, P});

      dim3 threadsPerBlock(8, 8, 8);
      dim3 blocks((P + threadsPerBlock.x - 1) / threadsPerBlock.x,
                  (M + threadsPerBlock.y - 1) / threadsPerBlock.y,
                  (B1 + threadsPerBlock.z - 1) / threadsPerBlock.z);
      matmul3DKernel<<<blocks, threadsPerBlock>>>(result.data, data, other.data, B1, M, N, P);
      cudaDeviceSynchronize();
      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess)
      {
        throw std::runtime_error("matmul3DKernel failed: " + std::string(cudaGetErrorString(err)));
      }
      return result;
    }

    throw std::invalid_argument("matmul only supports 2D or 3D tensors with matching inner dimensions");
  }

  static Tensor bernoulli(const std::vector<int> &shape, float p)
  {
    if (p < 0.0f || p > 1.0f)
    {
      throw std::invalid_argument("Probability p must be in [0, 1]");
    }
    int total = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    Tensor result(shape);

    int threadsPerBlock = 256;
    int blocks = (total + threadsPerBlock - 1) / threadsPerBlock;
    bernoulliKernel<<<blocks, threadsPerBlock>>>(result.data, p, total, static_cast<unsigned int>(time(nullptr)));
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
      throw std::runtime_error("bernoulliKernel failed: " + std::string(cudaGetErrorString(err)));
    }
    return result;
  }

  Tensor relu() const
  {
    Tensor out(shape);
    int threadsPerBlock = 256;
    int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    reluKernel<<<blocks, threadsPerBlock>>>(out.data, data, size);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
      throw std::runtime_error("reluKernel failed: " + std::string(cudaGetErrorString(err)));
    }
    return out;
  }

  Tensor relu_derivative() const
  {
    Tensor out(shape);
    int threadsPerBlock = 256;
    int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    reluDerivativeKernel<<<blocks, threadsPerBlock>>>(out.data, data, size);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
      throw std::runtime_error("reluDerivativeKernel failed: " + std::string(cudaGetErrorString(err)));
    }
    return out;
  }

  Tensor softmax(int dim) const
  {
    if (dim < 0 || dim >= static_cast<int>(shape.size()))
    {
      throw std::invalid_argument("Invalid dimension for softmax");
    }

    // Compute outer and inner sizes
    int outer = 1, inner = 1, dim_size = shape[dim];
    for (int i = 0; i < dim; ++i)
    {
      outer *= shape[i];
    }
    for (size_t i = dim + 1; i < shape.size(); ++i)
    {
      inner *= shape[i];
    }

    // Allocate output tensor and temporary max/sum arrays
    Tensor out(shape);
    float *max_vals;
    cudaMalloc(&max_vals, outer * inner * sizeof(float));
    float *sums;
    cudaMalloc(&sums, outer * inner * sizeof(float));

    // Launch kernels
    int threadsPerBlock = 256;
    int blocks = (outer * inner + threadsPerBlock - 1) / threadsPerBlock;

    // Find max
    softmaxMaxKernel<<<blocks, threadsPerBlock>>>(max_vals, data, outer, dim_size, inner);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
      cudaFree(max_vals);
      cudaFree(sums);
      throw std::runtime_error("softmaxMaxKernel failed: " + std::string(cudaGetErrorString(err)));
    }

    // Compute exp and sum
    softmaxExpSumKernel<<<blocks, threadsPerBlock>>>(out.data, sums, data, max_vals, outer, dim_size, inner);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
      cudaFree(max_vals);
      cudaFree(sums);
      throw std::runtime_error("softmaxExpSumKernel failed: " + std::string(cudaGetErrorString(err)));
    }

    // Normalize
    softmaxNormalizeKernel<<<blocks, threadsPerBlock>>>(out.data, sums, outer, dim_size, inner);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
      cudaFree(max_vals);
      cudaFree(sums);
      throw std::runtime_error("softmaxNormalizeKernel failed: " + std::string(cudaGetErrorString(err)));
    }

    // Free temporary arrays
    cudaFree(max_vals);
    cudaFree(sums);

    return out;
  }

  Tensor operator*(const Tensor &other) const
  {
    if (shape != other.shape)
    {
      throw std::invalid_argument("Tensor shapes must match for element-wise multiplication");
    }
    Tensor result(shape);
    int threadsPerBlock = 256;
    int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    elementwiseMulKernel<<<blocks, threadsPerBlock>>>(result.data, data, other.data, size);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
      throw std::runtime_error("elementwiseMulKernel failed: " + std::string(cudaGetErrorString(err)));
    }
    return result;
  }

  Tensor operator-(const Tensor &other) const
  {
    if (shape != other.shape)
    {
      throw std::invalid_argument("Tensor shapes must match for element-wise subtraction");
    }
    Tensor result(shape);
    int threadsPerBlock = 256;
    int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    elementwiseSubKernel<<<blocks, threadsPerBlock>>>(result.data, data, other.data, size);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
      throw std::runtime_error("elementwiseSubKernel failed: " + std::string(cudaGetErrorString(err)));
    }
    return result;
  }

  static Tensor softmax_backward(const Tensor &softmax, const Tensor &delta)
  {
    if (softmax.shape != delta.shape)
    {
      throw std::invalid_argument("Softmax and delta shapes must match");
    }
    Tensor sum = (delta * softmax).sum(-1, true); // [N, HW, 1]
    return softmax * (delta - sum);
  }

  Tensor operator+(const Tensor &other) const
  {
    // Case 1: Same shape
    if (shape == other.shape)
    {
      Tensor result(shape);
      int threadsPerBlock = 256;
      int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
      addKernel<<<blocks, threadsPerBlock>>>(result.data, data, other.data,
                                             shape.data(), shape.data(), shape.data(), shape.data(),
                                             shape.size(), size);
      cudaDeviceSynchronize();
      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess)
      {
        throw std::runtime_error("addKernel failed: " + std::string(cudaGetErrorString(err)));
      }
      return result;
    }

    // Compute strides
    std::vector<int> a_strides(shape.size(), 1);
    std::vector<int> b_strides(other.shape.size(), 1);
    for (int i = shape.size() - 2; i >= 0; --i)
    {
      a_strides[i] = a_strides[i + 1] * shape[i + 1];
    }
    for (int i = other.shape.size() - 2; i >= 0; --i)
    {
      b_strides[i] = b_strides[i + 1] * other.shape[i + 1];
    }

    // Case 2: Broadcasting [B, D] + [1, D]
    if (shape.size() == 2 && other.shape.size() == 2 &&
        other.shape[0] == 1 && shape[1] == other.shape[1])
    {
      Tensor result(shape);
      int *d_a_shape, *d_b_shape, *d_a_strides, *d_b_strides;
      cudaMalloc(&d_a_shape, shape.size() * sizeof(int));
      cudaMalloc(&d_b_shape, other.shape.size() * sizeof(int));
      cudaMalloc(&d_a_strides, shape.size() * sizeof(int));
      cudaMalloc(&d_b_strides, other.shape.size() * sizeof(int));
      cudaMemcpy(d_a_shape, shape.data(), shape.size() * sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy(d_b_shape, other.shape.data(), other.shape.size() * sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy(d_a_strides, a_strides.data(), shape.size() * sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy(d_b_strides, b_strides.data(), other.shape.size() * sizeof(int), cudaMemcpyHostToDevice);

      int threadsPerBlock = 256;
      int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
      addKernel<<<blocks, threadsPerBlock>>>(result.data, data, other.data,
                                             d_a_shape, d_b_shape, d_a_strides, d_b_strides,
                                             shape.size(), size);
      cudaDeviceSynchronize();
      cudaError_t err = cudaGetLastError();
      cudaFree(d_a_shape);
      cudaFree(d_b_shape);
      cudaFree(d_a_strides);
      cudaFree(d_b_strides);
      if (err != cudaSuccess)
      {
        throw std::runtime_error("addKernel failed: " + std::string(cudaGetErrorString(err)));
      }
      return result;
    }

    // Case 3: Broadcasting 4D [B, C, H, W] + [1, C, 1, 1]
    if (shape.size() == 4 && other.shape.size() == 4 &&
        other.shape[0] == 1 && other.shape[2] == 1 && other.shape[3] == 1 && shape[1] == other.shape[1])
    {
      Tensor result(shape);
      int *d_a_shape, *d_b_shape, *d_a_strides, *d_b_strides;
      cudaMalloc(&d_a_shape, shape.size() * sizeof(int));
      cudaMalloc(&d_b_shape, other.shape.size() * sizeof(int));
      cudaMalloc(&d_a_strides, shape.size() * sizeof(int));
      cudaMalloc(&d_b_strides, other.shape.size() * sizeof(int));
      cudaMemcpy(d_a_shape, shape.data(), shape.size() * sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy(d_b_shape, other.shape.data(), other.shape.size() * sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy(d_a_strides, a_strides.data(), shape.size() * sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy(d_b_strides, b_strides.data(), other.shape.size() * sizeof(int), cudaMemcpyHostToDevice);

      int threadsPerBlock = 256;
      int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
      addKernel<<<blocks, threadsPerBlock>>>(result.data, data, other.data,
                                             d_a_shape, d_b_shape, d_a_strides, d_b_strides,
                                             shape.size(), size);
      cudaDeviceSynchronize();
      cudaError_t err = cudaGetLastError();
      cudaFree(d_a_shape);
      cudaFree(d_b_shape);
      cudaFree(d_a_strides);
      cudaFree(d_b_strides);
      if (err != cudaSuccess)
      {
        throw std::runtime_error("addKernel failed: " + std::string(cudaGetErrorString(err)));
      }
      return result;
    }

    throw std::runtime_error("operator+: Incompatible shapes for addition with broadcasting");
  }

  Tensor operator+(float scalar) const
  {
    Tensor result(shape);
    int threadsPerBlock = 256;
    int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    scalarAddKernel<<<blocks, threadsPerBlock>>>(result.data, data, scalar, size);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
      throw std::runtime_error("scalarAddKernel failed: " + std::string(cudaGetErrorString(err)));
    }
    return result;
  }
  Tensor operator*(float scalar) const
  {
    Tensor result(shape);
    int threadsPerBlock = 256;
    int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    scalarMulKernel<<<blocks, threadsPerBlock>>>(result.data, data, scalar, size);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
      throw std::runtime_error("scalarMulKernel failed: " + std::string(cudaGetErrorString(err)));
    }
    return result;
  }

  // Método para copiar datos de GPU a CPU (para inspección o depuración)
  std::vector<float> to_host() const
  {
    std::vector<float> host_data(size);
    cudaError_t err = cudaMemcpy(host_data.data(), data, size * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
      throw std::runtime_error("cudaMemcpy failed: " + std::string(cudaGetErrorString(err)));
    }
    return host_data;
  }
};