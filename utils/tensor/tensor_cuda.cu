#include <cuda_runtime.h>
#include <random>
#include <vector>
#include <numeric>
#include <stdexcept>
#include "tensor_cuda.cuh"
#include "algorithm"
#include <iostream>
#include <cstring>
#include <cfloat>

#include <sstream>

static std::mt19937 global_gen;

// Kernel CUDA - solución 1: gridDim.z = N, loop interno sobre co
__global__ void conv2d_forward_kernel(const float *input, const float *weights, const float *biases,
                                      float *output,
                                      int N, int C_in, int H_in, int W_in,
                                      int C_out, int K_h, int K_w,
                                      int H_out, int W_out,
                                      int stride, int padding)
{
  int wo = threadIdx.x + blockIdx.x * blockDim.x;
  int ho = threadIdx.y + blockIdx.y * blockDim.y;
  int n = blockIdx.z; // cada bloque z es una imagen (batch)

  if (wo >= W_out || ho >= H_out || n >= N)
    return;

  // Loop sobre canales de salida (co)
  for (int co = 0; co < C_out; ++co)
  {
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
            int input_idx = ((n * C_in + ci) * H_in + h_in) * W_in + w_in;
            int weight_idx = ((co * C_in + ci) * K_h + kh) * K_w + kw;
            sum += input[input_idx] * weights[weight_idx];
          }
        }
      }
    }

    int output_idx = ((n * C_out + co) * H_out + ho) * W_out + wo;
    output[output_idx] = sum;
  }
}
__global__ void maxpool2d_backward_kernel(float *input_deltas, const float *delta, const int *max_indices,
                                          int N, int C, int H, int W, int H_out, int W_out)
{
  int n_c = blockIdx.z;
  int n = n_c / C;
  int c = n_c % C;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (n < N && c < C && i < H_out && j < W_out)
  {
    int idx = ((n * C + c) * H_out + i) * W_out + j;
    int max_pos = max_indices[idx];
    int h = max_pos / W;
    int w = max_pos % W;

    if (h >= 0 && h < H && w >= 0 && w < W)
    {
      int input_idx = ((n * C + c) * H + h) * W + w;
      input_deltas[input_idx] = delta[idx];
    }
  }
}
__global__ void maxpool2d_kernel(float *output, const float *input, int *max_indices,
                                 int N, int C, int H, int W, int H_out, int W_out,
                                 int kernel_size, int stride)
{
  int n_c = blockIdx.z;                          // Combined batch and channel index
  int n = n_c / C;                               // Batch index
  int c = n_c % C;                               // Channel index
  int i = blockIdx.x * blockDim.x + threadIdx.x; // Output height index
  int j = blockIdx.y * blockDim.y + threadIdx.y; // Output width index

  if (n < N && c < C && i < H_out && j < W_out)
  {
    float max_val = -FLT_MAX;
    int max_idx = -1;

    for (int ki = 0; ki < kernel_size; ++ki)
    {
      for (int kj = 0; kj < kernel_size; ++kj)
      {
        int h = i * stride + ki;
        int w = j * stride + kj;
        if (h >= 0 && h < H && w >= 0 && w < W)
        {
          int input_idx = ((n * C + c) * H + h) * W + w;
          float val = input[input_idx];
          if (val > max_val)
          {
            max_val = val;
            max_idx = h * W + w;
          }
        }
      }
    }

    int output_idx = ((n * C + c) * H_out + i) * W_out + j;
    output[output_idx] = max_val;
    max_indices[output_idx] = max_idx;
  }
}

__global__ void avgpool2d_backward_kernel(float *input_deltas, const float *delta,
                                          int N, int C, int H, int W, int H_out, int W_out,
                                          int kernel_size, int stride)
{
  int n_c = blockIdx.z;
  int n = n_c / C;
  int c = n_c % C;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (n < N && c < C && i < H_out && j < W_out)
  {
    int idx = ((n * C + c) * H_out + i) * W_out + j;
    float grad = delta[idx] / (kernel_size * kernel_size);

    for (int ki = 0; ki < kernel_size; ++ki)
    {
      for (int kj = 0; kj < kernel_size; ++kj)
      {
        int h = i * stride + ki;
        int w = j * stride + kj;
        if (h >= 0 && h < H && w >= 0 && w < W)
        {
          int input_idx = ((n * C + c) * H + h) * W + w;
          atomicAdd(&input_deltas[input_idx], grad);
        }
      }
    }
  }
}

// Kernel for grad_biases: Reduction sum over delta

__global__ void conv2d_grad_biases_kernel(float *grad_biases, const float *delta,
                                          int N, int C_out, int H_out, int W_out)
{
  extern __shared__ float sdata[];
  int co = blockIdx.y;
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  if (co >= C_out)
    return;

  float sum = 0.0f;
  for (int i = idx; i < N * H_out * W_out; i += stride)
  {
    int n = i / (H_out * W_out);
    int ho = (i % (H_out * W_out)) / W_out;
    int wo = i % W_out;
    int delta_idx = ((n * C_out + co) * H_out + ho) * W_out + wo;
    sum += delta[delta_idx];
  }
  sdata[tid] = sum;

  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1)
  {
    if (tid < s)
      sdata[tid] += sdata[tid + s];
    __syncthreads();
  }

  if (tid == 0)
    atomicAdd(&grad_biases[co], sdata[0]);
}

__global__ void conv2d_grad_weights_kernel(float *grad_weights, const float *padded_input, const float *delta,
                                           int N, int C_out, int C_in, int K_h, int K_w, int H_out, int W_out,
                                           int H_in_padded, int W_in_padded, int stride)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int co = idx / (C_in * K_h * K_w);
  int rest = idx % (C_in * K_h * K_w);
  int ci = rest / (K_h * K_w);
  rest = rest % (K_h * K_w);
  int kh = rest / K_w;
  int kw = rest % K_w;

  if (co < C_out && ci < C_in && kh < K_h && kw < K_w)
  {
    float sum = 0.0f;
    for (int n = 0; n < N; ++n)
    {
      for (int ho = 0; ho < H_out; ++ho)
      {
        for (int wo = 0; wo < W_out; ++wo)
        {
          int h_in = ho * stride + kh;
          int w_in = wo * stride + kw;
          if (h_in >= 0 && h_in < H_in_padded && w_in >= 0 && w_in < W_in_padded)
          {
            int input_idx = ((n * C_in + ci) * H_in_padded + h_in) * W_in_padded + w_in;
            int delta_idx = ((n * C_out + co) * H_out + ho) * W_out + wo;
            sum += padded_input[input_idx] * delta[delta_idx];
          }
        }
      }
    }
    grad_weights[((co * C_in + ci) * K_h + kh) * K_w + kw] = sum;
  }
}

__global__ void conv2d_input_deltas_kernel(float *input_deltas, const float *delta, const float *weights,
                                           int N, int C_out, int C_in, int K_h, int K_w,
                                           int H_in, int W_in, int H_out, int W_out, int stride, int padding)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int n = idx / (C_in * H_in * W_in);
  int rest = idx % (C_in * H_in * W_in);
  int ci = rest / (H_in * W_in);
  rest = rest % (H_in * W_in);
  int h_in = rest / W_in;
  int w_in = rest % W_in;

  if (n < N && ci < C_in && h_in < H_in && w_in < W_in)
  {
    float sum = 0.0f;
    for (int co = 0; co < C_out; ++co)
    {
      for (int kh = 0; kh < K_h; ++kh)
      {
        for (int kw = 0; kw < K_w; ++kw)
        {
          int ho = (h_in - kh + padding) / stride;
          int wo = (w_in - kw + padding) / stride;
          if ((h_in - kh + padding) % stride == 0 && (w_in - kw + padding) % stride == 0 &&
              ho >= 0 && ho < H_out && wo >= 0 && wo < W_out)
          {
            int delta_idx = ((n * C_out + co) * H_out + ho) * W_out + wo;
            int weight_idx = ((co * C_in + ci) * K_h + (K_h - 1 - kh)) * K_w + (K_w - 1 - kw);
            sum += delta[delta_idx] * weights[weight_idx];
          }
        }
      }
    }
    input_deltas[((n * C_in + ci) * H_in + h_in) * W_in + w_in] = sum;
  }
}

void conv2d_cuda_backward(const Tensor &delta,
                          const Tensor &input,
                          const Tensor &weights,
                          Tensor &grad_weights,
                          Tensor &grad_biases,
                          Tensor &input_deltas,
                          int stride,
                          int padding)
{
  int N = input.shape[0];
  int C_in = input.shape[1];
  int H_in = input.shape[2];
  int W_in = input.shape[3];
  int C_out = weights.shape[0];
  int K_h = weights.shape[2];
  int K_w = weights.shape[3];
  int H_out = delta.shape[2];
  int W_out = delta.shape[3];

  if (delta.shape[0] != N || delta.shape[1] != C_out ||
      grad_weights.shape != weights.shape || grad_biases.shape[0] != C_out ||
      input_deltas.shape != input.shape)
    throw std::runtime_error("conv2d_cuda_backward: Invalid shapes");

  if (!delta.data || !input.data || !weights.data || !grad_weights.data || !grad_biases.data || !input_deltas.data)
    throw std::runtime_error("conv2d_cuda_backward: Null pointer detected");

  Tensor padded_input = input;
  int H_in_padded = H_in;
  int W_in_padded = W_in;
  if (padding > 0)
  {

    padded_input = input.pad({0, 0, 0, 0, padding, padding, padding, padding});
    H_in_padded = padded_input.shape[2];
    W_in_padded = padded_input.shape[3];
    if (!padded_input.data)
      throw std::runtime_error("conv2d_cuda_backward: padded_input allocation failed");
  }

  dim3 threadsPerBlockBias(256);
  dim3 blocksBias(64, C_out); // Multiple blocks per channel for better parallelism

  conv2d_grad_biases_kernel<<<blocksBias, threadsPerBlockBias, threadsPerBlockBias.x * sizeof(float)>>>(grad_biases.data, delta.data,
                                                                                                        N, C_out, H_out, W_out);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    throw std::runtime_error("conv2d_grad_biases_kernel failed: " + std::string(cudaGetErrorString(err)));

  dim3 threadsPerBlockWeights(256);
  dim3 blocksWeights((C_out * C_in * K_h * K_w + threadsPerBlockWeights.x - 1) / threadsPerBlockWeights.x);

  conv2d_grad_weights_kernel<<<blocksWeights, threadsPerBlockWeights>>>(grad_weights.data, padded_input.data, delta.data,
                                                                        N, C_out, C_in, K_h, K_w, H_out, W_out,
                                                                        H_in_padded, W_in_padded, stride);
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  if (err != cudaSuccess)
    throw std::runtime_error("conv2d_grad_weights_kernel failed: " + std::string(cudaGetErrorString(err)));

  dim3 threadsPerBlockDeltas(256);
  dim3 blocksDeltas((N * C_in * H_in * W_in + threadsPerBlockDeltas.x - 1) / threadsPerBlockDeltas.x);

  conv2d_input_deltas_kernel<<<blocksDeltas, threadsPerBlockDeltas>>>(input_deltas.data, delta.data, weights.data,
                                                                      N, C_out, C_in, K_h, K_w, H_in, W_in,
                                                                      H_out, W_out, stride, padding);
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  if (err != cudaSuccess)
    throw std::runtime_error("conv2d_input_deltas_kernel failed: " + std::string(cudaGetErrorString(err)));
}

// Backward pass for average pooling (optional)
Tensor avgpool2d_cuda_backward(const Tensor &delta, const Tensor &input, int kernel_size, int stride)
{
  int N = input.shape[0];
  int C = input.shape[1];
  int H = input.shape[2];
  int W = input.shape[3];
  int H_out = delta.shape[2];
  int W_out = delta.shape[3];

  Tensor input_deltas({N, C, H, W}, true);
  cudaMemset(input_deltas.data, 0, input_deltas.size * sizeof(float));

  dim3 threadsPerBlock(16, 16);
  dim3 blocks((H_out + threadsPerBlock.x - 1) / threadsPerBlock.x,
              (W_out + threadsPerBlock.y - 1) / threadsPerBlock.y,
              N * C);

  avgpool2d_backward_kernel<<<blocks, threadsPerBlock>>>(input_deltas.data, delta.data,
                                                         N, C, H, W, H_out, W_out, kernel_size, stride);
  cudaDeviceSynchronize();

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    throw std::runtime_error("avgpool2d_backward_kernel failed: " + std::string(cudaGetErrorString(err)));

  return input_deltas;
}

__global__ void avgpool2d_kernel(float *output, const float *input,
                                 int N, int C, int H, int W, int H_out, int W_out,
                                 int kernel_size, int stride)
{
  int n_c = blockIdx.z;
  int n = n_c / C;
  int c = n_c % C;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (n < N && c < C && i < H_out && j < W_out)
  {
    float sum = 0.0f;
    int count = 0;

    for (int ki = 0; ki < kernel_size; ++ki)
    {
      for (int kj = 0; kj < kernel_size; ++kj)
      {
        int h = i * stride + ki;
        int w = j * stride + kj;
        if (h >= 0 && h < H && w >= 0 && w < W)
        {
          int input_idx = ((n * C + c) * H + h) * W + w;
          sum += input[input_idx];
          count++;
        }
      }
    }

    int output_idx = ((n * C + c) * H_out + i) * W_out + j;
    output[output_idx] = (count > 0) ? sum / count : 0.0f;
  }
}

// Forward pass for average pooling (optional)
Tensor avgpool2d_cuda_forward(const Tensor &input, int kernel_size, int stride)
{
  int N = input.shape[0];
  int C = input.shape[1];
  int H = input.shape[2];
  int W = input.shape[3];
  int H_out = (H - kernel_size) / stride + 1;
  int W_out = (W - kernel_size) / stride + 1;

  Tensor output({N, C, H_out, W_out}, true);

  dim3 threadsPerBlock(16, 16);
  dim3 blocks((H_out + threadsPerBlock.x - 1) / threadsPerBlock.x,
              (W_out + threadsPerBlock.y - 1) / threadsPerBlock.y,
              N * C);

  avgpool2d_kernel<<<blocks, threadsPerBlock>>>(output.data, input.data,
                                                N, C, H, W, H_out, W_out, kernel_size, stride);
  cudaDeviceSynchronize();

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    throw std::runtime_error("avgpool2d_kernel failed: " + std::string(cudaGetErrorString(err)));

  return output;
}
// Función auxiliar
Tensor conv2d_cuda_forward(const Tensor &input,
                           const Tensor &weights,
                           const Tensor &biases,
                           int stride,
                           int padding)
{
  const int N = input.shape[0];
  const int C_in = input.shape[1];
  const int H_in = input.shape[2];
  const int W_in = input.shape[3];
  const int C_out = weights.shape[0];
  const int K_h = weights.shape[2];
  const int K_w = weights.shape[3];

  const int H_out = (H_in + 2 * padding - K_h) / stride + 1;
  const int W_out = (W_in + 2 * padding - K_w) / stride + 1;

  Tensor temp_output = Tensor({N, C_out, H_out, W_out});
  Tensor output = temp_output.to_device(true);

  dim3 blockDim(16, 16);
  dim3 gridDim((W_out + 15) / 16, (H_out + 15) / 16, N);

  conv2d_forward_kernel<<<gridDim, blockDim>>>(
      input.data, weights.data, biases.data, output.data,
      N, C_in, H_in, W_in,
      C_out, K_h, K_w,
      H_out, W_out,
      stride, padding);

  // ⚠️ Chequeo de errores CUDA
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    std::cerr << "[CUDA ERROR] conv2d_forward_kernel: " << cudaGetErrorString(err) << "\n";
    throw std::runtime_error("CUDA kernel launch failed");
  }

  cudaDeviceSynchronize();

  if (!output.is_cuda)
    std::cerr << "[WARNING] output está en CPU, se esperaba en GPU\n";

  return output;
}

Tensor maxpool2d_cuda_forward(const Tensor &input, int *d_max_indices, int kernel_size, int stride)
{
  int N = input.shape[0];
  int C = input.shape[1];
  int H = input.shape[2];
  int W = input.shape[3];
  int H_out = (H - kernel_size) / stride + 1;
  int W_out = (W - kernel_size) / stride + 1;

  Tensor output({N, C, H_out, W_out}, true); // is_cuda=true

  dim3 threadsPerBlock(16, 16);
  dim3 blocks((H_out + threadsPerBlock.x - 1) / threadsPerBlock.x,
              (W_out + threadsPerBlock.y - 1) / threadsPerBlock.y,
              N * C);

  maxpool2d_kernel<<<blocks, threadsPerBlock>>>(output.data, input.data, d_max_indices,
                                                N, C, H, W, H_out, W_out, kernel_size, stride);
  cudaDeviceSynchronize();

  return output;
}

// Backward pass for max pooling
Tensor maxpool2d_cuda_backward(const Tensor &delta, const Tensor &input, const int *d_max_indices,
                               int kernel_size, int stride)
{
  int N = input.shape[0];
  int C = input.shape[1];
  int H = input.shape[2];
  int W = input.shape[3];
  int H_out = delta.shape[2];
  int W_out = delta.shape[3];

  Tensor input_deltas({N, C, H, W}, true);                             // is_cuda=true
  cudaMemset(input_deltas.data, 0, input_deltas.size * sizeof(float)); // Initialize to zero

  dim3 threadsPerBlock(16, 16);
  dim3 blocks((H_out + threadsPerBlock.x - 1) / threadsPerBlock.x,
              (W_out + threadsPerBlock.y - 1) / threadsPerBlock.y,
              N * C);

  maxpool2d_backward_kernel<<<blocks, threadsPerBlock>>>(input_deltas.data, delta.data, d_max_indices,
                                                         N, C, H, W, H_out, W_out);
  cudaDeviceSynchronize();

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    throw std::runtime_error("maxpool2d_backward_kernel failed: " + std::string(cudaGetErrorString(err)));

  return input_deltas;
}

__global__ void elementwiseMulKernelBroadcast(
    float *result, const float *a, const float *b,
    const int *result_shape, const int *a_shape, const int *b_shape,
    const int *a_strides, const int *b_strides,
    int dims, int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;

  int a_idx = 0, b_idx = 0;
  int temp = idx;
  for (int d = dims - 1; d >= 0; --d)
  {
    int coord = temp % result_shape[d];
    temp /= result_shape[d];
    a_idx += (a_shape[d] == 1 ? 0 : coord) * a_strides[d];
    b_idx += (b_shape[d] == 1 ? 0 : coord) * b_strides[d];
  }
  result[idx] = a[a_idx] * b[b_idx];
}

__global__ void broadcastKernel(float *out, const float *in,
                                const int *in_strides, const int *out_strides,
                                const int *in_shape, const int *out_shape,
                                int dims, int total)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total)
    return;

  int in_idx = 0;
  int remaining = idx;
  for (int i = 0; i < dims; ++i)
  {
    int coord = remaining / out_strides[i];
    remaining %= out_strides[i];
    int in_coord = (in_shape[i] == 1) ? 0 : coord;
    in_idx += in_coord * in_strides[i];
  }

  out[idx] = in[in_idx];
}
Tensor Tensor::broadcast_to(const std::vector<int> &target_shape) const
{

  if (target_shape.size() < shape.size())
  {
    throw std::invalid_argument("broadcast_to: target shape must be >= rank of original tensor");
  }

  // Ajustar shape con 1s al frente si es necesario
  std::vector<int> adjusted_shape = shape;
  while (adjusted_shape.size() < target_shape.size())
    adjusted_shape.insert(adjusted_shape.begin(), 1);

  // Validación de compatibilidad
  for (size_t i = 0; i < target_shape.size(); ++i)
  {
    if (adjusted_shape[i] != target_shape[i] && adjusted_shape[i] != 1)
    {
      std::cerr << "[ERROR] broadcast_to: incompatible at dim " << i
                << " (" << adjusted_shape[i] << " vs " << target_shape[i] << ")\n";
      throw std::invalid_argument("broadcast_to: incompatible shape for broadcasting");
    }
  }

  Tensor result(target_shape);
  int dims = target_shape.size();

  // Calcular strides
  std::vector<int> in_strides(dims), out_strides(dims);
  in_strides[dims - 1] = 1;
  out_strides[dims - 1] = 1;
  for (int i = dims - 2; i >= 0; --i)
  {
    in_strides[i] = in_strides[i + 1] * adjusted_shape[i + 1];
    out_strides[i] = out_strides[i + 1] * target_shape[i + 1];
  }

  // Enviar a GPU
  int *d_in_strides, *d_out_strides, *d_in_shape, *d_out_shape;
  cudaMalloc(&d_in_strides, dims * sizeof(int));
  cudaMalloc(&d_out_strides, dims * sizeof(int));
  cudaMalloc(&d_in_shape, dims * sizeof(int));
  cudaMalloc(&d_out_shape, dims * sizeof(int));

  cudaMemcpy(d_in_strides, in_strides.data(), dims * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_out_strides, out_strides.data(), dims * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_in_shape, adjusted_shape.data(), dims * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_out_shape, target_shape.data(), dims * sizeof(int), cudaMemcpyHostToDevice);

  // Ejecutar kernel
  int threadsPerBlock = 256;
  int blocks = (result.size + threadsPerBlock - 1) / threadsPerBlock;

  broadcastKernel<<<blocks, threadsPerBlock>>>(result.data, data,
                                               d_in_strides, d_out_strides,
                                               d_in_shape, d_out_shape,
                                               dims, result.size);

  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    std::cerr << "[ERROR] broadcastKernel failed: " << cudaGetErrorString(err) << "\n";
    throw std::runtime_error("broadcast_to: CUDA kernel launch failed");
  }

  cudaFree(d_in_strides);
  cudaFree(d_out_strides);
  cudaFree(d_in_shape);
  cudaFree(d_out_shape);

  return result;
}

__global__ void fillKernel(float *data, float value, int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
  {
    data[idx] = value;
  }
}

__global__ void compute_accuracy_kernel(const float *preds, const float *targets, int *correct, int num_classes, int batch_size)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= batch_size)
    return;

  // Encontrar argmax
  int pred_label = 0;
  float max_val = preds[i * num_classes];
  for (int j = 1; j < num_classes; ++j)
  {
    float val = preds[i * num_classes + j];
    if (val > max_val)
    {
      max_val = val;
      pred_label = j;
    }
  }

  int true_label = static_cast<int>(targets[i]);

  if (pred_label == true_label)
  {
    atomicAdd(correct, 1);
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
    result[idx] = input[idx] * scalar;
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
    for (int d = dims - 1; d >= 0; --d)
    {
      int coord = temp % a_shape[d];
      temp /= a_shape[d];
      a_idx += coord * a_strides[d];
      int coord_b = (b_shape[d] == 1) ? 0 : coord;
      b_idx += coord_b * b_strides[d];
    }
    result[idx] = a[a_idx] + b[b_idx];
  }
}

__global__ void simpleAddKernel(float *result, const float *a, const float *b, int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
  {
    result[idx] = a[idx] + b[idx];
  }
}

// Kernel for element-wise subtraction
__global__ void elementwiseSubKernel(float *output, const float *a, const float *b,
                                     const int *result_shape, const int *a_shape, const int *b_shape,
                                     int max_dims, size_t size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;

  size_t idx_a = 0, idx_b = 0;
  size_t tmp = idx;
  for (int d = 0; d < max_dims; ++d)
  {
    size_t coord = tmp % result_shape[d];
    tmp /= result_shape[d];
    idx_a += (a_shape[d] == 1 ? 0 : coord) * (d == max_dims - 1 ? 1 : a_shape[d + 1]);
    idx_b += (b_shape[d] == 1 ? 0 : coord) * (d == max_dims - 1 ? 1 : b_shape[d + 1]);
  }
  output[idx] = a[idx_a] - b[idx_b];
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

void Tensor::set_seed(unsigned int seed)
{
  global_gen.seed(seed);
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
    result[idx] = input[idx] / scalar;
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
                                   bool *reduce_axes, int total_size, int out_size, int dims, int out_dims)
{
  int flat_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (flat_idx >= total_size)
    return;

  int idx = flat_idx;
  int out_index = 0;
  int out_i = 0;

  for (int i = 0; i < dims; ++i)
  {
    int coord = idx / in_strides[i];
    idx %= in_strides[i];
    if (!reduce_axes[i])
    {
      if (out_i >= out_dims)
        return;

      out_index += coord * out_strides[out_i];
      ++out_i;
    }
  }

  if (out_index < out_size)
  {
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
    if (label < 0 || label >= num_classes)
    {
      printf("❌ [Kernel] Etiqueta inválida en idx %d: %d (num_classes=%d)\n", idx, label, num_classes);
      return;
    }

    // output[idx * num_classes + label] = 1.0f;
    if (idx < batch_size)
    {
      int label = static_cast<int>(labels[idx]);
      if (label < 0 || label >= num_classes)
      {
        printf("❌ [Kernel] Etiqueta inválida en idx %d: %d (num_classes=%d)\n", idx, label, num_classes);
        return;
      }

      output[idx * num_classes + label] = 1.0f;
    }
  }
}

// Kernel for padding operation
__global__ void padKernel(float *output, const float *input, int N, int C, int H, int W,
                          int new_N, int new_C, int new_H, int new_W,
                          int pad_N0, int pad_C0, int pad_H0, int pad_W0)
{
  int h = blockIdx.x * blockDim.x + threadIdx.x;
  int w = blockIdx.y * blockDim.y + threadIdx.y;

  if (h < new_H && w < new_W)
  {
    for (int n = 0; n < new_N; ++n)
      for (int c = 0; c < new_C; ++c)
      {
        int idx = ((n * new_C + c) * new_H + h) * new_W + w;
        if (n >= pad_N0 && n < (N + pad_N0) &&
            c >= pad_C0 && c < (C + pad_C0) &&
            h >= pad_H0 && h < (H + pad_H0) &&
            w >= pad_W0 && w < (W + pad_W0))
        {
          int in_n = n - pad_N0;
          int in_c = c - pad_C0;
          int in_h = h - pad_H0;
          int in_w = w - pad_W0;
          int in_idx = ((in_n * C + in_c) * H + in_h) * W + in_w;
          output[idx] = input[in_idx];
        }
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
    int *offsets,
    int *strides,
    int *shape,
    int dims)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_size)
    return;

  int temp = idx;
  int tensor_idx = 0;
  int d = dim;
  int coord_d = (temp / strides[d]) % shape[d];

  // Find which tensor corresponds to this coordinate in the concat dimension
  int input_offset = 0;
  while (tensor_idx < num_tensors - 1 && coord_d >= input_sizes[tensor_idx])
  {
    coord_d -= input_sizes[tensor_idx];
    input_offset += input_sizes[tensor_idx] * strides[d];
    tensor_idx++;
  }

  // Calculate the index within the source tensor
  int input_idx = 0;
  for (int i = dims - 1; i >= 0; --i)
  {
    int coord = (i == d) ? coord_d : (temp / strides[i]) % shape[i];
    input_idx += coord * strides[i];
  }

  // Validate input_idx
  int tensor_size = 1;
  for (int i = 0; i < dims; ++i)
  {
    if (i == d)
    {
      tensor_size *= input_sizes[tensor_idx];
    }
    else
    {
      tensor_size *= shape[i];
    }
  }
  if (tensor_idx < num_tensors && inputs[tensor_idx] != nullptr && input_idx < tensor_size)
  {
    output[idx] = inputs[tensor_idx][input_idx];
  }
  else
  {
    // Debug invalid access (won't print in kernel, but can trigger a check)
    output[idx] = 0.0f;
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

Tensor::Tensor() : data(nullptr), size(0), is_cuda(true) {}

Tensor::Tensor(const std::vector<int> &shape, bool is_cuda) : shape(shape), is_cuda(is_cuda)
{
  size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  data = nullptr; // Initialize data to nullptr
  if (size > 0)
  {
    if (is_cuda)
    {
      cudaError_t err = cudaMalloc(&data, size * sizeof(float));
      if (err != cudaSuccess)
      {
        throw std::runtime_error("cudaMalloc failed in Tensor constructor: " + std::string(cudaGetErrorString(err)));
      }
      cudaMemset(data, 0, size * sizeof(float));
    }
    else
    {
      data = new float[size];
      std::memset(data, 0, size * sizeof(float));
    }
  }
}

Tensor::Tensor(const std::vector<int> &shape, const std::vector<float> &host_data, bool use_cuda)
    : shape(shape), is_cuda(use_cuda)
{
  size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  if (host_data.size() != size)
    throw std::runtime_error("Data size does not match tensor shape");

  data = nullptr;
  if (size > 0)
  {
    if (is_cuda)
    {
      cudaError_t err = cudaMalloc(&data, size * sizeof(float));
      if (err != cudaSuccess)
        throw std::runtime_error("cudaMalloc failed: " + std::string(cudaGetErrorString(err)));
      err = cudaMemcpy(data, host_data.data(), size * sizeof(float), cudaMemcpyHostToDevice);
      if (err != cudaSuccess)
        throw std::runtime_error("cudaMemcpy failed: " + std::string(cudaGetErrorString(err)));
    }
    else
    {
      data = new float[size];
      std::memcpy(data, host_data.data(), size * sizeof(float));
    }
  }
}

// Add copy constructor
Tensor::Tensor(const Tensor &other) : shape(other.shape), size(other.size), is_cuda(other.is_cuda)
{
  data = nullptr;
  if (size > 0)
  {
    if (is_cuda)
    {
      cudaError_t err = cudaMalloc(&data, size * sizeof(float));
      if (err != cudaSuccess)
      {
        throw std::runtime_error("cudaMalloc failed in Tensor copy constructor: " + std::string(cudaGetErrorString(err)));
      }
      if (other.data)
      {
        err = cudaMemcpy(data, other.data, size * sizeof(float), cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess)
        {
          cudaFree(data);
          throw std::runtime_error("cudaMemcpy failed in Tensor copy constructor: " + std::string(cudaGetErrorString(err)));
        }
      }
      else
      {
        cudaMemset(data, 0, size * sizeof(float));
      }
    }
    else
    {
      data = new float[size];
      if (other.data)
      {
        std::memcpy(data, other.data, size * sizeof(float));
      }
      else
      {
        std::memset(data, 0, size * sizeof(float));
      }
    }
  }
}

// Add move constructor
Tensor::Tensor(Tensor &&other) noexcept : shape(std::move(other.shape)), size(other.size), is_cuda(other.is_cuda), data(other.data)
{
  other.data = nullptr;
  other.size = 0;
  other.shape.clear();
}

Tensor::~Tensor()
{
  if (data)
  {
    if (is_cuda)
      cudaFree(data);
    else
      delete[] data;
  }
}

Tensor &Tensor::operator=(Tensor &&other) noexcept
{
  if (this != &other)
  {
    if (data)
    {
      if (is_cuda)
        cudaFree(data);
      else
        delete[] data;
    }
    shape = std::move(other.shape);
    size = other.size;
    is_cuda = other.is_cuda;
    data = other.data;
    other.data = nullptr;
    other.size = 0;
    other.shape.clear();
    // std::cout << "[DEBUG] Tensor move assignment - data: " << data << " | is_cuda: " << is_cuda << "\n";   33599
  }
  return *this;
}

Tensor &Tensor::operator=(const Tensor &other)
{
  if (this == &other)
    return *this;

  if (data)
  {
    if (is_cuda)
      cudaFree(data);
    else
      delete[] data;
  }

  shape = other.shape;
  size = other.size;
  is_cuda = other.is_cuda;

  // std::cout << "[DEBUG] Assign - size: " << size << " | other.data: "<< other.data << " | is_cuda: " << is_cuda << "\n";

  data = nullptr;
  if (size > 0)
  {
    if (!other.data)
    {
      throw std::runtime_error("operator= ❌ 'other.data' is nullptr");
    }

    if (is_cuda)
    {
      cudaPointerAttributes attr_other;
      cudaError_t errAttrOther = cudaPointerGetAttributes(&attr_other, other.data);
      if (errAttrOther != cudaSuccess)
      {
        throw std::runtime_error("❌ cudaPointerGetAttributes failed (other.data): " + std::string(cudaGetErrorString(errAttrOther)));
      }
      if (attr_other.type != cudaMemoryTypeDevice)
      {
        throw std::runtime_error("❌ 'other.data' is not valid GPU memory for is_cuda=true");
      }

      cudaError_t err = cudaMalloc(&data, size * sizeof(float));
      if (err != cudaSuccess)
      {
        throw std::runtime_error("operator= ❌ cudaMalloc failed: " + std::string(cudaGetErrorString(err)));
      }
      err = cudaMemcpy(data, other.data, size * sizeof(float), cudaMemcpyDeviceToDevice);
      if (err != cudaSuccess)
      {
        cudaFree(data);
        throw std::runtime_error("operator= ❌ cudaMemcpy failed: " + std::string(cudaGetErrorString(err)));
      }
    }
    else
    {
      data = new float[size];
      std::memcpy(data, other.data, size * sizeof(float));
    }
  }

  return *this;
}

// -------------------- Inicialización --------------------
Tensor Tensor::zeros(const std::vector<int> &shape, bool is_cuda)
{
  return Tensor(shape, is_cuda); // Ya inicializa a cero en el constructor
}

Tensor Tensor::rand_uniform(const std::vector<int> &shape, float low, float high)
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

Tensor Tensor::rand_uniform_gpu(const std::vector<int> &shape, float low, float high)
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

Tensor Tensor::kaiming_normal(const std::vector<int> &shape, int fan_in, bool use_negative)
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

Tensor Tensor::xavier_normal(const std::vector<int> &shape, int fan_in, int fan_out)
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

Tensor Tensor::xavier_uniform(const std::vector<int> &shape, int fan_in, int fan_out)
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

Tensor Tensor::one_hot(const Tensor &labels, int num_classes)
{
  if (labels.shape.size() != 2 || labels.shape[1] != 1)
    throw std::invalid_argument("Labels tensor must have shape [batch_size, 1]");

  int batch_size = labels.shape[0];
  std::vector<int> new_shape = {batch_size, num_classes};

  // ✅ Siempre forzamos clon correcto hacia GPU (aunque ya esté en GPU)
  Tensor labels_gpu = labels.to_device(true);

  // ✅ DEBUG

  Tensor result(new_shape); // en GPU por defecto

  std::vector<float> host_labels = labels_gpu.to_host();

  for (int i = 0; i < batch_size; ++i)
  {
    int label = static_cast<int>(host_labels[i]);
    if (label < 0 || label >= num_classes)
      throw std::runtime_error("Etiqueta fuera de rango en one_hot");
  }

  // Lanzar kernel
  int threads = 256;
  int blocks = (batch_size + threads - 1) / threads;
  oneHotKernel<<<blocks, threads>>>(result.data, labels_gpu.data, batch_size, num_classes);
  cudaError_t errSync = cudaDeviceSynchronize();
  cudaError_t errAsync = cudaGetLastError();

  if (errAsync != cudaSuccess)
    throw std::runtime_error("❌ Kernel async error: " + std::string(cudaGetErrorString(errAsync)));

  if (errSync != cudaSuccess)
    throw std::runtime_error("❌ Kernel sync error: " + std::string(cudaGetErrorString(errSync)));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    throw std::runtime_error("❌ oneHotKernel failed: " + std::string(cudaGetErrorString(err)));

  return result;
}

Tensor Tensor::slice(const std::vector<int> &starts, const std::vector<int> &ends) const
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

Tensor Tensor::slice(int dim, int start, int end) const
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

Tensor Tensor::concat(const std::vector<Tensor> &tensors, int dim)
{
  if (tensors.empty())
    throw std::invalid_argument("No tensors provided for concat");

  // Verify shapes are compatible
  std::vector<int> new_shape = tensors[0].shape;
  int concat_dim_size = 0;
  for (const auto &t : tensors)
  {
    if (t.shape.size() != new_shape.size())
      throw std::invalid_argument("Tensor dimensions mismatch");
    if (!t.data)
      throw std::runtime_error("Tensor::concat: Null input tensor data");
    if (t.size == 0)
      throw std::runtime_error("Tensor::concat: Zero-sized input tensor");
    for (size_t i = 0; i < new_shape.size(); ++i)
    {
      if (i != dim && t.shape[i] != new_shape[i])
        throw std::invalid_argument("Tensor shapes mismatch in dimension " + std::to_string(i));
    }
    concat_dim_size += t.shape[dim];
    if (t.size != std::accumulate(t.shape.begin(), t.shape.end(), 1, std::multiplies<int>()))
      throw std::runtime_error("Tensor::concat: Input tensor size mismatch, expected " +
                               std::to_string(std::accumulate(t.shape.begin(), t.shape.end(), 1, std::multiplies<int>())) +
                               ", got " + std::to_string(t.size));
  }
  new_shape[dim] = concat_dim_size;

  // Allocate output tensor
  Tensor result(new_shape, tensors[0].is_cuda);
  if (!result.data)
    throw std::runtime_error("Tensor::concat: Result allocation failed");
  int total_size = result.size;
  if (total_size != std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies<int>()))
    throw std::runtime_error("Tensor::concat: Output tensor size mismatch");

  // Prepare data for kernel
  float **d_inputs;
  cudaMalloc(&d_inputs, tensors.size() * sizeof(float *));
  std::vector<float *> h_inputs(tensors.size());
  for (size_t i = 0; i < tensors.size(); ++i)
  {
    h_inputs[i] = tensors[i].data;
    if (!h_inputs[i])
      throw std::runtime_error("Tensor::concat: Null input tensor data at index " + std::to_string(i));
  }
  cudaMemcpy(d_inputs, h_inputs.data(), tensors.size() * sizeof(float *), cudaMemcpyHostToDevice);

  std::vector<int> input_sizes(tensors.size());
  std::vector<int> offsets(tensors.size());
  int current_offset = 0;
  for (size_t i = 0; i < tensors.size(); ++i)
  {
    input_sizes[i] = tensors[i].shape[dim];
    offsets[i] = current_offset;
    current_offset += input_sizes[i];
  }
  int *d_input_sizes, *d_offsets;
  cudaMalloc(&d_input_sizes, tensors.size() * sizeof(int));
  cudaMalloc(&d_offsets, tensors.size() * sizeof(int));
  cudaMemcpy(d_input_sizes, input_sizes.data(), tensors.size() * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_offsets, offsets.data(), tensors.size() * sizeof(int), cudaMemcpyHostToDevice);

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

  // Debug shapes

  // Launch kernel
  int threadsPerBlock = 256;
  int blocks = (total_size + threadsPerBlock - 1) / threadsPerBlock;
  concatKernel<<<blocks, threadsPerBlock>>>(result.data, d_inputs, d_input_sizes, tensors.size(), dim, total_size, d_offsets, d_strides, d_shape, new_shape.size());
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    cudaFree(d_inputs);
    cudaFree(d_input_sizes);
    cudaFree(d_offsets);
    cudaFree(d_strides);
    cudaFree(d_shape);
    throw std::runtime_error("concatKernel failed: " + std::string(cudaGetErrorString(err)));
  }

  // Free auxiliary memory
  cudaFree(d_inputs);
  cudaFree(d_input_sizes);
  cudaFree(d_offsets);
  cudaFree(d_strides);
  cudaFree(d_shape);

  return result;
}

Tensor Tensor::pad(const std::vector<int> &pads) const
{
  if (shape.size() != 4)
    throw std::invalid_argument("Tensor must have 4 dimensions [N, C, H, W]");
  if (pads.size() != 8)
    throw std::invalid_argument("Pads must have 8 values [N_pre, N_post, C_pre, C_post, H_pre, H_post, W_pre, W_post]");
  if (!data)
    throw std::runtime_error("Tensor::pad: Input data is null");

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
  Tensor result = Tensor::zeros(new_shape, is_cuda); // Initialize to zero

  if (!result.data)
    throw std::runtime_error("Tensor::pad: Failed to allocate result tensor");

  if (is_cuda)
  {
    dim3 threadsPerBlock(16, 16);
    dim3 blocks((new_H + threadsPerBlock.x - 1) / threadsPerBlock.x,
                (new_W + threadsPerBlock.y - 1) / threadsPerBlock.y);
    padKernel<<<blocks, threadsPerBlock>>>(result.data, data, N, C, H, W,
                                           new_N, new_C, new_H, new_W,
                                           pad_N0, pad_C0, pad_H0, pad_W0);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
      throw std::runtime_error("padKernel failed: " + std::string(cudaGetErrorString(err)));
  }
  else
  {
    for (int n = 0; n < new_N; ++n)
      for (int c = 0; c < new_C; ++c)
        for (int h = 0; h < new_H; ++h)
          for (int w = 0; w < new_W; ++w)
          {
            if (n >= pad_N0 && n < (N + pad_N0) &&
                c >= pad_C0 && c < (C + pad_C0) &&
                h >= pad_H0 && h < (H + pad_H0) &&
                w >= pad_W0 && w < (W + pad_W0))
            {
              int in_n = n - pad_N0;
              int in_c = c - pad_C0;
              int in_h = h - pad_H0;
              int in_w = w - pad_W0;
              result.at({n, c, h, w}) = at({in_n, in_c, in_h, in_w});
            }
          }
  }

  return result;
}

float Tensor::at(const std::vector<int> &index) const
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

float &Tensor::at(std::initializer_list<int> indices)
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

const float &Tensor::at(std::initializer_list<int> indices) const
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

Tensor Tensor::flatten(int start_dim) const
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

void Tensor::fill(float value)
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

Tensor Tensor::sum(int axis, bool keepdims) const
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

Tensor Tensor::sum(const std::vector<int> &dims) const
{
  // Identify axes to reduce
  std::vector<int> reduce_axes(shape.size(), 0);
  std::fill(reduce_axes.begin(), reduce_axes.end(), 0); // ← asegura que todas sean 0

  for (int d : dims)
  {
    if (d < 0)
      d += shape.size(); // Soporte para índices negativos
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

  cudaMemset(result.data, 0, result.size * sizeof(float)); // importantísimo
  sumMultiDimsKernel<<<blocks, threadsPerBlock>>>(
      result.data, data, d_in_strides, d_out_strides,
      d_reduce_axes, size, result.size,
      shape.size(), new_shape.size());

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

Tensor Tensor::mean(int dim, bool keepdim) const
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

Tensor Tensor::operator/(float scalar) const
{
  if (scalar == 0.0f)
    throw std::runtime_error("operator/(scalar): Division by zero");
  if (!data && size > 0)
    throw std::runtime_error("operator/(scalar): Input tensor data is null");
  if (size == 0)
    throw std::runtime_error("operator/(scalar): Tensor size is zero");

  Tensor result(shape, is_cuda);
  if (!result.data && size > 0)
    throw std::runtime_error("operator/(scalar): Result tensor allocation failed");

  if (is_cuda)
  {
    int threadsPerBlock = 256;
    int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    if (blocks == 0 || threadsPerBlock <= 0)
      throw std::runtime_error("operator/(scalar): Invalid kernel launch configuration (blocks=" + std::to_string(blocks) + ")");
    if (blocks * threadsPerBlock > 2147483647)
      throw std::runtime_error("operator/(scalar): Total threads exceed GPU limit");

    scalarDivKernel<<<blocks, threadsPerBlock>>>(result.data, data, scalar, size);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
      throw std::runtime_error("scalarDivKernel failed: " + std::string(cudaGetErrorString(err)));
  }
  else
  {
    for (size_t i = 0; i < size; ++i)
      result.data[i] = data[i] / scalar;
  }
  return result;
}

Tensor Tensor::reshape(const std::vector<int> &new_shape) const
{
  std::vector<int> adjusted_shape = new_shape;
  int minus_one_count = std::count(new_shape.begin(), new_shape.end(), -1);
  if (minus_one_count > 1)
  {
    throw std::invalid_argument("Reshape can only have one -1 dimension");
  }

  int new_size = 1;
  int minus_one_index = -1;
  for (size_t i = 0; i < new_shape.size(); ++i)
  {
    if (new_shape[i] == -1)
    {
      minus_one_index = i;
    }
    else
    {
      new_size *= new_shape[i];
    }
  }

  if (minus_one_index != -1)
  {
    if (size % new_size != 0)
    {
      throw std::invalid_argument("Cannot infer -1 dimension");
    }
    adjusted_shape[minus_one_index] = size / new_size;
    new_size = size; // Ensure total size matches
  }
  else
  {
    new_size = std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies<int>());
  }

  if (new_size != static_cast<int>(size))
  {
    throw std::invalid_argument("Reshape size must match original size");
  }

  if (is_cuda && size > 0 && data)
  {
    cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, data);
    if (err != cudaSuccess || attr.type != cudaMemoryTypeDevice)
    {
      throw std::runtime_error("reshape: data is not valid GPU memory: " + std::string(cudaGetErrorString(err)));
    }
  }

  Tensor result(adjusted_shape, is_cuda);
  if (size > 0 && data)
  {
    if (is_cuda)
    {
      cudaError_t err = cudaMemcpy(result.data, data, size * sizeof(float), cudaMemcpyDeviceToDevice);
      if (err != cudaSuccess)
      {
        throw std::runtime_error("reshape: cudaMemcpy failed: " + std::string(cudaGetErrorString(err)));
      }
    }
    else
    {
      std::memcpy(result.data, data, size * sizeof(float));
    }
  }

  return result;
}

Tensor Tensor::pow(float exponent) const
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

Tensor Tensor::sqrt() const
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

Tensor Tensor::reciprocal() const
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

Tensor Tensor::transpose(const std::vector<int> &axes) const
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

Tensor Tensor::permute(const std::vector<int> &dims) const
{
  // `permute` is identical to `transpose` in this implementation
  return transpose(dims);
}

Tensor Tensor::matmul(const Tensor &other) const
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
    int B1 = shape[0], M = shape[1], N = shape[2]; // Fix: Use shape[2]
    int B2 = other.shape[0], N2 = other.shape[1], P = other.shape[2];
    if (B1 != B2 || N != N2)
    {
      std::string error_msg = "Batch matrix dimensions must match for matmul: [" +
                              std::to_string(B1) + ", " + std::to_string(M) + ", " + std::to_string(N) +
                              "] × [" + std::to_string(B2) + ", " + std::to_string(N2) + ", " + std::to_string(P) + "]";
      throw std::invalid_argument(error_msg);
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

Tensor Tensor::bernoulli(const std::vector<int> &shape, float p)
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

Tensor Tensor::relu() const
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

Tensor Tensor::relu_derivative() const
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

Tensor Tensor::softmax(int dim) const
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

Tensor Tensor::operator*(const Tensor &other) const
{

  if (is_cuda != other.is_cuda)
    throw std::runtime_error("operator*: Both tensors must be on the same device");
  if (!data || !other.data)
    throw std::runtime_error("operator*: Input tensor data is null");

  // Determine output shape with broadcasting
  std::vector<int> result_shape = shape;
  std::vector<int> other_shape = other.shape;
  size_t max_dims = std::max(shape.size(), other.shape.size());
  result_shape.insert(result_shape.begin(), max_dims - shape.size(), 1);
  other_shape.insert(other_shape.begin(), max_dims - other.shape.size(), 1);

  // Check compatibility and compute output shape
  for (size_t i = 0; i < max_dims; ++i)
  {
    if (result_shape[i] != other_shape[i] && result_shape[i] != 1 && other_shape[i] != 1)
    {
      std::string error_msg = "operator*: Incompatible shapes for multiplication: [";
      for (size_t j = 0; j < shape.size(); ++j)
        error_msg += std::to_string(shape[j]) + (j < shape.size() - 1 ? ", " : "");
      error_msg += "] vs [";
      for (size_t j = 0; j < other.shape.size(); ++j)
        error_msg += std::to_string(other.shape[j]) + (j < other.shape.size() - 1 ? ", " : "");
      error_msg += "]";
      throw std::runtime_error(error_msg);
    }
    result_shape[i] = std::max(result_shape[i], other_shape[i]);
  }

  Tensor result(result_shape, is_cuda);

  if (!result.data && result.size > 0)
    throw std::runtime_error("operator*: result.data is nullptr");

  // Compute strides
  std::vector<int> a_strides(max_dims, 1);
  std::vector<int> b_strides(max_dims, 1);
  for (int i = max_dims - 2; i >= 0; --i)
  {
    a_strides[i] = a_strides[i + 1] * (shape.size() > i + 1 ? shape[i + 1] : 1);
    b_strides[i] = b_strides[i + 1] * (other.shape.size() > i + 1 ? other.shape[i + 1] : 1);
  }

  if (is_cuda)
  {
    // Allocate device memory for shapes and strides
    int *d_result_shape, *d_a_shape, *d_b_shape, *d_a_strides, *d_b_strides;
    cudaMalloc(&d_result_shape, max_dims * sizeof(int));
    cudaMalloc(&d_a_shape, max_dims * sizeof(int));
    cudaMalloc(&d_b_shape, max_dims * sizeof(int));
    cudaMalloc(&d_a_strides, max_dims * sizeof(int));
    cudaMalloc(&d_b_strides, max_dims * sizeof(int));

    cudaMemcpy(d_result_shape, result_shape.data(), max_dims * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_shape, shape.data(), max_dims * sizeof(int), cudaMemcpyHostToDevice); // Fixed: Use shape, not result_shape
    cudaMemcpy(d_b_shape, other_shape.data(), max_dims * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_strides, a_strides.data(), max_dims * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_strides, b_strides.data(), max_dims * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocks = (result.size + threadsPerBlock - 1) / threadsPerBlock;
    elementwiseMulKernelBroadcast<<<blocks, threadsPerBlock>>>(result.data, data, other.data,
                                                               d_result_shape, d_a_shape, d_b_shape,
                                                               d_a_strides, d_b_strides, max_dims, result.size);
    cudaDeviceSynchronize();

    cudaFree(d_result_shape);
    cudaFree(d_a_shape);
    cudaFree(d_b_shape);
    cudaFree(d_a_strides);
    cudaFree(d_b_strides);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
      throw std::runtime_error("elementwiseMulKernelBroadcast failed: " + std::string(cudaGetErrorString(err)));
  }
  else
  {
    for (size_t i = 0; i < result.size; ++i)
    {
      size_t a_idx = 0, b_idx = 0;
      size_t temp = i;
      for (int d = max_dims - 1; d >= 0; --d)
      {
        size_t coord = temp % result_shape[d];
        temp /= result_shape[d];
        a_idx += (result_shape[d] == 1 ? 0 : coord) * a_strides[d];
        b_idx += (other_shape[d] == 1 ? 0 : coord) * b_strides[d];
      }
      result.data[i] = data[a_idx] * other.data[b_idx];
    }
  }

  return result;
}

Tensor Tensor::operator-(const Tensor &other) const
{
  // Determine output shape with broadcasting
  std::vector<int> result_shape = shape;
  std::vector<int> other_shape = other.shape;

  // Pad shapes with 1s to match dimensions
  size_t max_dims = std::max(shape.size(), other.shape.size());
  result_shape.insert(result_shape.begin(), max_dims - shape.size(), 1);
  other_shape.insert(other_shape.begin(), max_dims - other.shape.size(), 1);

  // Check compatibility and compute output shape
  for (size_t i = 0; i < max_dims; ++i)
  {
    if (result_shape[i] != other_shape[i] && result_shape[i] != 1 && other_shape[i] != 1)
    {
      std::string error_msg = "Tensor shapes must match for element-wise subtraction: [";
      for (size_t j = 0; j < shape.size(); ++j)
        error_msg += std::to_string(shape[j]) + (j < shape.size() - 1 ? ", " : "");
      error_msg += "] vs [";
      for (size_t j = 0; j < other.shape.size(); ++j)
        error_msg += std::to_string(other.shape[j]) + (j < other.shape.size() - 1 ? ", " : "");
      error_msg += "]";
      throw std::invalid_argument(error_msg);
    }
    result_shape[i] = std::max(result_shape[i], other_shape[i]);
  }

  Tensor result(result_shape, is_cuda);
  if (is_cuda)
  {
    // Allocate device memory for shapes
    int *d_result_shape, *d_self_shape, *d_other_shape;
    cudaMalloc(&d_result_shape, max_dims * sizeof(int));
    cudaMalloc(&d_self_shape, max_dims * sizeof(int));
    cudaMalloc(&d_other_shape, max_dims * sizeof(int));

    cudaMemcpy(d_result_shape, result_shape.data(), max_dims * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_self_shape, result_shape.data(), max_dims * sizeof(int), cudaMemcpyHostToDevice); // Use padded shape
    cudaMemcpy(d_other_shape, other_shape.data(), max_dims * sizeof(int), cudaMemcpyHostToDevice);

    // Launch CUDA kernel
    int threadsPerBlock = 256;
    int blocks = (result.size + threadsPerBlock - 1) / threadsPerBlock;
    elementwiseSubKernel<<<blocks, threadsPerBlock>>>(result.data, data, other.data,
                                                      d_result_shape, d_self_shape, d_other_shape,
                                                      max_dims, result.size);
    cudaDeviceSynchronize();

    // Free device memory
    cudaFree(d_result_shape);
    cudaFree(d_self_shape);
    cudaFree(d_other_shape);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
      throw std::runtime_error("elementwiseSubKernel failed: " + std::string(cudaGetErrorString(err)));
    }
  }
  else
  {
    // CPU broadcasting (existing logic)
    std::vector<int> self_strides(max_dims, 1);
    std::vector<int> other_strides(max_dims, 1);
    for (int i = max_dims - 2; i >= 0; --i)
    {
      self_strides[i] = self_strides[i + 1] * result_shape[i + 1];
      other_strides[i] = other_strides[i + 1] * other_shape[i + 1];
    }

#pragma omp parallel for
    for (size_t i = 0; i < result.size; ++i)
    {
      size_t index_self = 0, index_other = 0;
      size_t rem = i;
      for (size_t d = 0; d < max_dims; ++d)
      {
        size_t coord = rem / self_strides[d];
        rem %= self_strides[d];
        index_self += (result_shape[d] == 1 ? 0 : coord) * self_strides[d];
        index_other += (other_shape[d] == 1 ? 0 : coord) * other_strides[d];
      }
      result.data[i] = data[index_self] - other.data[index_other];
    }
  }
  return result;
}

Tensor Tensor::softmax_backward(const Tensor &softmax, const Tensor &delta)
{
  if (softmax.shape != delta.shape)
  {
    throw std::invalid_argument("Softmax and delta shapes must match");
  }
  Tensor sum = (delta * softmax).sum(-1, true); // [N, HW, 1]
  return softmax * (delta - sum);
}

Tensor Tensor::operator+(const Tensor &other) const
{

  // Validate device consistency
  if (is_cuda != other.is_cuda)
  {
    throw std::runtime_error("operator+: Both tensors must be on the same device");
  }

  // Case 1: Same shape (simple element-wise addition)
  if (shape == other.shape)
  {
    // Create result tensor with same is_cuda flag
    Tensor result(shape, is_cuda);

    // Validate result.data
    if (!result.data && size > 0)
    {
      throw std::runtime_error("operator+: result.data is nullptr");
    }

    // Validate input pointers
    cudaPointerAttributes attr_this, attr_other;
    cudaError_t err_this = cudaPointerGetAttributes(&attr_this, data);
    cudaError_t err_other = cudaPointerGetAttributes(&attr_other, other.data);
    if (err_this != cudaSuccess || (is_cuda && attr_this.type != cudaMemoryTypeDevice))
    {
      throw std::runtime_error("operator+: this.data is not valid GPU memory: " + std::string(cudaGetErrorString(err_this)));
    }
    if (err_other != cudaSuccess || (is_cuda && attr_other.type != cudaMemoryTypeDevice))
    {
      throw std::runtime_error("operator+: other.data is not valid GPU memory: " + std::string(cudaGetErrorString(err_other)));
    }

    int threadsPerBlock = 256;
    int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    simpleAddKernel<<<blocks, threadsPerBlock>>>(result.data, data, other.data, size);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
      throw std::runtime_error("simpleAddKernel failed: " + std::string(cudaGetErrorString(err)));
    }

    // Validate result.data after kernel
    cudaPointerAttributes attr_result;
    cudaError_t err_result = cudaPointerGetAttributes(&attr_result, result.data);
    if (err_result != cudaSuccess || (is_cuda && attr_result.type != cudaMemoryTypeDevice))
    {
      throw std::runtime_error("operator+: result.data is not valid GPU memory after kernel");
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
    Tensor result(shape, is_cuda);

    if (!result.data && size > 0)
    {
      throw std::runtime_error("operator+: result.data is nullptr");
    }

    // Validate input pointers
    cudaPointerAttributes attr_this, attr_other;
    cudaError_t err_this = cudaPointerGetAttributes(&attr_this, data);
    cudaError_t err_other = cudaPointerGetAttributes(&attr_other, other.data);
    if (err_this != cudaSuccess || (is_cuda && attr_this.type != cudaMemoryTypeDevice))
    {
      throw std::runtime_error("operator+: this.data is not valid GPU memory: " + std::string(cudaGetErrorString(err_this)));
    }
    if (err_other != cudaSuccess || (is_cuda && attr_other.type != cudaMemoryTypeDevice))
    {
      throw std::runtime_error("operator+: other.data is not valid GPU memory: " + std::string(cudaGetErrorString(err_other)));
    }

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

    // Validate result.data after kernel
    cudaPointerAttributes attr_result;
    cudaError_t err_result = cudaPointerGetAttributes(&attr_result, result.data);
    if (err_result != cudaSuccess || (is_cuda && attr_result.type != cudaMemoryTypeDevice))
    {
      throw std::runtime_error("operator+: result.data is not valid GPU memory after kernel");
    }

    return result;
  }

  // Case 3: Broadcasting 4D [B, C, H, W] + [1, C, 1, 1]
  if (shape.size() == 4 && other.shape.size() == 4 &&
      other.shape[0] == 1 && other.shape[2] == 1 && other.shape[3] == 1 && shape[1] == other.shape[1])
  {
    Tensor result(shape, is_cuda);

    if (!result.data && size > 0)
    {
      throw std::runtime_error("operator+: result.data is nullptr");
    }

    // Validate input pointers
    cudaPointerAttributes attr_this, attr_other;
    cudaError_t err_this = cudaPointerGetAttributes(&attr_this, data);
    cudaError_t err_other = cudaPointerGetAttributes(&attr_other, other.data);
    if (err_this != cudaSuccess || (is_cuda && attr_this.type != cudaMemoryTypeDevice))
    {
      throw std::runtime_error("operator+: this.data is not valid GPU memory: " + std::string(cudaGetErrorString(err_this)));
    }
    if (err_other != cudaSuccess || (is_cuda && attr_other.type != cudaMemoryTypeDevice))
    {
      throw std::runtime_error("operator+: other.data is not valid GPU memory: " + std::string(cudaGetErrorString(err_other)));
    }

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

    // Validate result.data after kernel
    cudaPointerAttributes attr_result;
    cudaError_t err_result = cudaPointerGetAttributes(&attr_result, result.data);
    if (err_result != cudaSuccess || (is_cuda && attr_result.type != cudaMemoryTypeDevice))
    {
      throw std::runtime_error("operator+: result.data is not valid GPU memory after kernel");
    }

    return result;
  }

  throw std::runtime_error("operator+: Incompatible shapes for addition with broadcasting");
}

Tensor Tensor::operator+(float scalar) const
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
Tensor Tensor::operator*(float scalar) const
{
  if (!data && size > 0)
    throw std::runtime_error("operator*(scalar): Input tensor data is null");
  if (size == 0)
    throw std::runtime_error("operator*(scalar): Tensor size is zero");

  Tensor result(shape, is_cuda);
  if (!result.data && size > 0)
    throw std::runtime_error("operator*(scalar): Result tensor allocation failed");

  if (is_cuda)
  {
    int threadsPerBlock = 256;
    int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    if (blocks == 0 || threadsPerBlock <= 0)
      throw std::runtime_error("operator*(scalar): Invalid kernel launch configuration (blocks=" + std::to_string(blocks) + ")");
    if (blocks * threadsPerBlock > 2147483647) // Max threads limit
      throw std::runtime_error("operator*(scalar): Total threads exceed GPU limit");

    scalarMulKernel<<<blocks, threadsPerBlock>>>(result.data, data, scalar, size);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
      throw std::runtime_error("scalarMulKernel failed: " + std::string(cudaGetErrorString(err)));
  }
  else
  {
    for (size_t i = 0; i < size; ++i)
      result.data[i] = data[i] * scalar;
  }
  return result;
}
// Método para copiar datos de GPU a CPU (para inspección o depuración)
std::vector<float> Tensor::to_host() const
{

  std::vector<float> host_data(size);

  if (size == 0)
    throw std::runtime_error("❌ [to_host] size == 0");

  if (is_cuda)
  {
    if (data == nullptr)
      throw std::runtime_error("❌ [to_host] data == nullptr (esperado en GPU)");

    cudaError_t err = cudaMemcpy(host_data.data(), data, size * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
      std::cerr << "❌ [to_host] cudaMemcpy falló. data=" << static_cast<const void *>(data)
                << ", size=" << size << ", err=" << cudaGetErrorString(err) << "\n";
      throw std::runtime_error("cudaMemcpy failed: " + std::string(cudaGetErrorString(err)));
    }
  }
  else
  {
    if (data == nullptr)
      throw std::runtime_error("❌ [to_host] data == nullptr (esperado en CPU)");

    std::copy(data, data + size, host_data.begin());
  }

  return host_data;
}

// eRROR
Tensor Tensor::to_device(bool use_cuda) const
{
  if (this->size == 0)
    throw std::runtime_error("❌ [to_device] Tensor vacío (size == 0)");
  if (this->data == nullptr)
    throw std::runtime_error("❌ [to_device] data == nullptr");

  Tensor clone;
  clone.shape = this->shape;
  clone.size = this->size;
  clone.is_cuda = use_cuda;

  if (use_cuda)
  {
    cudaError_t err = cudaMalloc(&clone.data, size * sizeof(float));
    if (err != cudaSuccess)
      throw std::runtime_error("cudaMalloc failed: " + std::string(cudaGetErrorString(err)));

    if (this->is_cuda)
    {
      err = cudaMemcpy(clone.data, this->data, size * sizeof(float), cudaMemcpyDeviceToDevice);
    }
    else
    {
      err = cudaMemcpy(clone.data, this->data, size * sizeof(float), cudaMemcpyHostToDevice);
    }
    if (err != cudaSuccess)
      throw std::runtime_error("cudaMemcpy to_device failed: " + std::string(cudaGetErrorString(err)));
  }
  else
  {
    clone.data = new float[size];
    if (this->is_cuda)
    {
      cudaError_t err = cudaMemcpy(clone.data, this->data, size * sizeof(float), cudaMemcpyDeviceToHost);
      if (err != cudaSuccess)
        throw std::runtime_error("cudaMemcpy DeviceToHost failed: " + std::string(cudaGetErrorString(err)));
    }
    else
    {
      std::copy(this->data, this->data + size, clone.data);
    }
  }

  return clone;
}

float Tensor::compute_accuracy(const Tensor &preds, const Tensor &targets, int num_classes)
{
  int batch_size = preds.shape[0];
  int *d_correct;
  cudaMalloc(&d_correct, sizeof(int));
  cudaMemset(d_correct, 0, sizeof(int));

  int threads = 256;
  int blocks = (batch_size + threads - 1) / threads;

  compute_accuracy_kernel<<<blocks, threads>>>(
      preds.data, targets.data, d_correct, num_classes, batch_size);

  cudaDeviceSynchronize();

  int h_correct;
  cudaMemcpy(&h_correct, d_correct, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_correct);

  return static_cast<float>(h_correct) / batch_size;
}
__global__ void crossEntropyLossKernel(const float *preds, const float *targets, float *loss, int B, int C)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= B)
    return;

  float sample_loss = 0.0f;
  for (int j = 0; j < C; ++j)
  {
    float y = targets[idx * C + j];
    float p = preds[idx * C + j];
    sample_loss += -y * logf(p + 1e-8f);
  }

  atomicAdd(loss, sample_loss);
}

float Tensor::compute_loss(const Tensor &preds, const Tensor &targets_onehot)
{

  if (preds.shape.size() != 2 || targets_onehot.shape.size() != 2)
    throw std::runtime_error("compute_loss: Expected 2D tensors");
  if (preds.shape[0] != targets_onehot.shape[0] || preds.shape[1] != targets_onehot.shape[1])
    throw std::runtime_error("compute_loss: Shape mismatch");
  if (!preds.data || !targets_onehot.data)
    throw std::runtime_error("compute_loss: Null tensor data");
  if (preds.size == 0 || targets_onehot.size == 0)
    throw std::runtime_error("compute_loss: Zero-sized tensor");

  int B = preds.shape[0];
  int C = preds.shape[1];
  float loss = 0.0f;

  if (preds.is_cuda && targets_onehot.is_cuda)
  {
    // Allocate single float for loss
    float *d_loss;
    cudaMalloc(&d_loss, sizeof(float));
    cudaMemset(d_loss, 0, sizeof(float)); // Initialize to zero
    int threadsPerBlock = 256;
    int blocks = (B + threadsPerBlock - 1) / threadsPerBlock;
    crossEntropyLossKernel<<<blocks, threadsPerBlock>>>(preds.data, targets_onehot.data, d_loss, B, C);

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
      cudaFree(d_loss);
      throw std::runtime_error("crossEntropyLossKernel failed: " + std::string(cudaGetErrorString(err)));
    }

    // Copy result to host
    cudaMemcpy(&loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_loss);
    loss /= B; // muy importante
  }
  else
  {
    for (int i = 0; i < B; ++i)
    {
      for (int j = 0; j < C; ++j)
      {
        float y = targets_onehot.data[i * C + j];
        float p = preds.data[i * C + j];
        loss += -y * std::log(p + 1e-8f);
      }
    }
    loss /= B;
  }

  return loss;
}

bool Tensor::empty() const
{
  return data == nullptr || size == 0;
}

Tensor Tensor::mean(const std::vector<int> &dims, bool keepdims) const
{
  Tensor s = this->sum(dims); // Este `sum` debe funcionar en GPU si `this->data` está en device.

  int reduce_size = 1;
  for (int d : dims)
    reduce_size *= shape[d];

  s = s / static_cast<float>(reduce_size); // Esto debe funcionar con operador `/` en GPU.

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
    return s.reshape(new_shape); // reshape debe funcionar también en GPU.
  }

  return s;
}

std::string Tensor::printsummary(std::string name) const
{
  std::ostringstream oss;
  if (!name.empty())
    oss << "Tensor: " << name << "\n";

  oss << " " << shape.size() << ", shape: [";
  for (size_t i = 0; i < shape.size(); ++i)
  {
    oss << shape[i];
    if (i < shape.size() - 1)
      oss << ", ";
  }
  oss << "]";
  return oss.str();
}

// maxpool2d_cuda.cu

__global__ void maxpool2d_forward_kernell(float *output, const float *input, int *max_indices,
                                          int N, int C, int H, int W, int H_out, int W_out,
                                          int kernel_size, int stride)
{
  int n = blockIdx.z / C;
  int c = blockIdx.z % C;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (n < N && c < C && i < H_out && j < W_out)
  {
    float max_val = -FLT_MAX;
    int max_idx = -1;

    for (int ki = 0; ki < kernel_size; ++ki)
    {
      for (int kj = 0; kj < kernel_size; ++kj)
      {
        int h = i * stride + ki;
        int w = j * stride + kj;
        if (h >= 0 && h < H && w >= 0 && w < W)
        {
          int input_idx = ((n * C + c) * H + h) * W + w;
          float val = input[input_idx];
          if (val > max_val)
          {
            max_val = val;
            max_idx = h * W + w;
          }
        }
      }
    }

    int output_idx = ((n * C + c) * H_out + i) * W_out + j;
    output[output_idx] = max_val;
    max_indices[output_idx] = max_idx;
  }
}

__global__ void maxpool2d_backward_kernell(float *input_deltas, const float *delta,
                                           const int *max_indices,
                                           int N, int C, int H_out, int W_out, int H, int W)
{
  int n = blockIdx.z / C;
  int c = blockIdx.z % C;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (n < N && c < C && i < H_out && j < W_out)
  {
    int idx = ((n * C + c) * H_out + i) * W_out + j;
    int max_pos = max_indices[idx];
    int h = max_pos / W;
    int w = max_pos % W;
    int input_idx = ((n * C + c) * H + h) * W + w;
    atomicAdd(&input_deltas[input_idx], delta[idx]);
  }
}

Tensor maxpool2d_cuda_forwardd(const Tensor &input, Tensor &max_indices_tensor, int kernel_size, int stride)
{
  int N = input.shape[0];
  int C = input.shape[1];
  int H = input.shape[2];
  int W = input.shape[3];

  int H_out = (H - kernel_size) / stride + 1;
  int W_out = (W - kernel_size) / stride + 1;

  Tensor output({N, C, H_out, W_out}, true);

  dim3 blockDim(16, 16);
  dim3 gridDim((H_out + 15) / 16, (W_out + 15) / 16, N * C);

  maxpool2d_forward_kernell<<<gridDim, blockDim>>>(
      output.data, input.data, (int *)max_indices_tensor.data,
      N, C, H, W, H_out, W_out, kernel_size, stride);

  cudaDeviceSynchronize();
  return output;
}

Tensor maxpool2d_cuda_backwardd(const Tensor &delta, const Tensor &input, const Tensor &max_indices_tensor, int kernel_size, int stride)
{
  int N = input.shape[0];
  int C = input.shape[1];
  int H = input.shape[2];
  int W = input.shape[3];
  int H_out = delta.shape[2];
  int W_out = delta.shape[3];

  Tensor input_deltas = Tensor::zeros(input.shape, true);

  dim3 blockDim(16, 16);
  dim3 gridDim((H_out + 15) / 16, (W_out + 15) / 16, N * C);

  maxpool2d_backward_kernell<<<gridDim, blockDim>>>(
      input_deltas.data, delta.data,
      (int *)max_indices_tensor.data,
      N, C, H_out, W_out, H, W);

  cudaDeviceSynchronize();
  return input_deltas;
}


void print_tensor(const Tensor &t, const std::string &name, int max_elems)
{
    std::cout << "[DEBUG] " << name << " | shape: (";
    for (size_t i = 0; i < t.shape.size(); ++i)
    {
        std::cout << t.shape[i];
        if (i != t.shape.size() - 1) std::cout << ", ";
    }
    std::cout << ") | size: " << t.size << " | is_cuda: " << t.is_cuda << "\n";

    std::vector<float> host_data(t.size);

    if (t.is_cuda)
    {
        cudaMemcpy(host_data.data(), t.data, t.size * sizeof(float), cudaMemcpyDeviceToHost);
    }
    else
    {
        std::memcpy(host_data.data(), t.data, t.size * sizeof(float));
    }

    std::cout << "[DEBUG] " << name << " values (first " << max_elems << "): ";
    for (int i = 0; i < std::min((size_t)max_elems, t.size); ++i)
    {
        std::cout << host_data[i] << " ";
    }
    std::cout << "\n";
}
