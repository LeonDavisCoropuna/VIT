#pragma once

#include <vector>
#include <random>
#include <numeric>
#include <stdexcept>
#include <cuda_runtime.h>

class Tensor
{
public:
  std::vector<int> shape;
  float *data;
  size_t size;
  bool is_cuda = true;
  static std::mt19937 global_gen;

  static void set_seed(unsigned int seed);

  Tensor();
  Tensor(const std::vector<int> &shape, bool use_cuda = true);
  Tensor(const std::vector<int> &shape, const std::vector<float> &host_data, bool use_cuda = true);

  Tensor(const Tensor &other);
  Tensor(Tensor &&other) noexcept;

  ~Tensor();

  void fill(float value); // Método que ejecuta un kernel
  Tensor &operator=(const Tensor &other);
  Tensor &operator=(Tensor &&other) noexcept;

  static Tensor zeros(const std::vector<int> &shape, bool use_cuda = true);
  static Tensor rand_uniform(const std::vector<int> &shape, float low, float high);
  static Tensor rand_uniform_gpu(const std::vector<int> &shape, float low, float high);
  static Tensor xavier_normal(const std::vector<int> &shape, int fan_in, int fan_out);
  static Tensor xavier_uniform(const std::vector<int> &shape, int fan_in, int fan_out);
  static Tensor kaiming_normal(const std::vector<int> &shape, int fan_in, bool use_negative = true);
  static Tensor kaiming_uniform(const std::vector<int> &shape, int fan_in, bool use_negative = true);
  static Tensor one_hot(const Tensor &labels, int num_classes);
  Tensor slice(const std::vector<int> &starts, const std::vector<int> &ends) const;
  Tensor slice(int dim, int start, int end) const;
  static Tensor concat(const std::vector<Tensor> &tensors, int dim);
  Tensor pad(const std::vector<int> &pads) const;
  float at(const std::vector<int> &index) const;
  float &at(std::initializer_list<int> indices);
  const float &at(std::initializer_list<int> indices) const;
  Tensor flatten(int start_dim) const;
  Tensor sum(int axis, bool keepdims = false) const;
  Tensor sum(const std::vector<int> &dims) const;
  Tensor mean(int dim, bool keepdim) const;
  Tensor operator/(float value) const;
  Tensor reshape(const std::vector<int> &new_shape) const;
  Tensor pow(float exponent) const;
  Tensor sqrt() const;
  Tensor reciprocal() const;
  Tensor transpose(const std::vector<int> &axes) const;
  Tensor permute(const std::vector<int> &dims) const;
  Tensor matmul(const Tensor &other) const;
  Tensor mean(const std::vector<int> &dims, bool keepdims) const;
  static Tensor bernoulli(const std::vector<int> &shape, float p);
  Tensor relu() const;
  Tensor relu_derivative() const;
  Tensor softmax(int dim) const;
  Tensor operator*(const Tensor &other) const;
  Tensor operator-(const Tensor &other) const;
  static Tensor softmax_backward(const Tensor &softmax, const Tensor &delta);
  Tensor operator+(const Tensor &other) const;
  Tensor operator+(float scalar) const;
  Tensor operator*(float scalar) const;
  std::vector<float> to_host() const;
  Tensor to_device(bool use_cuda) const;
  static float compute_accuracy(const Tensor &preds, const Tensor &targets, int num_classes);
  static float compute_loss(const Tensor &preds, const Tensor &targets_onehot);
  bool empty() const;
  Tensor broadcast_to(const std::vector<int> &target_shape) const;
  std::string printsummary(std::string name = "") const;
};

#ifdef USE_CUDA
// Forward pass for convolution
Tensor conv2d_cuda_forward(const Tensor &input,
                           const Tensor &weights,
                           const Tensor &biases,
                           int stride,
                           int padding);

// Backward pass for convolution
void conv2d_cuda_backward(const Tensor &delta,
                          const Tensor &input,
                          const Tensor &weights,
                          Tensor &grad_weights,
                          Tensor &grad_biases,
                          Tensor &input_deltas,
                          int stride,
                          int padding);

// Existing maxpool and avgpool declarations
Tensor maxpool2d_cuda_forward(const Tensor &input, int *d_max_indices, int kernel_size, int stride);
Tensor maxpool2d_cuda_backward(const Tensor &delta, const Tensor &input, const int *d_max_indices,
                               int kernel_size, int stride);
Tensor avgpool2d_cuda_forward(const Tensor &input, int kernel_size, int stride);
Tensor avgpool2d_cuda_backward(const Tensor &delta, const Tensor &input, int kernel_size, int stride);

Tensor maxpool2d_cuda_forwardd(const Tensor &input, Tensor &max_indices_tensor, int kernel_size, int stride);
Tensor maxpool2d_cuda_backwardd(const Tensor &delta, const Tensor &input, Tensor &max_indices_tensor,
                                int kernel_size, int stride);

void print_tensor(const Tensor &t, const std::string &name = "Tensor", int max_elems = 20);
#endif
