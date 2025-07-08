#pragma once
#include <vector>
#include "../utils/tensor.hpp"
#include "../utils/optimizer.hpp"

class Layer
{
protected:
  bool isTraining = true;
  Optimizer *optimizer;

public:
  virtual ~Layer() = default;

  virtual std::vector<Tensor> forward(const std::vector<Tensor> &inputs) = 0;

  virtual void backward(const Tensor *targets = nullptr,
                        const Layer *nextLayer = nullptr) = 0;

  virtual void update_weights(float batchSize) = 0;

  virtual void zero_grad() = 0;

  virtual bool has_weights() const { return false; }

  virtual void set_training(bool isTraining) = 0;

  virtual const Tensor &get_input_deltas() const = 0;

  static std::mt19937 gen;

  virtual void set_optimizer(Optimizer *opt)
  {
    this->optimizer = opt;
  }
};
