#pragma once
#include "tensor.hpp"

class Optimizer
{
public:
  virtual ~Optimizer() = default;

  // Actualiza pesos y biases usando sus respectivos gradientes
  virtual void update(Tensor &weights, const Tensor &grad_weights,
                      Tensor &biases, const Tensor &grad_biases) = 0;
};

class SGD : public Optimizer
{
private:
  float learning_rate;

public:
  explicit SGD(float lr = 0.01f) : learning_rate(lr) {}

  void update(Tensor &weights, const Tensor &grad_weights,
              Tensor &biases, const Tensor &grad_biases) override
  {
    // w = w - lr * dw
    weights = weights - grad_weights * learning_rate;
    biases = biases - grad_biases * learning_rate;
  }
};