#pragma once
#include "layer.hpp"

class DenseLayer : public Layer
{
private:
  Tensor weights;
  Tensor biases;
  Tensor input;
  Tensor output;
  Tensor gradWeights;
  Tensor gradBiases;
  Tensor inputDeltas;
  Tensor outputDeltas;
  bool useBias;
  bool isTraining;

public:
  DenseLayer(int inFeatures, int outFeatures, bool useBias = true)
      : useBias(useBias), isTraining(true)
  {
    // Xavier initialization
    float limit = std::sqrt(6.0f / (inFeatures + outFeatures));
    weights = Tensor::rand_uniform({inFeatures, outFeatures}, -limit, limit);
    gradWeights = Tensor::zeros({inFeatures, outFeatures});

    if (useBias)
    {
      biases = Tensor::zeros({outFeatures});
      gradBiases = Tensor::zeros({outFeatures});
    }
  }

  std::vector<Tensor> forward(const std::vector<Tensor> &inputs) override
  {
    input = inputs[0];               // (N, inFeatures)
    output = input.matmul(weights);  // (N, outFeatures)
    if (useBias)
      output = output + biases.reshape({1, -1}); // broadcasting

    return {outputDeltas = output};
  }

  void backward(const Tensor *targets = nullptr,
                const Layer *next_layer = nullptr) override
  {
    Tensor delta;
    if (targets)
    {
      delta = *targets;
    }
    else
    {
      delta = next_layer->get_input_deltas();
    }

    Tensor inputTransposed = input.transpose({1, 0});     // (inFeatures, N)
    gradWeights = inputTransposed.matmul(delta);          // (inFeatures, outFeatures)

    if (useBias)
      gradBiases = delta.sum(0); // sum across batch

    Tensor weightsTransposed = weights.transpose({1, 0}); // (outFeatures, inFeatures)
    inputDeltas = delta.matmul(weightsTransposed);        // (N, inFeatures)
  }

  void update_weights(float batchSize) override
  {
    if (optimizer)
    {
      gradWeights = gradWeights / batchSize;
      gradBiases = gradBiases / batchSize;
      optimizer->update(weights, gradWeights,
                        biases, gradBiases);
    }
  }

  void zero_grad() override
  {
    gradWeights.fill(0.0f);
    if (useBias)
      gradBiases.fill(0.0f);
  }

  void set_training(bool training) override
  {
    isTraining = training;
  }

  const Tensor &get_output() const
  {
    return output;
  }

  const Tensor &get_input_deltas() const override
  {
    return inputDeltas;
  }

  const Tensor &get_last_input() const
  {
    return input;
  }

  bool has_weights() const override
  {
    return true;
  }
};
