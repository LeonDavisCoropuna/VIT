#pragma once
#include "layer.hpp"

class Conv2DLayer : public Layer
{
private:
  int inChannels, outChannels;
  int kernelHeight, kernelWidth;
  int stride, padding;

  Tensor weights;       // [outChannels, inChannels, kernelHeight, kernelWidth]
  Tensor biases;        // [outChannels]
  Tensor gradWeights;
  Tensor gradBiases;
  Tensor input;
  Tensor inputDeltas;
  Tensor output;

public:
  Conv2DLayer(int inChannels, int outChannels, int kernelHeight, int kernelWidth,
              int stride = 1, int padding = 0)
      : inChannels(inChannels), outChannels(outChannels),
        kernelHeight(kernelHeight), kernelWidth(kernelWidth),
        stride(stride), padding(padding)
  {
    int fanIn = inChannels * kernelHeight * kernelWidth;
    weights = Tensor::kaiming_normal({outChannels, inChannels, kernelHeight, kernelWidth}, fanIn);
    biases = Tensor::zeros({outChannels});
    gradWeights = Tensor::zeros(weights.shape);
    gradBiases = Tensor::zeros(biases.shape);
  }

  std::vector<Tensor> forward(const std::vector<Tensor> &inputs) override
  {
    input = inputs[0]; // [N, C_in, H_in, W_in]
    const int batchSize = input.shape[0];
    const int inputChannels = input.shape[1];
    const int inputHeight = input.shape[2];
    const int inputWidth = input.shape[3];

    const int outputChannels = weights.shape[0];
    const int kernelH = weights.shape[2];
    const int kernelW = weights.shape[3];

    const int outputHeight = (inputHeight + 2 * padding - kernelH) / stride + 1;
    const int outputWidth = (inputWidth + 2 * padding - kernelW) / stride + 1;

    output = Tensor({batchSize, outputChannels, outputHeight, outputWidth});
    output.fill(0.0f);

    for (int n = 0; n < batchSize; ++n)
    {
      for (int oc = 0; oc < outputChannels; ++oc)
      {
        for (int oh = 0; oh < outputHeight; ++oh)
        {
          for (int ow = 0; ow < outputWidth; ++ow)
          {
            float sum = biases.at({oc});
            for (int ic = 0; ic < inputChannels; ++ic)
            {
              for (int kh = 0; kh < kernelH; ++kh)
              {
                for (int kw = 0; kw < kernelW; ++kw)
                {
                  int ih = oh * stride + kh - padding;
                  int iw = ow * stride + kw - padding;

                  if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth)
                  {
                    sum += input.at({n, ic, ih, iw}) *
                           weights.at({oc, ic, kh, kw});
                  }
                }
              }
            }
            output.at({n, oc, oh, ow}) = sum;
          }
        }
      }
    }
    return {output};
  }

  void backward(const Tensor *targets = nullptr, const Layer *next_layer = nullptr) override
  {
    const Tensor &delta = next_layer->get_input_deltas(); // [N, C_out, H_out, W_out]

    const int batchSize = input.shape[0];
    const int inputChannels = input.shape[1];
    const int inputHeight = input.shape[2];
    const int inputWidth = input.shape[3];

    const int outputChannels = weights.shape[0];
    const int kernelH = weights.shape[2];
    const int kernelW = weights.shape[3];

    const int outputHeight = delta.shape[2];
    const int outputWidth = delta.shape[3];

    Tensor paddedInput = input;
    if (padding > 0)
      paddedInput = input.pad({0, 0, 0, 0, padding, padding, padding, padding});

    // gradBiases
    for (int n = 0; n < batchSize; ++n)
      for (int oc = 0; oc < outputChannels; ++oc)
        for (int oh = 0; oh < outputHeight; ++oh)
          for (int ow = 0; ow < outputWidth; ++ow)
            gradBiases.at({oc}) += delta.at({n, oc, oh, ow});

    // gradWeights
    for (int oc = 0; oc < outputChannels; ++oc)
    {
      for (int ic = 0; ic < inputChannels; ++ic)
      {
        for (int kh = 0; kh < kernelH; ++kh)
        {
          for (int kw = 0; kw < kernelW; ++kw)
          {
            float sum = 0.0f;
            for (int n = 0; n < batchSize; ++n)
            {
              for (int oh = 0; oh < outputHeight; ++oh)
              {
                for (int ow = 0; ow < outputWidth; ++ow)
                {
                  int ih = oh * stride + kh;
                  int iw = ow * stride + kw;
                  sum += paddedInput.at({n, ic, ih, iw}) * delta.at({n, oc, oh, ow});
                }
              }
            }
            gradWeights.at({oc, ic, kh, kw}) = sum;
          }
        }
      }
    }

    // inputDeltas
    Tensor paddedDeltas = Tensor::zeros({batchSize, inputChannels, inputHeight + 2 * padding, inputWidth + 2 * padding});

    for (int n = 0; n < batchSize; ++n)
    {
      for (int oc = 0; oc < outputChannels; ++oc)
      {
        for (int oh = 0; oh < outputHeight; ++oh)
        {
          for (int ow = 0; ow < outputWidth; ++ow)
          {
            for (int ic = 0; ic < inputChannels; ++ic)
            {
              for (int kh = 0; kh < kernelH; ++kh)
              {
                for (int kw = 0; kw < kernelW; ++kw)
                {
                  int ih = oh * stride + kh;
                  int iw = ow * stride + kw;
                  if (ih >= 0 && ih < paddedDeltas.shape[2] &&
                      iw >= 0 && iw < paddedDeltas.shape[3])
                  {
                    float flippedWeight = weights.at({oc, ic, kernelH - 1 - kh, kernelW - 1 - kw});
                    paddedDeltas.at({n, ic, ih, iw}) += delta.at({n, oc, oh, ow}) * flippedWeight;
                  }
                }
              }
            }
          }
        }
      }
    }

    if (padding > 0)
    {
      inputDeltas = paddedDeltas.slice(2, padding, padding + inputHeight)
                               .slice(3, padding, padding + inputWidth);
    }
    else
    {
      inputDeltas = paddedDeltas;
    }
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
    gradBiases.fill(0.0f);
  }

  void set_training(bool training) override
  {
    is_training = training;
  }

  const Tensor &get_input_deltas() const override
  {
    return inputDeltas;
  }

  bool has_weights() const override
  {
    return true;
  }
};
