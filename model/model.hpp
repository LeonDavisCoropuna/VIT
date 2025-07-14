#pragma once

#include "../layers/layer.hpp"

class Model
{
protected:
  std::vector<Layer *> layers;
  Tensor output;

public:
  virtual ~Model()
  {
    for (auto *layer : layers)
      delete layer;
  }
  size_t get_num_layers() const { return layers.size(); }
  Layer *get_layer(size_t i) const { return layers[i]; }

  virtual Tensor forward(const Tensor &input)
  {
    Tensor x = input;
    std::vector<Tensor> x_vec = {x};
    for (auto *layer : layers)
      x_vec[0] = layer->forward(x_vec)[0];
    output = x_vec[0];
    return output;
  }

  virtual void backward(const Tensor &targets)
  {
    const Tensor *target_ptr = &targets;
    for (int i = layers.size() - 1; i >= 0; --i)
      layers[i]->backward(target_ptr, i + 1 < layers.size() ? layers[i + 1] : nullptr);
  }

  virtual void update_weights(float batch_size)
  {
    for (auto *layer : layers)
      if (layer->has_weights())
        layer->update_weights(batch_size);
  }

  virtual void zero_grad()
  {
    for (auto *layer : layers)
      layer->zero_grad();
  }

  virtual void set_training(bool is_training)
  {
    for (auto *layer : layers)
      layer->set_training(is_training);
  }

  const Tensor &get_output() const
  {
    return output;
  }
};
