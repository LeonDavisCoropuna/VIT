#pragma once

#include <filesystem>
#include <vector>
#include <regex>
#include <string>
#include <iostream>
#include "tensor.hpp"
#include "fstream"

struct Dataset
{
  std::vector<Tensor> images;
  std::vector<Tensor> labels;
};

namespace fs = std::filesystem;
std::vector<Tensor> loadImages2D(const std::string &filename, int max_images = 9999999)
{
  std::ifstream file(filename, std::ios::binary);
  if (!file)
    throw std::runtime_error("No se pudo abrir el archivo de imágenes");

  int32_t magic = 0, num = 0, rows = 0, cols = 0;
  file.read(reinterpret_cast<char *>(&magic), 4);
  file.read(reinterpret_cast<char *>(&num), 4);
  file.read(reinterpret_cast<char *>(&rows), 4);
  file.read(reinterpret_cast<char *>(&cols), 4);

  magic = __builtin_bswap32(magic);
  num = __builtin_bswap32(num);
  rows = __builtin_bswap32(rows);
  cols = __builtin_bswap32(cols);

  if (max_images > 0 && max_images < num)
  {
    num = max_images;
  }

  std::vector<Tensor> images;
  images.reserve(num);

  for (int i = 0; i < num; ++i)
  {
    std::vector<float> data(rows * cols);
    for (int j = 0; j < rows * cols; ++j)
    {
      unsigned char pixel = 0;
      file.read(reinterpret_cast<char *>(&pixel), 1);
      data[j] = static_cast<float>(pixel) / 255.0f;
    }

    // Cada imagen tiene forma [1, rows, cols] → 1 canal
    Tensor image({1, rows, cols}, data);
    images.push_back(image);
  }

  return images;
}

std::vector<Tensor> loadLabelsAsTensors(const std::string &filename, int max_labels = 9999999)
{
  std::ifstream file(filename, std::ios::binary);
  if (!file)
    throw std::runtime_error("No se pudo abrir el archivo de etiquetas");

  int32_t magic = 0, num_labels = 0;
  file.read(reinterpret_cast<char *>(&magic), 4);
  file.read(reinterpret_cast<char *>(&num_labels), 4);

  magic = __builtin_bswap32(magic);
  num_labels = __builtin_bswap32(num_labels);

  if (max_labels > 0 && max_labels < num_labels)
    num_labels = max_labels;

  std::vector<Tensor> labels;
  labels.reserve(num_labels);

  for (int i = 0; i < num_labels; ++i)
  {
    unsigned char label = 0;
    file.read(reinterpret_cast<char *>(&label), 1);
    // Creamos un tensor escalar con el label
    labels.emplace_back(std::vector<int>{1}, std::vector<float>{static_cast<float>(label)});
  }

  return labels;
}

Dataset load_dataset(const std::string &image_path, const std::string &label_path, int max_samples = 9999999)
{
  std::vector<Tensor> images = loadImages2D(image_path, max_samples);
  std::vector<Tensor> labels = loadLabelsAsTensors(label_path, max_samples);

  if (images.size() != labels.size())
    throw std::runtime_error("El número de imágenes y etiquetas no coincide");

  return {images, labels};
}

class DataLoader
{
private:
  const std::vector<Tensor> &images;
  const std::vector<Tensor> &labels;
  size_t batch_size;
  size_t index;
  std::vector<size_t> indices;
  std::mt19937 rng;
  // ...

public:
  DataLoader(const std::vector<Tensor> &imgs, const std::vector<Tensor> &lbls, size_t b_size, unsigned int seed = 42)
      : images(imgs), labels(lbls), batch_size(b_size), index(0), rng(seed)
  {
    if (images.size() != labels.size())
      throw std::runtime_error("Imágenes y etiquetas no coinciden en tamaño");

    indices.resize(images.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);
  }

  std::pair<Tensor, Tensor> next_batch()
  {
    size_t end = std::min(index + batch_size, images.size());
    size_t actual_batch_size = end - index;

    const int C = images[0].shape[0];
    const int H = images[0].shape[1];
    const int W = images[0].shape[2];

    std::vector<float> batch_data(actual_batch_size * C * H * W);
    std::vector<float> batch_labels(actual_batch_size);

    for (size_t i = 0; i < actual_batch_size; ++i)
    {
      size_t idx = indices[index + i];
      const Tensor &img = images[idx];
      const Tensor &lbl = labels[idx];

      std::copy(img.data.begin(), img.data.end(),
                batch_data.begin() + i * C * H * W);

      batch_labels[i] = lbl.data[0]; // suponemos tensor {1}
    }

    Tensor batch_tensor({(int)actual_batch_size, C, H, W}, batch_data);
    Tensor label_tensor({(int)actual_batch_size, 1}, batch_labels);

    index = end;
    return {batch_tensor, label_tensor};
  }

  bool has_next() const
  {
    return index < images.size();
  }

  void reset()
  {
    index = 0;
    std::shuffle(indices.begin(), indices.end(), rng);
  }
  size_t total_batches() const
  {
    return (images.size() + batch_size - 1) / batch_size;
  }
};
