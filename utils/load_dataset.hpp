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
  std::vector<std::vector<float>> images; // CPU-only
  std::vector<int> labels;                // CPU-only
};

namespace fs = std::filesystem;

std::vector<std::vector<float>> loadImages2D(const std::string &filename, int max_images = 9999999)
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
    num = max_images;

  std::vector<std::vector<float>> images;
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
    images.push_back(std::move(data));
  }

  return images;
}

std::vector<int> loadLabels(const std::string &filename, int max_labels = 9999999)
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

  std::vector<int> labels;
  labels.reserve(num_labels);

  for (int i = 0; i < num_labels; ++i)
  {
    unsigned char label = 0;
    file.read(reinterpret_cast<char *>(&label), 1);
    labels.push_back(static_cast<int>(label));
  }

  return labels;
}

Dataset load_dataset(const std::string &image_path, const std::string &label_path, int max_samples = 9999999)
{
  std::vector<std::vector<float>> images = loadImages2D(image_path, max_samples);
  std::vector<int> labels = loadLabels(label_path, max_samples);

  if (images.size() != labels.size())
    throw std::runtime_error("El número de imágenes y etiquetas no coincide");

  return {images, labels};
}

class DataLoader
{
private:
  const std::vector<std::vector<float>> &images; // CPU
  const std::vector<int> &labels;                // CPU
  size_t batch_size;
  size_t index;
  std::vector<size_t> indices;
  std::mt19937 rng;

  int C, H, W, num_classes;

public:
  DataLoader(const std::vector<std::vector<float>> &imgs,
             const std::vector<int> &lbls,
             size_t b_size,
             int channels = 1, int height = 28, int width = 28,
             int num_cls = 10,
             unsigned int seed = 42)
      : images(imgs), labels(lbls), batch_size(b_size),
        index(0), C(channels), H(height), W(width), num_classes(num_cls), rng(seed)
  {
    if (images.size() != labels.size())
      throw std::runtime_error("Mismatch entre imágenes y etiquetas");

    indices.resize(images.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);
  }

  std::pair<Tensor, Tensor> next_batch()
  {
    size_t end = std::min(index + batch_size, images.size());
    size_t actual_batch_size = end - index;

    std::vector<float> batch_data(actual_batch_size * C * H * W);
    std::vector<float> batch_labels(actual_batch_size);

    for (size_t i = 0; i < actual_batch_size; ++i)
    {
      size_t idx = indices[index + i];
      std::copy(images[idx].begin(), images[idx].end(),
                batch_data.begin() + i * C * H * W);
      batch_labels[i] = static_cast<float>(labels[idx]);
    }

    index = end;

    // ⬇️ Estos son tensores en GPU
    Tensor X({(int)actual_batch_size, C, H, W}, batch_data); // copia a GPU
    Tensor y({(int)actual_batch_size, 1}, batch_labels);     // copia a GPU
    return {X, y};
  }

  bool has_next() const { return index < images.size(); }

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
