#pragma once
#include "tensor.hpp"
#include "load_dataset.hpp"
#include "../model/model.hpp"
#include <iostream>
#include <iomanip>

class Trainer
{
private:
  Model &model;
  DataLoader &train_loader;
  DataLoader &val_loader;
  int num_classes;
  int batch_size;

public:
  Trainer(Model &model, DataLoader &train_loader, DataLoader &val_loader,
          int num_classes, int batch_size)
      : model(model), train_loader(train_loader), val_loader(val_loader),
        num_classes(num_classes), batch_size(batch_size) {}

  float compute_accuracy(const Tensor &preds, const Tensor &targets)
  {
    int correct = 0;
    int total = preds.shape[0];

    for (int i = 0; i < total; ++i)
    {
      int pred_label = std::distance(preds.data.begin() + i * num_classes,
                                     std::max_element(preds.data.begin() + i * num_classes,
                                                      preds.data.begin() + (i + 1) * num_classes));
      int true_label = static_cast<int>(targets.data[i]);
      if (pred_label == true_label)
        ++correct;
    }

    return static_cast<float>(correct) / total;
  }

  float compute_loss(const Tensor &preds, const Tensor &targets_onehot)
  {
    // Cross entropy: -sum(y_true * log(pred)) / B
    float loss = 0.0f;
    int B = preds.shape[0];
    int C = preds.shape[1];

    for (int i = 0; i < B; ++i)
    {
      for (int j = 0; j < C; ++j)
      {
        float y = targets_onehot.data[i * C + j];
        float p = preds.data[i * C + j];
        loss += -y * std::log(p + 1e-8f);
      }
    }

    return loss / B;
  }
#include <chrono>
#include <iomanip>

  void train(int epochs)
  {
    for (int epoch = 0; epoch < epochs; ++epoch)
    {
      // Registrar tiempo de inicio
      auto start_time = std::chrono::high_resolution_clock::now();

      std::cout << "Epoch " << (epoch + 1) << "/" << epochs << "\n";
      train_loader.reset();

      float epoch_loss = 0.0f;
      float epoch_acc = 0.0f;
      int batches = 0;

      while (train_loader.has_next())
      {
        auto [X, y_tensor] = train_loader.next_batch();           // y_tensor: shape [B, 1]
        Tensor y_onehot = Tensor::one_hot(y_tensor, num_classes); // shape [B, C]

        Tensor preds = model.forward(X);  // preds: [B, C]
        model.backward(y_onehot);         // backprop
        model.update_weights(batch_size); // SGD update
        model.zero_grad();                // reset gradients

        float batch_loss = compute_loss(preds, y_onehot);
        float batch_acc = compute_accuracy(preds, y_tensor);

        epoch_loss += batch_loss;
        epoch_acc += batch_acc;
        ++batches;
      }

      // Calcular tiempo transcurrido
      auto end_time = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

      // Mostrar resultados
      std::cout << std::fixed << std::setprecision(4);
      std::cout << "Train Loss: " << (epoch_loss / batches)
                << " | Accuracy: " << (epoch_acc / batches) * 100.0f << "%"
                << " | Time: " << duration.count() << " ms\n";

      evaluate();
    }
  }
  void evaluate()
  {
    val_loader.reset();
    float val_loss = 0.0f;
    float val_acc = 0.0f;
    int batches = 0;

    model.set_training(false);

    while (val_loader.has_next())
    {
      auto [X, y_tensor] = val_loader.next_batch();
      Tensor y_onehot = Tensor::one_hot(y_tensor, num_classes);

      Tensor preds = model.forward(X);

      val_loss += compute_loss(preds, y_onehot);
      val_acc += compute_accuracy(preds, y_tensor);
      ++batches;
    }

    std::cout << "Val Loss: " << (val_loss / batches)
              << " | Accuracy: " << (val_acc / batches) * 100.0f << "%\n";
    model.set_training(true);
  }
};
