#pragma once
#include "tensor.hpp"
#include "load_dataset.hpp"
#include "../model/model.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>

class Trainer
{
private:
  Model &model;
  DataLoader &train_loader;
  DataLoader &val_loader;
  DataLoader *test_loader; // puntero para que sea opcional
  int num_classes;
  int batch_size;

public:
  Trainer(Model &model, DataLoader &train_loader, DataLoader &val_loader,
          int num_classes, int batch_size, DataLoader *test_loader = nullptr)
      : model(model), train_loader(train_loader), val_loader(val_loader),
        test_loader(test_loader), num_classes(num_classes), batch_size(batch_size) {}

  void train(int epochs, int log_every, std::string device)
  {
    for (int epoch = 0; epoch < epochs; ++epoch)
    {
      auto start_time = std::chrono::high_resolution_clock::now();
      std::cout << "Epoch " << (epoch + 1) << "/" << epochs << "\n";
      train_loader.reset();

      float epoch_loss = 0.0f;
      float epoch_acc = 0.0f;
      int batches = 0;

      int total_batches = train_loader.total_batches();

      while (train_loader.has_next())
      {
        auto [X, y_tensor] = train_loader.next_batch(); // y_tensor: shape [B, 1]
        if (device == "cuda")
        {
          X = X.to_device(true);
          y_tensor = y_tensor.to_device(true);
        }
        Tensor y_onehot = Tensor::one_hot(y_tensor, num_classes); // shape [B, C]

        Tensor preds = model.forward(X); // preds: [B, C]
        // print_tensor(preds, "preds");
        // print_tensor(y_onehot, "y_onehot");
        model.backward(y_onehot);         // backprop
        model.update_weights(batch_size); // SGD update
        model.zero_grad();                // reset gradients

        float batch_loss = Tensor::compute_loss(preds, y_onehot);
        float batch_acc = Tensor::compute_accuracy(preds, y_tensor, num_classes);

        epoch_loss += batch_loss;
        epoch_acc += batch_acc;
        ++batches;

        // Solo imprime si toca
        if (log_every > 0 && batches % log_every == 0)
        {
          std::cout << "  Batch " << batches << "/" << total_batches
                    << " | Loss: " << std::fixed << std::setprecision(4) << batch_loss
                    << " | Accuracy: " << std::setprecision(4) << batch_acc * 100.0f << "%\n";
        }
      }

      auto end_time = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> duration = end_time - start_time;

      std::cout << std::fixed << std::setprecision(4);
      std::cout << "Train Loss: " << (epoch_loss / batches)
                << " | Accuracy: " << (epoch_acc / batches) * 100.0f << "%"
                << " | Time: " << duration.count() << " s\n";

      evaluate();
    }

    // Evaluación final con test_loader (si se proporcionó)
    if (test_loader)
    {
      std::cout << "\n=== Evaluación final sobre conjunto de test ===\n";
      evaluate_test(); // ✅ solo se ejecuta una vez al final
    }
  }

  void evaluate_test()
  {
    if (!test_loader)
    {
      std::cout << "[WARN] No se proporcionó test_loader\n";
      return;
    }

    test_loader->reset();
    float test_loss = 0.0f;
    float test_acc = 0.0f;
    int batches = 0;

    model.set_training(false);
    while (test_loader->has_next())
    {
      auto [X, y_tensor] = test_loader->next_batch();
      Tensor y_onehot = Tensor::one_hot(y_tensor, num_classes);

      Tensor preds = model.forward(X);

      test_loss += Tensor::compute_loss(preds, y_onehot);
      test_acc += Tensor::compute_accuracy(preds, y_tensor, num_classes);
      ++batches;
    }

    std::cout << "[TEST] Final Evaluation — Loss: " << (test_loss / batches)
              << " | Accuracy: " << (test_acc / batches) * 100.0f << "%\n";
    model.set_training(true);
  }

  void evaluate()
  {
    val_loader.reset();
    float val_loss = 0.0f;
    float val_acc = 0.0f;
    int batches = 0;

    model.set_training(false);
    int total_batches = val_loader.total_batches();
    while (val_loader.has_next())
    {
      // std::cout << "🔍 [EVAL] Cargando batch " << (batches + 1) << "/" << total_batches << "...\n";
      auto [X, y_tensor] = val_loader.next_batch();
      Tensor y_onehot = Tensor::one_hot(y_tensor, num_classes);

      Tensor preds = model.forward(X);

      val_loss += Tensor::compute_loss(preds, y_onehot);
      val_acc += Tensor::compute_accuracy(preds, y_tensor, num_classes);
      ++batches;
    }

    std::cout << "Val Loss: " << (val_loss / batches)
              << " | Accuracy: " << (val_acc / batches) * 100.0f << "%\n";
    model.set_training(true);
  }
};
