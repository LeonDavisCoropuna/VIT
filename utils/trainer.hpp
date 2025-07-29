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
          // std::cout << "  Batch " << batches << "/" << total_batches
          //           << " | Loss: " << std::fixed << std::setprecision(4) << batch_loss
          //           << " | Accuracy: " << std::setprecision(4) << batch_acc * 100.0f << "%\n";
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

    std::vector<int> true_labels;
    std::vector<int> pred_labels;

    model.set_training(false);

    while (test_loader->has_next())
    {
      auto [X, y_tensor] = test_loader->next_batch();
      Tensor y_onehot = Tensor::one_hot(y_tensor, num_classes);

      Tensor preds = model.forward(X);

      test_loss += Tensor::compute_loss(preds, y_onehot);
      test_acc += Tensor::compute_accuracy(preds, y_tensor, num_classes);
      ++batches;

      // Recolectar predicciones y reales
      std::vector<int> y_pred = Tensor::argmax_batch(preds); // vector<int>
      std::vector<int> y_true = y_tensor.to_vector();        // vector<int>

      true_labels.insert(true_labels.end(), y_true.begin(), y_true.end());
      pred_labels.insert(pred_labels.end(), y_pred.begin(), y_pred.end());
    }

    std::cout << "[TEST] Final Evaluation — Loss: " << (test_loss / batches)
              << " | Accuracy: " << (test_acc / batches) * 100.0f << "%\n";

    // Compute metrics
    auto [f1_macro, f1_weighted, f1_scores] = Tensor::f1_score(true_labels, pred_labels, num_classes);
    auto [precision_macro, recall_macro, precisions, recalls] = Tensor::precision_recall(true_labels, pred_labels, num_classes);
    auto confusion = Tensor::confusion_matrix(true_labels, pred_labels, num_classes);

    // Print aggregate metrics
    std::cout << " - F1 Score (macro):    " << f1_macro << "\n";
    std::cout << " - F1 Score (weighted): " << f1_weighted << "\n";
    std::cout << " - Precision (macro):   " << precision_macro << "\n";
    std::cout << " - Recall (macro):      " << recall_macro << "\n";

    // Print per-class metrics
    std::cout << "\n📊 Per-Class Metrics:\n";
    std::cout << std::setw(8) << "Class" << std::setw(12) << "Precision" << std::setw(12) << "Recall" << std::setw(12) << "F1 Score" << "\n";
    std::cout << std::string(44, '-') << "\n";
    for (int i = 0; i < num_classes; ++i)
    {
      std::cout << std::setw(8) << i
                << std::fixed << std::setprecision(4)
                << std::setw(12) << precisions[i]
                << std::setw(12) << recalls[i]
                << std::setw(12) << f1_scores[i] << "\n";
    }

    // Print confusion matrix
    std::cout << "\n🧩 Matriz de Confusión:\n    ";
    for (int j = 0; j < num_classes; ++j)
      std::cout << std::setw(4) << j;
    std::cout << "\n";

    for (int i = 0; i < num_classes; ++i)
    {
      std::cout << std::setw(2) << i << ": ";
      for (int j = 0; j < num_classes; ++j)
        std::cout << std::setw(4) << confusion[i][j];
      std::cout << "\n";
    }

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

  int predict(const std::vector<float> &image_data)
  {
    if (image_data.size() != 28 * 28)
      throw std::runtime_error("La imagen debe ser de tamaño 28x28");

    Tensor input({1, 1, 28, 28}, image_data); // Crea un tensor con batch=1
    model.set_training(false);                // Desactiva dropout
    Tensor out = model.forward(input);        // Ejecuta forward

    out.printsummary();

    // Asumimos que out tiene forma [1, 10], queremos argmax sobre ese vector
    float max_val = out.data[0];
    int max_idx = 0;
    for (int i = 1; i < out.data.size(); ++i)
    {
      if (out.data[i] > max_val)
      {
        max_val = out.data[i];
        max_idx = i;
      }
    }

    return max_idx;
  }

  static float compute_accuracy(const Tensor &preds, const Tensor &targets, int num_classes)
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
};
