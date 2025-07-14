#include "utils/load_dataset.hpp"
#include "utils/trainer.hpp"
#include "model/cnn.hpp"
#include "model/mlp.hpp"
#include "model/vit_cnn.hpp"

std::mt19937 Tensor::global_gen(42); // semilla fija por defecto

int main()
{
  VTCNN vt_cnn(
      /* image_channels        */ 1,   // MNIST tiene 1 canal
      /* input_dim             */ 28,  // Ancho/alto de entrada (28x28)
      /* vt_layers_num         */ 1,   // Número de capas VisualTransformer (puedes variar)
      /* tokens                */ 16,  // Número de tokens visuales
      /* token_channels        */ 32,  // Dimensión de cada token
      /* vt_channels           */ 16,  // Salida de cada capa VT (C)
      /* transformer_enc_layers*/ 2,   // Capas encoder del transformer interno
      /* transformer_heads     */ 2,   // Número de cabezas de atención
      /* transformer_fc_dim    */ 64,  // Dimensión FC del Transformer
      /* transformer_dropout   */ 0.1, // Dropout en Transformer
      /* num_classes           */ 10   // Para clasificación binaria (sandal vs sneaker)
  );
  MLP mlp;
  // MLP vt_cnn;
  int batch_size = 32;
  int num_classes = 10;
  int max_samples = 5000;
  std::string path = "fashion_data/";
  std::string device = "cpu";
  // 📦 Cargar todo el dataset de entrenamiento
  Dataset full_train = load_dataset(path + "train-images-idx3-ubyte",
                                    path + "train-labels-idx1-ubyte",
                                    max_samples);

  // 📦 Cargar el dataset de test (t10k)
  Dataset dataset_test = load_dataset(path + "t10k-images-idx3-ubyte",
                                      path + "t10k-labels-idx1-ubyte",
                                      1000);

  // 📤 Dividir el dataset de entrenamiento en 80% train, 20% val
  size_t total = full_train.images.size();
  size_t train_count = static_cast<size_t>(total * 0.8);
  size_t val_count = total - train_count;

  Dataset train_dataset, val_dataset;
  train_dataset.images.assign(full_train.images.begin(), full_train.images.begin() + train_count);
  train_dataset.labels.assign(full_train.labels.begin(), full_train.labels.begin() + train_count);

  val_dataset.images.assign(full_train.images.begin() + train_count, full_train.images.end());
  val_dataset.labels.assign(full_train.labels.begin() + train_count, full_train.labels.end());

  // 📊 Imprimir cantidades
  std::cout << "📊 Datos cargados:\n";
  std::cout << "  - Train: " << train_dataset.images.size() << " muestras\n";
  std::cout << "  - Validation: " << val_dataset.images.size() << " muestras\n";
  std::cout << "  - Test: " << dataset_test.images.size() << " muestras\n";

  // 🔁 DataLoaders
  DataLoader train_loader(train_dataset.images, train_dataset.labels, batch_size);
  DataLoader val_loader(val_dataset.images, val_dataset.labels, batch_size);
  DataLoader test_loader(dataset_test.images, dataset_test.labels, batch_size);

  // Trainer trainer(vt_cnn, train_loader, val_loader, num_classes, batch_size);
  Trainer trainer(mlp, train_loader, val_loader, num_classes, batch_size);

  trainer.train(/*epochs=*/20, /*log_every=*/100, device);
  return 0;
}