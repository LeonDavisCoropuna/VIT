#include "utils/load_dataset.hpp"
#include "utils/trainer.hpp"
#include "model/cnn.hpp"
#include "model/mlp.hpp"
#include "model/vit_cnn.hpp"
#include "utils/load_image_stb.hpp"

std::mt19937 Tensor::global_gen(42); // semilla fija por defecto

int main()
{
  // Model parameters
  int image_channels = 1;
  int input_dim = 28;
  int vt_layers_num = 1;
  int tokens = 16;
  int token_channels = 32;
  int vt_channels = 16;
  int transformer_enc_layers = 2;
  int transformer_heads = 2;
  int transformer_fc_dim = 64;
  float transformer_dropout = 0.1;
  int num_classes = 10;
  int batch_size = 32;
  int max_samples = 60000;
  std::string path = "mnist_data/";
  std::string device = "cpu";

  // Load datasets
  Dataset full_train = load_dataset(
      path + "train-images.idx3-ubyte",
      path + "train-labels.idx1-ubyte",
      max_samples);
  Dataset dataset_test = load_dataset(
      path + "t10k-images.idx3-ubyte",
      path + "t10k-labels.idx1-ubyte",
      10000);

  // Split train/validation
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

  // DataLoaders
  DataLoader train_loader(train_dataset.images, train_dataset.labels, batch_size);
  DataLoader val_loader(val_dataset.images, val_dataset.labels, batch_size);
  DataLoader test_loader(dataset_test.images, dataset_test.labels, batch_size);

  // Initialize and train original model
  VTCNN vt_cnn(
      image_channels,
      input_dim,
      vt_layers_num,
      tokens,
      token_channels,
      vt_channels,
      transformer_enc_layers,
      transformer_heads,
      transformer_fc_dim,
      transformer_dropout,
      num_classes);
  Trainer trainer(vt_cnn, train_loader, val_loader, num_classes, batch_size, &test_loader);
  trainer.train(20, 100, device);
  vt_cnn.save("vt_cnn_model.bin");

  // Initialize model for loading
  VTCNN vt_cnn2(
      image_channels,
      input_dim,
      vt_layers_num,
      tokens,
      token_channels,
      vt_channels,
      transformer_enc_layers,
      transformer_heads,
      transformer_fc_dim,
      transformer_dropout,
      num_classes);

  // Load saved model
  try
  {
    vt_cnn2.load("vt_cnn_model.bin");
    std::cout << "Modelo cargado exitosamente desde 'vt_cnn_model.bin'\n";
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error al cargar el modelo: " << e.what() << "\n";
    return 1;
  }
  vt_cnn2.set_training(false); // Añade esta línea
  // Evaluate loaded model
  std::cout << "\n=== Verificación del modelo cargado ===\n";
  Trainer test_trainer(vt_cnn2, train_loader, val_loader, num_classes, batch_size, &test_loader);
  test_trainer.evaluate_test();

  // mlp.save("mlp_model.bin");
  /*
   // Recargar en nuevo modelo
   MLP mlp_loaded;
   mlp_loaded.load("mlp_model.bin");

   // Crear nuevo trainer con el modelo cargado
   Trainer test_trainer(mlp_loaded, train_loader, val_loader, num_classes, batch_size, &test_loader);

   std::cout << "\n=== Verificación del modelo cargado ===\n";
   //test_trainer.evaluate_test(); // ✅ usa el método que ya implementaste
*/
  // std::cout << "\n🔍 Cargando imágenes personalizadas desde 'custom_images/'...\n";
  Dataset custom_data = load_custom_images_from_folder("custom_images/");

  for (size_t i = 0; i < custom_data.images.size(); ++i)
  {
    int pred = test_trainer.predict(custom_data.images[i]);
    std::cout << "Imagen predicción: " << pred << "\n";
  }
  return 0;
}