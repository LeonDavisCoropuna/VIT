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

  // MLP vt_cnn;
  int batch_size = 128;
  int num_classes = 10;
  int max_samples = 5000;
  std::string path = "/home/leon/Documentos/UNSA/TOPICOS IA/VIT/fashion_data/";
  std::string device = "cpu";
  Dataset dataset_train = load_dataset(path + "train-images-idx3-ubyte",
                                       path + "train-labels-idx1-ubyte",
                                       max_samples);

  Dataset dataset_val = load_dataset(path + "t10k-images-idx3-ubyte",
                                     path + "t10k-labels-idx1-ubyte",
                                     1000);

  DataLoader train_loader(dataset_train.images, dataset_train.labels, batch_size);
  DataLoader val_loader(dataset_val.images, dataset_val.labels, batch_size);

  Trainer trainer(vt_cnn, train_loader, val_loader, num_classes, batch_size);

  trainer.train(/*epochs=*/10, /*log_every=*/5, device);
  return 0;
}