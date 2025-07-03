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
  std::string path = "/home/leon/Documentos/UNSA/TOPICOS IA/MLP-Multi-Layer-Perceptron/mnist_data/";

  Dataset dataset_train = load_dataset(path + "train-images.idx3-ubyte",
                                       path + "train-labels.idx1-ubyte",
                                       max_samples);

  Dataset dataset_val = load_dataset(path + "t10k-images.idx3-ubyte",
                                     path + "t10k-labels.idx1-ubyte",
                                     1000);

  DataLoader train_loader(dataset_train.images, dataset_train.labels, batch_size);
  DataLoader val_loader(dataset_val.images, dataset_val.labels, batch_size);

  Trainer trainer(vt_cnn, train_loader, val_loader, num_classes, batch_size);

  trainer.train(/*epochs=*/10, /*log_every=*/5);

  // Tensor t({3, 3});
  // t.data = {1, 2, 3, 4, 5, 6, 7, 8, 9}; // Matriz:
  //                                       // | 0 1 2 |
  //                                       // | 3 4 5 |

  // auto t1 = t.transpose({1, 0}); // General
  // auto t2 = t.transpose(0, 1);   // Optimizada 2D

  // std::cout << "Original: " << std::endl;

  // for (int i = 0; i < t.shape[0]; ++i)
  // {
  //   for (int j = 0; j < t.shape[1]; ++j)
  //   {
  //     std::cout << t.at({i, j}) << " ";
  //   }
  //   std::cout << "\n";
  // }
  // std::cout << "T1: " << std::endl;

  // for (int i = 0; i < t1.shape[0]; ++i)
  // {
  //   for (int j = 0; j < t1.shape[1]; ++j)
  //   {
  //     std::cout << t1.at({i, j}) << " ";
  //   }
  //   std::cout << "\n";
  // }
  // std::cout << "T2: " << std::endl;

  // for (int i = 0; i < t2.shape[0]; ++i)
  // {
  //   for (int j = 0; j < t2.shape[1]; ++j)
  //   {
  //     std::cout << t2.at({i, j}) << " ";
  //   }
  //   std::cout << "\n";
  // }
  return 0;
}
