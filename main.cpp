#include "utils/load_dataset.hpp"
#include "utils/trainer.hpp"
#include "model/cnn.hpp"
#include "model/mlp.hpp"
#include "model/vit_cnn.hpp"

std::mt19937 Tensor::global_gen(42); // semilla fija por defecto

int main()
{
  MLP vt_cnn;
  int batch_size = 128;
  int num_classes = 10;
  std::string path = "fashion_data/";
  std::string device = "cpu";
  Dataset dataset_train = load_dataset(path + "train-images-idx3-ubyte",
                                       path + "train-labels-idx1-ubyte");

  Dataset dataset_val = load_dataset(path + "t10k-images-idx3-ubyte",
                                     path + "t10k-labels-idx1-ubyte");

  DataLoader train_loader(dataset_train.images, dataset_train.labels, batch_size);
  DataLoader val_loader(dataset_val.images, dataset_val.labels, batch_size);

  Trainer trainer(vt_cnn, train_loader, val_loader, num_classes, batch_size);

  trainer.train(/*epochs=*/50, /*log_every=*/100, device);
  return 0;
}