#include "utils/load_dataset.hpp"
#include "utils/trainer.hpp"
#include "model/cnn.hpp"
#include "model/mlp.hpp"
#include "model/vit_cnn.hpp"
#include "utils/load_image_stb.hpp"

#include <iostream>
#include <string>
#include <algorithm>

std::mt19937 Tensor::global_gen(42);

enum DatasetType
{
    MNIST,
    BLOOD_MNIST,
    FASHION_MNIST,
    UNKNOWN
};

DatasetType parse_dataset_type(const std::string &name)
{
    if (name == "mnist")
        return MNIST;
    if (name == "bloodmnist")
        return BLOOD_MNIST;
    if (name == "fashionmnist")
        return FASHION_MNIST;
    return UNKNOWN;
}

std::string get_dataset_path(DatasetType type)
{
    switch (type)
    {
    case MNIST:
        return "mnist_data/";
    case BLOOD_MNIST:
        return "blood_data/";
    case FASHION_MNIST:
        return "fashion_data/";
    default:
        throw std::runtime_error("Dataset desconocido.");
    }
}

std::string get_model_filename(DatasetType type)
{
    switch (type)
    {
    case MNIST:
        return "model_mnist.bin";
    case BLOOD_MNIST:
        return "model_bloodmnist.bin";
    case FASHION_MNIST:
        return "model_fashionmnist.bin";
    default:
        throw std::runtime_error("Dataset desconocido.");
    }
}

void run_pipeline(DatasetType dataset_type, bool train_mode, const std::string &device, int epochs)
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

    // Load datasets
    std::string path = get_dataset_path(dataset_type);
    std::string model_file = get_model_filename(dataset_type);
    /*
        Dataset train_dataset, val_dataset, test_set;

        // Check if separate validation files exist for BLOOD_MNIST
        bool use_separate_val = (dataset_type == BLOOD_MNIST) &&
                                std::filesystem::exists(path + "val-images-idx3-ubyte") &&
                                std::filesystem::exists(path + "val-labels-idx1-ubyte");

        if (use_separate_val)
        {
            // Load separate train, val, and test sets for BLOOD_MNIST
            std::cout << "Loading separate validation set for BLOOD_MNIST\n";
            train_dataset = load_dataset(
                path + "train-images-idx3-ubyte",
                path + "train-labels-idx1-ubyte",
                max_samples);
            val_dataset = load_dataset(
                path + "val-images-idx3-ubyte",
                path + "val-labels-idx1-ubyte",
                max_samples);
            test_set = load_dataset(
                path + "t10k-images-idx3-ubyte",
                path + "t10k-labels-idx1-ubyte",
                1000);
        }
        else
        {
            // Load train and split into train/val (for MNIST, FASHION_MNIST, or if BLOOD_MNIST lacks val files)
            Dataset full_train = load_dataset(
                path + "train-images-idx3-ubyte",
                path + "train-labels-idx1-ubyte",
                max_samples);
            test_set = load_dataset(
                path + "t10k-images-idx3-ubyte",
                path + "t10k-labels-idx1-ubyte",
                1000);

            size_t total = full_train.images.size();
            size_t train_count = static_cast<size_t>(total * 0.8);
            size_t val_count = total - train_count;

            train_dataset.images.assign(full_train.images.begin(), full_train.images.begin() + train_count);
            train_dataset.labels.assign(full_train.labels.begin(), full_train.labels.begin() + train_count);
            val_dataset.images.assign(full_train.images.begin() + train_count, full_train.images.end());
            val_dataset.labels.assign(full_train.labels.begin() + train_count, full_train.labels.end());
        }*/

    Dataset full_train = load_dataset(
        path + "train-images-idx3-ubyte",
        path + "train-labels-idx1-ubyte",
        max_samples);
    Dataset test_set = load_dataset(
        path + "t10k-images-idx3-ubyte",
        path + "t10k-labels-idx1-ubyte",
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

    // Print dataset sizes
    std::cout << "📊 Datos cargados:\n";
    std::cout << "  - Train: " << train_dataset.images.size() << " muestras\n";
    std::cout << "  - Validation: " << val_dataset.images.size() << " muestras\n";
    std::cout << "  - Test: " << test_set.images.size() << " muestras\n";

    // DataLoaders
    DataLoader train_loader(train_dataset.images, train_dataset.labels, batch_size);
    DataLoader val_loader(val_dataset.images, val_dataset.labels, batch_size);
    DataLoader test_loader(test_set.images, test_set.labels, batch_size);

    // Initialize model
    VTCNN model(
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

    if (train_mode)
    {
        // Train model
        Trainer trainer(model, train_loader, val_loader, num_classes, batch_size, &test_loader);
        std::cout << "🚀 Entrenando modelo por " << epochs << " épocas...\n";
        trainer.train(epochs, 100, device);
        model.save(model_file);
        std::cout << "✅ Modelo guardado como: " << model_file << "\n";
    }
    else
    {
        // Load and evaluate model
        try
        {
            model.load(model_file);
            model.set_training(false);
            std::cout << "✅ Modelo cargado desde: " << model_file << "\n";
        }
        catch (const std::exception &e)
        {
            std::cerr << "❌ Error al cargar modelo: " << e.what() << "\n";
            return;
        }

        Trainer trainer(model, train_loader, val_loader, num_classes, batch_size, &test_loader);
        std::cout << "\n=== Evaluación del modelo cargado ===\n";
        trainer.evaluate_test();

        // Predict on custom images
        std::cout << "\n🔍 Cargando imágenes personalizadas desde 'custom_images/'...\n";
        Dataset custom_data = load_custom_images_from_folder("custom_images/");
        for (size_t i = 0; i < custom_data.images.size(); ++i)
        {
            if (custom_data.images[i].size() != 28 * 28)
            {
                std::cerr << "Error: Imagen personalizada #" << i << " tiene tamaño incorrecto.\n";
                continue;
            }
            int pred = trainer.predict(custom_data.images[i]); // Pass std::vector<float> directly
            std::cout << "Imagen predicción: " << pred << "\n";
        }
    }
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cerr << "Uso: ./main <mnist|bloodmnist|fashionmnist> <train|predict> [epocas]\n";
        return 1;
    }

    std::string dataset_str = argv[1];
    std::string mode_str = argv[2];

    std::transform(dataset_str.begin(), dataset_str.end(), dataset_str.begin(), ::tolower);
    std::transform(mode_str.begin(), mode_str.end(), mode_str.begin(), ::tolower);

    DatasetType dataset = parse_dataset_type(dataset_str);
    if (dataset == UNKNOWN)
    {
        std::cerr << "❌ Dataset inválido: " << dataset_str << "\n";
        return 1;
    }

    bool train_mode;
    if (mode_str == "train")
    {
        train_mode = true;
    }
    else if (mode_str == "predict")
    {
        train_mode = false;
    }
    else
    {
        std::cerr << "❌ Modo inválido (usa 'train' o 'predict'): " << mode_str << "\n";
        return 1;
    }

    int epochs = 5; // Default
    if (train_mode && argc >= 4)
    {
        try
        {
            epochs = std::stoi(argv[3]);
            if (epochs <= 0)
            {
                std::cerr << "❌ Número de épocas inválido.\n";
                return 1;
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << "❌ Error al parsear épocas: " << e.what() << "\n";
            return 1;
        }
    }

    std::string device = "cpu"; // or "cuda" if using GPU
    run_pipeline(dataset, train_mode, device, epochs);
    return 0;
}