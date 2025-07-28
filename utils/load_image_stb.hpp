#pragma once
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <string>
#include <vector>
#include <stdexcept>

std::vector<float> load_image_as_vector(const std::string &path, int target_w = 28, int target_h = 28)
{
    int width, height, channels;
    unsigned char *data = stbi_load(path.c_str(), &width, &height, &channels, 1); // fuerza 1 canal (grayscale)

    if (!data)
        throw std::runtime_error("No se pudo cargar la imagen: " + path);

    if (width != target_w || height != target_h)
        throw std::runtime_error("La imagen debe ser de tamaño 28x28");

    std::vector<float> image(target_w * target_h);
    for (int i = 0; i < target_w * target_h; ++i)
        image[i] = static_cast<float>(data[i]) / 255.0f;

    stbi_image_free(data);
    return image;
}
