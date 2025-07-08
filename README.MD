
# VTCNN: Visual Transformer CNN

Este proyecto implementa una arquitectura híbrida que combina **Convoluciones (CNN)** con **Transformers** para el procesamiento de imágenes, específicamente sobre el dataset **Fashion-MNIST**. La arquitectura principal se denomina `VTCNN` y se compone de:

- Bloque CNN inicial para extracción de características espaciales locales.
- Una o más capas Visual Transformer (`VisualTransformer`) que operan sobre embeddings.
- Capas densas para clasificación.

---

## 🧠 Arquitectura General

```

Input Image
↓
Conv2DLayer (conv\_layer)
↓
BatchNorm2DLayer (bn)
↓
\[reshape + transpose]
↓
VisualTransformer (1 o más capas)
↓
reshape → CNN format
↓
MaxPool2D
↓
Flatten
↓
Dense
↓
Softmax

```

---

## 🔍 Descripción de Componentes

### 🔹 CNN (Embedding Espacial)

La imagen de entrada es pasada por:

1. **Conv2DLayer**: extrae patrones locales (filtros 3x3 con padding).
2. **BatchNorm2DLayer**: estabiliza y normaliza la salida.
3. **Flatten + Transpose**: convierte el mapa 4D `[N, C, H, W]` a una secuencia `[N, HW, C]` para poder alimentar al Transformer.

Este proceso actúa como un **encoder convolucional** que genera un embedding espacial por píxel (o patch), antes de pasar a la parte atencional.

---

### 🔸 VisualTransformer

Este bloque introduce atención basada en **transformers**, pero con entradas visuales.

**Componentes internos:**

- `FilterTokenizer`: genera una secuencia de **tokens** desde el mapa de características de la imagen.
- `TransformerLayer`: aplica atención multi-cabeza sobre estos tokens, modelando relaciones a largo alcance.
- `ProjectorLayer`: fusiona los tokens y el mapa original si `is_projected = true`.

El flujo en un `VisualTransformer` es:

```

[feature map] → Tokenizer → Transformer → [opcional] Projector → salida

````

Múltiples bloques `VisualTransformer` pueden apilarse. El primer bloque recibe únicamente el mapa convolucional. Los siguientes bloques reciben tanto el mapa como los tokens anteriores.

---

### 🔹 MaxPool + Dense

Luego del procesamiento por Transformer, se reconstruye la estructura de imagen y se aplica:

- `MaxPool2DLayer`: reduce espacialmente (ej: de 28x28 a 14x14).
- `FlattenLayer`: aplana para pasar a capa densa.
- `DenseLayer`: capa de salida con `num_classes` neuronas.
- `SoftmaxLayer`: produce las probabilidades de clasificación.

---

## ⚙️ Entrenamiento

En `main.cpp`, el modelo es entrenado con el optimizador `SGD` y `cross-entropy loss`:

```cpp
trainer.train(/*epochs=*/50, /*log_every=*/100, device);
````

Se usa `Fashion-MNIST`, cargado con `load_dataset`.

---

## 🧪 Backward Propagation

El `backward` se implementa de forma personalizada y recorre en orden inverso todos los componentes, permitiendo entrenamiento extremo a extremo.

Para los Transformers, la retropropagación incluye:

* `projector->backward()`
* `transformer->backward()`
* `tokenizer->backward()`

Con manejo explícito de los gradientes para tokens y mapa de características.

---

## 🧠 ¿Por qué usar Visual Transformers?

Este enfoque combina:

* La **capacidad local** de las CNN para capturar texturas y bordes.
* La **capacidad global** de los Transformers para modelar relaciones espaciales complejas entre regiones lejanas.

Esto lo hace ideal para tareas donde los patrones pueden estar distribuidos espacialmente, como clasificación, segmentación o detección.

---

## 📁 Estructura de Carpetas

```
model/
│  ├── cnn.hpp
│  ├── mlp.hpp
│  ├── vit_cnn.hpp  ← clase principal VTCNN
│
layers/
│  ├── conv2d_layer.hpp
│  ├── batch_normalization_layer.hpp
│  ├── visual_transformer.hpp
│  ├── ...
│
utils/
│  ├── load_dataset.hpp
│  ├── trainer.hpp
│
main.cpp
```

---

## 🧩 Extensiones futuras

* Reemplazar `FilterTokenizer` con `RecurrentTokenizer`.
* Añadir posición codificada (positional encoding) a los tokens.
* Usar `Adam` como optimizador.
* Extender a otras tareas como segmentación o VQA.

---

## 🖼️ Ejemplo Visual (flujo de datos)

```
Image (28x28x1)
  ↓ Conv2D (1→4)  →  (28x28x4)
  ↓ BatchNorm
  ↓ Flatten to Sequence (784x4)
  ↓ Visual Transformer (→ token 16x16)
  ↓ Projector → Restore (28x28x8)
  ↓ MaxPool (14x14x8)
  ↓ Flatten
  ↓ Dense (→ 10)
  ↓ Softmax → Prediction
```

---

## ✍️ Autor

Desarrollado por \[Tu Nombre]
Implementación en C++ usando clases propias de `Tensor`, `Layer` y `Trainer`.

---
