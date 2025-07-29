# 🧠 VTCNN - Visual Transformer CNN Classifier

Este proyecto implementa un modelo híbrido basado en **Convolutional Neural Networks (CNNs)** y **Visual Transformers**, llamado `VTCNN`, para tareas de clasificación de imágenes.

Se incluyen:

- Soporte para datasets: **MNIST**, **FashionMNIST**, **BloodMNIST**.
- Ejecución en **CPU** o **GPU (CUDA)**.
- Interfaz frontend para visualizar métricas y predecir imágenes personalizadas.
- Entrenamiento desde línea de comandos con script automatizado (`run.sh`).

---

## 🌐 Visión General del Modelo

<img src="img/image.png" width="600"/>

1. CNN para extracción de características.
2. Transformer para modelar relaciones globales.
3. Capas densas para clasificación final.

---

## 📦 Requisitos

### 🔧 Backend (C++)

- CMake >= 3.10
- GCC o Clang compatible con C++17
- CUDA Toolkit (opcional)

### 💻 Frontend (Next.js)

```bash
cd frontend
npm install
```

---

## 🏗 Estructura del Proyecto

```
.
├── blood_data/              # Datos BloodMNIST (train, test, val)
├── fashion_data/            # Datos FashionMNIST
├── mnist_data/              # Datos MNIST
├── custom_images/           # Imágenes personalizadas para predicción
├── save_models/             # Modelos entrenados (.bin)
├── layers/                  # Implementación de capas (CNN, Transformer, etc.)
├── model/                   # Modelos (MLP, VTCNN)
├── utils/                   # Dataset loader, optimizer, tensor
├── frontend/                # App web (Next.js)
├── logs/                    # Logs de ejecución
├── run.sh                   # Script de entrenamiento/predicción
├── main.cpp                 # Punto de entrada
├── CMakeLists.txt           # Configuración CMake
└── img/                     # Gráficas, visuales y pipeline
```

---

## ⚙️ Compilación y Ejecución

### 🔨 Entrenamiento

```bash
./run.sh main <dataset> train <épocas> [--cuda]
```

#### Ejemplos:

```bash
./run.sh main mnist train 10
./run.sh main fashionmnist train 20 --cuda
./run.sh main bloodmnist train 30
```

### 🔎 Predicción

```bash
./run.sh main <dataset> predict 0 --no-build [--cuda]
```

```bash
./run.sh main mnist predict 0 --no-build
```

> **Nota:** El número de épocas (`0`) se ignora durante predicción, pero debe pasarse por sintaxis.




## Ejecución rápida

Para compilar y ejecutar, usa el script `run.sh`:

```bash
./run.sh <ejecutable> <dataset> <modo> <épocas> [--cuda] [--no-build]
````

| Parámetro      | Descripción                                            | Ejemplo    |
| -------------- | ------------------------------------------------------ | ---------- |
| `<ejecutable>` | Binario a ejecutar (normalmente `main`)                | `main`     |
| `<dataset>`    | Dataset a usar (`mnist`, `fashionmnist`, `bloodmnist`) | `mnist`    |
| `<modo>`       | `train` o `predict`                                    | `train`    |
| `<épocas>`     | Número de épocas (usar `0` en `predict`)               | `10`       |
| `--cuda`       | Usa compilación con soporte CUDA                       | (opcional) |
| `--no-build`   | Salta la compilación                                   | (opcional) |

## Ejemplos

### Entrenar MNIST por 15 épocas

```bash
./run.sh main mnist train 15
```

### Entrenar FashionMNIST con CUDA

```bash
./run.sh main fashionmnist train 10 --cuda
```

### Predecir usando un modelo previamente entrenado

```bash
./run.sh main mnist predict 0 --no-build
```

### Predecir desde tu frontend (ejecutado vía Node.js)

```ts
const child = spawn('bash', ['run.sh', 'main', 'mnist', 'predict', '0', '--no-build'], {
  env: { ...process.env, INPUT_IMAGE: imagePath }
});
```

> 📝 Asegúrate de haber entrenado el modelo previamente antes de ejecutar en modo `predict`.

---

## Dataset BloodMNIST

Si deseas usar **BloodMNIST**, asegúrate de haberlo convertido previamente al formato MNIST (imágenes `.png` de 28x28 y un archivo `labels.txt` correspondiente).

---

## Notas adicionales

* El script `run.sh` compila automáticamente el ejecutable si no se usa `--no-build`.
* Si usas `--cuda`, se compilará con soporte para GPU (requiere que CUDA esté correctamente instalado).
* Las predicciones pueden integrarse con aplicaciones web mediante llamadas a procesos como se muestra arriba.

---



## 🚀 Frontend: Visualización y Predicción Web

### Instalación

```bash
cd frontend
npm install
```

### Ejecución

```bash
npm run dev
```

Accede a: [http://localhost:3000](http://localhost:3000)

### Funcionalidades

- Ver gráficas de entrenamiento:
  - Accuracy / Loss
  - Matriz de confusión
- Predicción interactiva:
  - **MNIST**: dibuja un número.
  - **Otros**: sube una imagen personalizada.

### Ejemplos Visuales

#### Accuracy

| MNIST | FashionMNIST | BloodMNIST |
|-------|---------------|------------|
| <img src="./img/mnistacuracy.png" width="200"/> | <img src="./img/accuracy_fashionmnist.png" width="200"/> | <img src="./img/accuracy_bloodmnist.png" width="200"/> |

#### Matriz de Confusión (MNIST)

<img src="./img/confusion_mnist.png" width="400"/>

---

## 🧠 Arquitectura VTCNN (Resumen)

```
Input (28x28)
↓
Conv2D (1→4) + BatchNorm
↓
Reshape → Tokenizer (Transformer)
↓
TransformerLayer → Projector
↓
MaxPool → Flatten → Dense
↓
Softmax
↓
Predicción
```

Componentes implementados manualmente:

- `Conv2DLayer`, `BatchNorm2DLayer`
- `VisualTransformer` con atención multi-cabeza
- `Tokenizador` + `Projector`
- `Trainer` con retropropagación desde cero

---

## ✍️ Dataset Soportado

| Dataset      | Clases | Formato Esperado         |
|--------------|--------|---------------------------|
| MNIST        | 10     | ubyte                     |
| FashionMNIST | 10     | ubyte                     |
| BloodMNIST   | 8      | ubyte (formato extendido) |

> Los datasets deben colocarse en:
- `mnist_data/`
- `fashion_data/`
- `blood_data/`

---

## 💾 Guardado de Modelos

Después de entrenamiento, los modelos se guardan en:

```
save_models/
├── model_mnist.bin
├── model_fashionmnist.bin
├── model_bloodmnist.bin
```

---

## 🧪 Ejemplo Completo

```bash
./run.sh main fashionmnist train 20 --cuda    # Entrena en GPU
./run.sh main fashionmnist predict 0 --no-build
npm run dev                                   # Levanta frontend
```

---

## 🧹 Limpieza

```bash
rm -rf build/
```

---

## 🧩 Posibles Extensiones

- Tokenización recurrente
- Positional Encoding
- Soporte a VQA o segmentación
- Soporte ONNX export
- UI amigable para cargar modelos

---

## ✨ Créditos

Desarrollado con ❤️ por [Tu Nombre].  
Este proyecto demuestra cómo integrar **C++ backend personalizado**, **Transformers visuales** y **Frontend moderno con Next.js** en una solución completa.

---
