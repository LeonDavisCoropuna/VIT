#!/bin/bash

# Detectar flag --cuda
USE_CUDA_FLAG=""
if [[ "$2" == "--cuda" ]]; then
  echo "⚙️  Modo CUDA activado (USE_CUDA=ON)"
  USE_CUDA_FLAG="-DUSE_CUDA=ON"
else
  echo "⚙️  Modo CPU activado (USE_CUDA=OFF)"
  USE_CUDA_FLAG="-DUSE_CUDA=OFF"
fi

# Verificar si se pasó un nombre de ejecutable
if [ -z "$1" ]; then
  echo "❌ Error: Debes proporcionar el nombre del ejecutable como parámetro."
  echo "👉 Uso: ./run.sh <nombre_ejecutable> [--cuda]"
  exit 1
fi

EXECUTABLE_NAME=$1

# Configuración de directorios
PROJECT_ROOT=$(pwd)
BUILD_DIR="$PROJECT_ROOT/build"
MODEL_DIR="$PROJECT_ROOT/save_models"  # Asegura que save_models existe

echo "🔧 Creando carpetas necesarias..."
mkdir -p "$BUILD_DIR"
mkdir -p "$MODEL_DIR"

echo "📁 Configurando el proyecto..."
cd "$BUILD_DIR"
cmake .. $USE_CUDA_FLAG

echo "🛠️ Compilando con make..."
make

echo "🚀 Ejecutando el programa: $EXECUTABLE_NAME"
cd "$PROJECT_ROOT"
"$BUILD_DIR/$EXECUTABLE_NAME"
