#!/bin/bash

# Verificar si se pasó el nombre del ejecutable
if [ -z "$1" ]; then
  echo "Debes proporcionar el nombre del ejecutable como primer argumento."
  echo "Uso: ./run.sh <nombre_ejecutable> [--cuda]"
  exit 1
fi

EXECUTABLE="$1"
shift  # Avanzar a los siguientes argumentos

# Configuración
PROJECT_ROOT=$(pwd)
BUILD_DIR="${PROJECT_ROOT}/build"
MODEL_DIR="${PROJECT_ROOT}/save_models"
DEFAULT_MODE="CPU"

# Parsear argumentos adicionales
while [[ $# -gt 0 ]]; do
  case $1 in
    --cuda)
      USE_CUDA=ON
      MODE="CUDA"
      shift
      ;;
    *)
      echo "Argumento desconocido: $1"
      echo "Uso: ./run.sh <nombre_ejecutable> [--cuda]"
      exit 1
      ;;
  esac
done

# Configurar modo por defecto si no se especificó CUDA
MODE=${MODE:-$DEFAULT_MODE}
USE_CUDA=${USE_CUDA:-OFF}

echo "Configurando construcción (Modo: ${MODE})"

# Crear directorios necesarios
mkdir -p "${BUILD_DIR}"
mkdir -p "${MODEL_DIR}"

# Configurar y construir
cd "${BUILD_DIR}" || exit 1

echo "🔧 Ejecutando CMake..."
cmake .. -DUSE_CUDA=${USE_CUDA} -DCMAKE_BUILD_TYPE=Release

echo "🛠️  Compilando proyecto..."
if ! make -j$(nproc); then
  echo "Error en la compilación"
  exit 1
fi

echo "Ejecutando programa..."
cd "${PROJECT_ROOT}" || exit 1
time "${BUILD_DIR}/${EXECUTABLE}"

echo "Ejecución completada (Modo: ${MODE})"
