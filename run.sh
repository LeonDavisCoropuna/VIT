#!/bin/bash

# Verificar que se pasen los argumentos requeridos
if [ $# -lt 3 ]; then
  echo "Uso: ./run.sh <nombre_ejecutable> <dataset> <modo> <epocas> [--cuda] [--no-build]"
  echo "Ejemplo: ./run.sh main mnist train 10 --cuda"
  exit 1
fi

EXECUTABLE="$1"
DATASET="$2"
MODE_RUN="$3"
EPOCHS="$4"
shift 4  # Avanzar argumentos

# Configuración
PROJECT_ROOT=$(pwd)
BUILD_DIR="${PROJECT_ROOT}/build"
MODEL_DIR="${PROJECT_ROOT}/save_models"
DEFAULT_MODE="CPU"
SKIP_BUILD=false

# Parsear flags opcionales
while [[ $# -gt 0 ]]; do
  case $1 in
    --cuda)
      USE_CUDA=ON
      MODE="CUDA"
      shift
      ;;
    --no-build)
      SKIP_BUILD=true
      shift
      ;;
    *)
      echo "Argumento desconocido: $1"
      echo "Uso: ./run.sh <nombre_ejecutable> <dataset> <modo> <epocas> [--cuda] [--no-build]"
      exit 1
      ;;
  esac
done

# Configurar modo por defecto si no se especificó CUDA
MODE=${MODE:-$DEFAULT_MODE}
USE_CUDA=${USE_CUDA:-OFF}

echo "🔧 Configuración del entorno (Modo: ${MODE})"

# Crear directorios necesarios
mkdir -p "${BUILD_DIR}"
mkdir -p "${MODEL_DIR}"

# Compilar si no se pasó --no-build
if [ "$SKIP_BUILD" = false ]; then
  cd "${BUILD_DIR}" || exit 1

  echo "🔨 Ejecutando CMake..."
  cmake .. -DUSE_CUDA=${USE_CUDA} -DCMAKE_BUILD_TYPE=Release

  echo "📦 Compilando proyecto..."
  if ! make -j$(nproc); then
    echo "❌ Error en la compilación"
    exit 1
  fi

  cd "${PROJECT_ROOT}" || exit 1
else
  echo "🚫 Saltando compilación (--no-build)"
fi

echo "🚀 Ejecutando programa..."
time "${BUILD_DIR}/${EXECUTABLE}" "${DATASET}" "${MODE_RUN}" "${EPOCHS}"

echo "✅ Ejecución completada (Modo: ${MODE})"
