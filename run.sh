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
LOG_DIR="${PROJECT_ROOT}/logs"
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

# Crear directorios necesarios
mkdir -p "${BUILD_DIR}"
mkdir -p "${MODEL_DIR}"
mkdir -p "${LOG_DIR}"

# Crear nombre de log
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="${LOG_DIR}/run_${EXECUTABLE}_${MODE}_${TIMESTAMP}.log"

# Mostrar + guardar en log
echo "Configurando construcción (Modo: ${MODE})" | tee "${LOGFILE}"

# Configurar y construir
cd "${BUILD_DIR}" || exit 1

echo "🔧 Ejecutando CMake..." | tee -a "${LOGFILE}"
cmake .. -DUSE_CUDA=${USE_CUDA} -DCMAKE_BUILD_TYPE=Release 2>&1 | tee -a "${LOGFILE}"

echo "🛠️  Compilando proyecto..." | tee -a "${LOGFILE}"
if ! make -j$(nproc) 2>&1 | tee -a "${LOGFILE}"; then
  echo "❌ Error en la compilación" | tee -a "${LOGFILE}"
  exit 1
fi

echo "🚀 Ejecutando programa..." | tee -a "${LOGFILE}"
cd "${PROJECT_ROOT}" || exit 1

# Ejecutar y mostrar/guardar salida de programa
{ time "${BUILD_DIR}/${EXECUTABLE}" "$@"; } 2>&1 | tee -a "${LOGFILE}"

echo "✅ Ejecución completada (Modo: ${MODE})" | tee -a "${LOGFILE}"
echo "📄 Log guardado en: ${LOGFILE}"
