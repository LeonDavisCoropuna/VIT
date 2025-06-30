#!/bin/bash

# Verificar si se pasó un nombre de ejecutable
if [ -z "$1" ]; then
  echo "❌ Error: Debes proporcionar el nombre del ejecutable como parámetro."
  echo "👉 Uso: ./run.sh <nombre_ejecutable>"
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
cmake ..

echo "🛠️ Compilando con make..."
make

echo "🚀 Ejecutando el programa: $EXECUTABLE_NAME"
cd "$PROJECT_ROOT"

# Ejecutar el ejecutable desde el build
"$BUILD_DIR/$EXECUTABLE_NAME"