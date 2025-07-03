cmake_minimum_required(VERSION 3.10)
project(mlp LANGUAGES CXX)

option(USE_CUDA "Use CUDA implementation for Tensor" OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
  message(STATUS "OpenMP found: ${OpenMP_CXX_FLAGS}")
endif()

# ==== CUDA SUPPORT ====
if(USE_CUDA)
  enable_language(CUDA)  # 👈 importante
  message(STATUS "✅ CUDA activado")
endif()

# Include dirs
include_directories(
  ${CMAKE_SOURCE_DIR}/models
  ${CMAKE_SOURCE_DIR}/utils
  ${CMAKE_SOURCE_DIR}/utils/tensor  # asegúrate de tener esto si usas tensor_cuda
)

# Fuente base
set(SOURCES
  main.cpp
  layers/layer.hpp
)

# Añadir archivos según modo
if(USE_CUDA)
  list(APPEND SOURCES utils/tensor/tensor_cuda.cu)
else()
  list(APPEND SOURCES utils/tensor/tensor_cpu.hpp)
endif()

add_executable(main ${SOURCES})

# Flags Release/Debug
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  target_compile_options(main PRIVATE -O3 -march=native -DNDEBUG)
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_options(main PRIVATE -O0 -g)
endif()

# Link OpenMP
if(OpenMP_FOUND)
  target_link_libraries(main PRIVATE OpenMP::OpenMP_CXX)
endif()
