cmake_minimum_required(VERSION 3.18)  # Versión más reciente para mejor soporte CUDA
project(mlp LANGUAGES CXX)

option(USE_CUDA "Use CUDA implementation for Tensor" OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Configuración de builds
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# OpenMP para paralelización en CPU
find_package(OpenMP REQUIRED)
message(STATUS "OpenMP found: ${OpenMP_CXX_FLAGS}")

# ==== Configuración avanzada de CUDA ====
if(USE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  enable_language(CUDA)
  
  # Configuración automática de arquitecturas
  set(CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING "GPU architectures")
  
  # Opciones específicas para CUDA
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")
  set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3 -DNDEBUG")
  set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -O0")
  
  message(STATUS "CUDA activado - Versión ${CUDAToolkit_VERSION}")
  message(STATUS "Arquitecturas CUDA: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# Configuración de includes
include_directories(
  ${CMAKE_SOURCE_DIR}/models
  ${CMAKE_SOURCE_DIR}/utils
  ${CMAKE_SOURCE_DIR}/utils/tensor
)

# Fuentes comunes
set(COMMON_SOURCES
  main.cpp
  layers/layer.hpp
  utils/tensor.hpp  # Interfaz común CPU/GPU
)

# Fuentes específicas
if(USE_CUDA)
  list(APPEND SOURCES 
    ${COMMON_SOURCES}
    utils/tensor/tensor_cuda.cu     # Cambiado de .cuh a .cu
  )
  set_source_files_properties(
    utils/tensor/tensor_cuda.cu
    PROPERTIES LANGUAGE CUDA
  )
else()
  list(APPEND SOURCES 
    ${COMMON_SOURCES}
    utils/tensor/tensor_cpu.hpp    # Implementación CPU
  )
endif()

add_executable(main ${SOURCES})

# Configuración de flags
target_compile_options(main PRIVATE
  $<$<CONFIG:Release>:-O3 -march=native -DNDEBUG>
  $<$<CONFIG:Debug>:-O0 -g>
)

# Enlaces
target_link_libraries(main PRIVATE OpenMP::OpenMP_CXX)

if(USE_CUDA)
  target_link_libraries(main PRIVATE CUDA::cudart CUDA::cublas)
  target_compile_definitions(main PRIVATE USE_CUDA=1)
endif()

# Instalación de modelos
install(DIRECTORY ${CMAKE_SOURCE_DIR}/save_models DESTINATION .)