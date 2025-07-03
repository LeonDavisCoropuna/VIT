#pragma once

#ifdef USE_CUDA
#include "tensor/tensor_cuda.cu"
#else
#include "tensor/tensor_cpu.hpp"
#endif
