#include <iostream>
#include "utils/tensor.hpp"
std::mt19937 Tensor::global_gen(42); // semilla fija por defecto
int main()
{
  Tensor t({4, 4});
  t.fill(3.14f);
  std::cout << "Tensor CUDA llenado con 3.14\n";
  return 0;
}
