#ifndef KMCUDA_WRAPPERS_H
#define KMCUDA_WRAPPERS_H

#include <cuda_runtime_api.h>
#include <functional>
#include <memory>
#include <vector>

template <typename T>
using unique_devptr_parent = std::unique_ptr<T, std::function<void(T*)>>;

/// RAII for CUDA arrays. Calls cudaFree() on the bound pointer, but only
/// if it is not nullptr (funnily enough, CUDA segfaults otherwise).
/// As can be seen, inherits the rest of the methods from std::unique_ptr.
/// @param T The type of the array element.
template <typename T>
class unique_devptr : public unique_devptr_parent<T> {
 public:
  explicit unique_devptr(T *ptr, bool fake = false) : unique_devptr_parent<T>(
      ptr, fake? [](T*){} : [](T *p){ cudaFree(p); }) {}
};

/// std::vector of unique_devptr-s. Used to pass device arrays inside .cu
/// wrapping functions.
/// @param T The type of the array element.
template <class T>
using udevptrs = std::vector<unique_devptr<T>>;

#endif //KMCUDA_WRAPPERS_H
