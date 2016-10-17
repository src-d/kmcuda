#ifndef KMCUDA_WRAPPERS_H
#define KMCUDA_WRAPPERS_H

#include <cuda_runtime_api.h>
#include <memory>
#include <vector>

template <typename T>
using unique_devptr_parent = std::unique_ptr<T, std::function<void(T*)>>;

template <typename T>
class unique_devptr : public unique_devptr_parent<T> {
 public:
  explicit unique_devptr(T *ptr, bool fake = false) : unique_devptr_parent<T>(
      ptr, fake? [](T*){} : [](T *p){ cudaFree(p); }) {}
};

template <class T>
using udevptrs = std::vector<unique_devptr<T>>;

#endif //KMCUDA_WRAPPERS_H
