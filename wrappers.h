#ifndef KMCUDA_WRAPPERS_H
#define KMCUDA_WRAPPERS_H

#include <cuda_runtime_api.h>
#include <memory>

using unique_devptr_parent = std::unique_ptr<void, std::function<void(void*)>>;

class unique_devptr : public unique_devptr_parent {
 public:
  explicit unique_devptr(void *ptr) : unique_devptr_parent(
      ptr, [](void *p){ cudaFree(p); }) {}
};

using unique_devptrptr_parent = std::unique_ptr<void*, std::function<void(void**)>>;

class unique_devptrptr : public unique_devptrptr_parent {
 public:
  explicit unique_devptrptr(void **ptr) : unique_devptrptr_parent(
      ptr, [](void **p){ if (p) { cudaFree(*p); } }) {}
};

#endif //KMCUDA_WRAPPERS_H
