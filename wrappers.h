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

#endif //KMCUDA_WRAPPERS_H
