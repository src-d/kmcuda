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

class CudaTextureObject {
 public:
  explicit CudaTextureObject(cudaTextureObject_t obj) : _obj(obj) {}
  CudaTextureObject(CudaTextureObject&& other) : _obj(other._obj) {}
  CudaTextureObject(const CudaTextureObject&) = delete;
  CudaTextureObject operator=(const CudaTextureObject&) = delete;

  ~CudaTextureObject() {
    if (_obj) {
      cudaDestroyTextureObject(_obj);
    }
  }

  inline operator cudaTextureObject_t() {
    return _obj;
  }

 private:
  cudaTextureObject_t _obj;
};

#endif //KMCUDA_WRAPPERS_H
