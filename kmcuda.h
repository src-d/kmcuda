#ifndef KMCUDA_KMCUDA_H
#define KMCUDA_KMCUDA_H

#include <stdint.h>

enum KMCUDAResult {
  kmcudaSuccess,
  kmcudaInvalidArguments,
  kmcudaMemoryAllocationFailure,
  kmcudaRuntimeError,
  kmcudaMemoryCopyError
};

extern "C" {
int kmeans_cuda(uint32_t samples_size, uint16_t features_size,
                uint32_t clusters_size, int32_t verbosity, uint32_t seed,
                uint32_t block_size, const float *samples, float *centroids,
                uint32_t *assignments);
}

#endif //KMCUDA_KMCUDA_H
