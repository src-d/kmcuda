#ifndef KMCUDA_PRIVATE_H
#define KMCUDA_PRIVATE_H

#include "kmcuda.h"

enum KMCUDAInitMethod {
  kmcudaInitMethodRandom = 0,
  kmcudaInitMethodPlusPlus
};

#define RETERR(call, ...) do { \
  auto __r = call; \
  if (__r != kmcudaSuccess) { \
    __VA_ARGS__; \
    return __r; \
  } \
} while (false)

#define INFO(...) do { if (verbosity > 0) { printf(__VA_ARGS__); } } while (false)
#define DEBUG(...) do { if (verbosity > 1) { printf(__VA_ARGS__); } } while (false)

extern "C" {

KMCUDAResult kmeans_cuda_plus_plus(
    uint32_t samples_size, uint32_t cc, float *samples, float *centroids,
    float *dists, float *distssum, float **dev_sums);

KMCUDAResult kmeans_cuda_setup(uint32_t samples_size, uint16_t features_size,
                               uint32_t clusters_size, uint32_t yy_groups_size,
                               uint32_t device, int32_t verbosity);

KMCUDAResult kmeans_cuda_yy(
    float tolerance, uint32_t yinyang_groups, uint32_t samples_size_,
    uint32_t clusters_size_, uint16_t features_size, int32_t verbosity,
    const float *samples, float *centroids, uint32_t *ccounts,
    uint32_t *assignments_prev, uint32_t *assignments, uint32_t *assignments_yy,
    float *centroids_yy, float *bounds_yy, float *drifts_yy, uint32_t *passed_yy);

KMCUDAResult kmeans_init_centroids(
    KMCUDAInitMethod method, uint32_t samples_size, uint16_t features_size,
    uint32_t clusters_size, uint32_t seed, int32_t verbosity, float *samples,
    void *dists, float *centroids);
}

#endif //KMCUDA_PRIVATE_H
