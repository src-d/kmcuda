#ifndef KMCUDA_KMCUDA_H
#define KMCUDA_KMCUDA_H

#include <stdint.h>

enum KMCUDAResult {
  kmcudaSuccess = 0,
  kmcudaInvalidArguments,
  kmcudaNoSuchDevice,
  kmcudaMemoryAllocationFailure,
  kmcudaRuntimeError,
  kmcudaMemoryCopyError
};

extern "C" {
/// @brief Performs K-means clustering on GPU / CUDA.
/// @param kmpp indicates whether to do kmeans++ initialization. If false,
///             ordinary random centroids will be picked.
/// @param samples_size number of samples.
/// @param features_size number of features.
/// @param clusters_size number of clusters.
/// @param seed random generator seed passed to srand().
/// @param device CUDA device index - usually 0.
/// @param verbosity 0 - no output; 1 - progress output; >=2 - debug output.
/// @param samples input array of size samples_size x features_size in row major format.
/// @param centroids output array of centroids of size clusters_size x features_size
///                  in row major format.
/// @param assignments output array of cluster indices for each sample of size
///                    samples_size x 1.
/// @return KMCUDAResult.
int kmeans_cuda(bool kmpp, float tolerance, uint32_t samples_size,
                uint16_t features_size, uint32_t clusters_size, uint32_t seed,
                uint32_t device, int32_t verbosity, const float *samples,
                float *centroids, uint32_t *assignments);
}

#endif //KMCUDA_KMCUDA_H
