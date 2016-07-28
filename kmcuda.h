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
/// @param tolerance if the number of reassignments drop below this ratio, stop.
/// @param yinyang_t the relative number of cluster groups, usually 0.1.
/// @param samples_size number of samples.
/// @param features_size number of features.
/// @param clusters_size number of clusters.
/// @param seed random generator seed passed to srand().
/// @param device used CUDA device mask. E.g., 1 means #0, 2 means #1 and 3 means
///               #0 and #1. n-th bit corresponds to n-th device.
/// @param verbosity 0 - no output; 1 - progress output; >=2 - debug output.
/// @param device_ptrs If negative, input and output pointers are taken from host;
///                    otherwise, device number where to load and store data.
/// @param samples input array of size samples_size x features_size in row major format.
/// @param centroids output array of centroids of size clusters_size x features_size
///                  in row major format.
/// @param assignments output array of cluster indices for each sample of size
///                    samples_size x 1.
/// @return KMCUDAResult.
int kmeans_cuda(bool kmpp, float tolerance, float yinyang_t, uint32_t samples_size,
                uint16_t features_size, uint32_t clusters_size, uint32_t seed,
                uint32_t device, int32_t verbosity, int device_ptrs,
                const float *samples, float *centroids, uint32_t *assignments);
}

#endif //KMCUDA_KMCUDA_H
