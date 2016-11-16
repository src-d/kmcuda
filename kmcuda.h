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

enum KMCUDAInitMethod {
  kmcudaInitMethodRandom = 0,
  kmcudaInitMethodPlusPlus,
  kmcudaInitMethodImport
};

enum KMCUDADistanceMetric {
  kmcudaDistanceMetricL2,
  kmcudaDistanceMetricCosine
};

extern "C" {

/// @brief Performs K-means clustering on GPU / CUDA.
/// @param init centroids initialization method.
/// @param tolerance if the number of reassignments drop below this ratio, stop.
/// @param yinyang_t the relative number of cluster groups, usually 0.1.
/// @param metric the distance metric to use. The default is Euclidean (L2), can be
///               changed to cosine to behave as Spherical K-means with the angular
///               distance. Please note that samples *must* be normalized in that
///               case.
/// @param samples_size number of samples.
/// @param features_size number of features (vector dimensionality).
/// @param clusters_size number of clusters.
/// @param seed random generator seed passed to srand().
/// @param device used CUDA device mask. E.g., 1 means #0, 2 means #1 and 3 means
///               #0 and #1. n-th bit corresponds to n-th device.
/// @param device_ptrs If negative, input and output pointers are taken from host;
///                    otherwise, device number where to load and store data.
/// @param verbosity 0 - no output; 1 - progress output; >=2 - debug output.
/// @param samples input array of size samples_size x features_size in row major format.
/// @param centroids output array of centroids of size clusters_size x features_size
///                  in row major format.
/// @param assignments output array of cluster indices for each sample of size
///                    samples_size x 1.
/// @return KMCUDAResult.
KMCUDAResult kmeans_cuda(
    KMCUDAInitMethod init, float tolerance, float yinyang_t,
    KMCUDADistanceMetric metric, uint32_t samples_size, uint16_t features_size,
    uint32_t clusters_size, uint32_t seed, uint32_t device, int device_ptrs,
    int32_t verbosity, const float *samples, float *centroids, uint32_t *assignments);

}  // extern "C"

#endif //KMCUDA_KMCUDA_H
