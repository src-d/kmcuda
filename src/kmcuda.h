#ifndef KMCUDA_KMCUDA_H
#define KMCUDA_KMCUDA_H

/*! @mainpage KMeansCUDA documentation
 *
 * @section s1 Description
 *
 * K-means and K-nn on NVIDIA CUDA which are designed for production usage and
 * simplicity.
 *
 * K-means is based on ["Yinyang K-Means: A Drop-In Replacement
 * of the Classic K-Means with Consistent Speedup"](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ding15.pdf).
 * While it introduces some overhead and many conditional clauses
 * which are bad for CUDA, it still shows 1.6-2x speedup against the Lloyd
 * algorithm. K-nearest neighbors employ the same triangle inequality idea and
 * require precalculated centroids and cluster assignments, similar to the flattened
 * ball tree.
 *
 * Project: https://github.com/src-d/kmcuda
 *
 * README: @ref ignore_this_doxygen_anchor
 *
 * @section s2 C/C++ API
 *
 * kmcuda.h exports two functions: kmeans_cuda() and knn_cuda(). They are not
 * thread safe.
 *
 * @section s3 Python 3 API
 *
 * The shared library exports kmeans_cuda() and knn_cuda() Python wrappers.
 *
 * @section s4 R API
 *
 * The shared library exports kmeans_cuda() and knn_cuda() R wrappers (.External).
 *
 */

#include <stdint.h>

/// All possible error codes in public API.
typedef enum {
  /// Everything's all right.
  kmcudaSuccess = 0,
  /// Arguments which were passed into a function failed the validation.
  kmcudaInvalidArguments,
  /// The requested CUDA device does not exist.
  kmcudaNoSuchDevice,
  /// Failed to allocate memory on CUDA device. Too big size? Switch off Yinyang?
  kmcudaMemoryAllocationFailure,
  /// Something bad and unidentified happened on the CUDA side.
  kmcudaRuntimeError,
  /// Failed to copy memory to/from CUDA device.
  kmcudaMemoryCopyError
} KMCUDAResult;

/// Centroid initialization method.
typedef enum {
  /// Pick initial centroids randomly.
  kmcudaInitMethodRandom = 0,
  /// Use kmeans++ initialization method. Theoretically proven to yield
  /// better clustering than kmcudaInitMethodRandom. O(n * k) complexity.
  /// https://en.wikipedia.org/wiki/K-means%2B%2B
  kmcudaInitMethodPlusPlus,
  /// AFK-MC2 initialization method. Theoretically proven to yield
  /// better clustering results than kmcudaInitMethodRandom; matches
  /// kmcudaInitMethodPlusPlus asymptotically and fast. O(n + k) complexity.
  /// Use it when kmcudaInitMethodPlusPlus takes too long to finish.
  /// http://olivierbachem.ch/files/afkmcmc-oral-pdf.pdf
  kmcudaInitMethodAFKMC2,
  /// Take user supplied centroids.
  kmcudaInitMethodImport
} KMCUDAInitMethod;

/// Specifies how to calculate the distance between each pair of dots.
typedef enum {
  /// Mesasure the distance between dots using Euclidean distance.
  kmcudaDistanceMetricL2,
  /// Measure the distance between dots using the angle between them.
  /// @note This metric requires all the supplied data to be normalized by L2 to 1.
  kmcudaDistanceMetricCosine
} KMCUDADistanceMetric;

#ifdef __cplusplus
extern "C" {
#endif

/// @brief Performs K-means clustering on GPU / CUDA.
/// @param init centroids initialization method.
/// @param init_params pointer to a struct / number with centroid initialization
///                    parameters. Ignored unless init == kmcudaInitMethodAFKMC2.
///                    In case with kmcudaInitMethodAFKMC2 it is expected to be
///                    uint32_t* to m; m == 0 means the default value (200).
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
/// @param fp16x2 If true, the input is treated as half2 instead of float. In that case,
///               features_size must be 2 times smaller than the real size.
/// @param verbosity 0 - no output; 1 - progress output; >=2 - debug output.
/// @param samples input array of size samples_size x features_size in row major format.
/// @param centroids output array of centroids of size clusters_size x features_size
///                  in row major format.
/// @param assignments output array of cluster indices for each sample of size
///                    samples_size x 1.
/// @param average_distance output mean distance between cluster elements and
///                         the corresponding centroids. If nullptr, not calculated.
/// @return KMCUDAResult.
KMCUDAResult kmeans_cuda(
    KMCUDAInitMethod init, const void *init_params, float tolerance, float yinyang_t,
    KMCUDADistanceMetric metric, uint32_t samples_size, uint16_t features_size,
    uint32_t clusters_size, uint32_t seed, uint32_t device, int32_t device_ptrs,
    int32_t fp16x2, int32_t verbosity, const float *samples, float *centroids,
    uint32_t *assignments, float *average_distance);

/// @brief Calculates K nearest neighbors for every sample using
///        the precalculated K-means clusters.
/// @param k the number of neighbors to search for every dot.
/// @param metric the distance metric to use. The default is Euclidean (L2), can be
///               changed to cosine to behave as Spherical K-means with the angular
///               distance. Please note that samples *must* be normalized in that
///               case.
/// @param samples_size number of samples.
/// @param features_size number of features (vector dimensionality).
/// @param clusters_size number of clusters.
/// @param device used CUDA device mask. E.g., 1 means #0, 2 means #1 and 3 means
///               #0 and #1. n-th bit corresponds to n-th device.
/// @param device_ptrs If negative, input and output pointers are taken from host;
///                    otherwise, device number where to load and store data.
/// @param fp16x2 If true, the input is treated as half2 instead of float. In that case,
///               features_size must be 2 times smaller than the real size.
/// @param verbosity 0 - no output; 1 - progress output; >=2 - debug output.
/// @param samples input array of size samples_size x features_size in row major format.
/// @param centroids input array of centroids of size clusters_size x features_size
///                  in row major format.
/// @param assignments input array of cluster indices for each sample of size
///                    samples_size x 1.
/// @param neighbors output array with the nearest neighbors of size
///                  samples_size x k in row major format.
/// @return KMCUDAResult.
KMCUDAResult knn_cuda(
    uint16_t k, KMCUDADistanceMetric metric, uint32_t samples_size,
    uint16_t features_size, uint32_t clusters_size, uint32_t device,
    int32_t device_ptrs, int32_t fp16x2, int32_t verbosity,
    const float *samples, const float *centroids, const uint32_t *assignments,
    uint32_t *neighbors);

#ifdef __cplusplus
}  // extern "C"
#endif

#ifdef __cplusplus
#include <string>
#include <unordered_map>

namespace {
namespace kmcuda {
/// Mapping from strings to KMCUDAInitMethod - useful for wrappers.
const std::unordered_map<std::string, KMCUDAInitMethod> init_methods {
    {"kmeans++", kmcudaInitMethodPlusPlus},
    {"k-means++", kmcudaInitMethodPlusPlus},
    {"afkmc2", kmcudaInitMethodAFKMC2},
    {"afk-mc2", kmcudaInitMethodAFKMC2},
    {"random", kmcudaInitMethodRandom}
};

/// Mapping from strings to KMCUDADistanceMetric - useful for wrappers.
const std::unordered_map<std::string, KMCUDADistanceMetric> metrics {
    {"euclidean", kmcudaDistanceMetricL2},
    {"L2", kmcudaDistanceMetricL2},
    {"l2", kmcudaDistanceMetricL2},
    {"cos", kmcudaDistanceMetricCosine},
    {"cosine", kmcudaDistanceMetricCosine},
    {"angular", kmcudaDistanceMetricCosine}
};

/// Mapping from KMCUDAResult to strings - useful for wrappers.
const std::unordered_map<int, const char *> statuses {
    {kmcudaSuccess, "Success"},
    {kmcudaInvalidArguments, "InvalidArguments"},
    {kmcudaNoSuchDevice, "NoSuchDevice"},
    {kmcudaMemoryAllocationFailure, "MemoryAllocationFailure"},
    {kmcudaRuntimeError, "RuntimeError"},
    {kmcudaMemoryCopyError, "MemoryCopyError"}
};
}  // namespace kmcuda
}  // namespace
#endif  // __cplusplus

#endif //KMCUDA_KMCUDA_H
