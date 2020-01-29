#ifndef KMCUDA_PRIVATE_H
#define KMCUDA_PRIVATE_H

#include "kmcuda.h"
#include <cinttypes>
#include <tuple>
#include "wrappers.h"

#if CUDA_ARCH >= 60
typedef double atomic_float;
#else
typedef float atomic_float;
#endif


#if CUDART_VERSION >= 9000
#define shfl(...) __shfl_sync(0xFFFFFFFF, __VA_ARGS__)
#define ballot(...) __ballot_sync(0xFFFFFFFF, __VA_ARGS__)
#define shfl_down(...) __shfl_down_sync(0xFFFFFFFF, __VA_ARGS__)
// This one removes all the registry usage optimizations which helped in CUDA 8
#define volatile
#else
#define shfl __shfl
#define ballot __ballot
#define shfl_down __shfl_down
#endif

/// printf() under INFO log level (0).
#define INFO(...) do { if (verbosity > 0) { printf(__VA_ARGS__); } } while (false)
/// printf() under DEBUG log level (1).
#define DEBUG(...) do { if (verbosity > 1) { printf(__VA_ARGS__); } } while (false)
/// printf() under TRACE log level (2).
#define TRACE(...) do { if (verbosity > 2) { printf(__VA_ARGS__); } } while (false)

#define CUERRSTR() cudaGetErrorString(cudaGetLastError())

/// Checks the CUDA call for errors, in case of an error logs it and returns.
/// "return" forces this to be a macro.
#define CUCH(cuda_call, ret, ...) \
do { \
  auto __res = cuda_call; \
  if (__res != cudaSuccess) { \
    DEBUG("%s\n", #cuda_call); \
    INFO("%s:%d -> %s\n", __FILE__, __LINE__, cudaGetErrorString(__res)); \
    __VA_ARGS__; \
    return ret; \
  } \
} while (false)

/// Checks whether the call returns 0; if not, executes arbitrary code and returns.
/// "return" forces this to be a macro.
#define RETERR(call, ...) \
do { \
  auto __res = call; \
  if (__res != 0) { \
    __VA_ARGS__; \
    return __res; \
  } \
} while (false)

/// Executes arbitrary code for every CUDA device.
#define FOR_EACH_DEV(...) do { for (int dev : devs) { \
  cudaSetDevice(dev); \
  __VA_ARGS__; \
} } while(false)

/// Executes arbitrary code for every CUDA device and supplies the device index
/// into the scope.
#define FOR_EACH_DEVI(...) do { for (size_t devi = 0; devi < devs.size(); devi++) { \
  cudaSetDevice(devs[devi]); \
  __VA_ARGS__; \
} } while(false)

/// Invokes cudaDeviceSynchronize() on every CUDA device.
#define SYNC_ALL_DEVS do { \
if (devs.size() > 1) { \
FOR_EACH_DEV(CUCH(cudaDeviceSynchronize(), kmcudaRuntimeError)); \
} } while (false)

/// Copies memory from device to host asynchronously across all the CUDA devices.
#define CUMEMCPY_D2H_ASYNC(dst, dst_stride, src, src_offset, size) do { \
  FOR_EACH_DEVI(CUCH(cudaMemcpyAsync( \
      dst + dst_stride * devi, (src)[devi].get() + src_offset, \
      (size) * sizeof(typename std::remove_reference<decltype(src)>::type::value_type \
          ::element_type), \
      cudaMemcpyDeviceToHost), \
                     kmcudaMemoryCopyError)); \
} while(false)

/// Copies memory from device to host synchronously across all the CUDA devices.
#define CUMEMCPY_D2H(dst, src, size) do { \
  CUMEMCPY_D2H_ASYNC(dst, src, size); \
  FOR_EACH_DEV(CUCH(cudaDeviceSynchronize(), kmcudaMemoryCopyError)); \
} while(false)

/// Copies memory from host to device asynchronously across all the CUDA devices.
#define CUMEMCPY_H2D_ASYNC(dst, dst_offset, src, size) do { \
  FOR_EACH_DEVI(CUCH(cudaMemcpyAsync( \
      (dst)[devi].get() + dst_offset, src, \
      (size) * sizeof(typename std::remove_reference<decltype(dst)>::type::value_type \
          ::element_type), \
      cudaMemcpyHostToDevice), \
                     kmcudaMemoryCopyError)); \
} while(false)

/// Copies memory from host to device synchronously across all the CUDA devices.
#define CUMEMCPY_H2D(dst, src, size) do { \
  CUMEMCPY_H2D_ASYNC(dst, src, size); \
  FOR_EACH_DEV(CUCH(cudaDeviceSynchronize(), kmcudaMemoryCopyError)); \
} while(false)

/// Copies memory from device to device asynchronously across all the CUDA devices.
#define CUMEMCPY_D2D_ASYNC(dst, dst_offset, src, src_offset, size) do { \
  FOR_EACH_DEVI(CUCH(cudaMemcpyAsync( \
      (dst)[devi].get() + dst_offset, (src)[devi].get() + src_offset, \
      (size) * sizeof(typename std::remove_reference<decltype(dst)>::type::value_type \
          ::element_type), \
      cudaMemcpyDeviceToDevice), \
                     kmcudaMemoryCopyError)); \
} while(false)

/// Copies memory from device to host synchronously across all the CUDA devices.
#define CUMEMCPY_D2D(dst, dst_offset, src, src_offset, size) do { \
  CUMEMCPY_D2D_ASYNC(dst, dst_offset, src, src_offset, size); \
  FOR_EACH_DEV(CUCH(cudaDeviceSynchronize(), kmcudaMemoryCopyError)); \
} while(false)

/// Allocates memory on CUDA device and adds the created pointer to the list.
#define CUMALLOC_ONEN(dest, size, name, dev) do { \
  void *__ptr; \
  size_t __size = (size) * \
      sizeof(typename std::remove_reference<decltype(dest)>::type::value_type::element_type); \
  CUCH(cudaMalloc(&__ptr, __size), kmcudaMemoryAllocationFailure, \
       INFO("failed to allocate %zu bytes for " name "\n", __size)); \
  (dest).emplace_back(reinterpret_cast<typename std::remove_reference<decltype(dest)> \
      ::type::value_type::element_type *>(__ptr)); \
  TRACE("[%d] " name ": %p - %p (%zu)\n", dev, __ptr, \
        reinterpret_cast<char *>(__ptr) + __size, __size); \
} while(false)

/// Shortcut for CUMALLOC_ONEN which defines the log name.
#define CUMALLOC_ONE(dest, size, dev) CUMALLOC_ONEN(dest, size, #dest, dev)

/// Allocates memory on all CUDA devices.
#define CUMALLOCN(dest, size, name) do { \
  FOR_EACH_DEV(CUMALLOC_ONEN(dest, size, name, dev)); \
} while(false)

/// Allocates memory on all CUDA devices. Does not require the log name, infers
/// it from dest.
#define CUMALLOC(dest, size) CUMALLOCN(dest, size, #dest)

/// Invokes cudaMemsetAsync() on all CUDA devices.
#define CUMEMSET_ASYNC(dst, val, size) do { \
  FOR_EACH_DEVI(CUCH(cudaMemsetAsync( \
      (dst)[devi].get(), val, \
      size * sizeof(typename std::remove_reference<decltype(dst)>::type::value_type::element_type)), \
                     kmcudaRuntimeError)); \
} while(false)

/// Invokes cudaMemset() on all CUDA devices.
#define CUMEMSET(dst, val, size) do { \
  CUMEMSET_ASYNC(dst, val, size); \
  FOR_EACH_DEV(CUCH(cudaDeviceSynchronize(), kmcudaRuntimeError)); \
} while(false)

/// Executes the specified code on all devices except the given one - number devi.
#define FOR_OTHER_DEVS(...) do { \
  for (size_t odevi = 0; odevi < devs.size(); odevi++) { \
    if (odevi == devi) { \
      continue; \
    } \
    __VA_ARGS__; \
  } } while(false)

/// Copies memory peer to peer (device to other device device).
#define CUP2P(what, offset, size) do { \
  CUCH(cudaMemcpyPeerAsync( \
      (*what)[odevi].get() + offset, devs[odevi], (*what)[devi].get() + offset, \
      devs[devi], (size) * sizeof(typename std::remove_reference<decltype(*what)>::type \
      ::value_type::element_type)), \
       kmcudaMemoryCopyError); \
} while(false)

#if CUDA_ARCH >= 60
/// Bridges the code from single branch to multiple template branches.
#define KERNEL_SWITCH(f, ...) do { switch (metric) { \
  case kmcudaDistanceMetricL2: \
    if (!fp16x2) { \
        using F = float; \
        f<kmcudaDistanceMetricL2, float>__VA_ARGS__; \
    } else { \
        using F = half2; \
        f<kmcudaDistanceMetricL2, half2>__VA_ARGS__; \
    } \
    break; \
  case kmcudaDistanceMetricCosine: \
    if (!fp16x2) { \
        using F = float; \
        f<kmcudaDistanceMetricCosine, float>__VA_ARGS__; \
    } else { \
        using F = half2; \
        f<kmcudaDistanceMetricCosine, half2>__VA_ARGS__; \
    } \
    break; \
} } while(false)
#else
#define KERNEL_SWITCH(f, ...) do { switch (metric) { \
  case kmcudaDistanceMetricL2: \
    using F = float; \
    f<kmcudaDistanceMetricL2, float>__VA_ARGS__; \
    break; \
  case kmcudaDistanceMetricCosine: \
    using F = float; \
    f<kmcudaDistanceMetricCosine, float>__VA_ARGS__; \
    break; \
} } while(false)
#endif

/// Alternative to dupper() for host.
template <typename T>
inline T upper(T size, T each) {
  T div = size / each;
  if (div * each == size) {
    return div;
  }
  return div + 1;
}

using plan_t = std::vector<std::tuple<uint32_t, uint32_t>>;

/// @brief Generates the split across CUDA devices: (offset, size) pairs.
/// It aligns every chunk at 512 bytes without breaking elements.
/// @param amount The total work size - array size in elements.
/// @param size_each Element size in bytes. Thus the total memory size is
///                  amount * size_each.
/// @param devs The list with device numbers.
/// @return The list with offset-size pairs. The measurement unit is the element
/// size.
inline plan_t distribute(
    uint32_t amount, uint32_t size_each, const std::vector<int> &devs) {
  if (devs.size() == 0) {
    return {};
  }
  if (devs.size() == 1) {
    return {std::make_tuple(0, amount)};
  }
  const uint32_t alignment = 512;
  uint32_t a = size_each, b = alignment, gcd = 0;
  for (;;) {
    if (a == 0) {
      gcd = b;
      break;
    }
    b %= a;
    if (b == 0) {
      gcd = a;
      break;
    }
    a %= b;
  }
  uint32_t stride = alignment / gcd;
  uint32_t offset = 0;
  std::vector<std::tuple<uint32_t, uint32_t>> res;
  for (size_t i = 0; i < devs.size() - 1; i++) {
    float step = (amount - offset + .0f) / (devs.size() - i);
    uint32_t len = roundf(step / stride) * stride;
    res.emplace_back(offset, len);
    offset += len;
  }
  res.emplace_back(offset, amount - offset);
  return res;
}

/// Extracts the maximum split length from the device distribution plan.
/// It calls distribute() and finds the maximum size.
inline uint32_t max_distribute_length(
    uint32_t amount, uint32_t size_each, const std::vector<int> &devs) {
  auto plan = distribute(amount, size_each, devs);
  uint32_t max_length = 0;
  for (auto& p : plan) {
    uint32_t length = std::get<1>(p);
    if (length > max_length) {
      max_length = length;
    }
  }
  return max_length;
}

/// Dumps the device split distribution to stdout.
inline void print_plan(const char *name, const plan_t& plan) {
  printf("%s: [", name);
  bool first = true;
  for (auto& p : plan) {
    if (!first) {
      printf(", ");
    }
    first = false;
    printf("(%" PRIu32 ", %" PRIu32 ")", std::get<0>(p), std::get<1>(p));
  }
  printf("]\n");
}

extern "C" {

/// Copies the single sample within the same device. Defined in transpose.cu.
KMCUDAResult cuda_copy_sample_t(
    uint32_t index, uint32_t offset, uint32_t samples_size, uint16_t features_size,
    const std::vector<int> &devs, int verbosity, const udevptrs<float> &samples,
    udevptrs<float> *dest);

/// Copies the single sample from device to host. Defined in transpose.cu.
KMCUDAResult cuda_extract_sample_t(
    uint32_t index, uint32_t samples_size, uint16_t features_size,
    int verbosity, const float *samples, float *dest);

/// Transposes the samples matrix. Defined in transpose.cu.
KMCUDAResult cuda_transpose(
    uint32_t samples_size, uint16_t features_size, bool forward,
    const std::vector<int> &devs, int verbosity, udevptrs<float> *samples);

/// Invokes kmeans++ kernel. Defined in kmeans.cu.
KMCUDAResult kmeans_cuda_plus_plus(
    uint32_t samples_size, uint32_t features_size, uint32_t cc,
    KMCUDADistanceMetric metric, const std::vector<int> &devs, int fp16x2,
    int verbosity, const udevptrs<float> &samples, udevptrs<float> *centroids,
    udevptrs<float> *dists, float *host_dists, atomic_float *dists_sum);

/// Invokes afk-mc2 kernel "calc_q". Defined in kmeans.cu.
KMCUDAResult kmeans_cuda_afkmc2_calc_q(
    uint32_t samples_size, uint32_t features_size, uint32_t firstc,
    KMCUDADistanceMetric metric, const std::vector<int> &devs, int fp16x2,
    int verbosity, const udevptrs<float> &samples, udevptrs<float> *d_q,
    float *h_q);

/// Invokes afk-mc2 kernel "random_step". Defined in kmeans.cu.
KMCUDAResult kmeans_cuda_afkmc2_random_step(
    uint32_t k, uint32_t m, uint64_t seed, int verbosity, const float *q,
    uint32_t *d_choices, uint32_t *h_choices, float *d_samples, float *h_samples);

/// Invokes afk-mc2 kernel "min_dist". Defined in kmeans.cu.
KMCUDAResult kmeans_cuda_afkmc2_min_dist(
    uint32_t k, uint32_t m, KMCUDADistanceMetric metric, int fp16x2,
    int32_t verbosity, const float *samples, const uint32_t *choices,
    const float *centroids, float *d_min_dists, float *h_min_dists);

/// Initializes the CUDA environment, e.g. assigns values to symbols.
/// Defined in kmeans.cu.
KMCUDAResult kmeans_cuda_setup(
    uint32_t samples_size, uint16_t features_size, uint32_t clusters_size,
    uint32_t yy_groups_size, const std::vector<int> &devs, int32_t verbosity);

/// Performs the centroids initialization. Defined in kmcuda.cc.
KMCUDAResult kmeans_init_centroids(
    KMCUDAInitMethod method, const void *init_params, uint32_t samples_size,
    uint16_t features_size, uint32_t clusters_size, KMCUDADistanceMetric metric,
    uint32_t seed, const std::vector<int> &devs, int device_ptrs, int fp16x2,
    int32_t verbosity, const float *host_centroids,  const udevptrs<float> &samples,
    udevptrs<float> *dists, udevptrs<float> *aux, udevptrs<float> *centroids);

/// Complementing implementation of kmeans_cuda() which requires nvcc.
/// Defined in kmeans.cu.
KMCUDAResult kmeans_cuda_yy(
    float tolerance, uint32_t yy_groups_size, uint32_t samples_size,
    uint32_t clusters_size, uint16_t features_size, KMCUDADistanceMetric metric,
    const std::vector<int> &devs, int fp16x2, int32_t verbosity,
    const udevptrs<float> &samples, udevptrs<float> *centroids,
    udevptrs<uint32_t> *ccounts, udevptrs<uint32_t> *assignments_prev,
    udevptrs<uint32_t> *assignments, udevptrs<uint32_t> *assignments_yy,
    udevptrs<float> *centroids_yy, udevptrs<float> *bounds_yy,
    udevptrs<float> *drifts_yy, udevptrs<uint32_t> *passed_yy);

/// Calculates the average distance between cluster members and the corresponding
/// centroid. Defined in kmeans.cu.
KMCUDAResult kmeans_cuda_calc_average_distance(
    uint32_t samples_size, uint16_t features_size,
    KMCUDADistanceMetric metric, const std::vector<int> &devs, int fp16x2,
    int32_t verbosity, const udevptrs<float> &samples,
    const udevptrs<float> &centroids, const udevptrs<uint32_t> &assignments,
    float *average_distance);

/// Prepares the CUDA environment for K-nn calculation, e.g., assigns values to
/// symbols. Defined in knn.cu.
KMCUDAResult knn_cuda_setup(
    uint32_t samples_size, uint16_t features_size, uint32_t clusters_size,
    const std::vector<int> &devs, int32_t verbosity);

/// Complementing implementation of knn_cuda() which requires nvcc.
/// Defined in knn.cu.
KMCUDAResult knn_cuda_calc(
    uint16_t k, uint32_t h_samples_size, uint32_t h_clusters_size,
    uint16_t h_features_size, KMCUDADistanceMetric metric,
    const std::vector<int> &devs, int fp16x2, int verbosity,
    const udevptrs<float> &samples, const udevptrs<float> &centroids,
    const udevptrs<uint32_t> &assignments, const udevptrs<uint32_t> &inv_asses,
    const udevptrs<uint32_t> &inv_asses_offsets, udevptrs<float> *distances,
    udevptrs<float>* sample_dists, udevptrs<float> *radiuses,
    udevptrs<uint32_t> *neighbors);

/// Looks at the amount of available shared memory and decides on the
/// performance critical property of knn_cuda_calc() - which of the two variants
/// to follow.
int knn_cuda_neighbors_mem_multiplier(uint16_t k, int dev, int verbosity);
}  // extern "C"

#endif  // KMCUDA_PRIVATE_H
