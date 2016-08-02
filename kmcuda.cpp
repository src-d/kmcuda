#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cinttypes>
#include <cfloat>
#include <cmath>
#include <cassert>
#include <memory>

#include <cuda_runtime_api.h>

#include "wrappers.h"
#include "private.h"


#define CUMEMCPY(dst, src, size, flag) \
do { if (cudaMemcpy(dst, src, size, flag) != cudaSuccess) { \
  return kmcudaMemoryCopyError; \
} } while(false)

#define CUMEMCPY_ASYNC(dst, src, size, flag) \
do { if (cudaMemcpyAsync(dst, src, size, flag) != cudaSuccess) { \
  return kmcudaMemoryCopyError; \
} } while(false)

#define CUMALLOC(dest, size, name) do { \
  DEBUG(name ": %zu\n", size); \
  if (cudaMalloc(&dest, size) != cudaSuccess) { \
    INFO("failed to allocate %zu bytes for " name "\n", size); \
    return kmcudaMemoryAllocationFailure; \
  } \
} while(false)

static int check_args(
    float tolerance, float yinyang_t, uint32_t samples_size, uint16_t features_size,
    uint32_t clusters_size, const float *samples, float *centroids,
    uint32_t *assignments) {
  if (clusters_size < 2 || clusters_size == UINT32_MAX) {
    return kmcudaInvalidArguments;
  }
  if (features_size == 0) {
    return kmcudaInvalidArguments;
  }
  if (samples_size < clusters_size) {
    return kmcudaInvalidArguments;
  }
  if (samples == nullptr || centroids == nullptr || assignments == nullptr) {
    return kmcudaInvalidArguments;
  }
  if (tolerance < 0 || tolerance > 1) {
    return kmcudaInvalidArguments;
  }
  if (yinyang_t < 0 || yinyang_t > 0.5) {
    return kmcudaInvalidArguments;
  }
  return kmcudaSuccess;
}

static KMCUDAResult print_memory_stats() {
  size_t free_bytes, total_bytes;
  if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
    return kmcudaRuntimeError;
  }
  printf("GPU memory: used %zu bytes (%.1f%%), free %zu bytes, total %zu bytes\n",
         total_bytes - free_bytes, (total_bytes - free_bytes) * 100.0 / total_bytes,
         free_bytes, total_bytes);
  return kmcudaSuccess;
}

extern "C" {

KMCUDAResult kmeans_init_centroids(
    KMCUDAInitMethod method, uint32_t samples_size, uint16_t features_size,
    uint32_t clusters_size, uint32_t seed, int32_t verbosity, float *samples,
    void *dists, float *centroids) {
  uint32_t ssize = features_size * sizeof(float);
  srand(seed);
  switch (method) {
    case kmcudaInitMethodRandom:
      INFO("randomly picking initial centroids...\n");
      for (uint32_t c = 0; c < clusters_size; c++) {
        if ((c + 1) % 1000 == 0 || c == clusters_size - 1) {
          INFO("\rcentroid #%" PRIu32, c + 1);
          fflush(stdout);
          CUMEMCPY(centroids + c * features_size,
                   samples + (rand() % samples_size) * features_size,
                   ssize, cudaMemcpyDeviceToDevice);
        } else {
          CUMEMCPY_ASYNC(centroids + c * features_size,
                         samples + (rand() % samples_size) * features_size,
                         ssize, cudaMemcpyDeviceToDevice);
        }
      }
      break;
    case kmcudaInitMethodPlusPlus:
      INFO("performing kmeans++...\n");
      float smoke = NAN;
      uint32_t first_offset;
      while (smoke != smoke) {
        first_offset = (rand() % samples_size) * features_size;
        CUMEMCPY(&smoke, samples + first_offset, sizeof(float), cudaMemcpyDeviceToHost);
      }
      CUMEMCPY(centroids, samples + first_offset, ssize, cudaMemcpyDeviceToDevice);
      std::unique_ptr<float[]> host_dists(new float[samples_size]);
      float *dev_sums = NULL;
      unique_devptrptr dev_sums_sentinel(reinterpret_cast<void**>(&dev_sums));
      for (uint32_t i = 1; i < clusters_size; i++) {
        if (verbosity > 1 || (verbosity > 0 && (
              clusters_size < 100 || i % (clusters_size / 100) == 0))) {
          printf("\rstep %d", i);
          fflush(stdout);
        }
        float dist_sum = 0;
        RETERR(kmeans_cuda_plus_plus(
            samples_size, i, samples, centroids, reinterpret_cast<float*>(dists),
            &dist_sum, &dev_sums),
               DEBUG("\nkmeans_cuda_plus_plus failed\n"));
        assert(dist_sum == dist_sum);
        CUMEMCPY(host_dists.get(), dists, samples_size * sizeof(float),
                 cudaMemcpyDeviceToHost);
        double choice = ((rand() + .0) / RAND_MAX);
        uint32_t choice_approx = choice * samples_size;
        double choice_sum = choice * dist_sum;
        uint32_t j;
        if (choice_approx < 100) {
          double dist_sum2 = 0;
          for (j = 0; j < samples_size && dist_sum2 < choice_sum; j++) {
            dist_sum2 += host_dists[j];
          }
        } else {
          double dist_sum2 = 0;
          #pragma omp simd reduction(+:dist_sum2)
          for (uint32_t t = 0; t < choice_approx; t++) {
            dist_sum2 += host_dists[t];
          }
          if (dist_sum2 < choice_sum) {
            for (j = choice_approx; j < samples_size && dist_sum2 < choice_sum; j++) {
              dist_sum2 += host_dists[j];
            }
          } else {
            for (j = choice_approx; j > 1 && dist_sum2 >= choice_sum; j--) {
              dist_sum2 -= host_dists[j];
            }
            j++;
          }
        }
        assert(j > 0);
        CUMEMCPY_ASYNC(centroids + i * features_size,
                       samples + (j - 1) * features_size,
                       ssize, cudaMemcpyDeviceToDevice);
      }
      break;
  }

  INFO("\rdone            \n");
  return kmcudaSuccess;
}

int kmeans_cuda(bool kmpp, float tolerance, float yinyang_t, uint32_t samples_size,
                uint16_t features_size, uint32_t clusters_size, uint32_t seed,
                uint32_t device, int32_t verbosity, const float *samples,
                float *centroids, uint32_t *assignments) {
  DEBUG("arguments: %d %.3f %.2f %" PRIu32 " %" PRIu16 " %" PRIu32 " %" PRIu32
        " %" PRIu32 " %" PRIi32 " %p %p %p\n",
        kmpp, tolerance, yinyang_t, samples_size, features_size, clusters_size,
        seed, device, verbosity, samples, centroids, assignments);
  auto check_result = check_args(
      tolerance, yinyang_t, samples_size, features_size, clusters_size,
      samples, centroids, assignments);
  if (check_result != kmcudaSuccess) {
    return check_result;
  }
  if (cudaSetDevice(device) != cudaSuccess) {
    return kmcudaNoSuchDevice;
  }

  void *device_samples;
  size_t device_samples_size = samples_size;
  device_samples_size *= features_size * sizeof(float);
  CUMALLOC(device_samples, device_samples_size, "samples");
  CUMEMCPY(device_samples, samples, device_samples_size, cudaMemcpyHostToDevice);
  unique_devptr device_samples_sentinel(device_samples);

  void *device_centroids;
  size_t centroids_size = clusters_size * features_size * sizeof(float);
  CUMALLOC(device_centroids, centroids_size, "centroids");
  unique_devptr device_centroids_sentinel(device_centroids);

  void *device_assignments;
  size_t assignments_size = samples_size * sizeof(uint32_t);
  CUMALLOC(device_assignments, assignments_size, "assignments");
  unique_devptr device_assignments_sentinel(device_assignments);

  void *device_assignments_prev;
  CUMALLOC(device_assignments_prev, assignments_size, "assignments_prev");
  unique_devptr device_assignments_prev_sentinel(device_assignments_prev);

  void *device_ccounts;
  CUMALLOC(device_ccounts, clusters_size * sizeof(uint32_t), "ccounts");
  unique_devptr device_ccounts_sentinel(device_ccounts);

  uint32_t yinyang_groups = yinyang_t * clusters_size;
  DEBUG("yinyang groups: %" PRIu32 "\n", yinyang_groups);
  void *device_assignments_yy = NULL, *device_bounds_yy = NULL,
      *device_drifts_yy = NULL, *device_passed_yy = NULL,
      *device_centroids_yy = NULL;
  if (yinyang_groups >= 1) {
    CUMALLOC(device_assignments_yy, clusters_size * sizeof(uint32_t),
             "yinyang assignments");
    size_t yyb_size = samples_size;
    yyb_size *= (yinyang_groups + 1) * sizeof(float);
    CUMALLOC(device_bounds_yy, yyb_size, "yinyang bounds");
    CUMALLOC(device_drifts_yy, centroids_size + clusters_size * sizeof(float),
             "yinyang drifts");
    CUMALLOC(device_passed_yy, assignments_size, "yinyang passed");
    size_t yyc_size = yinyang_groups * features_size * sizeof(float);
    if (yyc_size + (clusters_size + yinyang_groups) * sizeof(uint32_t)
        <= assignments_size) {
      device_centroids_yy = device_passed_yy;
    } else {
      CUMALLOC(device_centroids_yy, yyc_size, "yinyang group centroids");
    }
  }
  unique_devptr device_centroids_yinyang_sentinel(
      (device_centroids_yy != device_passed_yy)? device_centroids_yy : NULL);
  unique_devptr device_assignments_yinyang_sentinel(device_assignments_yy);
  unique_devptr device_bounds_yinyang_sentinel(device_bounds_yy);
  unique_devptr device_drifts_yinyang_sentinel(device_drifts_yy);
  unique_devptr device_passed_yinyang_sentinel(device_passed_yy);

  if (verbosity > 1) {
    RETERR(print_memory_stats());
  }
  RETERR(kmeans_cuda_setup(samples_size, features_size, clusters_size,
                           yinyang_groups, device, verbosity),
         DEBUG("kmeans_cuda_setup failed: %s\n",
               cudaGetErrorString(cudaGetLastError())));
  RETERR(kmeans_init_centroids(
      static_cast<KMCUDAInitMethod>(kmpp), samples_size, features_size,
      clusters_size, seed, verbosity, reinterpret_cast<float*>(device_samples),
      device_assignments, reinterpret_cast<float*>(device_centroids)),
         DEBUG("kmeans_init_centroids failed: %s\n",
               cudaGetErrorString(cudaGetLastError())));
  RETERR(kmeans_cuda_yy(
      tolerance, yinyang_groups, samples_size, clusters_size, features_size, verbosity,
      reinterpret_cast<float*>(device_samples),
      reinterpret_cast<float*>(device_centroids),
      reinterpret_cast<uint32_t*>(device_ccounts),
      reinterpret_cast<uint32_t*>(device_assignments_prev),
      reinterpret_cast<uint32_t*>(device_assignments),
      reinterpret_cast<uint32_t*>(device_assignments_yy),
      reinterpret_cast<float*>(device_centroids_yy),
      reinterpret_cast<float*>(device_bounds_yy),
      reinterpret_cast<float*>(device_drifts_yy),
      reinterpret_cast<uint32_t*>(device_passed_yy)),
         DEBUG("kmeans_cuda_internal failed: %s\n",
               cudaGetErrorString(cudaGetLastError())));
  CUMEMCPY(centroids, device_centroids, centroids_size, cudaMemcpyDeviceToHost);
  CUMEMCPY(assignments, device_assignments, assignments_size, cudaMemcpyDeviceToHost);
  DEBUG("return kmcudaSuccess\n");
  return kmcudaSuccess;
}
}