#include "kmcuda.h"

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

extern "C" {

KMCUDAResult kmeans_cuda_plus_plus(
    uint32_t samples_size, uint32_t cc, float *samples, float *centroids,
    float *dists, float *distssum, float **dev_sums);

KMCUDAResult kmeans_cuda_setup(uint32_t samples_size, uint16_t features_size,
                               uint32_t clusters_size, uint32_t device,
                               int32_t verbosity);

KMCUDAResult kmeans_cuda_internal(
    uint32_t samples_size, uint32_t clusters_size, int32_t verbosity,
    float *samples, float *centroids, uint32_t *ccounts, uint32_t *assignments);

}

#define cumemcpy(dst, src, size, flag) \
do { if (cudaMemcpy(dst, src, size, flag) != cudaSuccess) { \
  return kmcudaMemoryCopyError; \
} } while(false)

static int check_args(uint32_t samples_size, uint16_t features_size,
                      uint32_t clusters_size, const float *samples,
                      float *centroids, uint32_t *assignments) {
  if (clusters_size < 2) {
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
  return kmcudaSuccess;
}

#if 0
static void init_centroids_cpu(uint32_t samples_size, uint16_t features_size,
                               uint32_t clusters_size, uint32_t seed,
                               int32_t verbosity, const float *samples,
                               float *centroids) {
  if (verbosity > 0) {
    printf("Performing kmeans++... ");
    fflush(stdout);
  }
  srand(seed);
  memcpy(centroids, &samples[(rand() % samples_size) * features_size],
         features_size * sizeof(float));
  std::unique_ptr<float[]> dists(new float[samples_size]);
  for (uint32_t i = 1; i < clusters_size; i++) {
    if (verbosity > 1) {
      printf("\nstep %d", i);
    }
    double dist_sum = 0;
    #pragma omp parallel for
    for (uint32_t j = 0; j < samples_size; j++) {
      float min_dist = FLT_MAX;
      for (uint32_t c = 0; c < i; c++) {
        float dist = 0;
        #pragma omp simd reduction(+:dist)
        for (uint16_t f = 0; f < features_size; f++) {
          float d = samples[j * features_size + f] - centroids[c * features_size + f];
          dist += d * d;
        }
        if (dist < min_dist) {
          min_dist = dist;
        }
      }
      min_dist = sqrtf(min_dist);
      dists[j] = min_dist;
      #pragma omp critical
      dist_sum += min_dist;
    }
    double choice = ((rand() + .0) / RAND_MAX) * dist_sum;
    dist_sum = 0;
    uint32_t j;
    for (j = 0; j < samples_size && dist_sum < choice; j++) {
      dist_sum += dists[j];
    }
    memcpy(&centroids[i * features_size], &samples[(j - 1) * features_size],
           features_size * sizeof(float));
  }
  if (verbosity > 0) {
    if (verbosity > 0) {
      printf("\n");
    }
    printf("done\n");
  }
}
#endif

static KMCUDAResult init_centroids_gpu(
    KMCUDAInitMethod method, uint32_t samples_size, uint16_t features_size,
    uint32_t clusters_size, uint32_t seed, int32_t verbosity, float *samples,
    float *centroids, void *dists) {
  srand(seed);
  switch (method) {
    case kmcudaInitMethodRandom:
      if (verbosity > 0) {
        printf("randomly picking initial centroids... ");
        fflush(stdout);
      }
      for (uint32_t c = 0; c < clusters_size; c++) {
        cumemcpy(centroids + c * features_size,
                 samples + (rand() % samples_size) * features_size,
                 features_size * sizeof(float), cudaMemcpyDeviceToDevice);
      }
      break;
    case kmcudaInitMethodPlusPlus:
      if (verbosity > 0) {
        printf("Performing kmeans++... ");
        fflush(stdout);
      }
      cumemcpy(centroids, samples + (rand() % samples_size) * features_size,
               features_size * sizeof(float), cudaMemcpyDeviceToDevice);
      std::unique_ptr<float[]> host_dists(new float[samples_size]);
      float *dev_sums = NULL;
      unique_devptrptr dev_sums_sentinel(reinterpret_cast<void**>(&dev_sums));
      for (uint32_t i = 1; i < clusters_size; i++) {
        if (verbosity > 1 || (verbosity > 0 && (
              clusters_size < 100 || i % (clusters_size / 100) == 0))) {
          printf("\nstep %d", i);
          fflush(stdout);
        }
        float dist_sum = 0;
        auto result = kmeans_cuda_plus_plus(
            samples_size, i, samples, centroids, reinterpret_cast<float*>(dists),
            &dist_sum, &dev_sums);
        if (result != kmcudaSuccess) {
          if (verbosity > 1) {
            printf("\nkmeans_cuda_plus_plus failed\n");
          }
          return result;
        }
        cumemcpy(host_dists.get(), dists, samples_size * sizeof(float),
                 cudaMemcpyDeviceToHost);
        double choice = ((rand() + .0) / RAND_MAX) * dist_sum;
        double dist_sum2 = 0;
        uint32_t j;
        for (j = 0; j < samples_size && dist_sum2 < choice; j++) {
          dist_sum2 += host_dists[j];
        }
        assert(j > 0);
        cumemcpy(centroids + i * features_size, samples + (j - 1) * features_size,
                 features_size * sizeof(float), cudaMemcpyDeviceToDevice);
      }
      break;
  }

  if (verbosity > 0) {
    if (verbosity > 0) {
      printf("\n");
    }
    printf("done\n");
  }
  return kmcudaSuccess;
}

extern "C" {

int kmeans_cuda(bool kmpp, uint32_t samples_size, uint16_t features_size,
                uint32_t clusters_size, int32_t verbosity, uint32_t seed,
                uint32_t device, const float *samples, float *centroids,
                uint32_t *assignments) {
  if (verbosity > 1) {
    printf("arguments: %d %" PRIu32 " %" PRIu16 " %" PRIu32 " %" PRIi32
           " %" PRIu32 " %" PRIu32 " %p %p %p\n",
           kmpp, samples_size, features_size, clusters_size, verbosity, seed,
           device, samples, centroids, assignments);
  }
  auto check_result = check_args(samples_size, features_size, clusters_size,
                                 samples, centroids, assignments);
  if (check_result != kmcudaSuccess) {
    return check_result;
  }
  if (cudaSetDevice(device) != cudaSuccess) {
    return kmcudaNoSuchDevice;
  }
  void *device_samples;
  size_t device_samples_size = samples_size * features_size * sizeof(float);
  if (cudaMalloc(&device_samples, device_samples_size) != cudaSuccess) {
    if (verbosity > 0) {
      printf("failed to allocate %zu bytes for samples\n", device_samples_size);
    }
    return kmcudaMemoryAllocationFailure;
  }
  cumemcpy(device_samples, samples, device_samples_size, cudaMemcpyHostToDevice);
  unique_devptr device_samples_sentinel(device_samples);
  void *device_centroids;
  size_t centroids_size = clusters_size * features_size * sizeof(float);
  if (cudaMalloc(&device_centroids, centroids_size)
      != cudaSuccess) {
    if (verbosity > 0) {
      printf("failed to allocate %zu bytes for centroids\n", centroids_size);
    }
    return kmcudaMemoryAllocationFailure;
  }
  unique_devptr device_centroids_sentinel(device_centroids);
  void *device_assignments;
  size_t assignments_size = samples_size * sizeof(uint32_t);
  if (cudaMalloc(&device_assignments, assignments_size)
      != cudaSuccess) {
    if (verbosity > 0) {
      printf("failed to allocate %zu bytes for assignments\n", assignments_size);
    }
    return kmcudaMemoryAllocationFailure;
  }
  unique_devptr device_assignments_sentinel(device_assignments);
  void *device_ccounts;
  if (cudaMalloc(&device_ccounts, clusters_size * sizeof(uint32_t))
      != cudaSuccess) {
    if (verbosity > 0) {
      printf("failed to allocate %zu bytes for ccounts\n",
             clusters_size * sizeof(uint32_t));
    }
    return kmcudaMemoryAllocationFailure;
  }
  unique_devptr device_ccounts_sentinel(device_ccounts);
  if (verbosity > 1) {
    size_t free_bytes, total_bytes;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
      return kmcudaRuntimeError;
    }
    printf("GPU memory: used %zu bytes (%.1f%%), free %zu bytes, total %zu bytes\n",
           total_bytes - free_bytes, (total_bytes - free_bytes) * 100.0 / total_bytes,
           free_bytes, total_bytes);
  }
  auto result = kmeans_cuda_setup(samples_size, features_size, clusters_size,
                                  device, verbosity);
  if (result != kmcudaSuccess) {
    if (verbosity > 1) {
      printf("kmeans_cuda_setup failed: %s\n",
             cudaGetErrorString(cudaGetLastError()));
    }
    return result;
  }
  result = init_centroids_gpu(
      static_cast<KMCUDAInitMethod>(kmpp), samples_size, features_size,
      clusters_size, seed, verbosity, reinterpret_cast<float*>(device_samples),
      reinterpret_cast<float*>(device_centroids), device_assignments);
  if (result != kmcudaSuccess) {
    if (verbosity > 1) {
      printf("\ninit_centroids_gpu failed: %s\n",
             cudaGetErrorString(cudaGetLastError()));
    }
    return result;
  }
  if (cudaMemset(device_assignments, 0, assignments_size) != cudaSuccess) {
    return kmcudaRuntimeError;
  }
  result = kmeans_cuda_internal(
      samples_size, clusters_size, verbosity,
      reinterpret_cast<float*>(device_samples),
      reinterpret_cast<float*>(device_centroids),
      reinterpret_cast<uint32_t*>(device_ccounts),
      reinterpret_cast<uint32_t*>(device_assignments));
  if (result != kmcudaSuccess) {
    if (verbosity > 1) {
      printf("kmeans_cuda_internal failed: %s\n",
             cudaGetErrorString(cudaGetLastError()));
    }
    return result;
  }
  cumemcpy(centroids, device_centroids, centroids_size, cudaMemcpyDeviceToHost);
  cumemcpy(assignments, device_assignments, assignments_size, cudaMemcpyDeviceToHost);
  if (verbosity > 1) {
    printf("return kmcudaSuccess\n");
  }
  return kmcudaSuccess;
}
}