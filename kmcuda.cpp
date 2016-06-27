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
                               uint32_t clusters_size, uint32_t yy_groups_size,
                               uint32_t device, int32_t verbosity);

KMCUDAResult kmeans_cuda_yy(
    float tolerance, uint32_t yinyang_groups, uint32_t samples_size_,
    uint32_t clusters_size_, uint16_t features_size, int32_t verbosity,
    float *samples, float *centroids, uint32_t *ccounts,
    uint32_t *assignments_prev, uint32_t *assignments,
    uint32_t *assignments_yy, float *bounds_yy, float *drifts_yy);

}

#define cumemcpy(dst, src, size, flag) \
do { if (cudaMemcpy(dst, src, size, flag) != cudaSuccess) { \
  return kmcudaMemoryCopyError; \
} } while(false)

#define cumemcpy_async(dst, src, size, flag) \
do { if (cudaMemcpyAsync(dst, src, size, flag) != cudaSuccess) { \
  return kmcudaMemoryCopyError; \
} } while(false)

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
      if (verbosity > 0) {
        printf("randomly picking initial centroids...\n");
        fflush(stdout);
      }
      for (uint32_t c = 0; c < clusters_size; c++) {
        if ((c + 1) % 1000 == 0 || c == clusters_size - 1) {
          if (verbosity > 0) {
            printf("\rcentroid #%" PRIu32, c + 1);
          }
          cumemcpy(centroids + c * features_size,
                   samples + (rand() % samples_size) * features_size,
                   ssize, cudaMemcpyDeviceToDevice);
        } else {
          cumemcpy_async(centroids + c * features_size,
                         samples + (rand() % samples_size) * features_size,
                         ssize, cudaMemcpyDeviceToDevice);
        }
      }
      break;
    case kmcudaInitMethodPlusPlus:
      if (verbosity > 0) {
        printf("Performing kmeans++...\n");
        fflush(stdout);
      }
      cumemcpy(centroids, samples + (rand() % samples_size) * features_size,
               ssize, cudaMemcpyDeviceToDevice);
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
        auto result = kmeans_cuda_plus_plus(
            samples_size, i, samples, centroids, reinterpret_cast<float*>(dists),
            &dist_sum, &dev_sums);
        if (result != kmcudaSuccess) {
          if (verbosity > 1) {
            printf("\nkmeans_cuda_plus_plus failed\n");
          }
          return result;
        }
        assert(dist_sum == dist_sum);
        cumemcpy(host_dists.get(), dists, samples_size * sizeof(float),
                 cudaMemcpyDeviceToHost);
        double choice = ((rand() + .0) / RAND_MAX);
        uint32_t choice_approx = choice * samples_size;
        double choice_sum = choice * dist_sum;
        uint32_t j;
        {
          double dist_sum2 = 0;
          for (j = 0; j < samples_size && dist_sum2 < choice_sum; j++) {
            dist_sum2 += host_dists[j];
          }
        }
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
        cumemcpy_async(centroids + i * features_size,
                       samples + (j - 1) * features_size,
                       ssize, cudaMemcpyDeviceToDevice);
      }
      break;
  }

  if (verbosity > 0) {
    printf("\rdone            \n");
  }
  return kmcudaSuccess;
}

int kmeans_cuda(bool kmpp, float tolerance, float yinyang_t, uint32_t samples_size,
                uint16_t features_size, uint32_t clusters_size, uint32_t seed,
                uint32_t device, int32_t verbosity, const float *samples,
                float *centroids, uint32_t *assignments) {
  if (verbosity > 1) {
    printf("arguments: %d %.3f %.2f %" PRIu32 " %" PRIu16 " %" PRIu32 " %" PRIu32
           " %" PRIu32 " %" PRIi32 " %p %p %p\n",
           kmpp, tolerance, yinyang_t, samples_size, features_size, clusters_size,
           seed, device, verbosity, samples, centroids, assignments);
  }
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
  if (verbosity > 1) {
    printf("samples: %zu\n", device_samples_size);
  }
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
  if (verbosity > 1) {
    printf("centroids: %zu\n", centroids_size);
  }
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
  if (verbosity > 1) {
    printf("assignments: %zu\n", assignments_size);
  }
  if (cudaMalloc(&device_assignments, assignments_size)
      != cudaSuccess) {
    if (verbosity > 0) {
      printf("failed to allocate %zu bytes for assignments\n", assignments_size);
    }
    return kmcudaMemoryAllocationFailure;
  }
  unique_devptr device_assignments_sentinel(device_assignments);

  void *device_assignments_prev;
  if (verbosity > 1) {
    printf("assignments_prev: %zu\n", assignments_size);
  }
  if (cudaMalloc(&device_assignments_prev, assignments_size)
      != cudaSuccess) {
    if (verbosity > 0) {
      printf("failed to allocate %zu bytes for assignments_prev\n",
             assignments_size);
    }
    return kmcudaMemoryAllocationFailure;
  }
  unique_devptr device_assignments_prev_sentinel(device_assignments_prev);

  void *device_ccounts;
  size_t ccounts_size = clusters_size * sizeof(uint32_t);
  if (verbosity > 1) {
    printf("ccounts: %zu\n", ccounts_size);
  }
  if (cudaMalloc(&device_ccounts, ccounts_size)
      != cudaSuccess) {
    if (verbosity > 0) {
      printf("failed to allocate %zu bytes for ccounts\n", ccounts_size);
    }
    return kmcudaMemoryAllocationFailure;
  }
  unique_devptr device_ccounts_sentinel(device_ccounts);

  uint32_t yinyang_groups = yinyang_t * clusters_size;
  if (verbosity > 1) {
    printf("yinyang groups: %" PRIu32 "\n", yinyang_groups);
  }
  void *device_assignments_yy = NULL, *device_bounds_yy = NULL,
      *device_drifts_yy = NULL;
  if (yinyang_groups >= 1) {
    size_t yya_size = clusters_size * sizeof(uint32_t);
    if (verbosity > 1) {
      printf("yinyang assignments: %zu\n", yya_size);
    }
    if (cudaMalloc(&device_assignments_yy, yya_size) != cudaSuccess) {
      if (verbosity > 0) {
        printf("failed to allocate %zu bytes for yinyang assignments\n", yya_size);
      }
      return kmcudaMemoryAllocationFailure;
    }

    size_t yyb_size = samples_size;
    yyb_size *= (yinyang_groups + 1) * sizeof(float);
    if (verbosity > 1) {
      printf("yinyang bounds: %zu\n", yyb_size);
    }
    if (cudaMalloc(&device_bounds_yy, yyb_size) != cudaSuccess) {
      if (verbosity > 0) {
        printf("failed to allocate %zu bytes for yinyang bounds\n", yyb_size);
      }
      return kmcudaMemoryAllocationFailure;
    }

    size_t yyd_size = centroids_size + clusters_size * sizeof(float);
    if (verbosity > 1) {
      printf("yinyang drifts: %zu\n", yyd_size);
    }
    if (cudaMalloc(&device_drifts_yy, yyd_size) != cudaSuccess) {
      if (verbosity > 0) {
        printf("failed to allocate %zu bytes for yinyang drifts\n", yyd_size);
      }
      return kmcudaMemoryAllocationFailure;
    }
  }
  unique_devptr device_assignments_yinyang_sentinel(device_assignments_yy);
  unique_devptr device_bounds_yinyang_sentinel(device_bounds_yy);
  unique_devptr device_drifts_yinyang_sentinel(device_drifts_yy);

  if (verbosity > 1) {
    auto pmr = print_memory_stats();
    if (pmr != kmcudaSuccess) {
      return pmr;
    }
  }
  auto result = kmeans_cuda_setup(samples_size, features_size, clusters_size,
                                  yinyang_groups, device, verbosity);
  if (result != kmcudaSuccess) {
    if (verbosity > 1) {
      printf("kmeans_cuda_setup failed: %s\n",
             cudaGetErrorString(cudaGetLastError()));
    }
    return result;
  }
  result = kmeans_init_centroids(
      static_cast<KMCUDAInitMethod>(kmpp), samples_size, features_size,
      clusters_size, seed, verbosity, reinterpret_cast<float*>(device_samples),
      device_assignments, reinterpret_cast<float*>(device_centroids));
  if (result != kmcudaSuccess) {
    if (verbosity > 1) {
      printf("kmeans_init_centroids failed: %s\n",
             cudaGetErrorString(cudaGetLastError()));
    }
    return result;
  }
  result = kmeans_cuda_yy(
      tolerance, yinyang_groups, samples_size, clusters_size, features_size, verbosity,
      reinterpret_cast<float*>(device_samples),
      reinterpret_cast<float*>(device_centroids),
      reinterpret_cast<uint32_t*>(device_ccounts),
      reinterpret_cast<uint32_t*>(device_assignments_prev),
      reinterpret_cast<uint32_t*>(device_assignments),
      reinterpret_cast<uint32_t*>(device_assignments_yy),
      reinterpret_cast<float*>(device_bounds_yy),
      reinterpret_cast<float*>(device_drifts_yy));
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