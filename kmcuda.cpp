#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cinttypes>
#include <cfloat>
#include <cmath>
#include <cassert>
#include <memory>

#include <cuda_runtime_api.h>
#ifdef PROFILE
#include <cuda_profiler_api.h>
#endif

#include "private.h"


static int check_args(
    float tolerance, float yinyang_t, uint32_t samples_size, uint16_t features_size,
    uint32_t clusters_size, uint32_t device, const float *samples, float *centroids,
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
  if (device == 0) {
    return kmcudaNoSuchDevice;
  }
  int devices = 0;
  cudaGetDeviceCount(&devices);
  if (device > (1u << devices)) {
    return kmcudaNoSuchDevice;
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

static std::vector<int> setup_devices(uint32_t device, int device_ptrs, int verbosity) {
  std::vector<int> devs;
  for (int dev = 0; device; dev++) {
    if (device & 1) {
      devs.push_back(dev);
      if (cudaSetDevice(dev) != cudaSuccess) {
        INFO("failed to validate device %d", dev);
        devs.pop_back();
      }
    }
    device >>= 1;
  }
  if (device_ptrs >= 0 && !(device & (1 << device_ptrs))) {
    // enable p2p for device_ptrs which is not in the devices list
    devs.push_back(device_ptrs);
  }
  if (devs.size() > 1) {
    for (int dev1 : devs) {
      for (int dev2 : devs) {
        if (dev1 <= dev2) {
          continue;
        }
        int access = 0;
        cudaDeviceCanAccessPeer(&access, dev1, dev2);
        if (!access) {
          INFO("warning: p2p %d <-> %d is impossible\n", dev1, dev2);
        }
      }
    }
    for (int dev : devs) {
      cudaSetDevice(dev);
      for (int odev : devs) {
        if (dev == odev) {
          continue;
        }
        if (cudaDeviceEnablePeerAccess(odev, 0) != cudaSuccess) {
          INFO("warning: failed to enable p2p on gpu #%d: %s\n", dev,
               CUERRSTR());
        }
      }
    }
  }
  if (device_ptrs >= 0 && !(device & (1 << device_ptrs))) {
    // remove device_ptrs - it is not in the devices list
    devs.pop_back();
  }
  return std::move(devs);
}

static KMCUDAResult print_memory_stats(const std::vector<int> &devs) {
  FOR_EACH_DEV(
    size_t free_bytes, total_bytes;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
      return kmcudaRuntimeError;
    }
    printf("GPU #%d memory: used %zu bytes (%.1f%%), free %zu bytes, "
           "total %zu bytes\n",
           dev, total_bytes - free_bytes,
           (total_bytes - free_bytes) * 100.0 / total_bytes,
           free_bytes, total_bytes);
  );
  return kmcudaSuccess;
}

extern "C" {

KMCUDAResult kmeans_init_centroids(
    KMCUDAInitMethod method, uint32_t samples_size, uint16_t features_size,
    uint32_t clusters_size, uint32_t seed, const std::vector<int> &devs,
    int device_ptrs, int32_t verbosity, const float *host_centroids,
    const udevptrs<float> &samples, udevptrs<float> *dists,
    udevptrs<float> *dev_sums, udevptrs<float> *centroids) {
  srand(seed);
  switch (method) {
    case kmcudaInitMethodImport:
      if (device_ptrs < 0) {
        CUMEMCPY_H2D_ASYNC(*centroids, 0, host_centroids,
                           clusters_size * features_size);
      } else {
        long long origin_devi = -1;
        FOR_EACH_DEVI(
          if (devs[devi] == device_ptrs) {
            origin_devi = devi;
          }
        );
        FOR_EACH_DEVI(
          if (static_cast<long long>(devi) != origin_devi) {
            CUCH(cudaMemcpyPeerAsync(
                (*centroids)[devi].get(), devs[devi], host_centroids,
                device_ptrs, clusters_size * features_size * sizeof(float)),
                 kmcudaMemoryCopyError);
          }
        );
      }
    case kmcudaInitMethodRandom:
      INFO("randomly picking initial centroids...\n");
      for (uint32_t c = 0; c < clusters_size; c++) {
        if ((c + 1) % 1000 == 0 || c == clusters_size - 1) {
          INFO("\rcentroid #%" PRIu32, c + 1);
          fflush(stdout);
          CUMEMCPY_D2D(
              *centroids, c * features_size, samples,
              (rand() % samples_size) * features_size, features_size);
        } else {
          CUMEMCPY_D2D_ASYNC(
              *centroids, c * features_size, samples,
              (rand() % samples_size) * features_size,
              features_size);
        }
      }
      break;
    case kmcudaInitMethodPlusPlus:
      INFO("performing kmeans++...\n");
      float smoke = NAN;
      uint32_t first_offset;
      while (smoke != smoke) {
        first_offset = (rand() % samples_size) * features_size;
        cudaSetDevice(devs[0]);
        CUCH(cudaMemcpy(&smoke, samples[0].get() + first_offset, sizeof(float),
                        cudaMemcpyDeviceToHost), kmcudaMemoryCopyError);
      }
      CUMEMCPY_D2D_ASYNC(*centroids, 0, samples, first_offset, features_size);
      std::unique_ptr<float[]> host_dists(new float[samples_size]);
      for (uint32_t i = 1; i < clusters_size; i++) {
        if (verbosity > 1 || (verbosity > 0 && (
              clusters_size < 100 || i % (clusters_size / 100) == 0))) {
          printf("\rstep %d", i);
          fflush(stdout);
        }
        float dist_sum = 0;
        RETERR(kmeans_cuda_plus_plus(
            samples_size, features_size, i, devs, verbosity, samples, centroids,
            dists, dev_sums, host_dists.get(), &dist_sum),
               DEBUG("\nkmeans_cuda_plus_plus failed\n"));
        assert(dist_sum == dist_sum);
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
        CUMEMCPY_D2D_ASYNC(*centroids, i * features_size, samples,
                           (j - 1) * features_size, features_size);
      }
      break;
  }
  INFO("\rdone            \n");
  return kmcudaSuccess;
}

int kmeans_cuda(
    KMCUDAInitMethod init, float tolerance, float yinyang_t,
    uint32_t samples_size, uint16_t features_size, uint32_t clusters_size,
    uint32_t seed, uint32_t device, int device_ptrs, int32_t verbosity,
    const float *samples, float *centroids, uint32_t *assignments) {
  DEBUG("arguments: %d %.3f %.2f %" PRIu32 " %" PRIu16 " %" PRIu32 " %" PRIu32
        " %" PRIu32 " %" PRIi32 " %p %p %p\n",
        init, tolerance, yinyang_t, samples_size, features_size, clusters_size,
        seed, device, verbosity, samples, centroids, assignments);
  RETERR(check_args(
      tolerance, yinyang_t, samples_size, features_size, clusters_size,
      device, samples, centroids, assignments));
  INFO("reassignments threshold: %" PRIu32 "\n", uint32_t(tolerance * samples_size));
  auto devs = setup_devices(device, device_ptrs, verbosity);
  if (devs.empty()) {
    return kmcudaNoSuchDevice;
  }
  udevptrs<float> device_samples;
  size_t device_samples_size = static_cast<size_t>(samples_size) * features_size;
  long long origin_devi = -1;
  FOR_EACH_DEVI(
    if (devs[devi] == device_ptrs) {
      device_samples.emplace_back(const_cast<float*>(samples), true);
      origin_devi = devi;
    } else {
      CUMALLOC_ONE(device_samples, device_samples_size);
    }
  );
  if (device_ptrs < 0) {
    CUMEMCPY_H2D_ASYNC(device_samples, 0, samples, device_samples_size);
  } else {
    FOR_EACH_DEVI(
      if (static_cast<long long>(devi) != origin_devi) {
        CUCH(cudaMemcpyPeerAsync(
            device_samples[devi].get(), devs[devi], samples,
            device_ptrs, device_samples_size * sizeof(float)),
             kmcudaMemoryCopyError);
      }
    );
  }
  udevptrs<float> device_centroids;
  size_t centroids_size = static_cast<size_t>(clusters_size) * features_size;
  FOR_EACH_DEV(
    if (dev == device_ptrs) {
      device_centroids.emplace_back(centroids, true);
    } else {
      CUMALLOC_ONE(device_centroids, centroids_size);
    }
  );
  udevptrs<uint32_t> device_assignments;
  FOR_EACH_DEV(
    if (dev == device_ptrs) {
      device_assignments.emplace_back(assignments, true);
    } else {
      CUMALLOC_ONE(device_assignments, samples_size);
    }
  );
  udevptrs<uint32_t> device_assignments_prev;
  CUMALLOC(device_assignments_prev, samples_size);
  udevptrs<uint32_t> device_ccounts;
  CUMALLOC(device_ccounts, clusters_size);

  uint32_t yy_groups_size = yinyang_t * clusters_size;
  DEBUG("yinyang groups: %" PRIu32 "\n", yy_groups_size);
  udevptrs<uint32_t> device_assignments_yy, device_passed_yy;
  udevptrs<float> device_bounds_yy, device_drifts_yy, device_centroids_yy;
  if (yy_groups_size >= 1) {
    CUMALLOC(device_assignments_yy, clusters_size);
    uint32_t max_length = max_distribute_length(
        samples_size, features_size * sizeof(float), devs);
    size_t yyb_size = static_cast<size_t>(max_length) * (yy_groups_size + 1);
    CUMALLOC(device_bounds_yy, yyb_size);
    CUMALLOC(device_drifts_yy, centroids_size + clusters_size);
    max_length = std::max(max_length, clusters_size + yy_groups_size);
    CUMALLOC(device_passed_yy, max_length);
    size_t yyc_size = yy_groups_size * features_size;
    if (yyc_size + (clusters_size + yy_groups_size) <= max_length) {
      for (auto &p : device_passed_yy) {
        device_centroids_yy.emplace_back(
            reinterpret_cast<float*>(p.get()), true);
      }
    } else {
      CUMALLOC(device_centroids_yy, yyc_size);
    }
  }

  if (verbosity > 1) {
    RETERR(print_memory_stats(devs));
  }
  RETERR(kmeans_cuda_setup(samples_size, features_size, clusters_size,
                           yy_groups_size, devs, verbosity),
         DEBUG("kmeans_cuda_setup failed: %s\n", CUERRSTR()));
  #ifdef PROFILE
  FOR_EACH_DEV(cudaProfilerStart());
  #endif
  RETERR(kmeans_init_centroids(
      init, samples_size, features_size, clusters_size, seed, devs, device_ptrs,
      verbosity, centroids, device_samples,
      reinterpret_cast<udevptrs<float>*>(&device_assignments),
      reinterpret_cast<udevptrs<float>*>(&device_assignments_prev),
      &device_centroids),
         DEBUG("kmeans_init_centroids failed: %s\n", CUERRSTR()));
  RETERR(kmeans_cuda_yy(
      tolerance, yy_groups_size, samples_size, clusters_size, features_size,
      devs, verbosity, device_samples, &device_centroids, &device_ccounts,
      &device_assignments_prev, &device_assignments, &device_assignments_yy,
      &device_centroids_yy, &device_bounds_yy, &device_drifts_yy, &device_passed_yy),
         DEBUG("kmeans_cuda_internal failed: %s\n", CUERRSTR()));
  #ifdef PROFILE
  FOR_EACH_DEV(cudaProfilerStop());
  #endif
  if (origin_devi < 0) {
    if (device_ptrs < 0) {
      CUCH(cudaMemcpy(centroids, device_centroids[devs.back()].get(),
                      centroids_size * sizeof(float), cudaMemcpyDeviceToHost),
           kmcudaMemoryCopyError);
      CUCH(cudaMemcpy(assignments, device_assignments[devs.back()].get(),
                      samples_size * sizeof(uint32_t), cudaMemcpyDeviceToHost),
           kmcudaMemoryCopyError);
    } else {
      CUCH(cudaMemcpyPeer(centroids, device_ptrs, device_centroids[devs.back()].get(),
                          devs[0], centroids_size * sizeof(float)),
           kmcudaMemoryCopyError);
      CUCH(cudaMemcpyPeer(assignments, device_ptrs, device_assignments[devs.back()].get(),
                          devs[0], samples_size * sizeof(uint32_t)),
           kmcudaMemoryCopyError);
    }
  }
  DEBUG("return kmcudaSuccess\n");
  return kmcudaSuccess;
}
}