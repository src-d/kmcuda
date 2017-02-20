#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cinttypes>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <map>
#include <memory>

#include <cuda_runtime_api.h>
#ifdef PROFILE
#include <cuda_profiler_api.h>
#endif

#include "private.h"


static KMCUDAResult check_kmeans_args(
    float tolerance,
    float yinyang_t,
    uint32_t samples_size,
    uint16_t features_size,
    uint32_t clusters_size,
    uint32_t device,
    bool fp16x2,
    int verbosity,
    const float *samples,
    float *centroids,
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
#if CUDA_ARCH < 60
  if (fp16x2) {
    INFO("CUDA device arch %d does not support fp16\n", CUDA_ARCH);
    return kmcudaInvalidArguments;
  }
#endif
  return kmcudaSuccess;
}

static std::vector<int> setup_devices(uint32_t device, int device_ptrs, int verbosity) {
  std::vector<int> devs;
  if (device == 0) {
    cudaGetDeviceCount(reinterpret_cast<int *>(&device));
    if (device == 0) {
      return devs;
    }
    device = (1u << device) - 1;
  }
  for (int dev = 0; device; dev++) {
    if (device & 1) {
      devs.push_back(dev);
      if (cudaSetDevice(dev) != cudaSuccess) {
        INFO("failed to cudaSetDevice(%d)\n", dev);
        devs.pop_back();
      }
      cudaDeviceProp props;
      auto err = cudaGetDeviceProperties(&props, dev);
      if (err != cudaSuccess) {
        INFO("failed to cudaGetDeviceProperties(%d): %s\n",
             dev, cudaGetErrorString(err));
        devs.pop_back();
      }
      if (props.major != (CUDA_ARCH / 10) || props.minor != (CUDA_ARCH % 10)) {
        INFO("compute capability mismatch for device %d: wanted %d.%d, have "
             "%d.%d\n>>>> you may want to build kmcuda with -DCUDA_ARCH=%d "
             "(refer to \"Building\" in README.md)\n",
             dev, CUDA_ARCH / 10, CUDA_ARCH % 10, props.major, props.minor,
             props.major * 10 + props.minor);
        devs.pop_back();
      }
    }
    device >>= 1;
  }
  bool p2p_dp = (device_ptrs >= 0 && !(device & (1 << device_ptrs)));
  if (p2p_dp) {
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
        auto err = cudaDeviceEnablePeerAccess(odev, 0);
        if (err == cudaErrorPeerAccessAlreadyEnabled) {
          INFO("p2p is already enabled on gpu #%d\n", dev);
        } else if (err != cudaSuccess) {
          INFO("warning: failed to enable p2p on gpu #%d: %s\n", dev,
               cudaGetErrorString(err));
        }
      }
    }
  }
  if (p2p_dp) {
    // remove device_ptrs - it is not in the devices list
    devs.pop_back();
  }
  return devs;
}

template <typename T>
static KMCUDAResult init_udevptrs(
    uint32_t length, uint32_t size_each,
    int32_t device_ptrs, const std::vector<int> &devs, int verbosity,
    const T *source, udevptrs<T> *dest, int32_t *origin_devi_ptr = nullptr) {
  size_t device_size = static_cast<size_t>(length) * size_each;
  int32_t origin_devi = -1;
  FOR_EACH_DEVI(
      if (devs[devi] == device_ptrs) {
        dest->emplace_back(const_cast<T*>(source), true);
        origin_devi = devi;
      } else {
        CUMALLOC_ONE(*dest, device_size, devs[devi]);
      }
  );
  if (origin_devi_ptr != nullptr) {
    *origin_devi_ptr = origin_devi;
  }
  if (device_ptrs < 0) {
    CUMEMCPY_H2D_ASYNC(*dest, 0, source, device_size);
  } else {
    FOR_EACH_DEVI(
        if (static_cast<int32_t>(devi) != origin_devi) {
          CUCH(cudaMemcpyPeerAsync(
              (*dest)[devi].get(), devs[devi], source,
              device_ptrs, device_size * sizeof(T)),
               kmcudaMemoryCopyError);
        }
    );
  }
  return kmcudaSuccess;
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
    KMCUDAInitMethod method, const void *init_params, uint32_t samples_size,
    uint16_t features_size, uint32_t clusters_size, KMCUDADistanceMetric metric,
    uint32_t seed, const std::vector<int> &devs, int device_ptrs, int fp16x2,
    int32_t verbosity, const float *host_centroids, const udevptrs<float> &samples,
    udevptrs<float> *dists, udevptrs<float> *aux, udevptrs<float> *centroids) {
  if (metric == kmcudaDistanceMetricCosine) {
    // 3 sanity checks
    float *probe;
    CUCH(cudaMallocManaged(reinterpret_cast<void**>(&probe),
                           static_cast<uint32_t>(features_size) * sizeof(float)),
         kmcudaMemoryAllocationFailure);
    unique_devptr<float> managed(probe);
    cudaSetDevice(devs[0]);
    for (uint32_t s : {0u, samples_size / 2, samples_size - 1}) {
      RETERR(cuda_extract_sample_t(
          s, samples_size, features_size, verbosity, samples[0].get(), probe));
      double norm = 0;
      #pragma omp simd
      for (uint16_t i = 0; i < features_size; i++) {
        float v = probe[i];
        norm += v * v;
      }
      if (norm > 1.0001 || norm < 0.9999) {
        INFO("error: angular distance: samples[%" PRIu32 "] has L2 norm = %f "
             "which is outside [0.9999, 1.0001]\n", s, norm);
        return kmcudaInvalidArguments;
      }
    }
  }

  srand(seed);
  switch (method) {
    case kmcudaInitMethodImport:
      if (device_ptrs < 0) {
        CUMEMCPY_H2D_ASYNC(*centroids, 0, host_centroids,
                           clusters_size * features_size);
      } else {
        int32_t origin_devi = -1;
        FOR_EACH_DEVI(
          if (devs[devi] == device_ptrs) {
            origin_devi = devi;
          }
        );
        FOR_EACH_DEVI(
          if (static_cast<int32_t>(devi) != origin_devi) {
            CUCH(cudaMemcpyPeerAsync(
                (*centroids)[devi].get(), devs[devi], host_centroids,
                device_ptrs, clusters_size * features_size * sizeof(float)),
                 kmcudaMemoryCopyError);
          }
        );
      }
      break;
    case kmcudaInitMethodRandom: {
      INFO("randomly picking initial centroids...\n");
      std::vector<uint32_t> chosen(samples_size);
      #pragma omp parallel for
      for (uint32_t s = 0; s < samples_size; s++) {
        chosen[s] = s;
      }
      std::random_shuffle(chosen.begin(), chosen.end());
      DEBUG("shuffle complete, copying to device(s)...\n");
      for (uint32_t c = 0; c < clusters_size; c++) {
        RETERR(cuda_copy_sample_t(
            chosen[c], c * features_size, samples_size, features_size, devs,
            verbosity, samples, centroids));
      }
      SYNC_ALL_DEVS;
      break;
    }
    case kmcudaInitMethodPlusPlus: {
      float smoke = NAN;
      uint32_t first_index;
      while (smoke != smoke) {
        first_index = rand() % samples_size;
        cudaSetDevice(devs[0]);
        CUCH(cudaMemcpy(&smoke, samples[0].get() + first_index, sizeof(float),
                        cudaMemcpyDeviceToHost), kmcudaMemoryCopyError);
      }
      RETERR(cuda_copy_sample_t(
            first_index, 0, samples_size, features_size, devs, verbosity,
            samples, centroids));
      INFO("performing kmeans++...\n");
      std::unique_ptr<float[]> host_dists(new float[samples_size]);
      if (verbosity > 2) {
        printf("kmeans++: dump %" PRIu32 " %" PRIu32 " %p\n",
               samples_size, features_size, host_dists.get());
        FOR_EACH_DEVI(
          printf("kmeans++: dev #%d: %p %p %p\n", devs[devi],
                 samples[devi].get(), (*centroids)[devi].get(),
                 (*dists)[devi].get());
        );
      }
      for (uint32_t i = 1; i < clusters_size; i++) {
        if (verbosity > 1 || (verbosity > 0 && (
              clusters_size < 100 || i % (clusters_size / 100) == 0))) {
          printf("\rstep %d", i);
          fflush(stdout);
        }
        atomic_float dist_sum = 0;
        RETERR(kmeans_cuda_plus_plus(
            samples_size, features_size, i, metric, devs, fp16x2, verbosity,
            samples, centroids, dists, host_dists.get(), &dist_sum),
               DEBUG("\nkmeans_cuda_plus_plus failed\n"));
        if (dist_sum != dist_sum) {
          assert(dist_sum == dist_sum);
          INFO("\ninternal bug inside kmeans_init_centroids: dist_sum is NaN\n");
        }
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
        if (j == 0 || j > samples_size) {
          assert(j > 0 && j <= samples_size);
          INFO("\ninternal bug in kmeans_init_centroids: j = %" PRIu32 "\n", j);
        }
        RETERR(cuda_copy_sample_t(
            j - 1, i * features_size, samples_size, features_size, devs,
            verbosity, samples, centroids));
      }
      SYNC_ALL_DEVS;
      break;
    }
    case kmcudaInitMethodAFKMC2: {
      uint32_t m = *reinterpret_cast<const uint32_t*>(init_params);
      if (m == 0) {
        m = 200;
      } else if (m > samples_size / 2) {
        INFO("afkmc2: m > %" PRIu32 " is not supported (got %" PRIu32 ")\n",
             samples_size / 2, m);
        return kmcudaInvalidArguments;
      }
      float smoke = NAN;
      uint32_t first_index;
      while (smoke != smoke) {
        first_index = rand() % samples_size;
        cudaSetDevice(devs[0]);
        CUCH(cudaMemcpy(&smoke, samples[0].get() + first_index, sizeof(float),
                        cudaMemcpyDeviceToHost), kmcudaMemoryCopyError);
      }
      INFO("afkmc2: calculating q (c0 = %" PRIu32 ")... ",
           first_index / features_size);
      RETERR(cuda_copy_sample_t(
            first_index, 0, samples_size, features_size, devs, verbosity,
            samples, centroids));
      auto q = std::unique_ptr<float[]>(new float[samples_size]);
      kmeans_cuda_afkmc2_calc_q(
          samples_size, features_size, first_index / features_size, metric,
          devs, fp16x2, verbosity, samples, dists, q.get());
      INFO("done\n");
      auto cand_ind = std::unique_ptr<uint32_t[]>(new uint32_t[m]);
      auto rand_a = std::unique_ptr<float[]>(new float[m]);
      auto p_cand = std::unique_ptr<float[]>(new float[m]);
      for (uint32_t k = 1; k < clusters_size; k++) {
        if (verbosity > 1 || (verbosity > 0 && (
              clusters_size < 100 || k % (clusters_size / 100) == 0))) {
          printf("\rstep %d", k);
          fflush(stdout);
        }
        RETERR(kmeans_cuda_afkmc2_random_step(
            k, m, seed, verbosity, dists->back().get(),
            reinterpret_cast<uint32_t*>(aux->back().get()),
            cand_ind.get(), aux->back().get() + m, rand_a.get()));
        RETERR(kmeans_cuda_afkmc2_min_dist(
            k, m, metric, fp16x2, verbosity, samples.back().get(),
            reinterpret_cast<uint32_t*>(aux->back().get()),
            centroids->back().get(), aux->back().get() + m, p_cand.get()));
        float curr_prob = 0;
        uint32_t curr_ind = 0;
        for (uint32_t j = 0; j < m; j++) {
          auto cand_prob = p_cand[j] / q[cand_ind[j]];
          if (curr_prob == 0 || cand_prob / curr_prob > rand_a[j]) {
            curr_ind = j;
            curr_prob = cand_prob;
          }
        }
        RETERR(cuda_copy_sample_t(
            cand_ind[curr_ind], k * features_size, samples_size, features_size, devs,
            verbosity, samples, centroids));
      }
      SYNC_ALL_DEVS;
      break;
    }
  }
  INFO("\rdone            \n");
  return kmcudaSuccess;
}

KMCUDAResult kmeans_cuda(
    KMCUDAInitMethod init, const void *init_params, float tolerance, float yinyang_t,
    KMCUDADistanceMetric metric, uint32_t samples_size, uint16_t features_size,
    uint32_t clusters_size, uint32_t seed, uint32_t device, int32_t device_ptrs,
    int32_t fp16x2, int32_t verbosity, const float *samples, float *centroids,
    uint32_t *assignments, float *average_distance) {
  DEBUG("arguments: %d %p %.3f %.2f %d %" PRIu32 " %" PRIu16 " %" PRIu32 " %"
        PRIu32 " %" PRIu32 " %d %" PRIi32 " %p %p %p %p\n", init, init_params,
        tolerance, yinyang_t, metric, samples_size, features_size, clusters_size,
        seed, device, fp16x2, verbosity, samples, centroids, assignments,
        average_distance);
  RETERR(check_kmeans_args(
      tolerance, yinyang_t, samples_size, features_size, clusters_size,
      device, fp16x2, verbosity, samples, centroids, assignments));
  INFO("reassignments threshold: %" PRIu32 "\n", uint32_t(tolerance * samples_size));
  uint32_t yy_groups_size = yinyang_t * clusters_size;
  DEBUG("yinyang groups: %" PRIu32 "\n", yy_groups_size);
  auto devs = setup_devices(device, device_ptrs, verbosity);
  if (devs.empty()) {
    return kmcudaNoSuchDevice;
  }
  udevptrs<float> device_samples;
  int32_t origin_devi;
  RETERR(init_udevptrs(samples_size, features_size, device_ptrs, devs,
                       verbosity, samples, &device_samples, &origin_devi));
  udevptrs<float> device_centroids;
  size_t centroids_size = static_cast<size_t>(clusters_size) * features_size;
  FOR_EACH_DEV(
    if (dev == device_ptrs) {
      device_centroids.emplace_back(centroids, true);
    } else {
      CUMALLOC_ONE(device_centroids, centroids_size, dev);
    }
  );
  udevptrs<uint32_t> device_assignments;
  FOR_EACH_DEV(
    if (dev == device_ptrs) {
      device_assignments.emplace_back(assignments, true);
    } else {
      CUMALLOC_ONE(device_assignments, samples_size, dev);
    }
  );
  udevptrs<uint32_t> device_assignments_prev;
  CUMALLOC(device_assignments_prev, samples_size);
  udevptrs<uint32_t> device_ccounts;
  CUMALLOC(device_ccounts, clusters_size);

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
    if (yyc_size <= max_length) {
      DEBUG("reusing passed_yy for centroids_yy\n");
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
  RETERR(cuda_transpose(
      samples_size, features_size, true, devs, verbosity, &device_samples));
  RETERR(kmeans_init_centroids(
      init, init_params, samples_size, features_size, clusters_size, metric,
      seed, devs, device_ptrs, fp16x2, verbosity, centroids, device_samples,
      reinterpret_cast<udevptrs<float>*>(&device_assignments),
      reinterpret_cast<udevptrs<float>*>(&device_assignments_prev),
      &device_centroids),
         DEBUG("kmeans_init_centroids failed: %s\n", CUERRSTR()));
  RETERR(kmeans_cuda_yy(
      tolerance, yy_groups_size, samples_size, clusters_size, features_size,
      metric, devs, fp16x2, verbosity, device_samples, &device_centroids, &device_ccounts,
      &device_assignments_prev, &device_assignments, &device_assignments_yy,
      &device_centroids_yy, &device_bounds_yy, &device_drifts_yy, &device_passed_yy),
         DEBUG("kmeans_cuda_yy failed: %s\n", CUERRSTR()));
  if (average_distance) {
    RETERR(kmeans_cuda_calc_average_distance(
        samples_size, features_size, metric, devs, fp16x2, verbosity,
        device_samples, device_centroids, device_assignments, average_distance),
           DEBUG("kmeans_cuda_calc_average_distance failed: %s\n", CUERRSTR()));
  }
  #ifdef PROFILE
  FOR_EACH_DEV(cudaProfilerStop());
  #endif
  if (origin_devi >= 0 || device_ptrs >= 0) {
    RETERR(cuda_transpose(
        samples_size, features_size, false, devs, verbosity, &device_samples));
  }
  if (origin_devi < 0) {
    if (device_ptrs < 0) {
      CUCH(cudaMemcpy(centroids, device_centroids[devs.back()].get(),
                      centroids_size * sizeof(float), cudaMemcpyDeviceToHost),
           kmcudaMemoryCopyError);
      CUCH(cudaMemcpy(assignments, device_assignments[devs.back()].get(),
                      samples_size * sizeof(uint32_t), cudaMemcpyDeviceToHost),
           kmcudaMemoryCopyError);
    } else {
      CUCH(cudaMemcpyPeerAsync(centroids, device_ptrs,
                               device_centroids[devs.size() - 1].get(),
                               devs.back(), centroids_size * sizeof(float)),
           kmcudaMemoryCopyError);
      CUCH(cudaMemcpyPeerAsync(assignments, device_ptrs,
                               device_assignments[devs.size() - 1].get(),
                               devs.back(), samples_size * sizeof(uint32_t)),
           kmcudaMemoryCopyError);
      SYNC_ALL_DEVS;
    }
  }
  DEBUG("return kmcudaSuccess\n");
  return kmcudaSuccess;
}

////////////--------------------------------------------------------------------
/// K-nn ///--------------------------------------------------------------------
////////////--------------------------------------------------------------------

static KMCUDAResult check_knn_args(
    uint16_t k, uint32_t samples_size, uint16_t features_size,
    uint32_t clusters_size, uint32_t device, int32_t fp16x2, int32_t verbosity,
    const float *samples, const float *centroids, const uint32_t *assignments,
    uint32_t *neighbors) {
  if (k == 0) {
    return kmcudaInvalidArguments;
  }
  if (clusters_size < 2 || clusters_size == UINT32_MAX) {
    return kmcudaInvalidArguments;
  }
  if (features_size == 0) {
    return kmcudaInvalidArguments;
  }
  if (samples_size < clusters_size) {
    return kmcudaInvalidArguments;
  }
  int devices = 0;
  cudaGetDeviceCount(&devices);
  if (device > (1u << devices)) {
    return kmcudaNoSuchDevice;
  }
  if (samples == nullptr || centroids == nullptr || assignments == nullptr ||
      neighbors == nullptr) {
    return kmcudaInvalidArguments;
  }
#if CUDA_ARCH < 60
  if (fp16x2) {
    INFO("CUDA device arch %d does not support fp16\n", CUDA_ARCH);
    return kmcudaInvalidArguments;
  }
#endif
  return kmcudaSuccess;
}

KMCUDAResult knn_cuda(
    uint16_t k, KMCUDADistanceMetric metric, uint32_t samples_size,
    uint16_t features_size, uint32_t clusters_size, uint32_t device,
    int32_t device_ptrs, int32_t fp16x2, int32_t verbosity,
    const float *samples, const float *centroids, const uint32_t *assignments,
    uint32_t *neighbors) {
  DEBUG("arguments: %" PRIu16 " %d %" PRIu32 " %" PRIu16 " %" PRIu32 " %" PRIu32
        " %" PRIi32 " %" PRIi32 " %" PRIi32 " %p %p %p %p\n",
        k, metric, samples_size, features_size, clusters_size, device,
        device_ptrs, fp16x2, verbosity, samples, centroids, assignments,
        neighbors);
  check_knn_args(k, samples_size, features_size, clusters_size, device, fp16x2,
                 verbosity, samples, centroids, assignments, neighbors);
  auto devs = setup_devices(device, device_ptrs, verbosity);
  if (devs.empty()) {
    return kmcudaNoSuchDevice;
  }
  udevptrs<float> device_samples;
  udevptrs<float> device_centroids;
  udevptrs<uint32_t> device_assignments;
  int32_t origin_devi;
  RETERR(init_udevptrs(samples_size, features_size, device_ptrs, devs,
                       verbosity, samples, &device_samples, &origin_devi));
  RETERR(init_udevptrs(clusters_size, features_size, device_ptrs, devs,
                       verbosity, centroids, &device_centroids));
  RETERR(init_udevptrs(samples_size, 1, device_ptrs, devs,
                       verbosity, assignments, &device_assignments));
  udevptrs<uint32_t> device_inv_asses, device_inv_asses_offsets;
  CUMALLOC(device_inv_asses, samples_size);
  CUMALLOC(device_inv_asses_offsets, clusters_size + 1);
  udevptrs<uint32_t> device_neighbors;
  auto nplan = distribute(samples_size, features_size * sizeof(float), devs);
  size_t neighbors_size = 0;
  for (auto &p : nplan) {
    auto length = std::get<1>(p);
    if (length > neighbors_size) {
      neighbors_size = length;
    }
  }
  neighbors_size *= k;
  FOR_EACH_DEVI(
    if (devs[devi] == device_ptrs) {
      if (knn_cuda_neighbors_mem_multiplier(k, devs[devi], 0) == 2) {
        INFO("warning: x2 memory is required for neighbors, using the "
             "external pointer and not able to check the size\n");
      }
      device_neighbors.emplace_back(
          neighbors + std::get<0>(nplan[devi]) * k, true);
    } else {
      CUMALLOC_ONE(
          device_neighbors,
          neighbors_size * knn_cuda_neighbors_mem_multiplier(k, devs[devi], 0),
          devs[devi]);
    }
  );
  udevptrs<float> device_cluster_distances;
  CUMALLOC(device_cluster_distances, clusters_size * clusters_size);
  udevptrs<float> device_sample_dists;
  if (clusters_size * clusters_size < samples_size) {
    CUMALLOC(device_sample_dists, samples_size);
  } else {
    DEBUG("using the centroid distances matrix as the sample distances temporary\n");
  }
  udevptrs<float> device_cluster_radiuses;
  CUMALLOC(device_cluster_radiuses, clusters_size);
  if (verbosity > 1) {
    RETERR(print_memory_stats(devs));
  }
  RETERR(knn_cuda_setup(samples_size, features_size, clusters_size,
                        devs, verbosity),
         DEBUG("knn_cuda_setup failed: %s\n", CUERRSTR()));
  #ifdef PROFILE
  FOR_EACH_DEV(cudaProfilerStart());
  #endif
  RETERR(cuda_transpose(
      samples_size, features_size, true, devs, verbosity, &device_samples));
  {
    INFO("initializing the inverse assignments...\n");
    auto asses_with_idxs = std::unique_ptr<std::tuple<uint32_t, uint32_t>[]>(
        new std::tuple<uint32_t, uint32_t>[samples_size]);
    if (device_ptrs < 0) {
      #pragma omp parallel for
      for (uint32_t s = 0; s < samples_size; s++) {
        asses_with_idxs[s] = std::make_tuple(assignments[s], s);
      }
    } else {
      auto asses_on_host =
          std::unique_ptr<uint32_t[]>(new uint32_t[samples_size]);
      cudaSetDevice(device_ptrs);
      CUCH(cudaMemcpy(
          asses_on_host.get(), assignments, samples_size * sizeof(uint32_t),
          cudaMemcpyDeviceToHost), kmcudaRuntimeError);
      #pragma omp parallel for
      for (uint32_t s = 0; s < samples_size; s++) {
        asses_with_idxs[s] = std::make_tuple(asses_on_host[s], s);
      }
    }
    std::sort(asses_with_idxs.get(), asses_with_idxs.get() + samples_size);
    auto asses_sorted =
        std::unique_ptr<uint32_t[]>(new uint32_t[samples_size]);
    auto asses_offsets =
        std::unique_ptr<uint32_t[]>(new uint32_t[clusters_size + 1]);
    uint32_t cls = 0;
    asses_offsets[0] = 0;
    for (uint32_t s = 0; s < samples_size; s++) {
      uint32_t newcls;
      std::tie(newcls, asses_sorted[s]) = asses_with_idxs[s];
      if (newcls != cls) {
        for (auto icls = newcls; icls > cls; icls--) {
          asses_offsets[icls] = s;
        }
        cls = newcls;
      }
    }
    for (auto icls = clusters_size; icls > cls; icls--) {
      asses_offsets[icls] = samples_size;
    }
    CUMEMCPY_H2D_ASYNC(device_inv_asses, 0, asses_sorted.get(), samples_size);
    CUMEMCPY_H2D_ASYNC(device_inv_asses_offsets, 0, asses_offsets.get(), clusters_size + 1);
  }
  CUMEMSET_ASYNC(device_cluster_distances, 0, clusters_size * clusters_size);
  if (clusters_size * clusters_size < samples_size) {
    CUMEMSET_ASYNC(device_sample_dists, 0, samples_size);
  }
  RETERR(knn_cuda_calc(
      k, samples_size, clusters_size, features_size, metric, devs, fp16x2,
      verbosity, device_samples, device_centroids, device_assignments,
      device_inv_asses, device_inv_asses_offsets, &device_cluster_distances,
      &device_sample_dists, &device_cluster_radiuses, &device_neighbors));
  #ifdef PROFILE
  FOR_EACH_DEV(cudaProfilerStop());
  #endif

  if (device_ptrs < 0) {
    FOR_EACH_DEVI(
      CUCH(cudaMemcpyAsync(neighbors + std::get<0>(nplan[devi]) * k,
                           device_neighbors[devi].get(),
                           std::get<1>(nplan[devi]) * k * sizeof(float),
                           cudaMemcpyDeviceToHost),
           kmcudaMemoryCopyError);
    );
  } else {
    RETERR(cuda_transpose(
        samples_size, features_size, false, devs, verbosity, &device_samples));
    FOR_EACH_DEVI(
      if (static_cast<int32_t>(devi) == origin_devi) {
        continue;
      }
      CUCH(cudaMemcpyPeerAsync(
          neighbors + std::get<0>(nplan[devi]) * k, device_ptrs,
          device_neighbors[devi].get(), devs[devi],
          std::get<1>(nplan[devi]) * k * sizeof(float)),
           kmcudaMemoryCopyError);
    );
  }
  SYNC_ALL_DEVS;
  DEBUG("return kmcudaSuccess\n");
  return kmcudaSuccess;
}

}  // extern "C"
