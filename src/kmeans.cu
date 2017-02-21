#include <cassert>
#include <cstdio>
#include <cfloat>
#include <cinttypes>
#include <algorithm>
#include <memory>

#include <curand_kernel.h>

#include "private.h"
#include "metric_abstraction.h"
#include "tricks.cuh"

#define BS_KMPP 1024
#define BS_AFKMC2_Q 512
#define BS_AFKMC2_R 512
#define BS_AFKMC2_MDT 512
#define BS_LL_ASS 128
#define BS_LL_CNT 256
#define BS_YY_INI 128
#define BS_YY_GFL 512
#define BS_YY_LFL 512
#define BLOCK_SIZE 1024  // for all the rest of the kernels
#define SHMEM_AFKMC2_RC 8191  // in float-s, the actual value is +1
#define SHMEM_AFKMC2_MT 8192

#define YINYANG_GROUP_TOLERANCE 0.02
#define YINYANG_DRAFT_REASSIGNMENTS 0.11
#define YINYANG_REFRESH_EPSILON 1e-4

__device__ uint32_t d_changed_number;
__device__ uint32_t d_passed_number;
__constant__ uint32_t d_samples_size;
__constant__ uint32_t d_clusters_size;
__constant__ uint32_t d_yy_groups_size;
__constant__ int d_shmem_size;

//////////////////////----------------------------------------------------------
// Device functions //----------------------------------------------------------
//////////////////////----------------------------------------------------------

template <KMCUDADistanceMetric M, typename F>
__global__ void kmeans_plus_plus(
    const uint32_t offset, const uint32_t length, const uint32_t cc,
    const F *__restrict__ samples, const F *__restrict__ centroids,
    float *__restrict__ dists, atomic_float *__restrict__ dists_sum) {
  uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  float dist = 0;
  if (sample < length) {
    centroids += (cc - 1) * d_features_size;
    const uint32_t local_sample = sample + offset;
    if (_eq(samples[local_sample], samples[local_sample])) {
      dist = METRIC<M, F>::distance_t(
          samples, centroids, d_samples_size, local_sample);
    }
    float prev_dist;
    if (cc == 1 || dist < (prev_dist = dists[sample])) {
      dists[sample] = dist;
    } else {
      dist = prev_dist;
    }
  }
  dist = warpReduceSum(dist);
  if (threadIdx.x % 32 == 0) {
    atomicAdd(dists_sum, dist);
  }
}

template <KMCUDADistanceMetric M, typename F>
__global__ void kmeans_afkmc2_calc_q_dists(
    const uint32_t offset, const uint32_t length, uint32_t c1_index,
    const F *__restrict__ samples, float *__restrict__ dists,
    atomic_float *__restrict__ dsum) {
  volatile uint64_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  float dist = 0;
  if (sample < length) {
    sample += offset;
    extern __shared__ float shmem_afkmc2[];
    auto c1 = reinterpret_cast<F*>(shmem_afkmc2);
    uint16_t size_each = dupper(d_features_size, static_cast<uint16_t>(blockDim.x));
    for (uint16_t i = size_each * threadIdx.x;
         i < min(size_each * (threadIdx.x + 1), d_features_size); i++) {
      c1[i] = samples[static_cast<uint64_t>(c1_index) * d_features_size + i];
    }
    __syncthreads();
    dist = METRIC<M, F>::distance_t(samples, c1, d_samples_size, sample);
    dist *= dist;
    dists[sample] = dist;
  }
  float sum = warpReduceSum(dist);
  if (threadIdx.x % 32 == 0) {
    atomicAdd(dsum, sum);
  }
}

__global__ void kmeans_afkmc2_calc_q(
    const uint32_t offset, const uint32_t length,
    float dsum, float *__restrict__ q) {
  volatile uint64_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample >= length) {
    return;
  }
  sample += offset;
  q[sample] = 1 / (2.f * d_samples_size) + q[sample] / (2 * dsum);
}

__global__ void kmeans_afkmc2_random_step(
    const uint32_t m, const uint64_t seed, const uint64_t seq,
    const float *__restrict__ q, uint32_t *__restrict__ choices,
    float *__restrict__ samples) {
  volatile uint32_t ti = blockIdx.x * blockDim.x + threadIdx.x;
  curandState_t state;
  curand_init(seed, ti, seq, &state);
  float part = curand_uniform(&state);
  if (ti < m) {
    samples[ti] = curand_uniform(&state);
  }
  float accum = 0, corr = 0;
  bool found = false;
  __shared__ float shared_q[SHMEM_AFKMC2_RC + 1];
  int32_t *all_found = reinterpret_cast<int32_t*>(shared_q + SHMEM_AFKMC2_RC);
  *all_found = blockDim.x;
  const uint32_t size_each = dupper(
      static_cast<unsigned>(SHMEM_AFKMC2_RC), blockDim.x);
  for (uint32_t sample = 0; sample < d_samples_size; sample += SHMEM_AFKMC2_RC) {
    __syncthreads();
    if (*all_found == 0) {
      return;
    }
    for (uint32_t i = 0, si = threadIdx.x * size_each;
         i < size_each && (si = threadIdx.x * size_each + i) < SHMEM_AFKMC2_RC
         && (sample + si) < d_samples_size;
         i++) {
      shared_q[si] = q[sample + si];
    }
    __syncthreads();
    if (!found) {
      int i = 0;
      #pragma unroll 4
      for (; i < SHMEM_AFKMC2_RC && accum < part && sample + i < d_samples_size;
           i++) {
        // Kahan summation with inverted c
        float y = _add(corr, shared_q[i]);
        float t = accum + y;
        corr = y - (t - accum);
        accum = t;
      }
      if (accum >= part) {
        if (ti < m) {
          choices[ti] = sample + i - 1;
        }
        found = true;
        atomicSub(all_found, 1);
      }
    }
  }
}

template <KMCUDADistanceMetric M, typename F>
__global__ void kmeans_afkmc2_min_dist(
    const uint32_t m, const uint32_t k, const F *__restrict__ samples,
    const uint32_t *__restrict__ choices, const F *__restrict__ centroids,
    float *__restrict__ min_dists) {
  uint32_t chi = blockIdx.x * blockDim.x + threadIdx.x;
  if (chi >= m) {
    return;
  }
  float min_dist = FLT_MAX;
  for (uint32_t c = 0; c < k; c++) {
    float dist = METRIC<M, F>::distance_t(
        samples, centroids + c * d_features_size, d_samples_size, choices[chi]);
    if (dist < min_dist) {
      min_dist = dist;
    }
  }
  min_dists[chi] = min_dist * min_dist;
}

// min_dists must be set to FLT_MAX or +inf or NAN!
template <KMCUDADistanceMetric M, typename F>
__global__ void kmeans_afkmc2_min_dist_transposed(
    const uint32_t m, const uint32_t k, const F *__restrict__ samples,
    const uint32_t *__restrict__ choices, const F *__restrict__ centroids,
    float *__restrict__ min_dists) {
  uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
  extern __shared__ float shared_min_dists[];
  uint32_t size_each = dupper(m, blockDim.x);
  for (uint32_t i = size_each * threadIdx.x;
       i < min(size_each * (threadIdx.x + 1), m);
       i++) {
    shared_min_dists[i] = FLT_MAX;
  }
  __syncthreads();
  for (uint32_t chi = 0; chi < m; chi++) {
    float dist = FLT_MAX;
    if (c < k) {
      dist = METRIC<M, F>::distance_t(
          samples, centroids + c * d_features_size, d_samples_size, choices[chi]);
    }
    float warp_min = warpReduceMin(dist);
    warp_min *= warp_min;
    if (threadIdx.x % 32 == 0 && c < k) {
      atomicMin(shared_min_dists + chi, warp_min);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (uint32_t chi = 0; chi < m; chi++) {
      atomicMin(min_dists + chi, shared_min_dists[chi]);
    }
  }
}

template <KMCUDADistanceMetric M, typename F>
__global__ void kmeans_assign_lloyd_smallc(
    const uint32_t offset, const uint32_t length, const F *__restrict__ samples,
    const F *__restrict__ centroids, uint32_t *__restrict__ assignments_prev,
    uint32_t * __restrict__ assignments) {
  using HF = typename HALF<F>::type;
  uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample >= length) {
    return;
  }
  HF min_dist = _fmax<HF>();
  uint32_t nearest = UINT32_MAX;
  extern __shared__ float _shared_samples[];
  F *shared_samples = reinterpret_cast<F *>(_shared_samples);
  F *shared_centroids = shared_samples + blockDim.x * d_features_size;
  const uint32_t cstep = (d_shmem_size - blockDim.x * d_features_size) /
      (d_features_size + 1);
  F *csqrs = shared_centroids + cstep * d_features_size;
  const uint32_t size_each = cstep /
      min(blockDim.x, length - blockIdx.x * blockDim.x) + 1;
  const uint32_t local_sample = sample + offset;
  bool insane = _neq(samples[local_sample], samples[local_sample]);
  const uint32_t soffset = threadIdx.x * d_features_size;
  if (!insane) {
    for (uint64_t f = 0; f < d_features_size; f++) {
      shared_samples[soffset + f] = samples[f * d_samples_size + local_sample];
    }
  }

  for (uint32_t gc = 0; gc < d_clusters_size; gc += cstep) {
    uint32_t coffset = gc * d_features_size;
    __syncthreads();
    for (uint32_t i = 0; i < size_each; i++) {
      uint32_t ci = threadIdx.x * size_each + i;
      uint32_t local_offset = ci * d_features_size;
      uint32_t global_offset = coffset + local_offset;
      if (global_offset < d_clusters_size * d_features_size && ci < cstep) {
        csqrs[ci] = METRIC<M, F>::sum_squares(
            centroids + global_offset, shared_centroids + local_offset);
      }
    }
    __syncthreads();
    if (insane) {
      continue;
    }
    for (uint32_t c = gc; c < gc + cstep && c < d_clusters_size; c++) {
      F product = _const<F>(0), corr = _const<F>(0);
      coffset = (c - gc) * d_features_size;
      #pragma unroll 4
      for (int f = 0; f < d_features_size; f++) {
        F y = _fma(corr, shared_samples[soffset + f], shared_centroids[coffset + f]);
        F t = _add(product, y);
        corr = _sub(y, _sub(t, product));
        product = t;
      }
      HF dist = METRIC<M, F>::distance(_const<F>(0), csqrs[c - gc], product);
      if (_lt(dist, min_dist)) {
        min_dist = dist;
        nearest = c;
      }
    }
  }
  if (nearest == UINT32_MAX) {
    if (!insane) {
      printf("CUDA kernel kmeans_assign: nearest neighbor search failed for "
             "sample %" PRIu32 "\n", sample);
      return;
    } else {
      nearest = d_clusters_size;
    }
  }
  uint32_t ass = assignments[sample];
  assignments_prev[sample] = ass;
  if (ass != nearest) {
    assignments[sample] = nearest;
    atomicAggInc(&d_changed_number);
  }
}

template <KMCUDADistanceMetric M, typename F>
__global__ void kmeans_assign_lloyd(
    const uint32_t offset, const uint32_t length, const F *__restrict__ samples,
    const F *__restrict__ centroids, uint32_t *__restrict__ assignments_prev,
    uint32_t * __restrict__ assignments) {
  using HF = typename HALF<F>::type;
  uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample >= length) {
    return;
  }
  HF min_dist = _fmax<HF>();
  uint32_t nearest = UINT32_MAX;
  extern __shared__ float _shared_centroids[];
  F *shared_centroids = reinterpret_cast<F *>(_shared_centroids);
  const uint32_t cstep = d_shmem_size / (d_features_size + 1);
  F *csqrs = shared_centroids + cstep * d_features_size;
  const uint32_t size_each = cstep /
      min(blockDim.x, length - blockIdx.x * blockDim.x) + 1;
  const uint32_t local_sample = sample + offset;
  bool insane = _neq(samples[local_sample], samples[local_sample]);

  for (uint32_t gc = 0; gc < d_clusters_size; gc += cstep) {
    uint32_t coffset = gc * d_features_size;
    __syncthreads();
    for (uint32_t i = 0; i < size_each; i++) {
      uint32_t ci = threadIdx.x * size_each + i;
      uint32_t local_offset = ci * d_features_size;
      uint32_t global_offset = coffset + local_offset;
      if (global_offset < d_clusters_size * d_features_size && ci < cstep) {
        csqrs[ci] = METRIC<M, F>::sum_squares(
            centroids + global_offset, shared_centroids + local_offset);
      }
    }
    __syncthreads();
    if (insane) {
      continue;
    }
    for (uint32_t c = gc; c < gc + cstep && c < d_clusters_size; c++) {
      F product = _const<F>(0), corr = _const<F>(0);
      coffset = (c - gc) * d_features_size;
      #pragma unroll 4
      for (uint64_t f = 0; f < d_features_size; f++) {
        F y = _fma(corr,
                   samples[static_cast<uint64_t>(d_samples_size) * f + local_sample],
                   shared_centroids[coffset + f]);
        F t = _add(product, y);
        corr = _sub(y, _sub(t, product));
        product = t;
      }
      HF dist = METRIC<M, F>::distance(_const<F>(0), csqrs[c - gc], product);
      if (_lt(dist, min_dist)) {
        min_dist = dist;
        nearest = c;
      }
    }
  }
  if (nearest == UINT32_MAX) {
    if (!insane) {
      printf("CUDA kernel kmeans_assign: nearest neighbor search failed for "
             "sample %" PRIu32 "\n", sample);
      return;
    } else {
      nearest = d_clusters_size;
    }
  }
  uint32_t ass = assignments[sample];
  assignments_prev[sample] = ass;
  if (ass != nearest) {
    assignments[sample] = nearest;
    atomicAggInc(&d_changed_number);
  }
}

template <KMCUDADistanceMetric M, typename F>
__global__ void kmeans_adjust(
    const uint32_t coffset, const uint32_t length,
    const F *__restrict__ samples,
    const uint32_t *__restrict__ assignments_prev,
    const uint32_t *__restrict__ assignments,
    F *__restrict__ centroids, uint32_t *__restrict__ ccounts) {
  uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= length) {
    return;
  }
  c += coffset;
  uint32_t my_count = ccounts[c];
  {
    F fmy_count = _const<F>(my_count);
    centroids += c * d_features_size;
    for (int f = 0; f < d_features_size; f++) {
      centroids[f] = _mul(centroids[f], fmy_count);
    }
  }
  extern __shared__ uint32_t ass[];
  int step = d_shmem_size / 2;
  F corr = _const<F>(0);
  for (uint32_t sbase = 0; sbase < d_samples_size; sbase += step) {
    __syncthreads();
    if (threadIdx.x == 0) {
      int pos = sbase;
      for (int i = 0; i < step && sbase + i < d_samples_size; i++) {
        ass[2 * i] = assignments[pos + i];
        ass[2 * i + 1] = assignments_prev[pos + i];
      }
    }
    __syncthreads();
    for (int i = 0; i < step && sbase + i < d_samples_size; i++) {
      uint32_t this_ass = ass[2 * i];
      uint32_t  prev_ass = ass[2 * i + 1];
      int sign = 0;
      if (prev_ass == c && this_ass != c) {
        sign = -1;
        my_count--;
      } else if (prev_ass != c && this_ass == c) {
        sign = 1;
        my_count++;
      }
      if (sign != 0) {
        F fsign = _const<F>(sign);
        #pragma unroll 4
        for (uint64_t f = 0; f < d_features_size; f++) {
          F centroid = centroids[f];
          F y = _fma(corr,
                     samples[static_cast<uint64_t>(d_samples_size) * f + sbase + i],
                     fsign);
          F t = _add(centroid, y);
          corr = _sub(y, _sub(t, centroid));
          centroids[f] = t;
        }
      }
    }
  }
  // my_count can be 0 => we get NaN with L2 and never use this cluster again
  // this is a feature, not a bug
  METRIC<M, F>::normalize(my_count, centroids);
  ccounts[c] = my_count;
}

template <KMCUDADistanceMetric M, typename F>
__global__ void kmeans_yy_init(
    const uint32_t offset, const uint32_t length, const F *__restrict__ samples,
    const F *__restrict__ centroids, const uint32_t *__restrict__ assignments,
    const uint32_t *__restrict__ groups, float *__restrict__ volatile bounds) {
  volatile uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample >= length) {
    return;
  }
  for (uint32_t i = 0; i < d_yy_groups_size + 1; i++) {
    bounds[static_cast<uint64_t>(length) * i + sample] = FLT_MAX;
  }
  uint32_t nearest = assignments[sample];
  extern __shared__ float shared_memory[];
  F *volatile shared_centroids = reinterpret_cast<F*>(shared_memory);
  const uint32_t cstep = d_shmem_size / d_features_size;
  const uint32_t size_each = cstep /
      min(blockDim.x, length - blockIdx.x * blockDim.x) + 1;

  for (uint32_t gc = 0; gc < d_clusters_size; gc += cstep) {
    uint32_t coffset = gc * d_features_size;
    __syncthreads();
    for (uint32_t i = 0; i < size_each; i++) {
      uint32_t ci = threadIdx.x * size_each + i;
      uint32_t local_offset = ci * d_features_size;
      uint32_t global_offset = coffset + local_offset;
      if (global_offset < d_clusters_size * d_features_size && ci < cstep) {
        #pragma unroll 4
        for (int f = 0; f < d_features_size; f++) {
          shared_centroids[local_offset + f] = centroids[global_offset + f];
        }
      }
    }
    __syncthreads();

    for (uint32_t c = gc; c < gc + cstep && c < d_clusters_size; c++) {
      uint32_t group = groups[c];
      if (group >= d_yy_groups_size) {
        // this may happen if the centroid is insane (NaN)
        continue;
      }
      float dist = METRIC<M, F>::distance_t(
          samples, shared_centroids + (c - gc) * d_features_size,
          d_samples_size, sample + offset);
      if (c != nearest) {
        uint64_t gindex = static_cast<uint64_t>(length) * (1 + group) + sample;
        if (dist < bounds[gindex]) {
          bounds[gindex] = dist;
        }
      } else {
        bounds[sample] = dist;
      }
    }
  }
}

template <KMCUDADistanceMetric M, typename F>
__global__ void kmeans_yy_calc_drifts(
    const uint32_t offset, const uint32_t length,
    const F *__restrict__ centroids, F *__restrict__ drifts) {
  uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= length) {
    return;
  }
  c += offset;
  uint32_t coffset = c * d_features_size;
  (reinterpret_cast<float *>(drifts))[d_clusters_size * d_features_size + c] =
      METRIC<M, F>::distance(centroids + coffset, drifts + coffset);
}

__global__ void kmeans_yy_find_group_max_drifts(
    const uint32_t offset, const uint32_t length,
    const uint32_t *__restrict__ groups, float *__restrict__ drifts) {
  uint32_t group = blockIdx.x * blockDim.x + threadIdx.x;
  if (group >= length) {
    return;
  }
  group += offset;
  const uint32_t doffset = d_clusters_size * d_features_size;
  const uint32_t step = d_shmem_size / 2;
  const uint32_t size_each = d_shmem_size /
      (2 * min(blockDim.x, length - blockIdx.x * blockDim.x));
  extern __shared__ uint32_t shmem[];
  float *cd = (float *)shmem;
  uint32_t *cg = shmem + step;
  float my_max = -FLT_MAX;
  for (uint32_t offset = 0; offset < d_clusters_size; offset += step) {
    __syncthreads();
    for (uint32_t i = 0; i < size_each; i++) {
      uint32_t local_offset = threadIdx.x * size_each + i;
      uint32_t global_offset = offset + local_offset;
      if (global_offset < d_clusters_size && local_offset < step) {
        cd[local_offset] = drifts[doffset + global_offset];
        cg[local_offset] = groups[global_offset];
      }
    }
    __syncthreads();
    for (uint32_t i = 0; i < step; i++) {
      if (cg[i] == group) {
        float d = cd[i];
        if (my_max < d) {
          my_max = d;
        }
      }
    }
  }
  drifts[group] = my_max;
}

template <KMCUDADistanceMetric M, typename F>
__global__ void kmeans_yy_global_filter(
    const uint32_t offset, const uint32_t length, const F *__restrict__ samples,
    const F *__restrict__ centroids, const uint32_t *__restrict__ groups,
    const float *__restrict__ drifts, const uint32_t *__restrict__ assignments,
    uint32_t *__restrict__ assignments_prev, float *__restrict__ bounds,
    uint32_t *__restrict__ passed) {
  volatile uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample >= length) {
    return;
  }
  uint32_t cluster = assignments[sample];
  assignments_prev[sample] = cluster;
  float upper_bound = bounds[sample];
  uint32_t doffset = d_clusters_size * d_features_size;
  float cluster_drift = drifts[doffset + cluster];
  upper_bound += cluster_drift;
  float min_lower_bound = FLT_MAX;
  for (uint32_t g = 0; g < d_yy_groups_size; g++) {
    uint64_t gindex = static_cast<uint64_t>(length) * (1 + g) + sample;
    float lower_bound = bounds[gindex] - drifts[g];
    bounds[gindex] = lower_bound;
    if (lower_bound < min_lower_bound) {
      min_lower_bound = lower_bound;
    }
  }
  // group filter try #1
  if (min_lower_bound >= upper_bound) {
    bounds[sample] = upper_bound;
    return;
  }
  upper_bound = 0;
  upper_bound = METRIC<M, F>::distance_t(
      samples, centroids + cluster * d_features_size,
      d_samples_size, sample + offset);
  bounds[sample] = upper_bound;
  // group filter try #2
  if (min_lower_bound >= upper_bound) {
    return;
  }
  // d'oh!
  passed[atomicAggInc(&d_passed_number)] = sample;
}

template <KMCUDADistanceMetric M, typename F>
__global__ void kmeans_yy_local_filter(
    const uint32_t offset, const uint32_t length, const F *__restrict__ samples,
    const uint32_t *__restrict__ passed, const F *__restrict__ centroids,
    const uint32_t *__restrict__ groups, const float *__restrict__ drifts,
    uint32_t *__restrict__ assignments, float *__restrict__ bounds) {
  volatile uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample >= d_passed_number) {
    return;
  }
  sample = passed[sample];
  float upper_bound = bounds[sample];
  uint32_t cluster = assignments[sample];
  uint32_t doffset = d_clusters_size * d_features_size;
  float min_dist = upper_bound, second_min_dist = FLT_MAX;
  uint32_t nearest = cluster;
  extern __shared__ float shared_memory[];
  F *volatile shared_centroids = reinterpret_cast<F*>(shared_memory);
  const uint32_t cstep = d_shmem_size / d_features_size;
  const uint32_t size_each = cstep /
      min(blockDim.x, d_passed_number - blockIdx.x * blockDim.x) + 1;

  for (uint32_t gc = 0; gc < d_clusters_size; gc += cstep) {
    uint32_t coffset = gc * d_features_size;
    __syncthreads();
    for (uint32_t i = 0; i < size_each; i++) {
      uint32_t ci = threadIdx.x * size_each + i;
      uint32_t local_offset = ci * d_features_size;
      uint32_t global_offset = coffset + local_offset;
      if (global_offset < d_clusters_size * d_features_size && ci < cstep) {
        #pragma unroll 4
        for (int f = 0; f < d_features_size; f++) {
          shared_centroids[local_offset + f] = centroids[global_offset + f];
        }
      }
    }
    __syncthreads();

    for (uint32_t c = gc; c < gc + cstep && c < d_clusters_size; c++) {
      if (c == cluster) {
        continue;
      }
      uint32_t group = groups[c];
      if (group >= d_yy_groups_size) {
        // this may happen if the centroid is insane (NaN)
        continue;
      }
      float lower_bound = bounds[
          static_cast<uint64_t>(length) * (1 + group) + sample];
      if (lower_bound >= upper_bound) {
        if (lower_bound < second_min_dist) {
          second_min_dist = lower_bound;
        }
        continue;
      }
      lower_bound += drifts[group] - drifts[doffset + c];
      if (second_min_dist < lower_bound) {
        continue;
      }
      float dist = METRIC<M, F>::distance_t(
          samples, shared_centroids + (c - gc) * d_features_size,
          d_samples_size, sample + offset);
      if (dist < min_dist) {
        second_min_dist = min_dist;
        min_dist = dist;
        nearest = c;
      } else if (dist < second_min_dist) {
        second_min_dist = dist;
      }
    }
  }
  uint32_t nearest_group = groups[nearest];
  uint32_t previous_group = groups[cluster];
  bounds[static_cast<uint64_t>(length) * (1 + nearest_group) + sample] =
      second_min_dist;
  if (nearest_group != previous_group) {
    uint64_t gindex =
        static_cast<uint64_t>(length) * (1 + previous_group) + sample;
    float pb = bounds[gindex];
    if (pb > upper_bound) {
      bounds[gindex] = upper_bound;
    }
  }
  bounds[sample] = min_dist;
  if (cluster != nearest) {
    assignments[sample] = nearest;
    atomicAggInc(&d_changed_number);
  }
}

template <KMCUDADistanceMetric M, typename F>
__global__ void kmeans_calc_average_distance(
    uint32_t offset, uint32_t length, const F *__restrict__ samples,
    const F *__restrict__ centroids, const uint32_t *__restrict__ assignments,
    atomic_float *distance) {
  volatile uint64_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  float dist = 0;
  if (sample < length) {
    sample += offset;
    dist = METRIC<M, F>::distance_t(
        samples, centroids + assignments[sample] * d_features_size,
        d_samples_size, sample);
  }
  float sum = warpReduceSum(dist);
  if (threadIdx.x % 32 == 0) {
    atomicAdd(distance, sum);
  }
}

////////////////////------------------------------------------------------------
// Host functions //------------------------------------------------------------
////////////////////------------------------------------------------------------

static int check_changed(int iter, float tolerance, uint32_t h_samples_size,
                         const std::vector<int> &devs, int32_t verbosity) {
  uint32_t overall_changed = 0;
  FOR_EACH_DEV(
    uint32_t my_changed = 0;
    CUCH(cudaMemcpyFromSymbol(&my_changed, d_changed_number, sizeof(my_changed)),
         kmcudaMemoryCopyError);
    overall_changed += my_changed;
  );
  INFO("iteration %d: %" PRIu32 " reassignments\n", iter, overall_changed);
  if (overall_changed <= tolerance * h_samples_size) {
    return -1;
  }
  assert(overall_changed <= h_samples_size);
  uint32_t zero = 0;
  FOR_EACH_DEV(
    CUCH(cudaMemcpyToSymbolAsync(d_changed_number, &zero, sizeof(zero)),
         kmcudaMemoryCopyError);
  );
  return kmcudaSuccess;
}

static KMCUDAResult prepare_mem(
    uint32_t h_samples_size, uint32_t h_clusters_size, bool resume,
    const std::vector<int> &devs, int verbosity, udevptrs<uint32_t> *ccounts,
    udevptrs<uint32_t> *assignments, udevptrs<uint32_t> *assignments_prev,
    std::vector<uint32_t> *shmem_sizes) {
  uint32_t zero = 0;
  shmem_sizes->clear();
  FOR_EACH_DEVI(
    uint32_t h_shmem_size;
    CUCH(cudaMemcpyFromSymbol(&h_shmem_size, d_shmem_size, sizeof(h_shmem_size)),
         kmcudaMemoryCopyError);
    shmem_sizes->push_back(h_shmem_size * sizeof(uint32_t));
    CUCH(cudaMemcpyToSymbolAsync(d_changed_number, &zero, sizeof(zero)),
         kmcudaMemoryCopyError);
    if (!resume) {
      CUCH(cudaMemsetAsync((*ccounts)[devi].get(), 0,
                           h_clusters_size * sizeof(uint32_t)),
           kmcudaRuntimeError);
      CUCH(cudaMemsetAsync((*assignments)[devi].get(), 0xff,
                           h_samples_size * sizeof(uint32_t)),
           kmcudaRuntimeError);
      CUCH(cudaMemsetAsync((*assignments_prev)[devi].get(), 0xff,
                           h_samples_size * sizeof(uint32_t)),
           kmcudaRuntimeError);
    }
  );
  return kmcudaSuccess;
}

extern "C" {

KMCUDAResult kmeans_cuda_setup(
    uint32_t h_samples_size, uint16_t h_features_size, uint32_t h_clusters_size,
    uint32_t h_yy_groups_size, const std::vector<int> &devs, int32_t verbosity) {
  FOR_EACH_DEV(
    CUCH(cudaMemcpyToSymbol(d_samples_size, &h_samples_size, sizeof(h_samples_size)),
         kmcudaMemoryCopyError);
    CUCH(cudaMemcpyToSymbol(d_features_size, &h_features_size, sizeof(h_features_size)),
         kmcudaMemoryCopyError);
    CUCH(cudaMemcpyToSymbol(d_clusters_size, &h_clusters_size, sizeof(h_clusters_size)),
         kmcudaMemoryCopyError);
    CUCH(cudaMemcpyToSymbol(d_yy_groups_size, &h_yy_groups_size, sizeof(h_yy_groups_size)),
         kmcudaMemoryCopyError);
    cudaDeviceProp props;
    CUCH(cudaGetDeviceProperties(&props, dev), kmcudaRuntimeError);
    int h_shmem_size = static_cast<int>(props.sharedMemPerBlock);
    DEBUG("GPU #%" PRIu32 " has %d bytes of shared memory per block\n",
          dev, h_shmem_size);
    h_shmem_size /= sizeof(uint32_t);
    CUCH(cudaMemcpyToSymbol(d_shmem_size, &h_shmem_size, sizeof(h_shmem_size)),
         kmcudaMemoryCopyError);
  );
  return kmcudaSuccess;
}

KMCUDAResult kmeans_cuda_plus_plus(
    uint32_t h_samples_size, uint32_t h_features_size, uint32_t cc,
    KMCUDADistanceMetric metric, const std::vector<int> &devs, int fp16x2,
    int verbosity, const udevptrs<float> &samples, udevptrs<float> *centroids,
    udevptrs<float> *dists, float *host_dists, atomic_float *dist_sum) {
  auto plan = distribute(h_samples_size, h_features_size * sizeof(float), devs);
  uint32_t max_len = 0;
  for (auto &p : plan) {
    auto len = std::get<1>(p);
    if (max_len < len) {
      max_len = len;
    }
  }
  udevptrs<atomic_float> dev_dists;
  CUMALLOC(dev_dists, sizeof(atomic_float));
  CUMEMSET_ASYNC(dev_dists, 0, sizeof(atomic_float));
  FOR_EACH_DEVI(
    uint32_t offset, length;
    std::tie(offset, length) = plan[devi];
    if (length == 0) {
      continue;
    }
    dim3 block(BS_KMPP, 1, 1);
    dim3 grid(upper(length, block.x), 1, 1);
    KERNEL_SWITCH(kmeans_plus_plus, <<<grid, block>>>(
        offset, length, cc,
        reinterpret_cast<const F*>(samples[devi].get()),
        reinterpret_cast<const F*>((*centroids)[devi].get()),
        (*dists)[devi].get(), dev_dists[devi].get()));
  );
  uint32_t dist_offset = 0;
  FOR_EACH_DEVI(
    uint32_t offset, length;
    std::tie(offset, length) = plan[devi];
    dim3 block(BS_KMPP, 1, 1);
    dim3 grid(upper(length, block.x), 1, 1);
    CUCH(cudaMemcpyAsync(
        host_dists + offset, (*dists)[devi].get(),
        length * sizeof(float), cudaMemcpyDeviceToHost), kmcudaMemoryCopyError);
    dist_offset += grid.x;
  );
  atomic_float sum = 0;
  FOR_EACH_DEVI(
    if (std::get<1>(plan[devi]) == 0) {
      continue;
    }
    atomic_float hdist;
    CUCH(cudaMemcpy(&hdist, dev_dists[devi].get(), sizeof(atomic_float),
                    cudaMemcpyDeviceToHost),
         kmcudaMemoryCopyError);
    sum += hdist;
  );
  *dist_sum = sum;
  return kmcudaSuccess;
}

KMCUDAResult kmeans_cuda_afkmc2_calc_q(
    uint32_t h_samples_size, uint32_t h_features_size, uint32_t firstc,
    KMCUDADistanceMetric metric, const std::vector<int> &devs, int fp16x2,
    int verbosity, const udevptrs<float> &samples, udevptrs<float> *d_q,
    float *h_q) {
  auto plan = distribute(h_samples_size, h_features_size * sizeof(float), devs);
  udevptrs<atomic_float> dev_dists;
  CUMALLOC(dev_dists, sizeof(atomic_float));
  CUMEMSET_ASYNC(dev_dists, 0, sizeof(atomic_float));
  FOR_EACH_DEVI(
    uint32_t offset, length;
    std::tie(offset, length) = plan[devi];
    if (length == 0) {
      continue;
    }
    dim3 block(BS_AFKMC2_Q, 1, 1);
    dim3 grid(upper(length, block.x), 1, 1);
    int shmem = std::max(
        BS_AFKMC2_Q, static_cast<int>(h_features_size)) * sizeof(float);
    KERNEL_SWITCH(kmeans_afkmc2_calc_q_dists,
                  <<<grid, block, shmem>>>(
        offset, length, firstc,
        reinterpret_cast<const F*>(samples[devi].get()),
        (*d_q)[devi].get(), dev_dists[devi].get()));

  );
  atomic_float dists_sum = 0;
  FOR_EACH_DEVI(
    if (std::get<1>(plan[devi]) == 0) {
      continue;
    }
    atomic_float hdist;
    CUCH(cudaMemcpy(&hdist, dev_dists[devi].get(), sizeof(atomic_float),
                    cudaMemcpyDeviceToHost),
         kmcudaMemoryCopyError);
    dists_sum += hdist;
  );
  FOR_EACH_DEVI(
    uint32_t offset, length;
    std::tie(offset, length) = plan[devi];
    if (length == 0) {
      continue;
    }
    dim3 block(BLOCK_SIZE, 1, 1);
    dim3 grid(upper(length, block.x), 1, 1);
    kmeans_afkmc2_calc_q<<<grid, block>>>(
        offset, length, dists_sum, (*d_q)[devi].get());
  );
  FOR_EACH_DEVI(
    uint32_t offset, length;
    std::tie(offset, length) = plan[devi];
    CUCH(cudaMemcpyAsync(h_q + offset, (*d_q)[devi].get() + offset,
                         length * sizeof(float), cudaMemcpyDeviceToHost),
         kmcudaMemoryCopyError);
    FOR_OTHER_DEVS(
      CUP2P(d_q, offset, length);
    );
  );
  SYNC_ALL_DEVS;
  return kmcudaSuccess;
}

KMCUDAResult kmeans_cuda_afkmc2_random_step(
    uint32_t k, uint32_t m, uint64_t seed, int verbosity, const float *q,
    uint32_t *d_choices, uint32_t *h_choices, float *d_samples, float *h_samples) {
  dim3 block(BS_AFKMC2_R, 1, 1);
  dim3 grid(upper(m, block.x), 1, 1);
  kmeans_afkmc2_random_step<<<grid, block>>>(
      m, seed, k, q, d_choices, d_samples);
  CUCH(cudaMemcpy(h_choices, d_choices, m * sizeof(uint32_t),
                  cudaMemcpyDeviceToHost),
       kmcudaMemoryCopyError);
  CUCH(cudaMemcpy(h_samples, d_samples, m * sizeof(float),
                  cudaMemcpyDeviceToHost),
       kmcudaMemoryCopyError);
  return kmcudaSuccess;
}

KMCUDAResult kmeans_cuda_afkmc2_min_dist(
    uint32_t k, uint32_t m, KMCUDADistanceMetric metric, int fp16x2,
    int32_t verbosity, const float *samples, const uint32_t *choices,
    const float *centroids, float *d_min_dists, float *h_min_dists) {
  if (m > k || m > SHMEM_AFKMC2_MT) {
    dim3 block(BLOCK_SIZE, 1, 1);
    dim3 grid(upper(m, block.x), 1, 1);
    KERNEL_SWITCH(kmeans_afkmc2_min_dist, <<<grid, block>>>(
        m, k, reinterpret_cast<const F*>(samples), choices,
        reinterpret_cast<const F*>(centroids), d_min_dists));
  } else {
    dim3 block(BS_AFKMC2_MDT, 1, 1);
    dim3 grid(upper(k, block.x), 1, 1);
    CUCH(cudaMemsetAsync(d_min_dists, 0xff, m * sizeof(float)),
         kmcudaRuntimeError);
    KERNEL_SWITCH(kmeans_afkmc2_min_dist_transposed,
        <<<grid, block, m * sizeof(float)>>>(
        m, k, reinterpret_cast<const F*>(samples), choices,
        reinterpret_cast<const F*>(centroids), d_min_dists));
  }
  CUCH(cudaMemcpy(h_min_dists, d_min_dists, m * sizeof(float),
                  cudaMemcpyDeviceToHost),
       kmcudaMemoryCopyError);
  return kmcudaSuccess;
}

KMCUDAResult kmeans_cuda_lloyd(
    float tolerance, uint32_t h_samples_size, uint32_t h_clusters_size,
    uint16_t h_features_size, KMCUDADistanceMetric metric, bool resume,
    const std::vector<int> &devs, int fp16x2, int32_t verbosity,
    const udevptrs<float> &samples, udevptrs<float> *centroids,
    udevptrs<uint32_t> *ccounts, udevptrs<uint32_t> *assignments_prev,
    udevptrs<uint32_t> *assignments, int *iterations = nullptr) {
  std::vector<uint32_t> shmem_sizes;
  RETERR(prepare_mem(h_samples_size, h_clusters_size, resume, devs, verbosity,
                     ccounts, assignments, assignments_prev, &shmem_sizes));
  auto plans = distribute(h_samples_size, h_features_size * sizeof(float), devs);
  auto planc = distribute(h_clusters_size, h_features_size * sizeof(float), devs);
  if (verbosity > 1) {
    print_plan("plans", plans);
    print_plan("planc", planc);
  }
  dim3 sblock(BS_LL_ASS, 1, 1);
  dim3 cblock(BS_LL_CNT, 1, 1);
  for (int iter = 1; ; iter++) {
    if (!resume || iter > 1) {
      FOR_EACH_DEVI(
        uint32_t offset, length;
        std::tie(offset, length) = plans[devi];
        if (length == 0) {
          continue;
        }
        dim3 sgrid(upper(length, sblock.x), 1, 1);
        int shmem_size = shmem_sizes[devi];
        int64_t ssqrmem = sblock.x * h_features_size * sizeof(float);
        if (shmem_size > ssqrmem && shmem_size - ssqrmem >=
            static_cast<int>((h_features_size + 1) * sizeof(float))) {
          KERNEL_SWITCH(kmeans_assign_lloyd_smallc, <<<sgrid, sblock, shmem_size>>>(
              offset, length,
              reinterpret_cast<const F*>(samples[devi].get()),
              reinterpret_cast<const F*>((*centroids)[devi].get()),
              (*assignments_prev)[devi].get() + offset,
              (*assignments)[devi].get() + offset));
        } else {
          KERNEL_SWITCH(kmeans_assign_lloyd, <<<sgrid, sblock, shmem_size>>>(
              offset, length,
              reinterpret_cast<const F*>(samples[devi].get()),
              reinterpret_cast<const F*>((*centroids)[devi].get()),
              (*assignments_prev)[devi].get() + offset,
              (*assignments)[devi].get() + offset));
        }
      );
      FOR_EACH_DEVI(
        uint32_t offset, length;
        std::tie(offset, length) = plans[devi];
        if (length == 0) {
          continue;
        }
        FOR_OTHER_DEVS(
          CUP2P(assignments_prev, offset, length);
          CUP2P(assignments, offset, length);
        );
      );
      int status = check_changed(iter, tolerance, h_samples_size, devs, verbosity);
      if (status < kmcudaSuccess) {
        if (iterations) {
          *iterations = iter;
        }
        return kmcudaSuccess;
      }
      if (status != kmcudaSuccess) {
        return static_cast<KMCUDAResult>(status);
      }
    }
    FOR_EACH_DEVI(
      uint32_t offset, length;
      std::tie(offset, length) = planc[devi];
      if (length == 0) {
        continue;
      }
      dim3 cgrid(upper(length, cblock.x), 1, 1);
      KERNEL_SWITCH(kmeans_adjust, <<<cgrid, cblock, shmem_sizes[devi]>>>(
          offset, length, reinterpret_cast<const F*>(samples[devi].get()),
          (*assignments_prev)[devi].get(), (*assignments)[devi].get(),
          reinterpret_cast<F*>((*centroids)[devi].get()), (*ccounts)[devi].get()));
    );
    FOR_EACH_DEVI(
      uint32_t offset, length;
      std::tie(offset, length) = planc[devi];
      if (length == 0) {
        continue;
      }
      FOR_OTHER_DEVS(
        CUP2P(ccounts, offset, length);
        CUP2P(centroids, offset * h_features_size, length * h_features_size);
      );
    );
  }
}

KMCUDAResult kmeans_cuda_yy(
    float tolerance, uint32_t h_yy_groups_size, uint32_t h_samples_size,
    uint32_t h_clusters_size, uint16_t h_features_size, KMCUDADistanceMetric metric,
    const std::vector<int> &devs, int fp16x2, int32_t verbosity,
    const udevptrs<float> &samples, udevptrs<float> *centroids,
    udevptrs<uint32_t> *ccounts, udevptrs<uint32_t> *assignments_prev,
    udevptrs<uint32_t> *assignments, udevptrs<uint32_t> *assignments_yy,
    udevptrs<float> *centroids_yy, udevptrs<float> *bounds_yy,
    udevptrs<float> *drifts_yy, udevptrs<uint32_t> *passed_yy) {
  if (h_yy_groups_size == 0 || YINYANG_DRAFT_REASSIGNMENTS <= tolerance) {
    if (verbosity > 0) {
      if (h_yy_groups_size == 0) {
        printf("too few clusters for this yinyang_t => Lloyd\n");
      } else {
        printf("tolerance is too high (>= %.2f) => Lloyd\n",
               YINYANG_DRAFT_REASSIGNMENTS);
      }
    }
    return kmeans_cuda_lloyd(
        tolerance, h_samples_size, h_clusters_size, h_features_size, metric,
        false, devs, fp16x2, verbosity, samples, centroids, ccounts,
        assignments_prev, assignments);
  }
  INFO("running Lloyd until reassignments drop below %" PRIu32 "\n",
       (uint32_t)(YINYANG_DRAFT_REASSIGNMENTS * h_samples_size));
  int iter;
  RETERR(kmeans_cuda_lloyd(
      YINYANG_DRAFT_REASSIGNMENTS, h_samples_size, h_clusters_size,
      h_features_size, metric, false, devs, fp16x2, verbosity, samples,
      centroids, ccounts, assignments_prev, assignments, &iter));
  if (check_changed(iter, tolerance, h_samples_size, devs, 0) < kmcudaSuccess) {
    return kmcudaSuccess;
  }
  // map each centroid to yinyang group -> assignments_yy
  FOR_EACH_DEV(
    CUCH(cudaMemcpyToSymbol(d_samples_size, &h_clusters_size, sizeof(h_samples_size)),
         kmcudaMemoryCopyError);
    CUCH(cudaMemcpyToSymbol(d_clusters_size, &h_yy_groups_size, sizeof(h_yy_groups_size)),
         kmcudaMemoryCopyError);
  );
  {
    udevptrs<float> tmpbufs, tmpbufs2;
    auto max_slength = max_distribute_length(
        h_samples_size, h_features_size * sizeof(float), devs);
    for (auto &pyy : *passed_yy) {
      // max_slength is guaranteed to be greater than or equal to
      // h_clusters_size + h_yy_groups_size
      tmpbufs.emplace_back(reinterpret_cast<float*>(pyy.get()) +
          max_slength - h_clusters_size - h_yy_groups_size, true);
      tmpbufs2.emplace_back(tmpbufs.back().get() + h_clusters_size, true);
    }
    RETERR(cuda_transpose(
        h_clusters_size, h_features_size, true, devs, verbosity, centroids));
    RETERR(kmeans_init_centroids(
        kmcudaInitMethodPlusPlus, nullptr, h_clusters_size, h_features_size,
        h_yy_groups_size, metric, 0, devs, -1, fp16x2, verbosity, nullptr,
        *centroids, &tmpbufs, nullptr, centroids_yy),
           INFO("kmeans_init_centroids() failed for yinyang groups: %s\n",
                cudaGetErrorString(cudaGetLastError())));
    RETERR(kmeans_cuda_lloyd(
        YINYANG_GROUP_TOLERANCE, h_clusters_size, h_yy_groups_size, h_features_size,
        metric, false, devs, fp16x2, verbosity, *centroids, centroids_yy,
        reinterpret_cast<udevptrs<uint32_t> *>(&tmpbufs2),
        reinterpret_cast<udevptrs<uint32_t> *>(&tmpbufs), assignments_yy));
    RETERR(cuda_transpose(
        h_clusters_size, h_features_size, false, devs, verbosity, centroids));
  }
  FOR_EACH_DEV(
    CUCH(cudaMemcpyToSymbol(d_samples_size, &h_samples_size, sizeof(h_samples_size)),
         kmcudaMemoryCopyError);
    CUCH(cudaMemcpyToSymbol(d_clusters_size, &h_clusters_size, sizeof(h_clusters_size)),
         kmcudaMemoryCopyError);
  );
  std::vector<uint32_t> shmem_sizes;
  RETERR(prepare_mem(h_samples_size, h_clusters_size, true, devs, verbosity,
                     ccounts, assignments, assignments_prev, &shmem_sizes));
  dim3 siblock(BS_YY_INI, 1, 1);
  dim3 sgblock(BS_YY_GFL, 1, 1);
  dim3 slblock(BS_YY_LFL, 1, 1);
  dim3 cblock(BS_LL_CNT, 1, 1);
  dim3 gblock(BLOCK_SIZE, 1, 1);
  auto plans = distribute(h_samples_size, h_features_size * sizeof(float), devs);
  auto planc = distribute(h_clusters_size, h_features_size * sizeof(float), devs);
  auto plang = distribute(h_yy_groups_size, h_features_size * sizeof(float), devs);
  if (verbosity > 1) {
    print_plan("plans", plans);
    print_plan("planc", planc);
    print_plan("plang", plang);
  }
  bool refresh = true;
  uint32_t h_passed_number = 0;
  for (; ; iter++) {
    if (!refresh) {
      int status = check_changed(iter, tolerance, h_samples_size, devs, verbosity);
      if (status < kmcudaSuccess) {
        return kmcudaSuccess;
      }
      if (status != kmcudaSuccess) {
        return static_cast<KMCUDAResult>(status);
      }
      FOR_EACH_DEV(
        uint32_t local_passed;
        CUCH(cudaMemcpyFromSymbol(&local_passed, d_passed_number,
                                  sizeof(h_passed_number)),
             kmcudaMemoryCopyError);
        h_passed_number += local_passed;
      );
      DEBUG("passed number: %" PRIu32 "\n", h_passed_number);
      if (1.f - (h_passed_number + 0.f) / h_samples_size < YINYANG_REFRESH_EPSILON) {
        refresh = true;
      }
      h_passed_number = 0;
    }
    if (refresh) {
      INFO("refreshing Yinyang bounds...\n");
      FOR_EACH_DEVI(
        uint32_t offset, length;
        std::tie(offset, length) = plans[devi];
        if (length == 0) {
          continue;
        }
        dim3 sigrid(upper(length, siblock.x), 1, 1);
        KERNEL_SWITCH(kmeans_yy_init, <<<sigrid, siblock, shmem_sizes[devi]>>>(
            offset, length,
            reinterpret_cast<const F*>(samples[devi].get()),
            reinterpret_cast<const F*>((*centroids)[devi].get()),
            (*assignments)[devi].get() + offset,
            (*assignments_yy)[devi].get(), (*bounds_yy)[devi].get()));
      );
      refresh = false;
    }
    CUMEMCPY_D2D_ASYNC(*drifts_yy, 0, *centroids, 0, h_clusters_size * h_features_size);
    FOR_EACH_DEVI(
      uint32_t offset, length;
      std::tie(offset, length) = planc[devi];
      if (length == 0) {
        continue;
      }
      dim3 cgrid(upper(length, cblock.x), 1, 1);
      KERNEL_SWITCH(kmeans_adjust, <<<cgrid, cblock, shmem_sizes[devi]>>>(
          offset, length, reinterpret_cast<const F*>(samples[devi].get()),
          (*assignments_prev)[devi].get(), (*assignments)[devi].get(),
          reinterpret_cast<F*>((*centroids)[devi].get()), (*ccounts)[devi].get()));
    );
    FOR_EACH_DEVI(
      uint32_t offset, length;
      std::tie(offset, length) = planc[devi];
      if (length == 0) {
        continue;
      }
      FOR_OTHER_DEVS(
        CUP2P(ccounts, offset, length);
        CUP2P(centroids, offset * h_features_size, length * h_features_size);
      );
    );
    FOR_EACH_DEVI(
      uint32_t offset, length;
      std::tie(offset, length) = planc[devi];
      if (length == 0) {
        continue;
      }
      dim3 cgrid(upper(length, cblock.x), 1, 1);
      KERNEL_SWITCH(kmeans_yy_calc_drifts, <<<cgrid, cblock>>>(
          offset, length, reinterpret_cast<const F*>((*centroids)[devi].get()),
          reinterpret_cast<F*>((*drifts_yy)[devi].get())));
    );
    FOR_EACH_DEVI(
      uint32_t offset, length;
      std::tie(offset, length) = planc[devi];
      if (length == 0) {
        continue;
      }
      FOR_OTHER_DEVS(
        CUP2P(drifts_yy, h_clusters_size * h_features_size + offset, length);
      );
    );
    FOR_EACH_DEVI(
      uint32_t offset, length;
      std::tie(offset, length) = plang[devi];
      if (length == 0) {
        continue;
      }
      dim3 ggrid(upper(length, gblock.x), 1, 1);
      kmeans_yy_find_group_max_drifts<<<ggrid, gblock, shmem_sizes[devi]>>>(
          offset, length, (*assignments_yy)[devi].get(),
          (*drifts_yy)[devi].get());
    );
    FOR_EACH_DEVI(
      uint32_t offset, length;
      std::tie(offset, length) = plang[devi];
      if (length == 0) {
        continue;
      }
      FOR_OTHER_DEVS(
        CUP2P(drifts_yy, offset, length);
      );
    );
    FOR_EACH_DEV(
      CUCH(cudaMemcpyToSymbolAsync(d_passed_number, &h_passed_number,
                                   sizeof(h_passed_number)),
           kmcudaMemoryCopyError);
    );
    FOR_EACH_DEVI(
      uint32_t offset, length;
      std::tie(offset, length) = plans[devi];
      if (length == 0) {
        continue;
      }
      dim3 sggrid(upper(length, sgblock.x), 1, 1);
      KERNEL_SWITCH(kmeans_yy_global_filter, <<<sggrid, sgblock>>>(
          offset, length,
          reinterpret_cast<const F*>(samples[devi].get()),
          reinterpret_cast<const F*>((*centroids)[devi].get()),
          (*assignments_yy)[devi].get(), (*drifts_yy)[devi].get(),
          (*assignments)[devi].get() + offset, (*assignments_prev)[devi].get() + offset,
          (*bounds_yy)[devi].get(), (*passed_yy)[devi].get()));
      dim3 slgrid(upper(length, slblock.x), 1, 1);
      KERNEL_SWITCH(kmeans_yy_local_filter, <<<slgrid, slblock, shmem_sizes[devi]>>>(
          offset, length, reinterpret_cast<const F*>(samples[devi].get()),
          (*passed_yy)[devi].get(), reinterpret_cast<const F*>((*centroids)[devi].get()),
          (*assignments_yy)[devi].get(), (*drifts_yy)[devi].get(),
          (*assignments)[devi].get() + offset, (*bounds_yy)[devi].get()));
    );
    FOR_EACH_DEVI(
      uint32_t offset, length;
      std::tie(offset, length) = plans[devi];
      if (length == 0) {
        continue;
      }
      FOR_OTHER_DEVS(
        CUP2P(assignments_prev, offset, length);
        CUP2P(assignments, offset, length);
      );
    );
  }
}

KMCUDAResult kmeans_cuda_calc_average_distance(
    uint32_t h_samples_size, uint16_t h_features_size,
    KMCUDADistanceMetric metric, const std::vector<int> &devs, int fp16x2,
    int32_t verbosity, const udevptrs<float> &samples,
    const udevptrs<float> &centroids, const udevptrs<uint32_t> &assignments,
    float *average_distance) {
  INFO("calculating the average distance...\n");
  auto plans = distribute(h_samples_size, h_features_size * sizeof(float), devs);
  udevptrs<atomic_float> dev_dists;
  CUMALLOC(dev_dists, sizeof(atomic_float));
  CUMEMSET_ASYNC(dev_dists, 0, sizeof(atomic_float));
  FOR_EACH_DEVI(
    uint32_t offset, length;
    std::tie(offset, length) = plans[devi];
    if (length == 0) {
      continue;
    }
    dim3 block(BLOCK_SIZE, 1, 1);
    dim3 grid(upper(length, block.x), 1, 1);
    KERNEL_SWITCH(kmeans_calc_average_distance,
                  <<<grid, block, block.x * sizeof(float)>>>(
        offset, length, reinterpret_cast<const F*>(samples[devi].get()),
        reinterpret_cast<const F*>(centroids[devi].get()),
        assignments[devi].get(), dev_dists[devi].get()));
  );
  atomic_float sum = 0;
  FOR_EACH_DEVI(
    atomic_float hdist;
    CUCH(cudaMemcpy(&hdist, dev_dists[devi].get(), sizeof(atomic_float),
                    cudaMemcpyDeviceToHost),
         kmcudaMemoryCopyError);
    sum += hdist;
  );
  *average_distance = sum / h_samples_size;
  return kmcudaSuccess;
}

}  // extern "C"
