#include <cassert>
#include <cstdio>
#include <cfloat>
#include <cinttypes>
#include <cinttypes>
#include <algorithm>
#include <memory>

#include "private.h"

#define BS_KMPP 512
#define BS_LL_ASS 256
#define BS_LL_CNT 256
#define BS_YY_INI 256
#define BS_YY_GFL 512
#define BS_YY_LFL 512
#define BLOCK_SIZE 1024  // for all the rest of the kernels

#define YINYANG_GROUP_TOLERANCE 0.02
#define YINYANG_DRAFT_REASSIGNMENTS 0.11
#define YINYANG_REFRESH_EPSILON 1e-4

__device__ uint32_t d_changed_number;
__device__ uint32_t d_passed_number;
__constant__ uint32_t d_samples_size;
__constant__ uint16_t d_features_size;
__constant__ uint32_t d_clusters_size;
__constant__ uint32_t d_yy_groups_size;
__constant__ int d_shmem_size;

__global__ void kmeans_plus_plus(
    const uint32_t border, const uint32_t cc, const float *__restrict__ samples,
    const float *__restrict__ centroids, float *__restrict__ dists,
    float *__restrict__ dist_sums) {
  uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample >= border) {
    return;
  }
  samples += static_cast<uint64_t>(sample) * d_features_size;
  extern __shared__ float local_dists[];
  float dist = 0;
  if (samples[0] == samples[0]) {
    uint32_t coffset = (cc - 1) * d_features_size;
    #pragma unroll 4
    for (uint16_t f = 0; f < d_features_size; f++) {
      float d = samples[f] - centroids[coffset + f];
      dist += d * d;
    }
    dist = sqrt(dist);
  }
  float prev_dist = dists[sample];
  if (dist < prev_dist || cc == 1) {
    dists[sample] = dist;
  } else {
    dist = prev_dist;
  }
  local_dists[threadIdx.x] = dist;
  uint32_t end = blockDim.x;
  if ((blockIdx.x + 1) * blockDim.x > border) {
    end = border - blockIdx.x * blockDim.x;
  }
  __syncthreads();
  if (threadIdx.x % 16 == 0) {
    float psum = 0;
    for (uint32_t i = threadIdx.x; i < end && i < threadIdx.x + 16; i++) {
      psum += local_dists[i];
    }
    local_dists[threadIdx.x] = psum;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float block_sum = 0;
    for (uint32_t i = 0; i < end; i += 16) {
      block_sum += local_dists[i];
    }
    dist_sums[blockIdx.x] = block_sum;
  }
}

__global__ void kmeans_assign_lloyd_smallc(
    const uint32_t border, const float *__restrict__ samples,
    const float *__restrict__ centroids, uint32_t *__restrict__ assignments_prev,
    uint32_t * __restrict__ assignments) {
  uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample >= border) {
    return;
  }
  samples += static_cast<uint64_t>(sample) * d_features_size;
  float min_dist = FLT_MAX;
  uint32_t nearest = UINT32_MAX;
  extern __shared__ float shared_samples[];
  float *shared_centroids = shared_samples + blockDim.x * d_features_size;
  const uint32_t cstep = (d_shmem_size - blockDim.x * d_features_size) /
      (d_features_size + 1);
  float *csqrs = shared_centroids + cstep * d_features_size;
  const uint32_t size_each = cstep /
      min(blockDim.x, border - blockIdx.x * blockDim.x) + 1;
  bool insane = samples[0] != samples[0];
  float ssqr = 0;
  const uint32_t soffset = threadIdx.x * d_features_size;
  if (!insane) {
    #pragma unroll 4
    for (int f = 0; f < d_features_size; f++) {
      float v = samples[f];
      shared_samples[soffset + f] = v;
      ssqr += v * v;
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
        float csqr = 0;
        #pragma unroll 4
        for (int f = 0; f < d_features_size; f++) {
          float v = centroids[global_offset + f];
          shared_centroids[local_offset + f] = v;
          csqr += v * v;
        }
        csqrs[ci] = csqr;
      }
    }
    __syncthreads();
    if (insane) {
      continue;
    }
    for (uint32_t c = gc; c < gc + cstep && c < d_clusters_size; c++) {
      float pair = 0;
      coffset = (c - gc) * d_features_size;
      #pragma unroll 4
      for (int f = 0; f < d_features_size; f++) {
        pair += shared_samples[soffset + f] * shared_centroids[coffset + f];
      }
      float dist = ssqr + csqrs[c - gc] - 2 * pair;
      if (dist < min_dist) {
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
    atomicAdd(&d_changed_number, 1);
  }
}

__global__ void kmeans_assign_lloyd(
    const uint32_t border, const float *__restrict__ samples,
    const float *__restrict__ centroids, uint32_t *__restrict__ assignments_prev,
    uint32_t * __restrict__ assignments) {
  uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample >= border) {
    return;
  }
  samples += static_cast<uint64_t>(sample) * d_features_size;
  float min_dist = FLT_MAX;
  uint32_t nearest = UINT32_MAX;
  extern __shared__ float shared_centroids[];
  const uint32_t cstep = d_shmem_size / (d_features_size + 1);
  float *csqrs = shared_centroids + cstep * d_features_size;
  const uint32_t size_each = cstep /
      min(blockDim.x, border - blockIdx.x * blockDim.x) + 1;
  bool insane = samples[0] != samples[0];
  float ssqr = 0;
  if (!insane) {
    #pragma unroll 4
    for (int f = 0; f < d_features_size; f++) {
      float v = samples[f];
      ssqr += v * v;
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
        float csqr = 0;
        #pragma unroll 4
        for (int f = 0; f < d_features_size; f++) {
          float v = centroids[global_offset + f];
          shared_centroids[local_offset + f] = v;
          csqr += v * v;
        }
        csqrs[ci] = csqr;
      }
    }
    __syncthreads();
    if (insane) {
      continue;
    }
    for (uint32_t c = gc; c < gc + cstep && c < d_clusters_size; c++) {
      float pair = 0;
      coffset = (c - gc) * d_features_size;
      #pragma unroll 4
      for (int f = 0; f < d_features_size; f++) {
        pair += samples[f] * shared_centroids[coffset + f];
      }
      float dist = ssqr + csqrs[c - gc] - 2 * pair;
      if (dist < min_dist) {
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
    atomicAdd(&d_changed_number, 1);
  }
}

__global__ void kmeans_adjust(
    const uint32_t border, const uint32_t coffset,
    const float *__restrict__ samples,
    const uint32_t *__restrict__ assignments_prev,
    const uint32_t *__restrict__ assignments,
    float *__restrict__ centroids, uint32_t *__restrict__ ccounts) {
  uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= border) {
    return;
  }
  c += coffset;
  uint32_t my_count = ccounts[c];
  centroids += c * d_features_size;
  for (int f = 0; f < d_features_size; f++) {
    centroids[f] *= my_count;
  }
  extern __shared__ uint32_t ass[];
  int step = d_shmem_size / 2;
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
      float sign = 0;
      if (prev_ass == c && this_ass != c) {
        sign = -1;
        my_count--;
      } else if (prev_ass != c && this_ass == c) {
        sign = 1;
        my_count++;
      }
      if (sign != 0) {
        uint64_t soffset = sbase + i;
        soffset *= d_features_size;
        #pragma unroll 4
        for (int f = 0; f < d_features_size; f++) {
          centroids[f] += samples[soffset + f] * sign;
        }
      }
    }
  }
  // my_count can be 0 => we get NaN and never use this cluster again
  // this is a feature, not a bug
  #pragma unroll 4
  for (int f = 0; f < d_features_size; f++) {
    centroids[f] /= my_count;
  }
  ccounts[c] = my_count;
}

__global__ void kmeans_yy_init(
    const uint32_t border, const float *__restrict__ samples,
    const float *__restrict__ centroids, const uint32_t *__restrict__ assignments,
    const uint32_t *__restrict__ groups, float *__restrict__ bounds) {
  uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample >= border) {
    return;
  }
  bounds += static_cast<uint64_t>(sample) * (d_yy_groups_size + 1);
  for (uint32_t i = 0; i < d_yy_groups_size + 1; i++) {
    bounds[i] = FLT_MAX;
  }
  bounds++;
  samples += static_cast<uint64_t>(sample) * d_features_size;
  uint32_t nearest = assignments[sample];
  extern __shared__ float shared_centroids[];
  const uint32_t cstep = d_shmem_size / d_features_size;
  const uint32_t size_each = cstep /
      min(blockDim.x, border - blockIdx.x * blockDim.x) + 1;

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
      float dist = 0;
      coffset = (c - gc) * d_features_size;
      uint32_t group = groups[c];
      if (group >= d_yy_groups_size) {
        // this may happen if the centroid is insane (NaN)
        continue;
      }
      #pragma unroll 4
      for (int f = 0; f < d_features_size; f++) {
        float d = samples[f] - shared_centroids[coffset + f];
        dist += d * d;
      }
      dist = sqrt(dist);
      if (c != nearest) {
        if (dist < bounds[group]) {
          bounds[group] = dist;
        }
      } else {
        bounds[-1] = dist;
      }
    }
  }
}

__global__ void kmeans_yy_calc_drifts(
    const uint32_t border, const uint32_t offset,
    const float *__restrict__ centroids, float *__restrict__ drifts) {
  uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= border) {
    return;
  }
  c += offset;
  uint32_t coffset = c * d_features_size;
  float sum = 0;
  #pragma unroll 4
  for (uint32_t f = coffset; f < coffset + d_features_size; f++) {
    float d = centroids[f] - drifts[f];
    sum += d * d;
  }
  drifts[d_clusters_size * d_features_size + c] = sqrt(sum);
}

__global__ void kmeans_yy_find_group_max_drifts(
    const uint32_t border, const uint32_t offset,
    const uint32_t *__restrict__ groups, float *__restrict__ drifts) {
  uint32_t group = blockIdx.x * blockDim.x + threadIdx.x;
  if (group >= border) {
    return;
  }
  group += offset;
  const uint32_t doffset = d_clusters_size * d_features_size;
  const uint32_t size_each = d_shmem_size /
      (2 * min(blockDim.x, border - blockIdx.x * blockDim.x));
  const uint32_t step = size_each * blockDim.x;
  extern __shared__ uint32_t shmem[];
  float *cd = (float *)shmem;
  uint32_t *cg = shmem + d_shmem_size / 2;
  float my_max = FLT_MIN;
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

__global__ void kmeans_yy_global_filter(
    const uint32_t border, const float *__restrict__ samples,
    const float *__restrict__ centroids, const uint32_t *__restrict__ groups,
    const float *__restrict__ drifts, const uint32_t *__restrict__ assignments,
    uint32_t *__restrict__ assignments_prev, float *__restrict__ bounds,
    uint32_t *__restrict__ passed) {
  uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample >= border) {
    return;
  }
  bounds += static_cast<uint64_t>(sample) * (d_yy_groups_size + 1);
  uint32_t cluster = assignments[sample];
  assignments_prev[sample] = cluster;
  float upper_bound = bounds[0];
  uint32_t doffset = d_clusters_size * d_features_size;
  float cluster_drift = drifts[doffset + cluster];
  upper_bound += cluster_drift;
  bounds++;
  float min_lower_bound = FLT_MAX;
  for (uint32_t g = 0; g < d_yy_groups_size; g++) {
    float lower_bound = bounds[g] - drifts[g];
    bounds[g] = lower_bound;
    if (lower_bound < min_lower_bound) {
      min_lower_bound = lower_bound;
    }
  }
  bounds--;
  // group filter try #1
  if (min_lower_bound >= upper_bound) {
    bounds[0] = upper_bound;
    return;
  }
  upper_bound = 0;
  samples += static_cast<uint64_t>(sample) * d_features_size;
  uint32_t coffset = cluster * d_features_size;
  #pragma unroll 4
  for (uint32_t f = 0; f < d_features_size; f++) {
    float d = samples[f] - centroids[coffset + f];
    upper_bound += d * d;
  }
  upper_bound = sqrt(upper_bound);
  bounds[0] = upper_bound;
  // group filter try #2
  if (min_lower_bound >= upper_bound) {
    return;
  }
  // D'oh!
  passed[atomicAdd(&d_passed_number, 1)] = sample;
}

__global__ void kmeans_yy_local_filter(
    const float *__restrict__ samples,
    const uint32_t *__restrict__ passed, const float *__restrict__ centroids,
    const uint32_t *__restrict__ groups, const float *__restrict__ drifts,
    uint32_t *__restrict__ assignments, float *__restrict__ bounds) {
  uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample >= d_passed_number) {
    return;
  }
  sample = passed[sample];
  samples += static_cast<uint64_t>(sample) * d_features_size;
  bounds += static_cast<uint64_t>(sample) * (d_yy_groups_size + 1);
  float upper_bound = bounds[0];
  bounds++;
  uint32_t cluster = assignments[sample];
  uint32_t doffset = d_clusters_size * d_features_size;
  float min_dist = upper_bound, second_min_dist = FLT_MAX;
  uint32_t nearest = cluster;
  extern __shared__ float shared_centroids[];
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
      float lower_bound = bounds[group];
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
      float dist = 0;
      uint32_t coffset = (c - gc) * d_features_size;
      #pragma unroll 4
      for (int f = 0; f < d_features_size; f++) {
        float d = samples[f] - shared_centroids[coffset + f];
        dist += d * d;
      }
      dist = sqrt(dist);
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
  bounds[nearest_group] = second_min_dist;
  if (nearest_group != previous_group) {
    float pb = bounds[previous_group];
    if (pb > upper_bound) {
      bounds[previous_group] = upper_bound;
    }
  }
  bounds[-1] = min_dist;
  if (cluster != nearest) {
    assignments[sample] = nearest;
    atomicAdd(&d_changed_number, 1);
  }
}

////////////////////
// Host functions //
////////////////////

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
    udevptrs<uint32_t> *assignments,  std::vector<uint32_t> *shmem_sizes) {
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
    }
  );
  return kmcudaSuccess;
}

static void print_plan(
    const char *name, const std::vector<std::tuple<uint32_t, uint32_t>>& plan) {
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
    const std::vector<int> &devs, int verbosity, const udevptrs<float> &samples,
    udevptrs<float> *centroids, udevptrs<float> *dists,
    udevptrs<float> *dev_sums, float *host_dists, float *dist_sum) {
  auto plan = distribute(h_samples_size, h_features_size * sizeof(float), devs);
  uint32_t max_len = 0;
  for (auto &p : plan) {
    auto len = std::get<1>(p);
    if (max_len < len) {
      max_len = len;
    }
  }
  CUMEMSET(*dev_sums, 0, max_len / BS_KMPP + 1);
  size_t dist_sums_size = h_samples_size / BS_KMPP + devs.size();
  std::unique_ptr<float[]> dist_sums(new float[dist_sums_size]);
  memset(dist_sums.get(), 0, dist_sums_size * sizeof(float));
  FOR_EACH_DEVI(
    uint32_t offset, length;
    std::tie(offset, length) = plan[devi];
    dim3 block(BS_KMPP, 1, 1);
    dim3 grid(length / block.x + 1, 1, 1);
    kmeans_plus_plus<<<grid, block, block.x * sizeof(float)>>>(
        length, cc, samples[devi].get() + offset * h_features_size,
        (*centroids)[devi].get(), (*dists)[devi].get(), (*dev_sums)[devi].get());
  );
  uint32_t dist_offset = 0;
  FOR_EACH_DEVI(
    uint32_t offset, length;
    std::tie(offset, length) = plan[devi];
    dim3 block(BS_KMPP, 1, 1);
    dim3 grid(length / block.x + 1, 1, 1);
    CUCH(cudaMemcpyAsync(
        host_dists + offset, (*dists)[devi].get(),
        length * sizeof(float), cudaMemcpyDeviceToHost), kmcudaMemoryCopyError);
    CUCH(cudaMemcpyAsync(
        dist_sums.get() + dist_offset, (*dev_sums)[devi].get(),
        grid.x * sizeof(float), cudaMemcpyDeviceToHost), kmcudaMemoryCopyError);
    dist_offset += grid.x;
  );
  SYNC_ALL_DEVS;
  double ds = 0;
  #pragma omp simd reduction(+:ds)
  for (uint32_t i = 0; i < dist_offset; i++) {
    ds += dist_sums[i];
  }
  *dist_sum = ds;
  return kmcudaSuccess;
}

KMCUDAResult kmeans_cuda_lloyd(
    float tolerance, uint32_t h_samples_size, uint32_t h_clusters_size,
    uint16_t h_features_size, bool resume, const std::vector<int> &devs,
    int32_t verbosity, const udevptrs<float> &samples,
    udevptrs<float> *centroids, udevptrs<uint32_t> *ccounts,
    udevptrs<uint32_t> *assignments_prev, udevptrs<uint32_t> *assignments,
    int *iterations = nullptr) {
  std::vector<uint32_t> shmem_sizes;
  RETERR(prepare_mem(h_samples_size, h_clusters_size, resume, devs, verbosity,
                     ccounts, assignments, &shmem_sizes));
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
        dim3 sgrid(length / sblock.x + 1, 1, 1);
        int shmem_size = shmem_sizes[devi];
        auto assigner = kmeans_assign_lloyd;
        if (shmem_size - sblock.x * h_features_size * sizeof(float) >=
            (h_features_size + 1) * sizeof(float)) {
          assigner = kmeans_assign_lloyd_smallc;
        }
        assigner<<<sgrid, sblock, shmem_size>>>(
            length, samples[devi].get() + offset * h_features_size,
            (*centroids)[devi].get(), (*assignments_prev)[devi].get() + offset,
            (*assignments)[devi].get() + offset);
      );
      FOR_EACH_DEVI(
        uint32_t offset, length;
        std::tie(offset, length) = plans[devi];
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
      dim3 cgrid(length / cblock.x + 1, 1, 1);
      kmeans_adjust<<<cblock, cgrid, shmem_sizes[devi]>>>(
          length, offset, samples[devi].get(),
          (*assignments_prev)[devi].get(), (*assignments)[devi].get(),
          (*centroids)[devi].get(), (*ccounts)[devi].get());
    );
    FOR_EACH_DEVI(
      uint32_t offset, length;
      std::tie(offset, length) = planc[devi];
      FOR_OTHER_DEVS(
        CUP2P(ccounts, offset, length);
        CUP2P(centroids, offset * h_features_size, length * h_features_size);
      );
    );
  }
}

KMCUDAResult kmeans_cuda_yy(
    float tolerance, uint32_t h_yy_groups_size, uint32_t h_samples_size,
    uint32_t h_clusters_size, uint16_t h_features_size, const std::vector<int> &devs,
    int32_t verbosity, const udevptrs<float> &samples, udevptrs<float> *centroids,
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
        tolerance, h_samples_size, h_clusters_size, h_features_size, false, devs,
        verbosity, samples, centroids, ccounts, assignments_prev, assignments);
  }
  INFO("running Lloyd until reassignments drop below %" PRIu32 "\n",
       (uint32_t)(YINYANG_DRAFT_REASSIGNMENTS * h_samples_size));
  int iter;
  RETERR(kmeans_cuda_lloyd(
      YINYANG_DRAFT_REASSIGNMENTS, h_samples_size, h_clusters_size, h_features_size,
      false, devs, verbosity, samples, centroids, ccounts, assignments_prev,
      assignments, &iter));
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
    RETERR(kmeans_init_centroids(
        kmcudaInitMethodPlusPlus, h_clusters_size, h_features_size, h_yy_groups_size,
        0, devs, -1, verbosity, nullptr, *centroids, &tmpbufs, drifts_yy, centroids_yy),
        INFO("kmeans_init_centroids() failed for yinyang groups: %s\n",
           cudaGetErrorString(cudaGetLastError())));
    RETERR(kmeans_cuda_lloyd(
        YINYANG_GROUP_TOLERANCE, h_clusters_size, h_yy_groups_size, h_features_size,
        false, devs, verbosity, *centroids, centroids_yy,
        reinterpret_cast<udevptrs<uint32_t> *>(&tmpbufs2),
        reinterpret_cast<udevptrs<uint32_t> *>(&tmpbufs), assignments_yy));
  }
  FOR_EACH_DEV(
    CUCH(cudaMemcpyToSymbol(d_samples_size, &h_samples_size, sizeof(h_samples_size)),
         kmcudaMemoryCopyError);
    CUCH(cudaMemcpyToSymbol(d_clusters_size, &h_clusters_size, sizeof(h_clusters_size)),
         kmcudaMemoryCopyError);
  );
  std::vector<uint32_t> shmem_sizes;
  RETERR(prepare_mem(h_samples_size, h_clusters_size, true, devs, verbosity,
                     ccounts, assignments, &shmem_sizes));
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
        dim3 sigrid(length / siblock.x + 1, 1, 1);
        kmeans_yy_init<<<sigrid, siblock, shmem_sizes[devi]>>>(
            length, samples[devi].get() + offset * h_features_size,
            (*centroids)[devi].get(), (*assignments)[devi].get() + offset,
            (*assignments_yy)[devi].get(), (*bounds_yy)[devi].get());
      );
      refresh = false;
    }
    CUMEMCPY_D2D_ASYNC(*drifts_yy, 0, *centroids, 0, h_clusters_size * h_features_size);
    FOR_EACH_DEVI(
      uint32_t offset, length;
      std::tie(offset, length) = planc[devi];
      dim3 cgrid(length / cblock.x + 1, 1, 1);
      kmeans_adjust<<<cblock, cgrid, shmem_sizes[devi]>>>(
          length, offset, samples[devi].get(),
          (*assignments_prev)[devi].get(), (*assignments)[devi].get(),
          (*centroids)[devi].get(), (*ccounts)[devi].get());
    );
    FOR_EACH_DEVI(
      uint32_t offset, length;
      std::tie(offset, length) = planc[devi];
      FOR_OTHER_DEVS(
        CUP2P(ccounts, offset, length);
        CUP2P(centroids, offset * h_features_size, length * h_features_size);
      );
    );
    FOR_EACH_DEVI(
      uint32_t offset, length;
      std::tie(offset, length) = planc[devi];
      dim3 cgrid(length / cblock.x + 1, 1, 1);
      kmeans_yy_calc_drifts<<<cblock, cgrid>>>(
          length, offset, (*centroids)[devi].get(), (*drifts_yy)[devi].get());
    );
    FOR_EACH_DEVI(
      uint32_t offset, length;
      std::tie(offset, length) = planc[devi];
      FOR_OTHER_DEVS(
        CUP2P(drifts_yy, h_clusters_size * h_features_size + offset, length);
      );
    );
    FOR_EACH_DEVI(
      uint32_t offset, length;
      std::tie(offset, length) = plang[devi];
      dim3 ggrid(length / gblock.x + 1, 1, 1);
      kmeans_yy_find_group_max_drifts<<<gblock, ggrid, shmem_sizes[devi]>>>(
          length, offset, (*assignments_yy)[devi].get(),
          (*drifts_yy)[devi].get());
    );
    FOR_EACH_DEVI(
      uint32_t offset, length;
      std::tie(offset, length) = plang[devi];
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
      dim3 sggrid(length / sgblock.x + 1, 1, 1);
      kmeans_yy_global_filter<<<sggrid, sgblock>>>(
          length, samples[devi].get() + offset * h_features_size,
          (*centroids)[devi].get(), (*assignments_yy)[devi].get(),
          (*drifts_yy)[devi].get(), (*assignments)[devi].get() + offset,
          (*assignments_prev)[devi].get() + offset,
          (*bounds_yy)[devi].get(), (*passed_yy)[devi].get());
      dim3 slgrid(length / slblock.x + 1, 1, 1);
      kmeans_yy_local_filter<<<slgrid, slblock, shmem_sizes[devi]>>>(
          samples[devi].get() + offset * h_features_size,
          (*passed_yy)[devi].get(), (*centroids)[devi].get(),
          (*assignments_yy)[devi].get(), (*drifts_yy)[devi].get(),
          (*assignments)[devi].get() + offset, (*bounds_yy)[devi].get());
    );
    FOR_EACH_DEVI(
      uint32_t offset, length;
      std::tie(offset, length) = plans[devi];
      FOR_OTHER_DEVS(
        CUP2P(assignments_prev, offset, length);
        CUP2P(assignments, offset, length);
      );
    );
  }
}
}  // extern "C"
