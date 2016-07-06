#include <cassert>
#include <cstdio>
#include <cfloat>
#include <cinttypes>
#include <cinttypes>
#include <algorithm>
#include <memory>
#include <cuda_profiler_api.h>
#include "kmcuda.h"

#define BLOCK_SIZE 1024
#define YINYANG_GROUP_TOLERANCE 0.025

#define CUCH(cuda_call, ret) \
do { \
  auto __res = cuda_call; \
  if (__res != cudaSuccess) { \
    printf("%s:%d -> %s\n", __FILE__, __LINE__, cudaGetErrorString(__res)); \
    return ret; \
  } \
} while (false)

__device__ uint32_t changed;
__constant__ uint32_t samples_size;
__constant__ uint16_t features_size;
__constant__ uint32_t clusters_size;
__constant__ uint32_t yy_groups_size;
__constant__ int shmem_size;

__global__ void kmeans_plus_plus(
    uint32_t cc, float *samples, float *centroids, float *dists,
    float *dist_sums) {
  uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample >= samples_size) {
    return;
  }
  uint64_t soffset = sample;
  soffset *= features_size;
  extern __shared__ float local_dists[];
  float dist = 0;
  uint32_t coffset = (cc - 1) * features_size;
  for (uint16_t f = 0; f < features_size; f++) {
    float myf = samples[soffset + f];
    float d = myf - centroids[coffset + f];
    dist += d * d;
  }
  dist = sqrt(dist);
  float prev_dist = dists[sample];
  if (dist < prev_dist || cc == 1) {
    dists[sample] = dist;
  } else {
    dist = prev_dist;
  }
  local_dists[threadIdx.x] = dist;
  __syncthreads();
  if (threadIdx.x == 0) {
    uint32_t end = blockDim.x;
    if ((blockIdx.x + 1) * blockDim.x > samples_size) {
      end = samples_size - blockIdx.x * blockDim.x;
    }
    float block_sum = 0;
    for (uint32_t i = 0; i < end; i++) {
      block_sum += local_dists[i];
    }
    dist_sums[blockIdx.x] = block_sum;
  }
}

__global__ void kmeans_assign_lloyd(float *samples, float *centroids,
                                    uint32_t *assignments_prev,
                                    uint32_t *assignments) {
  uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample >= samples_size) {
    return;
  }
  uint64_t soffset = sample;
  soffset *= features_size;
  float min_dist = FLT_MAX;
  uint32_t nearest = UINT32_MAX;
  extern __shared__ float shared_centroids[];
  const uint32_t cstep = shmem_size / features_size;
  const uint32_t size_each = cstep / blockDim.x + 1;

  for (uint32_t gc = 0; gc < clusters_size; gc += cstep) {
    uint32_t coffset = gc * features_size;
    if (threadIdx.x * size_each < cstep) {
      for (uint32_t i = 0; i < size_each; i++) {
        uint32_t local_offset = (threadIdx.x * size_each + i) * features_size;
        uint32_t global_offset = coffset + local_offset;
        if (global_offset < clusters_size * features_size) {
          for (int f = 0; f < features_size; f++) {
            shared_centroids[local_offset + f] = centroids[global_offset + f];
          }
        }
      }
    }
    __syncthreads();
    for (uint32_t c = gc; c < gc + cstep && c < clusters_size; c++) {
      float dist = 0;
      coffset = (c - gc) * features_size;
      for (int f = 0; f < features_size; f++) {
        float myf = samples[soffset + f];
        float d = myf - shared_centroids[coffset + f];
        dist += d * d;
      }
      if (dist < min_dist) {
        min_dist = dist;
        nearest = c;
      }
    }
  }
  if (nearest == UINT32_MAX) {
    printf("CUDA kernel kmeans_assign: nearest neighbor search failed for "
           "sample %" PRIu32 "\n", sample);
    return;
  }
  uint32_t ass = assignments[sample];
  assignments_prev[sample] = ass;
  if (ass != nearest) {
    assignments[sample] = nearest;
    atomicAdd(&changed, 1);
  }
}

__global__ void kmeans_adjust(
    float *samples, float *centroids, uint32_t *assignments_prev,
    uint32_t *assignments, uint32_t *ccounts) {
  uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= clusters_size) {
    return;
  }
  uint32_t coffset = c * features_size;
  uint32_t my_count = ccounts[c];
  for (int f = 0; f < features_size; f++) {
    centroids[coffset + f] *= my_count;
  }
  extern __shared__ uint32_t ass[];
  int step = shmem_size / 2;
  for (uint32_t sbase = 0; sbase < samples_size; sbase += step) {
    __syncthreads();
    if (threadIdx.x == 0) {
      int pos = sbase;
      for (int i = 0; i < step && sbase + i < samples_size; i++) {
        ass[2 * i] = assignments[pos + i];
        ass[2 * i + 1] = assignments_prev[pos + i];
      }
    }
    __syncthreads();
    for (int i = 0; i < step && sbase + i < samples_size; i++) {
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
        soffset *= features_size;
        for (int f = 0; f < features_size; f++) {
          centroids[coffset + f] += samples[soffset + f] * sign;
        }
      }
    }
  }
  // my_count can be 0 => we get NaN and never use this cluster again
  // this is a feature, not a bug
  for (int f = 0; f < features_size; f++) {
    centroids[coffset + f] /= my_count;
  }
  ccounts[c] = my_count;
}

__global__ void kmeans_yy_init(
    float *samples, float *centroids, uint32_t *assignments_prev,
    uint32_t *assignments, uint32_t *groups, float *bounds) {
  uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample >= samples_size) {
    return;
  }
  uint64_t boffset = sample;
  boffset *= yy_groups_size + 1;
  for (uint32_t i = 0; i < yy_groups_size + 1; i++) {
    bounds[boffset + i] = FLT_MAX;
  }
  boffset++;
  uint64_t soffset = sample;
  soffset *= features_size;
  float min_dist = FLT_MAX;
  uint32_t nearest = UINT32_MAX;
  extern __shared__ float shared_centroids[];
  const uint32_t cstep = shmem_size / features_size;
  const uint32_t size_each = cstep / blockDim.x + 1;

  for (uint32_t gc = 0; gc < clusters_size; gc++) {
    uint32_t coffset = gc * features_size;
    if (threadIdx.x * size_each < cstep) {
      for (uint32_t i = 0; i < size_each; i++) {
        uint32_t local_offset = (threadIdx.x * size_each + i) * features_size;
        uint32_t global_offset = coffset + local_offset;
        if (global_offset < clusters_size * features_size) {
          for (int f = 0; f < features_size; f++) {
            shared_centroids[local_offset + f] = centroids[global_offset + f];
          }
        }
      }
    }
    __syncthreads();

    for (uint32_t c = gc; c < gc + cstep && c < clusters_size; c++) {
      float dist = 0;
      coffset = (c - gc) * features_size;
      uint32_t group = groups[c];
      for (int f = 0; f < features_size; f++) {
        float d = samples[soffset + f] - shared_centroids[coffset + f];
        dist += d * d;
      }
      dist = sqrt(dist);
      if (dist < bounds[boffset + group]) {
        bounds[boffset + group] = dist;
      }
      if (dist < min_dist) {
        min_dist = dist;
        nearest = c;
      }
    }
  }
  bounds[boffset - 1] = min_dist;
  uint32_t nearest_group = groups[nearest];
  min_dist = FLT_MAX;
  for (uint32_t c = 0; c < clusters_size; c++) {
    if (c == nearest || groups[c] != nearest_group) {
      continue;
    }
    float dist = 0;
    uint32_t coffset = c * features_size;
    for (int f = 0; f < features_size; f++) {
      float myf = samples[soffset + f];
      float d = myf - centroids[coffset + f];
      dist += d * d;
    }
    dist = sqrt(dist);
    if (dist < min_dist) {
      min_dist = dist;
    }
  }
  bounds[boffset + nearest_group] = min_dist;

  if (nearest == UINT32_MAX) {
    printf("CUDA kernel kmeans_assign: nearest neighbor search failed for "
           "sample %" PRIu32 "\n", sample);
    return;
  }
  uint32_t ass = assignments[sample];
  assignments_prev[sample] = ass;
  assignments[sample] = nearest;
  atomicAdd(&changed, 1);
}

__global__ void kmeans_yy_calc_drifts(
    float *centroids, float *drifts) {
  uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= clusters_size) {
    return;
  }
  uint32_t coffset = c * features_size;
  float sum = 0;
  for (uint32_t f = coffset; f < coffset + features_size; f++) {
    float d = centroids[f] - drifts[f];
    sum += d * d;
  }
  drifts[clusters_size * features_size + c] = sqrt(sum);
}

__global__ void kmeans_yy_find_group_max_drifts(uint32_t *groups, float *drifts) {
  uint32_t group = blockIdx.x * blockDim.x + threadIdx.x;
  if (group >= yy_groups_size) {
    return;
  }
  const uint32_t doffset = clusters_size * features_size;
  const uint32_t size_each = shmem_size / (2 * blockDim.x);
  const uint32_t step = size_each * blockDim.x;
  extern __shared__ uint32_t shmem[];
  float *cd = (float *)shmem;
  uint32_t *cg = shmem + shmem_size / 2;
  float my_max = FLT_MIN;
  for (uint32_t offset = 0; offset < clusters_size; offset += step) {
    for (uint32_t i = 0; i < size_each; i++) {
      uint32_t local_offset = threadIdx.x * size_each + i;
      uint32_t global_offset = offset + local_offset;
      if (global_offset < clusters_size) {
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

__global__ void kmeans_yy_filter_assign(
    float *samples, float *centroids, uint32_t *assignments_prev,
    uint32_t *assignments, uint32_t *groups, float *bounds, float *drifts) {
  uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample >= samples_size) {
    return;
  }

  uint64_t boffset = sample;
  boffset *= yy_groups_size + 1;
  uint32_t cluster = assignments[sample];
  assignments_prev[sample] = cluster;
  float upper_bound = bounds[boffset];
  uint32_t doffset = clusters_size * features_size;
  float cluster_drift = drifts[doffset + cluster];
  upper_bound += cluster_drift;
  boffset++;
  float min_lower_bound = FLT_MAX;
  for (uint32_t g = 0; g < yy_groups_size; g++) {
    float lower_bound = bounds[boffset + g] - drifts[g];
    bounds[boffset + g] = lower_bound;
    if (lower_bound < min_lower_bound) {
      min_lower_bound = lower_bound;
    }
  }
  boffset--;
  // group filter try #1
  if (min_lower_bound >= upper_bound) {
    bounds[boffset] = upper_bound;
    return;
  }
  upper_bound = 0;
  uint64_t soffset = sample;
  soffset *= features_size;
  uint32_t coffset = cluster * features_size;
  for (uint32_t f = 0; f < features_size; f++) {
    float d = samples[soffset + f] - centroids[coffset + f];
    upper_bound += d * d;
  }
  upper_bound = sqrt(upper_bound);
  // group filter try #2
  if (min_lower_bound >= upper_bound) {
    bounds[boffset] = upper_bound;
    return;
  }

  // D'oh!
  boffset++;
  float min_dist = upper_bound, second_min_dist = FLT_MAX;
  uint32_t nearest = cluster;
  for (uint32_t c = 0; c < clusters_size; c++) {
    if (c == cluster) {
      continue;
    }
    uint32_t group = groups[c];
    float lower_bound = bounds[boffset + group];
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
    coffset = c * features_size;
    for (int f = 0; f < features_size; f++) {
      float d = samples[soffset + f] - centroids[coffset + f];
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
  uint32_t nearest_group = groups[nearest];
  uint32_t previous_group = groups[cluster];
  bounds[boffset + nearest_group] = second_min_dist;
  if (nearest_group != previous_group) {
    float pb = bounds[boffset + previous_group];
    if (pb > upper_bound) {
      bounds[boffset + previous_group] = upper_bound;
    }
  }
  bounds[boffset - 1] = min_dist;
  if (cluster != nearest) {
    assignments[sample] = nearest;
    atomicAdd(&changed, 1);
  }
}

static int check_changed(int iter, float tolerance, uint32_t samples_size,
                         int32_t verbosity) {
  uint32_t my_changed = 0;
  CUCH(cudaMemcpyFromSymbol(&my_changed, changed, sizeof(my_changed)),
       kmcudaMemoryCopyError);
  if (verbosity > 0) {
    printf("iteration %d: %" PRIu32 " reassignments\n", iter, my_changed);
  }
  uint32_t zero = 0;
  CUCH(cudaMemcpyToSymbolAsync(changed, &zero, sizeof(my_changed)),
       kmcudaMemoryCopyError);
  if (my_changed <= tolerance * samples_size) {
    return -1;
  }
  assert(my_changed <= samples_size);
  return kmcudaSuccess;
}

static KMCUDAResult prepare_mem(uint32_t *ccounts, uint32_t *assignments,
                                uint32_t samples_size, uint32_t clusters_size,
                                uint32_t *my_shmem_size) {
  CUCH(cudaMemcpyFromSymbol(my_shmem_size, shmem_size, sizeof(shmem_size)),
       kmcudaMemoryCopyError);
  *my_shmem_size *= sizeof(uint32_t);
  CUCH(cudaMemsetAsync(ccounts, 0, clusters_size * sizeof(uint32_t)),
       kmcudaRuntimeError);
  CUCH(cudaMemsetAsync(assignments, 0xff, samples_size * sizeof(uint32_t)),
       kmcudaRuntimeError);
  return kmcudaSuccess;
}


extern "C" {

KMCUDAResult kmeans_init_centroids(
    KMCUDAInitMethod method, uint32_t samples_size, uint16_t features_size,
    uint32_t clusters_size, uint32_t seed, int32_t verbosity, float *samples,
    void *dists, float *centroids);

KMCUDAResult kmeans_cuda_setup(uint32_t samples_size_, uint16_t features_size_,
                               uint32_t clusters_size_, uint32_t yy_groups_size_,
                               uint32_t device, int32_t verbosity) {
  CUCH(cudaMemcpyToSymbol(samples_size, &samples_size_, sizeof(samples_size)),
       kmcudaMemoryCopyError);
  CUCH(cudaMemcpyToSymbol(features_size, &features_size_, sizeof(features_size)),
       kmcudaMemoryCopyError);
  CUCH(cudaMemcpyToSymbol(clusters_size, &clusters_size_, sizeof(clusters_size)),
       kmcudaMemoryCopyError);
  CUCH(cudaMemcpyToSymbol(yy_groups_size, &yy_groups_size_, sizeof(yy_groups_size)),
       kmcudaMemoryCopyError);
  cudaDeviceProp props;
  CUCH(cudaGetDeviceProperties(&props, device), kmcudaRuntimeError);
  int my_shmem_size = static_cast<int>(props.sharedMemPerBlock);
  if (verbosity > 1) {
    printf("GPU #%" PRIu32 " has %d bytes of shared memory per block\n",
           device, my_shmem_size);
  }
  my_shmem_size /= sizeof(uint32_t);
  CUCH(cudaMemcpyToSymbol(shmem_size, &my_shmem_size, sizeof(my_shmem_size)),
       kmcudaMemoryCopyError);
  return kmcudaSuccess;
}

KMCUDAResult kmeans_cuda_plus_plus(
    uint32_t samples_size, uint32_t cc, float *samples, float *centroids,
    float *dists, float *dist_sum, float **dev_sums) {
  dim3 block(BLOCK_SIZE, 1, 1);
  dim3 grid(samples_size / block.x + 1, 1, 1);
  if (*dev_sums == NULL) {
    CUCH(cudaMalloc(reinterpret_cast<void**>(dev_sums), grid.x * sizeof(float)),
         kmcudaMemoryAllocationFailure);
  } else {
    CUCH(cudaMemset(*dev_sums, 0, grid.x * sizeof(float)), kmcudaRuntimeError);
  }
  kmeans_plus_plus<<<grid, block, block.x * sizeof(float)>>>(
      cc, samples, centroids, dists, *dev_sums);
  std::unique_ptr<float[]> host_dist_sums(new float[grid.x]);
  CUCH(cudaMemcpy(host_dist_sums.get(), *dev_sums, grid.x * sizeof(float),
                  cudaMemcpyDeviceToHost), kmcudaMemoryCopyError);
  float ds = 0;
  #pragma omp simd reduction(+:ds)
  for (uint32_t i = 0; i < grid.x; i++) {
    ds += host_dist_sums[i];
  }
  *dist_sum = ds;
  return kmcudaSuccess;
}

KMCUDAResult kmeans_cuda_lloyd(
    float tolerance, uint32_t samples_size, uint32_t clusters_size,
    uint16_t features_size, int32_t verbosity,
    float *samples, float *centroids, uint32_t *ccounts,
    uint32_t *assignments_prev, uint32_t *assignments) {
  dim3 sblock(BLOCK_SIZE, 1, 1);
  dim3 sgrid(samples_size / sblock.x + 1, 1, 1);
  dim3 cblock(BLOCK_SIZE, 1, 1);
  dim3 cgrid(clusters_size / cblock.x + 1, 1, 1);
  uint32_t my_shmem_size;
  auto pr = prepare_mem(ccounts, assignments, samples_size, clusters_size,
                        &my_shmem_size);
  if (pr != kmcudaSuccess) {
    return pr;
  }
  for (int i = 1; ; i++) {
    kmeans_assign_lloyd<<<sgrid, sblock, my_shmem_size>>>(
        samples, centroids, assignments_prev, assignments);
    int status = check_changed(i, tolerance, samples_size, verbosity);
    if (status < kmcudaSuccess) {
      break;
    }
    if (status != kmcudaSuccess) {
      return static_cast<KMCUDAResult>(status);
    }
    kmeans_adjust<<<cblock, cgrid, my_shmem_size>>>(
        samples, centroids, assignments_prev, assignments, ccounts);
  }
  return kmcudaSuccess;
}

KMCUDAResult kmeans_cuda_yy(
    float tolerance, uint32_t yinyang_groups, uint32_t samples_size_,
    uint32_t clusters_size_, uint16_t features_size, int32_t verbosity,
    float *samples, float *centroids, uint32_t *ccounts,
    uint32_t *assignments_prev, uint32_t *assignments,
    uint32_t *assignments_yy, float *bounds_yy, float *drifts_yy) {
  if (yinyang_groups == 0) {
    if (verbosity > 0) {
      printf("too few clusters for this yinyang_t => Lloyd\n");
    }
    return kmeans_cuda_lloyd(
        tolerance, samples_size_, clusters_size_, features_size, verbosity,
        samples, centroids, ccounts, assignments_prev, assignments);
  }

  // map each centroid to yinyang group -> assignments_yy
  CUCH(cudaMemcpyToSymbol(samples_size, &clusters_size_, sizeof(samples_size_)),
       kmcudaMemoryCopyError);
  CUCH(cudaMemcpyToSymbol(clusters_size, &yinyang_groups, sizeof(clusters_size_)),
       kmcudaMemoryCopyError);
  auto result = kmeans_init_centroids(
      kmcudaInitMethodPlusPlus, clusters_size_, features_size, yinyang_groups,
      0, verbosity, centroids, assignments,
      reinterpret_cast<float*>(assignments_prev));
  if (result != kmcudaSuccess) {
    if (verbosity > 0) {
      printf("kmeans_init_centroids() failed for yinyang groups: %s\n",
             cudaGetErrorString(cudaGetLastError()));
    }
    return result;
  }
  kmeans_cuda_lloyd(
      YINYANG_GROUP_TOLERANCE, clusters_size_, yinyang_groups, features_size,
      verbosity, centroids, reinterpret_cast<float*>(assignments_prev),
      ccounts, assignments, assignments_yy);
  CUCH(cudaMemcpyToSymbol(samples_size, &samples_size_, sizeof(samples_size_)),
       kmcudaMemoryCopyError);
  CUCH(cudaMemcpyToSymbol(clusters_size, &clusters_size_, sizeof(clusters_size_)),
       kmcudaMemoryCopyError);
  cudaProfilerStart();
  if (verbosity > 0) {
    printf("Initializing Yinyang bounds...\n");
  }
  std::unique_ptr<uint32_t[]> groups(new uint32_t[clusters_size_]);
  CUCH(cudaMemcpyAsync(groups.get(), assignments_yy, clusters_size_ * sizeof(uint32_t),
                       cudaMemcpyDeviceToHost), kmcudaMemoryCopyError);
  uint32_t my_shmem_size;
  auto pr = prepare_mem(ccounts, assignments, samples_size_, clusters_size_,
                        &my_shmem_size);
  if (pr != kmcudaSuccess) {
    return pr;
  }
  dim3 sblock(BLOCK_SIZE, 1, 1);
  dim3 sgrid(samples_size_ / sblock.x + 1, 1, 1);
  dim3 cblock(BLOCK_SIZE, 1, 1);
  dim3 cgrid(clusters_size_ / cblock.x + 1, 1, 1);
  dim3 gblock(BLOCK_SIZE, 1, 1);
  dim3 ggrid(yinyang_groups / cblock.x + 1, 1, 1);
  kmeans_yy_init<<<sgrid, sblock, my_shmem_size>>>(
      samples, centroids, assignments_prev, assignments, assignments_yy, bounds_yy);
  for (int iter = 1; ; iter++) {
    int status = check_changed(iter, tolerance, samples_size_, verbosity);
    if (status < kmcudaSuccess) {
      cudaProfilerStop();
      return kmcudaSuccess;
    }
    if (status != kmcudaSuccess) {
      return static_cast<KMCUDAResult>(status);
    }
    CUCH(cudaMemcpyAsync(
        drifts_yy, centroids, clusters_size_ * features_size * sizeof(float),
        cudaMemcpyDeviceToDevice), kmcudaMemoryCopyError);
    kmeans_adjust<<<cblock, cgrid, my_shmem_size>>>(
          samples, centroids, assignments_prev, assignments, ccounts);
    kmeans_yy_calc_drifts<<<cblock, cgrid>>>(centroids, drifts_yy);
    kmeans_yy_find_group_max_drifts<<<gblock, ggrid, my_shmem_size>>>(
        assignments_yy, drifts_yy);
    kmeans_yy_filter_assign<<<sgrid, sblock>>>(samples, centroids, assignments_prev,
        assignments, assignments_yy, bounds_yy, drifts_yy);
  }
}
}
