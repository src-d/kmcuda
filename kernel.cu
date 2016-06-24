#include <stdio.h>
#include <float.h>
#include <inttypes.h>
#include <vector_types.h>
#include <bits/unique_ptr.h>
#include "kmcuda.h"

#define BLOCK_SIZE 1024
#define YINYANG_GROUP_TOLERANCE 0.05

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
  uint32_t soffset = sample * features_size;
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
  uint32_t soffset = sample * features_size;
  float min_dist = FLT_MAX;
  uint32_t nearest = UINT32_MAX;
  for (uint32_t c = 0; c < clusters_size; c++) {
    float dist = 0;
    uint32_t coffset = c * features_size;
    for (int f = 0; f < features_size; f++) {
      float myf = samples[soffset + f];
      float d = myf - centroids[coffset + f];
      dist += d * d;
    }
    if (dist < min_dist) {
      min_dist = dist;
      nearest = c;
    }
  }
  if (nearest == UINT32_MAX) {
    printf("CUDA kernel kmeans_assign: nearest neighbor search failed for"
           "sample %" PRIu32, samples);
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
        uint32_t soffset = (sbase + i) * features_size;
        for (int f = 0; f < features_size; f++) {
          centroids[coffset + f] += samples[soffset + f] * sign;
        }
      }
    }
  }
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
  uint32_t boffset = sample * (yy_groups_size + 1);
  for (uint32_t i = 0; i < yy_groups_size + 1; i++) {
    bounds[boffset + i] = FLT_MAX;
  }
  boffset++;
  uint32_t soffset = sample * features_size;
  float min_dist = FLT_MAX;
  uint32_t nearest = UINT32_MAX;
  for (uint32_t c = 0; c < clusters_size; c++) {
    float dist = 0;
    uint32_t group = groups[c];
    uint32_t coffset = c * features_size;
    for (int f = 0; f < features_size; f++) {
      float d = samples[soffset + f] - centroids[coffset + f];
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
    printf("CUDA kernel kmeans_assign: nearest neighbor search failed for"
           "sample %" PRIu32, samples);
    return;
  }
  uint32_t ass = assignments[sample];
  assignments_prev[sample] = ass;
  if (ass != nearest) {
    assignments[sample] = nearest;
    atomicAdd(&changed, 1);
  }
}

__global__ void kmeans_calc_drifts_yy(
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

__global__ void kmeans_filter_assign_yy(
    float *samples, float *centroids, uint32_t *assignments_prev,
    uint32_t *assignments, uint32_t *groups, float *bounds, float *drifts) {
  uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample >= samples_size) {
    return;
  }

  uint32_t boffset = sample * (yy_groups_size + 1);
  uint32_t cluster = assignments[sample];
  assignments_prev[sample] = cluster;
  float upper_bound = bounds[boffset];
  float cluster_drift = drifts[clusters_size * features_size + cluster];
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
  if (min_lower_bound >= upper_bound) {
    bounds[boffset] = upper_bound;
    return;
  }
  upper_bound = 0;
  uint32_t soffset = sample * features_size;
  uint32_t coffset = cluster * features_size;
  for (uint32_t f = 0; f < features_size; f++) {
    float d = samples[soffset + f] - centroids[coffset + f];
    upper_bound += d * d;
  }
  upper_bound = sqrt(upper_bound);
  if (min_lower_bound >= upper_bound) {
    bounds[boffset] = upper_bound;
    return;
  }

  // D'oh!

  boffset++;
  float min_dist = FLT_MAX;
  uint32_t nearest = UINT32_MAX, second_nearest = UINT32_MAX;
  for (uint32_t c = 0; c < clusters_size; c++) {
    uint32_t group = groups[c];
    float lower_bound = bounds[boffset + group];
    if (lower_bound >= upper_bound) {
      continue;
    }
    lower_bound += drifts[group] - cluster_drift;
    if (second_nearest < lower_bound) {
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
      second_nearest = min_dist;
      min_dist = dist;
      nearest = c;
    }
  }
  if (nearest == UINT32_MAX) {
    printf("CUDA kernel kmeans_assign: nearest neighbor search failed for"
           "sample %" PRIu32, samples);
    return;
  }

  if (cluster != nearest) {
    bounds[boffset - 1] = min_dist;
    assignments[sample] = nearest;
    atomicAdd(&changed, 1);

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
  }
}

extern "C" {

KMCUDAResult kmeans_cuda_setup(uint32_t samples_size_, uint16_t features_size_,
                               uint32_t clusters_size_, uint32_t yy_groups_size_,
                               uint32_t device, int32_t verbosity) {
  if (cudaMemcpyToSymbol(samples_size, &samples_size_, sizeof(samples_size))
      != cudaSuccess) {
    return kmcudaMemoryCopyError;
  }
  if (cudaMemcpyToSymbol(features_size, &features_size_, sizeof(features_size))
      != cudaSuccess) {
    return kmcudaMemoryCopyError;
  }
  if (cudaMemcpyToSymbol(clusters_size, &clusters_size_, sizeof(clusters_size))
      != cudaSuccess) {
    return kmcudaMemoryCopyError;
  }
  if (cudaMemcpyToSymbol(yy_groups_size, &yy_groups_size_, sizeof(yy_groups_size))
      != cudaSuccess) {
    return kmcudaMemoryCopyError;
  }
  cudaDeviceProp props;
  if (cudaGetDeviceProperties(&props, device) != cudaSuccess) {
    return kmcudaRuntimeError;
  }
  int my_shmem_size = static_cast<int>(props.sharedMemPerBlock);
  if (verbosity > 1) {
    printf("GPU #%" PRIu32 " has %d bytes of shared memory per block\n",
           device, my_shmem_size);
  }
  my_shmem_size /= sizeof(uint32_t);
  if (cudaMemcpyToSymbol(shmem_size, &my_shmem_size, sizeof(my_shmem_size))
      != cudaSuccess) {
    return kmcudaMemoryCopyError;
  }
  return kmcudaSuccess;
}

KMCUDAResult kmeans_cuda_plus_plus(
    uint32_t samples_size, uint32_t cc, float *samples, float *centroids,
    float *dists, float *dist_sum, float **dev_sums) {
  dim3 block(BLOCK_SIZE, 1, 1);
  dim3 grid(samples_size / block.x + 1, 1, 1);
  if (*dev_sums == NULL) {
    if (cudaMalloc(reinterpret_cast<void**>(dev_sums),
                   grid.x * sizeof(float)) != cudaSuccess) {
      return kmcudaMemoryAllocationFailure;
    }
  } else {
    if (cudaMemset(*dev_sums, 0, grid.x * sizeof(float)) != cudaSuccess) {
      return kmcudaRuntimeError;
    }
  }
  kmeans_plus_plus<<<grid, block, block.x * sizeof(float)>>>(
      cc, samples, centroids, dists, *dev_sums);
  std::unique_ptr<float[]> host_dist_sums(new float[grid.x]);
  if (cudaMemcpy(host_dist_sums.get(), *dev_sums, grid.x * sizeof(float),
                 cudaMemcpyDeviceToHost) != cudaSuccess) {
    return kmcudaMemoryCopyError;
  }
  float ds = 0;
  #pragma omp simd reduction(+:ds)
  for (uint32_t i = 0; i < grid.x; i++) {
    ds += host_dist_sums[i];
  }
  *dist_sum = ds;
  return kmcudaSuccess;
}

static int check_changed(int iter, float tolerance, uint32_t samples_size,
                         int32_t verbosity) {
  uint32_t changed_ = 0;
  if (cudaMemcpyFromSymbol(&changed_, changed, sizeof(changed_))
      != cudaSuccess) {
    return kmcudaMemoryCopyError;
  }
  if (verbosity > 0) {
    printf("iteration %d: %" PRIu32 " reassignments\n", iter, changed_);
  }
  if (changed_ <= tolerance * samples_size) {
    return -1;
  }
  changed_ = 0;
  if (cudaMemcpyToSymbolAsync(changed, &changed_, sizeof(changed_))
      != cudaSuccess) {
    return kmcudaMemoryCopyError;
  }
  return kmcudaSuccess;
}

static KMCUDAResult prepare_mem(uint32_t *ccounts, uint32_t *assignments,
                                uint32_t samples_size, uint32_t clusters_size,
                                uint32_t *my_shmem_size) {
  if (cudaMemcpyFromSymbol(my_shmem_size, shmem_size, sizeof(my_shmem_size))
      != cudaSuccess) {
    return kmcudaMemoryCopyError;
  }
  *my_shmem_size *= sizeof(uint32_t);
  if (cudaMemsetAsync(ccounts, 0, clusters_size * sizeof(uint32_t)) != cudaSuccess) {
    return kmcudaRuntimeError;
  }
  if (cudaMemsetAsync(assignments, 0xff, samples_size * sizeof(uint32_t)) != cudaSuccess) {
    return kmcudaRuntimeError;
  }
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
    kmeans_assign_lloyd<<<sgrid, sblock>>>(
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

KMCUDAResult kmeans_cuda_internal(
    float tolerance, uint32_t yinyang_groups, uint32_t samples_size,
    uint32_t clusters_size, uint16_t features_size, int32_t verbosity,
    float *samples, float *centroids, uint32_t *ccounts,
    uint32_t *assignments_prev, uint32_t *assignments,
    uint32_t *assignments_yy, float *bounds_yy, float *drifts_yy) {
  if (yinyang_groups == 0) {
    if (verbosity > 0) {
      printf("too few clusters for this yinyang_t => Lloyd\n");
    }
    return kmeans_cuda_lloyd(
        tolerance, samples_size, clusters_size, features_size, verbosity,
        samples, centroids, ccounts, assignments_prev, assignments);
  }
  // map each centroid to yinyang group -> assignments_yy
  kmeans_cuda_lloyd(
      YINYANG_GROUP_TOLERANCE, clusters_size, yinyang_groups, features_size,
      verbosity, centroids, reinterpret_cast<float*>(assignments_prev),
      ccounts, assignments, assignments_yy);
  dim3 sblock(BLOCK_SIZE, 1, 1);
  dim3 sgrid(samples_size / sblock.x + 1, 1, 1);
  dim3 cblock(BLOCK_SIZE, 1, 1);
  dim3 cgrid(clusters_size / cblock.x + 1, 1, 1);
  uint32_t centroids_size = clusters_size * features_size;
  uint32_t my_shmem_size;
  auto pr = prepare_mem(ccounts, assignments, samples_size, clusters_size,
                        &my_shmem_size);
  if (pr != kmcudaSuccess) {
    return pr;
  }
  std::unique_ptr<uint32_t[]> groups(new uint32_t[clusters_size]);
  if (cudaMemcpyAsync(groups.get(), assignments_yy, clusters_size * sizeof(uint32_t),
                      cudaMemcpyDeviceToHost) != cudaSuccess) {
    return kmcudaMemoryCopyError;
  }
  std::unique_ptr<float[]> drifts(new float[clusters_size]);
  std::unique_ptr<float[]> max_drifts(new float[yinyang_groups]);
  auto max_drifts_ptr = max_drifts.get();
  kmeans_yy_init<<<sgrid, sblock>>>(
      samples, centroids, assignments_prev, assignments, assignments_yy, bounds_yy);
  while (true) {
    int status = check_changed(1, tolerance, samples_size, verbosity);
    if (status < kmcudaSuccess) {
      return kmcudaSuccess;
    }
    if (status != kmcudaSuccess) {
      return static_cast<KMCUDAResult>(status);
    }
    if (cudaMemcpyAsync(drifts_yy, centroids, centroids_size * sizeof(float),
                        cudaMemcpyDeviceToDevice) != cudaSuccess) {
      return kmcudaMemoryCopyError;
    }
    kmeans_adjust<<<cblock, cgrid, my_shmem_size>>>(
          samples, centroids, assignments_prev, assignments, ccounts);
    kmeans_calc_drifts_yy<<<cblock, cgrid>>>(centroids, drifts_yy);
    if (cudaMemcpy(drifts.get(), drifts_yy + centroids_size,
                   clusters_size * sizeof(float), cudaMemcpyDeviceToHost)
        != cudaSuccess) {
      return kmcudaMemoryCopyError;
    }
    #pragma omp for simd
    for (uint32_t i = 0; i < yinyang_groups; i++) {
      max_drifts_ptr[i] = FLT_MAX;
    }
    for (uint32_t c = 0; c < clusters_size; c++) {
      uint32_t cg = groups[c];
      float d = drifts[centroids_size + c];
      if (max_drifts_ptr[cg] > d) {
        max_drifts_ptr[cg] = d;
      }
    }
    if (cudaMemcpyAsync(max_drifts_ptr, drifts_yy, yinyang_groups * sizeof(float),
                        cudaMemcpyHostToDevice) != cudaSuccess) {
      return kmcudaMemoryCopyError;
    }
    kmeans_filter_assign_yy<<<sgrid, sblock>>>(samples, centroids, assignments_prev,
        assignments, assignments_yy, bounds_yy, drifts_yy);
  }
}
}
