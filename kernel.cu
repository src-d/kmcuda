#include <stdio.h>
#include <float.h>
#include <inttypes.h>
#include "kmcuda.h"

#define BLOCK_SIZE 1024

__device__ float dist_sum;
__device__ uint32_t changed;
__constant__ uint32_t samples_size;
__constant__ uint16_t features_size;
__constant__ uint32_t clusters_size;

__global__ void kmeans_plus_plus(uint32_t cc, float *samples,
                                 float *centroids, float *dists) {
  uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample >= samples_size) {
    return;
  }
  uint32_t soffset = sample * features_size;
  extern __shared__ float local_dists[];
  float min_dist = FLT_MAX;
  for (uint32_t c = 0; c < cc; c++) {
    float dist = 0;
    uint32_t coffset = c * features_size;
    for (uint16_t f = 0; f < features_size; f++) {
      float myf = samples[soffset + f];
      float d = myf - centroids[coffset + f];
      dist += d * d;
    }
    if (dist < min_dist) {
      min_dist = dist;
    }
  }
  min_dist = sqrt(min_dist);
  dists[sample] = min_dist;
  local_dists[threadIdx.x] = min_dist;
  __syncthreads();
  if (threadIdx.x == 0) {
    float local_sum = 0;
    for (int i = 0; i < blockDim.x; i++) {
      local_sum += local_dists[i];
    }
    dist_sum += local_sum;
  }
}

__global__ void kmeans_assign(float *samples, float *centroids,
                              uint32_t *ccounts, uint32_t *assignments) {
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
  if (assignments[sample] != nearest) {
    assignments[sample] = nearest;
    atomicAdd(&changed, 1);
  }
  atomicAdd(&ccounts[nearest], 1);
}

__global__ void kmeans_sum(float *samples, float *centroids,
                           uint32_t *assignments) {
  uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample >= samples_size) {
    return;
  }
  uint32_t soffset = sample * features_size;
  uint32_t coffset = assignments[sample] * features_size;
  for (int f = 0; f < features_size; f++) {
    float myf = samples[soffset + f];
    centroids[coffset + f] += myf;
  }
}

__global__ void kmeans_adjust(float *centroids, uint32_t *ccounts) {
  uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= clusters_size) {
    return;
  }
  uint32_t coffset = c * features_size;
  for (int f = 0; f < features_size; f++) {
    centroids[coffset + f] /= ccounts[c];
  }
  ccounts[c] = 0;
}

extern "C" {

KMCUDAResult kmeans_cuda_setup(uint32_t samples_size_, uint16_t features_size_,
                               uint32_t clusters_size_) {
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
  return kmcudaSuccess;
}

KMCUDAResult kmeans_cuda_plus_plus(
    uint32_t samples_size, uint32_t cc, float *samples, float *centroids,
    float *dists, float *dist_sum_) {
  float zero = 0;
  if (cudaMemcpyToSymbol(dist_sum, &zero, sizeof(zero)) != cudaSuccess) {
    return kmcudaMemoryCopyError;
  }
  dim3 block(BLOCK_SIZE, 1, 1);
  dim3 grid(samples_size / block.x + 1, 1, 1);
  kmeans_plus_plus<<<grid, block, block.x * sizeof(float)>>>(
      cc, samples, centroids, dists);
  cudaDeviceSynchronize();
  if (cudaMemcpyFromSymbol(dist_sum_, dist_sum, sizeof(float))
      != cudaSuccess) {
    return kmcudaMemoryCopyError;
  }
  return kmcudaSuccess;
}

KMCUDAResult kmeans_cuda_internal(
    uint32_t samples_size, uint32_t clusters_size, int32_t verbosity,
    float *samples, float *centroids, uint32_t *ccounts, uint32_t *assignments) {
  dim3 sblock(BLOCK_SIZE, 1, 1);
  dim3 sgrid(samples_size / sblock.x + 1, 1, 1);
  dim3 cblock(BLOCK_SIZE, 1, 1);
  dim3 cgrid(clusters_size / cblock.x + 1, 1, 1);
  for (int i = 1; ; i++) {
    kmeans_assign<<<sgrid, sblock>>>(samples, centroids, ccounts, assignments);
    uint32_t changed_ = 0;
    if (cudaMemcpyFromSymbol(&changed_, changed, sizeof(changed_))
        != cudaSuccess) {
      return kmcudaMemoryCopyError;
    }
    if (verbosity > 0) {
      printf("iteration %d: %" PRIu32 " reassignments\n", i, changed_);
    }
    if (!changed_) {
      break;
    }
    changed_ = 0;
    if (cudaMemcpyToSymbol(changed, &changed_, sizeof(changed_))
        != cudaSuccess) {
      return kmcudaMemoryCopyError;
    }
    kmeans_sum<<<sblock, sgrid>>>(samples, centroids, assignments);
    kmeans_adjust<<<cblock, cgrid>>>(centroids, ccounts);
  }
  return kmcudaSuccess;
}
}
