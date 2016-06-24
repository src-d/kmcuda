#include <stdio.h>
#include <float.h>
#include <inttypes.h>
#include <vector_types.h>
#include <bits/unique_ptr.h>
#include "kmcuda.h"

#define BLOCK_SIZE 1024

__device__ uint32_t changed;
__constant__ uint32_t samples_size;
__constant__ uint16_t features_size;
__constant__ uint32_t clusters_size;
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

__global__ void kmeans_assign(float *samples, float *centroids,
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

extern "C" {

KMCUDAResult kmeans_cuda_setup(uint32_t samples_size_, uint16_t features_size_,
                               uint32_t clusters_size_, uint32_t device,
                               int32_t verbosity) {
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

KMCUDAResult kmeans_cuda_internal(
    float tolerance, uint32_t samples_size, uint32_t clusters_size,
    uint16_t features_size, int32_t verbosity,
    float *samples, float *centroids, uint32_t *ccounts,
    uint32_t *assignments_prev, uint32_t *assignments) {
  dim3 sblock(BLOCK_SIZE, 1, 1);
  dim3 sgrid(samples_size / sblock.x + 1, 1, 1);
  dim3 cblock(BLOCK_SIZE, 1, 1);
  dim3 cgrid(clusters_size / cblock.x + 1, 1, 1);
  uint32_t my_shmem_size;
  if (cudaMemcpyFromSymbol(&my_shmem_size, shmem_size, sizeof(my_shmem_size))
      != cudaSuccess) {
    return kmcudaMemoryCopyError;
  }
  my_shmem_size *= sizeof(uint32_t);
  if (cudaMemsetAsync(ccounts, 0, clusters_size * sizeof(uint32_t)) != cudaSuccess) {
    return kmcudaRuntimeError;
  }
  if (cudaMemsetAsync(assignments, 0xff, samples_size * sizeof(uint32_t)) != cudaSuccess) {
    return kmcudaRuntimeError;
  }
  for (int i = 1; ; i++) {
    kmeans_assign<<<sgrid, sblock>>>(
        samples, centroids, assignments_prev, assignments);
    uint32_t changed_ = 0;
    if (cudaMemcpyFromSymbol(&changed_, changed, sizeof(changed_))
        != cudaSuccess) {
      return kmcudaMemoryCopyError;
    }
    if (verbosity > 0) {
      printf("iteration %d: %" PRIu32 " reassignments\n", i, changed_);
    }
    if (changed_ <= tolerance * samples_size) {
      break;
    }
    changed_ = 0;
    if (cudaMemcpyToSymbolAsync(changed, &changed_, sizeof(changed_))
        != cudaSuccess) {
      return kmcudaMemoryCopyError;
    }
    kmeans_adjust<<<cblock, cgrid, my_shmem_size>>>(
        samples, centroids, assignments_prev, assignments, ccounts);
  }
  return kmcudaSuccess;
}
}
