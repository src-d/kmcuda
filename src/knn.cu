#include <cfloat>

#include "private.h"
#include "metric_abstraction.h"
#include "tricks.cuh"

#define CLUSTER_DISTANCES_BLOCK_SIZE 512
#define CLUSTER_DISTANCES_SHMEM 12288  // in float-s
#define CLUSTER_RADIUSES_BLOCK_SIZE 512
#define CLUSTER_RADIUSES_SHMEM 8192  // in float-s
#define KNN_BLOCK_SIZE_SHMEM 512
#define KNN_BLOCK_SIZE_GMEM 1024

__constant__ uint32_t d_samples_size;
__constant__ uint32_t d_clusters_size;
__device__ unsigned long long int d_dists_calced;

/// sample_dists musr be zero-ed!
template <KMCUDADistanceMetric M, typename F>
__global__ void knn_calc_cluster_radiuses(
    uint32_t offset, uint32_t length, const uint32_t *__restrict__ inv_asses,
    const uint32_t *__restrict__ inv_asses_offsets,
    const F *__restrict__ centroids, const F *__restrict__ samples,
    float *__restrict__ sample_dists, float *__restrict__ radiuses) {
  volatile uint32_t ci = blockIdx.x * blockDim.x + threadIdx.x;
  if (ci >= length) {
    return;
  }
  ci += offset;

  // stage 1 - accumulate partial distances for every sample
  __shared__ F shcents[CLUSTER_RADIUSES_SHMEM];
  volatile const int cent_step = min(
      CLUSTER_RADIUSES_SHMEM / blockDim.x, static_cast<unsigned>(d_features_size));
  F *volatile const my_cent = shcents + cent_step * threadIdx.x;
  for (int cfi = 0; cfi < d_features_size; cfi += cent_step) {
    const int fsize = min(cent_step, d_features_size - cfi);
    for (int f = 0; f < fsize; f++) {
      my_cent[f] = centroids[ci * d_features_size + cfi + f];
    }
    for (uint32_t ass = inv_asses_offsets[ci]; ass < inv_asses_offsets[ci + 1];
         ass++) {
       uint64_t sample = inv_asses[ass];  // uint64_t!
       sample_dists[sample] += METRIC<M, F>::partial_t(
           samples, my_cent, fsize, d_samples_size, cfi, sample);
    }
  }
  // stage 2 - find the maximum distance
  float max_dist = -1;
  for (uint32_t ass = inv_asses_offsets[ci]; ass < inv_asses_offsets[ci + 1];
       ass++) {
    float dist = METRIC<M, F>::finalize(sample_dists[inv_asses[ass]]);
    if (dist > max_dist) {
      max_dist = dist;
    }
  }
  radiuses[ci] = max_dist > -1? max_dist : NAN;
}

/// distances must be zero-ed!
template <KMCUDADistanceMetric M, typename F>
__global__ void knn_calc_cluster_distances(
    uint32_t offset, const F *__restrict__ centroids, float *distances) {
  volatile const uint32_t bi = blockIdx.x + offset;
  const uint32_t bs = CLUSTER_DISTANCES_BLOCK_SIZE;
  uint32_t x, y;
  const uint32_t n = dupper(d_clusters_size, bs);
  {
    float tmp = n + 0.5;
    float d = _sqrt(tmp * tmp - 2 * bi);
    y = tmp - d;
    x = bi + y + (n - y) * (n - y + 1) / 2 - n * (n + 1) / 2;
  }
  __shared__ F shcents[CLUSTER_DISTANCES_SHMEM];
  const uint32_t fstep = CLUSTER_DISTANCES_SHMEM / bs;
  F *volatile my_cent = shcents + fstep * threadIdx.x;

  // stage 1 - accumulate distances
  for (uint16_t fpos = 0; fpos < d_features_size; fpos += fstep) {
    __syncthreads();
    const uint16_t fsize = min(
        fstep, static_cast<uint32_t>(d_features_size - fpos));
    uint32_t cbase = x * bs + threadIdx.x;
    if (cbase < d_clusters_size) {
      for (uint16_t f = 0; f < fsize; f++) {
        my_cent[f] = centroids[cbase * d_features_size + fpos + f];
      }
    }
    __syncthreads();
    for (uint32_t ti = 0; ti < bs; ti++) {
      if ((y * bs + threadIdx.x) < d_clusters_size
          && (x * bs + ti) < d_clusters_size) {
        auto other_cent = d_clusters_size <= bs?
            shcents + (y * bs + threadIdx.x) * fstep
            :
            centroids + (y * bs + threadIdx.x) * d_features_size + fpos;
        distances[(y * bs + threadIdx.x) * d_clusters_size + x * bs + ti] +=
            METRIC<M, F>::partial(other_cent, shcents + ti * fstep, fsize);
      }
    }
  }

  // stage 2 - finalize the distances
  for (uint32_t ti = 0; ti < bs; ti++) {
    if ((y * bs + threadIdx.x) < d_clusters_size
        && (x * bs + ti) < d_clusters_size) {
      uint32_t di = (y * bs + threadIdx.x) * d_clusters_size + x * bs + ti;
      float dist = distances[di];
      dist = METRIC<M, F>::finalize(dist);
      distances[di] = dist;
    }
  }
}

__global__ void knn_mirror_cluster_distances(float *__restrict__ distances) {
  const uint32_t bs = CLUSTER_DISTANCES_BLOCK_SIZE;
  uint32_t x, y;
  const uint32_t n = dupper(d_clusters_size, bs);
  {
    float tmp = n + 0.5;
    float d = _sqrt(tmp * tmp - 2 * blockIdx.x);
    y = tmp - d;
    x = blockIdx.x + y + (n - y) * (n - y + 1) / 2 - n * (n + 1) / 2;
  }
  for (uint32_t ti = 0; ti < bs; ti++) {
    if ((y * bs + threadIdx.x) < d_clusters_size && (x * bs + ti) < d_clusters_size) {
      distances[(x * bs + ti) * d_clusters_size + y * bs + threadIdx.x] =
          distances[(y * bs + threadIdx.x) * d_clusters_size + x * bs + ti];
    }
  }
}

FPATTR void push_sample(uint16_t k, float dist, uint32_t index, float *heap) {
  uint16_t pos = 0;
  while (true) {
    float left, right;
    bool left_le, right_le;
    if ((2 * pos + 1) < k) {
      left = heap[4 * pos + 2];
      left_le = dist >= left;
    } else {
      left_le = true;
    }
    if ((2 * pos + 2) < k) {
      right = heap[4 * pos + 4];
      right_le = dist >= right;
    } else {
      right_le = true;
    }
    if (left_le && right_le) {
      heap[2 * pos] = dist;
      *reinterpret_cast<uint32_t *>(heap + 2 * pos + 1) = index;
      break;
    }
    if (!left_le && !right_le) {
      if (left <= right) {
        heap[2 * pos] = right;
        heap[2 * pos + 1] = heap[4 * pos + 5];
        pos = 2 * pos + 2;
      } else {
        heap[2 * pos] = left;
        heap[2 * pos + 1] = heap[4 * pos + 3];
        pos = 2 * pos + 1;
      }
    } else if (left_le) {
      heap[2 * pos] = right;
      heap[2 * pos + 1] = heap[4 * pos + 5];
      pos = 2 * pos + 2;
    } else {
      heap[2 * pos] = left;
      heap[2 * pos + 1] = heap[4 * pos + 3];
      pos = 2 * pos + 1;
    }
  }
}

template <KMCUDADistanceMetric M, typename F>
__global__ void knn_assign_shmem(
    uint32_t offset, uint32_t length, uint16_t k,
    const float *__restrict__ cluster_distances,
    const float *__restrict__ cluster_radiuses,
    const F *__restrict__ samples, const F *__restrict__ centroids,
    const uint32_t *assignments, const uint32_t *inv_asses,
    const uint32_t *inv_asses_offsets, uint32_t *neighbors) {
  volatile uint64_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample >= length) {
    return;
  }
  sample += offset;
  volatile uint32_t mycls = assignments[sample];
  volatile float mydist = METRIC<M, F>::distance_t(
      samples, centroids + mycls * d_features_size, d_samples_size, sample);
  extern __shared__ float buffer[];
  float *volatile mynearest = buffer + k * 2 * threadIdx.x;
  volatile float mndist = FLT_MAX;
  for (int i = 0; i < static_cast<int>(k); i++) {
    mynearest[i * 2] = FLT_MAX;
  }
  uint32_t pos_start = inv_asses_offsets[mycls];
  uint32_t pos_finish = inv_asses_offsets[mycls + 1];
  atomicAdd(&d_dists_calced, pos_finish - pos_start);
  for (uint32_t pos = pos_start; pos < pos_finish; pos++) {
    uint64_t other_sample = inv_asses[pos];
    if (sample == other_sample) {
      continue;
    }
    float dist = METRIC<M, F>::distance_tt(
        samples, d_samples_size, sample, other_sample);
    if (dist <= mndist) {
      push_sample(k, dist, other_sample, mynearest);
      mndist = mynearest[0];
    }
  }
  for (uint32_t cls = 0; cls < d_clusters_size; cls++) {
    if (cls == mycls) {
      continue;
    }
    float cdist = cluster_distances[cls * d_clusters_size + mycls];
    if (cdist != cdist) {
      continue;
    }
    float dist = cdist - mydist - cluster_radiuses[cls];
    if (dist > mndist) {
      continue;
    }
    uint32_t pos_start = inv_asses_offsets[cls];
    uint32_t pos_finish = inv_asses_offsets[cls + 1];
    atomicAdd(&d_dists_calced, pos_finish - pos_start);
    for (uint32_t pos = pos_start; pos < pos_finish; pos++) {
      uint64_t other_sample = inv_asses[pos];
      dist = METRIC<M, F>::distance_tt(
          samples, d_samples_size, sample, other_sample);
      if (dist <= mndist) {
        push_sample(k, dist, other_sample, mynearest);
        mndist = mynearest[0];
      }
    }
  }
  for (int i = k - 1; i >= 0; i--) {
    neighbors[(sample - offset) * k + i] = reinterpret_cast<uint32_t*>(mynearest)[1];
    push_sample(k, -1, UINT32_MAX, mynearest);
  }
}

template <KMCUDADistanceMetric M, typename F>
__global__ void knn_assign_gmem(
    uint32_t offset, uint32_t length, uint16_t k,
    const float *__restrict__ cluster_distances,
    const float *__restrict__ cluster_radiuses,
    const F *__restrict__ samples, const F *__restrict__ centroids,
    const uint32_t *assignments, const uint32_t *inv_asses,
    const uint32_t *inv_asses_offsets, uint32_t *neighbors) {
  volatile uint64_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample >= length) {
    return;
  }
  sample += offset;
  volatile uint32_t mycls = assignments[sample];
  volatile float mydist = METRIC<M, F>::distance_t(
      samples, centroids + mycls * d_features_size, d_samples_size, sample);
  float *volatile mynearest =
      reinterpret_cast<float*>(neighbors) + (sample - offset) * k * 2;
  volatile float mndist = FLT_MAX;
  for (int i = 0; i < static_cast<int>(k); i++) {
    mynearest[i * 2] = FLT_MAX;
  }
  uint32_t pos_start = inv_asses_offsets[mycls];
  uint32_t pos_finish = inv_asses_offsets[mycls + 1];
  atomicAdd(&d_dists_calced, pos_finish - pos_start);
  for (uint32_t pos = pos_start; pos < pos_finish; pos++) {
    uint64_t other_sample = inv_asses[pos];
    if (sample == other_sample) {
      continue;
    }
    float dist = METRIC<M, F>::distance_tt(
        samples, d_samples_size, sample, other_sample);
    if (dist <= mndist) {
      push_sample(k, dist, other_sample, mynearest);
      mndist = mynearest[0];
    }
  }
  for (uint32_t cls = 0; cls < d_clusters_size; cls++) {
    if (cls == mycls) {
      continue;
    }
    float cdist = cluster_distances[cls * d_clusters_size + mycls];
    if (cdist != cdist) {
      continue;
    }
    float dist = cdist - mydist - cluster_radiuses[cls];
    if (dist > mndist) {
      continue;
    }
    pos_start = inv_asses_offsets[cls];
    pos_finish = inv_asses_offsets[cls + 1];
    atomicAdd(&d_dists_calced, pos_finish - pos_start);
    for (uint32_t pos = pos_start; pos < pos_finish; pos++) {
      uint64_t other_sample = inv_asses[pos];
      dist = METRIC<M, F>::distance_tt(
          samples, d_samples_size, sample, other_sample);
      if (dist <= mndist) {
        push_sample(k, dist, other_sample, mynearest);
        mndist = mynearest[0];
      }
    }
  }
  for (int i = 0; i < k; i++) {
    uint32_t imax = reinterpret_cast<uint32_t*>(mynearest)[1];
    push_sample(k - i - 1, mynearest[2 * k - 2 * i - 2],
                reinterpret_cast<uint32_t*>(mynearest)[2 * k - 2 * i - 1],
                mynearest);
    reinterpret_cast<uint32_t*>(mynearest)[2 * k - 2 * i - 1] = imax;
  }
  for (int i = 0; i < k; i++) {
    reinterpret_cast<uint32_t*>(mynearest)[i] =
        reinterpret_cast<uint32_t*>(mynearest)[2 * i + 1];
  }
}

__global__ void knn_assign_gmem_deinterleave1(
    uint32_t length, uint16_t k, uint32_t *neighbors) {
  volatile uint64_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample >= length) {
    return;
  }
  if (sample % 2 == 1) {
    for (int i = 0; i < k; i++) {
      neighbors[sample * k + i] = neighbors[sample * 2 * k + i];
    }
  } else {
    for (int i = 0; i < k; i++) {
      neighbors[(length + sample) * k + k + i] = neighbors[sample * 2 * k + i];
    }
  }
}

__global__ void knn_assign_gmem_deinterleave2(
    uint32_t length, uint16_t k, uint32_t *neighbors) {
  volatile uint64_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  sample *= 2;
  if (sample >= length) {
    return;
  }
  for (int i = 0; i < k; i++) {
    neighbors[sample * k + i] = neighbors[(length + sample) * k + k + i];
  }
}

extern "C" {

KMCUDAResult knn_cuda_setup(
    uint32_t h_samples_size, uint16_t h_features_size, uint32_t h_clusters_size,
    const std::vector<int> &devs, int32_t verbosity) {
  FOR_EACH_DEV(
    CUCH(cudaMemcpyToSymbol(d_samples_size, &h_samples_size, sizeof(h_samples_size)),
         kmcudaMemoryCopyError);
    CUCH(cudaMemcpyToSymbol(d_features_size, &h_features_size, sizeof(h_features_size)),
         kmcudaMemoryCopyError);
    CUCH(cudaMemcpyToSymbol(d_clusters_size, &h_clusters_size, sizeof(h_clusters_size)),
         kmcudaMemoryCopyError);
    uint64_t zero = 0;
    CUCH(cudaMemcpyToSymbol(d_dists_calced, &zero, sizeof(d_dists_calced)),
         kmcudaMemoryCopyError);
  );
  return kmcudaSuccess;
}

int knn_cuda_neighbors_mem_multiplier(uint16_t k, int dev, int verbosity) {
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, dev);
  int shmem_size = static_cast<int>(props.sharedMemPerBlock);
  int needed_shmem_size = KNN_BLOCK_SIZE_SHMEM * 2 * k * sizeof(uint32_t);
  if (needed_shmem_size > shmem_size) {
    INFO("device #%d: needed shmem size %d > %d => using global memory\n",
         dev, needed_shmem_size, shmem_size);
    return 2;
  }
  return 1;
}

KMCUDAResult knn_cuda_calc(
    uint16_t k, uint32_t h_samples_size, uint32_t h_clusters_size,
    uint16_t h_features_size, KMCUDADistanceMetric metric,
    const std::vector<int> &devs, int fp16x2, int verbosity,
    const udevptrs<float> &samples, const udevptrs<float> &centroids,
    const udevptrs<uint32_t> &assignments, const udevptrs<uint32_t> &inv_asses,
    const udevptrs<uint32_t> &inv_asses_offsets, udevptrs<float> *distances,
    udevptrs<float>* sample_dists, udevptrs<float> *radiuses,
    udevptrs<uint32_t> *neighbors) {
  auto plan = distribute(h_clusters_size, h_features_size * sizeof(float), devs);
  if (verbosity > 1) {
    print_plan("plan_calc_radiuses", plan);
  }
  INFO("calculating the cluster radiuses...\n");
  FOR_EACH_DEVI(
    uint32_t offset, length;
    std::tie(offset, length) = plan[devi];
    if (length == 0) {
      continue;
    }
    dim3 block(CLUSTER_RADIUSES_BLOCK_SIZE, 1, 1);
    dim3 grid(upper(h_clusters_size, block.x), 1, 1);
    float *dsd;
    if (h_clusters_size * h_clusters_size >= h_samples_size) {
      dsd = (*distances)[devi].get();
    } else {
      dsd = (*sample_dists)[devi].get();
    }
    KERNEL_SWITCH(knn_calc_cluster_radiuses, <<<grid, block>>>(
        offset, length, inv_asses[devi].get(), inv_asses_offsets[devi].get(),
        reinterpret_cast<const F*>(centroids[devi].get()),
        reinterpret_cast<const F*>(samples[devi].get()),
        dsd, (*radiuses)[devi].get()));
  );
  FOR_EACH_DEVI(
    uint32_t offset, length;
    std::tie(offset, length) = plan[devi];
    FOR_OTHER_DEVS(
      CUP2P(radiuses, offset, length);
    );
  );
  if (h_clusters_size * h_clusters_size >= h_samples_size) {
    CUMEMSET_ASYNC(*distances, 0, h_samples_size);
  }
  uint32_t dist_blocks_dim = upper(
      h_clusters_size, static_cast<uint32_t>(CLUSTER_DISTANCES_BLOCK_SIZE));
  uint32_t dist_blocks_n = (2 * dist_blocks_dim + 1) * (2 * dist_blocks_dim + 1) / 8;
  plan = distribute(dist_blocks_n, 512, devs);
  {  // align across CLUSTER_DISTANCES_BLOCK_SIZE horizontal boundaries
    uint32_t align = 0;
    for (auto& p : plan) {
      uint32_t offset, length;
      std::tie(offset, length) = p;
      offset += align;
      std::get<0>(p) = offset;
      uint32_t n = dist_blocks_dim;
      float tmp = n + 0.5;
      float d = sqrt(tmp * tmp - 2 * (offset + length));
      uint32_t y = tmp - d;
      uint32_t x = offset + length + (n - y) * (n - y + 1) / 2 - n * (n + 1) / 2;
      if (x > 0) {
        align = n - y - x;
        std::get<1>(p) += align;
      }
    }
  }
  if (verbosity > 1) {
    print_plan("plan_calc_cluster_distances", plan);
  }
  INFO("calculating the centroid distance matrix...\n");
  FOR_EACH_DEVI(
    uint32_t offset, length;
    std::tie(offset, length) = plan[devi];
    if (length == 0) {
      continue;
    }
    dim3 block(CLUSTER_DISTANCES_BLOCK_SIZE, 1, 1);
    dim3 grid(length, 1, 1);
    KERNEL_SWITCH(knn_calc_cluster_distances, <<<grid, block>>>(
        offset, reinterpret_cast<const F*>(centroids[devi].get()),
        (*distances)[devi].get()));
  );
  FOR_EACH_DEVI(
    uint32_t y_start, y_finish;
    {
      uint32_t offset, length;
      std::tie(offset, length) = plan[devi];
      float tmp = dist_blocks_dim + 0.5;
      float d = sqrt(tmp * tmp - 2 * offset);
      y_start = tmp - d;
      d = sqrt(tmp * tmp - 2 * (offset + length));
      y_finish = tmp - d;
    }
    if (y_finish == y_start) {
      continue;
    }
    uint32_t p_offset = y_start * h_clusters_size * CLUSTER_DISTANCES_BLOCK_SIZE;
    uint32_t p_size = (y_finish - y_start) * h_clusters_size * CLUSTER_DISTANCES_BLOCK_SIZE;
    p_size = std::min(p_size, h_clusters_size * h_clusters_size - p_offset);
    FOR_OTHER_DEVS(
      CUP2P(distances, p_offset, p_size);
    );
  );
  FOR_EACH_DEVI(
    dim3 block(CLUSTER_DISTANCES_BLOCK_SIZE, 1, 1);
    dim3 grid(dist_blocks_n, 1, 1);
    knn_mirror_cluster_distances<<<grid, block>>>((*distances)[devi].get());
  );
  plan = distribute(h_samples_size, h_features_size * sizeof(float), devs);
  INFO("searching for the nearest neighbors...\n");
  FOR_EACH_DEVI(
    uint32_t offset, length;
    std::tie(offset, length) = plan[devi];
    if (knn_cuda_neighbors_mem_multiplier(k, devs[devi], 1) == 2) {
      dim3 block(KNN_BLOCK_SIZE_GMEM, 1, 1);
      dim3 grid(upper(h_samples_size, block.x), 1, 1);
      KERNEL_SWITCH(knn_assign_gmem, <<<grid, block>>>(
          offset, length, k, (*distances)[devi].get(), (*radiuses)[devi].get(),
          reinterpret_cast<const F*>(samples[devi].get()),
          reinterpret_cast<const F*>(centroids[devi].get()),
          assignments[devi].get(), inv_asses[devi].get(),
          inv_asses_offsets[devi].get(), (*neighbors)[devi].get()));
      knn_assign_gmem_deinterleave1<<<grid, block>>>(
          length, k, (*neighbors)[devi].get());
      dim3 grid2(upper(h_samples_size, 2 * block.x), 1, 1);
      knn_assign_gmem_deinterleave2<<<grid2, block>>>(
          length, k, (*neighbors)[devi].get());
    } else {
      dim3 block(KNN_BLOCK_SIZE_SHMEM, 1, 1);
      dim3 grid(upper(h_samples_size, block.x), 1, 1);
      KERNEL_SWITCH(
          knn_assign_shmem,
          <<<grid, block, KNN_BLOCK_SIZE_SHMEM * 2 * k * sizeof(uint32_t)>>>(
              offset, length, k, (*distances)[devi].get(), (*radiuses)[devi].get(),
              reinterpret_cast<const F*>(samples[devi].get()),
              reinterpret_cast<const F*>(centroids[devi].get()),
              assignments[devi].get(), inv_asses[devi].get(),
              inv_asses_offsets[devi].get(), (*neighbors)[devi].get()));
    }
  );
  uint64_t dists_calced = 0;
  FOR_EACH_DEV(
    uint64_t h_dists_calced = 0;
    CUCH(cudaMemcpyFromSymbol(&h_dists_calced, d_dists_calced, sizeof(h_dists_calced)),
         kmcudaMemoryCopyError);
    DEBUG("#%d dists_calced: %" PRIu64 "\n", dev, h_dists_calced);
    dists_calced += h_dists_calced;
  );
  uint64_t max_dists_calced = static_cast<uint64_t>(h_samples_size) * h_samples_size;
  INFO("calculated %f of all the distances\n", (dists_calced + .0) / max_dists_calced);
  return kmcudaSuccess;
}

}  // extern "C"
