#include "private.h"

#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void copy_sample_t(
    uint32_t index, uint32_t samples_size, uint16_t features_size,
    const float *__restrict__ samples, float *__restrict__ dest) {
  uint32_t ti = blockIdx.x * blockDim.x + threadIdx.x;
  if (ti >= features_size) {
    return;
  }
  dest[ti] = samples[static_cast<uint64_t>(samples_size) * static_cast<uint64_t>(ti) + index];
}

__global__ void transpose(
    const float *__restrict__ input, uint32_t rows, uint32_t cols,
    float *__restrict__ output) {
  __shared__ float tile[TILE_DIM][TILE_DIM + 1];
  volatile uint64_t x = blockIdx.x * TILE_DIM + threadIdx.x;
  volatile uint64_t y = blockIdx.y * TILE_DIM + threadIdx.y;

  if (x < cols && y < rows) {
    for (uint32_t j = 0;
         j < min(static_cast<unsigned long>(TILE_DIM), rows - y);
         j += BLOCK_ROWS) {
      tile[threadIdx.y + j][threadIdx.x] = input[
          static_cast<uint64_t>(y + j) * cols + x];
    }
  }

  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  if (x < rows && y < cols) {
    for (uint32_t j = 0;
         j < min(static_cast<unsigned long>(TILE_DIM), cols - y);
         j += BLOCK_ROWS) {
      output[static_cast<uint64_t>(y + j) * rows + x] =
          tile[threadIdx.x][threadIdx.y + j];
    }
  }
}

extern "C" {

KMCUDAResult cuda_copy_sample_t(
    uint32_t index, uint32_t offset, uint32_t samples_size, uint16_t features_size,
    const std::vector<int> &devs, int verbosity, const udevptrs<float> &samples,
    udevptrs<float> *dest) {
  FOR_EACH_DEVI(
    dim3 block(min(1024, features_size), 1, 1);
    dim3 grid(upper(static_cast<unsigned>(features_size), block.x), 1, 1);
    copy_sample_t<<<grid, block>>>(
        index, samples_size, features_size, samples[devi].get(),
        (*dest)[devi].get() + offset);
  );
  return kmcudaSuccess;
}

KMCUDAResult cuda_transpose(
    uint32_t samples_size, uint16_t features_size, bool forward,
    const std::vector<int> &devs, int verbosity, udevptrs<float> *samples) {
  INFO("transposing the samples...\n");
  uint64_t size = static_cast<uint64_t>(samples_size) * features_size * sizeof(float);
  float *ptr;
  CUCH(cudaMallocManaged(&ptr, size), kmcudaMemoryAllocationFailure);
  unique_devptr<float> managed(ptr);
  cudaSetDevice(devs[0]);
  CUCH(cudaMemcpy(ptr, (*samples)[0].get(), size, cudaMemcpyDefault), kmcudaMemoryCopyError);
  uint32_t cols, rows;
  if (forward) {
    cols = features_size;
    rows = samples_size;
  } else {
    cols = samples_size;
    rows = features_size;
  }
  dim3 block(TILE_DIM, BLOCK_ROWS, 1);
  dim3 grid(upper(cols, static_cast<uint32_t>(TILE_DIM)), upper(rows, static_cast<uint32_t>(TILE_DIM)), 1);
  DEBUG("transpose <<<(%d, %d), (%d, %d)>>> %" PRIu32 ", %" PRIu32 "\n",
        grid.x, grid.y, block.x, block.y, rows, cols);
  FOR_EACH_DEVI(
    transpose<<<grid, block>>>(ptr, rows, cols, (*samples)[devi].get());
  );
  SYNC_ALL_DEVS;
  return kmcudaSuccess;
}

}  // extern "C"