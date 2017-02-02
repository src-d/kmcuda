#include "private.h"

__global__ void copy_sample_t(
    uint32_t index, uint32_t samples_size, uint16_t features_size,
    const float *__restrict__ samples, float *__restrict__ dest) {
  uint32_t ti = blockIdx.x * blockDim.x + threadIdx.x;
  if (ti >= features_size) {
    return;
  }
  dest[ti] = samples[static_cast<uint64_t>(samples_size) * static_cast<uint64_t>(ti) + index];
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

KMCUDAResult cuda_inplace_transpose(
    uint32_t samples_size, uint16_t features_size, const std::vector<int> &devs,
    int verbosity, udevptrs<float> *samples) {
  INFO("transposing the samples inplace...\n");
  FOR_EACH_DEVI(
    //inplace_transpose((*samples)[devi].get(), samples_size, features_size);
  );
  SYNC_ALL_DEVS;
  return kmcudaSuccess;
}

}  // extern "C"