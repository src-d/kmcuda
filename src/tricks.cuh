#include <cstdint>

#define warpSize 32

template <typename T>
__device__ __forceinline__ T dupper(T size, T each) {
  T div = size / each;
  if (div * each == size) {
    return div;
  }
  return div + 1;
}

/*template <typename T>
__device__ __forceinline__ T dmin(T a, T b) {
  return a <= b? a : b;
}*/

// https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/
__device__ __forceinline__ uint32_t atomicAggInc(uint32_t *ctr) {
  int mask = __ballot(1);
  int leader = __ffs(mask) - 1;
  uint32_t res;
  if ((threadIdx.x % warpSize) == leader) {
    res = atomicAdd(ctr, __popc(mask));
  }
  res = __shfl(res, leader);
  return res + __popc(mask & ((1 << (threadIdx.x % warpSize)) - 1));
}

// https://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
template <typename T>
__device__ __forceinline__ T warpReduceSum(T val) {
  #pragma unroll
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    val += __shfl_down(val, offset);
  }
  return val;
}

template <typename T>
__device__ __forceinline__ T warpReduceMin(T val) {
  #pragma unroll
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    val = min(val, __shfl_down(val, offset));
  }
  return val;
}

// https://github.com/parallel-forall/code-samples/blob/master/posts/cuda-aware-mpi-example/src/Device.cu#L53
__device__ __forceinline__ void atomicMin(
    float *const address, const float value) {
	if (*address <= value) {
		return;
	}

	int32_t *const address_as_i = reinterpret_cast<int32_t*>(address);
	int32_t old = *address_as_i, assumed;

	do {
		assumed = old;
		if (__int_as_float(assumed) <= value) {
			break;
		}
		old = atomicCAS(address_as_i, assumed, __float_as_int(value));
	} while (assumed != old);
}
