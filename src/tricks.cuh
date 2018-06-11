#include <cstdint>

#define warpSize 32

/// Inline function which rounds the ratio between size and each to the nearest
/// greater than or equal integer.
/// @param T Any integer type. Calling dupper() on floating point types is useless.
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

/// Optimized aggregation, equivalent to and a drop-in replacement for atomicInc.
/// https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/
__device__ __forceinline__ uint32_t atomicAggInc(uint32_t *ctr) {
  int mask = ballot(1);
  int leader = __ffs(mask) - 1;
  uint32_t res;
  if ((threadIdx.x % warpSize) == leader) {
    res = atomicAdd(ctr, __popc(mask));
  }
  res = shfl(res, leader);
  return res + __popc(mask & ((1 << (threadIdx.x % warpSize)) - 1));
}

/// Optimized sum reduction, sums all the values across the warp.
/// https://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
template <typename T>
__device__ __forceinline__ T warpReduceSum(T val) {
  #pragma unroll
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    val += shfl_down(val, offset);
  }
  return val;
}

/// Optimized minimum reduction, finds the minimum across the values in the warp.
template <typename T>
__device__ __forceinline__ T warpReduceMin(T val) {
  #pragma unroll
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    val = min(val, shfl_down(val, offset));
  }
  return val;
}

/// This is how would atomicMin() for float-s look like.
/// https://github.com/parallel-forall/code-samples/blob/master/posts/cuda-aware-mpi-example/src/Device.cu#L53
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
