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
  if ((threadIdx.x % 32) == leader) {
    res = atomicAdd(ctr, __popc(mask));
  }
  res = __shfl(res, leader);
  return res + __popc(mask & ((1 << (threadIdx.x % 32)) - 1));
}

// https://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
template <typename T>
__device__ __forceinline__ T warpReduceSum(T val) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_down(val, offset);
  }
  return val;
}