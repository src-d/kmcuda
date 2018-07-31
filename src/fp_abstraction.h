//
// half, half2 and float functions.
//

#ifndef KMCUDA_FP_ABSTRACTION_H
#define KMCUDA_FP_ABSTRACTION_H

#include <cfloat>

#if CUDA_ARCH >= 60
#undef volatile
#define volatile volatile
#include <cuda_fp16.h>
#undef volatile
#define volatile
#endif

#define FPATTR __device__ __forceinline__

template <typename F>
struct HALF;

template <>
struct HALF<float> {
    using type = float;
};

template <typename HF>
FPATTR HF _fmax();

template <>
FPATTR float _fmax() {
  return FLT_MAX;
}

template <typename F>
FPATTR F _const(int);

template <>
FPATTR float _const(int v) {
  return v;
}

FPATTR float _fin(float v) {
  return v;
}

FPATTR float _fout(float v) {
  return v;
}

FPATTR float _float(float v) {
  return v;
}

template <class F>
FPATTR typename HALF<F>::type _half(float v);

template <>
FPATTR float _half<float>(float v) {
  return v;
}

FPATTR bool _eq(float v1, float v2) {
  return v1 == v2;
}

FPATTR bool _neq(float v1, float v2) {
  return v1 != v2;
}

FPATTR float _add(float v1, float v2) {
  return v1 + v2;
}

FPATTR float _sub(float v1, float v2) {
  return v1 - v2;
}

FPATTR float _mul(float v1, float v2) {
  return v1 * v2;
}

FPATTR float _reciprocal(float v) {
  return __frcp_rn(v);
}

FPATTR float _fma(float acc, float v1, float v2) {
  return __fmaf_rd(v1, v2, acc);
}

FPATTR bool _lt(float v1, float v2) {
  return v1 < v2;
}

FPATTR float _sqrt(float v) {
  return __fsqrt_rn(v);
}

#if CUDA_ARCH >= 60
template <>
struct HALF<half2> {
  using type = half;
};

template <>
FPATTR half _fmax() {
  return __int2half_rd(65504);
}

template <>
FPATTR half2 _const(int v) {
  return __half2half2(__int2half_rd(v));
}

template <>
FPATTR half _const(int v) {
  return __int2half_rd(v);
}

FPATTR half _fin(half2 v) {
  return __hadd(__high2half(v), __low2half(v));
}

FPATTR half2 _fout(half v) {
  return __half2half2(v);
}

FPATTR float _float(half v) {
  return __half2float(v);
}

template <>
FPATTR half _half<half2>(float v) {
  return __float2half(v);
}

FPATTR bool _eq(half2 v1, half2 v2) {
  return __hbeq2(v1, v2);
}

FPATTR bool _neq(half2 v1, half2 v2) {
  return !__hbeq2(v1, v2);
}

FPATTR half2 _add(half2 v1, half2 v2) {
  return __hadd2(v1, v2);
}

FPATTR half2 _sub(half2 v1, half2 v2) {
  return __hsub2(v1, v2);
}

FPATTR half2 _mul(half2 v1, half2 v2) {
  return __hmul2(v1, v2);
}

FPATTR half _reciprocal(half v) {
  return hrcp(v);
}

FPATTR half2 _reciprocal(half2 v) {
  return h2rcp(v);
}

FPATTR half2 _fma(half2 acc, half2 v1, half2 v2) {
  return __hfma2(v1, v2, acc);
}

FPATTR bool _lt(half v1, half v2) {
  return __hlt(v1, v2);
}

FPATTR half _sqrt(half v) {
  return hsqrt(v);
}

FPATTR half2 _sqrt(half2 v) {
  return h2sqrt(v);
}

#endif // CUDA_ARCH >= 60

#endif // KMCUDA_FP_ABSTRACTION_H
