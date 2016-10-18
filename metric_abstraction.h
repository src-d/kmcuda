//
// distance and normalization functions.
//

#ifndef KMCUDA_METRIC_ABSTRACTION_H
#define KMCUDA_METRIC_ABSTRACTION_H

#include "fp_abstraction.h"

__constant__ uint16_t d_features_size;

template <KMCUDADistanceMetric M, typename F>
struct METRIC {
  FPATTR static typename HALF<F>::type distance(F sqr1, F sqr2, F prod);
  FPATTR static float distance(const F *__restrict__ v1, const F *__restrict__ v2);
  FPATTR static void normalize(uint32_t count, F *vec);
};

template <typename F>
struct METRIC<kmcudaDistanceMetricL2, F> {
  FPATTR static typename HALF<F>::type distance(F sqr1, F sqr2, F prod) {
    return _fin(_fma(_add(sqr1, sqr2), _const<F>(-2), prod));
  }

  FPATTR static float distance(const F *__restrict__ v1, const F *__restrict__ v2) {
    F dist = _const<F>(0);
    #pragma unroll 4
    for (uint16_t f = 0; f < d_features_size; f++) {
      F d = _sub(v1[f], v2[f]);
      dist = _fma(dist, d, d);
    }
    return sqrt(_float(_fin(dist)));
  }

  FPATTR static void normalize(uint32_t count, F *vec) {
    F rc = _reciprocal(_const<F>(count));
    #pragma unroll 4
    for (int f = 0; f < d_features_size; f++) {
      vec[f] = _mul(vec[f], rc);
    }
  }
};

template <typename F>
struct METRIC<kmcudaDistanceMetricCosine, F> {
  FPATTR static typename HALF<F>::type distance(F sqr1, F sqr2, F prod) {
    float fsqr1 = _float(_fin(sqr1)), fsqr2 = _float(_fin(sqr2)),
        fprod = _float(_fin(prod));
    return _half<F>(acos(fprod / sqrt(fsqr1 * fsqr2)));
  }

  FPATTR static float distance(const F *__restrict__ v1, const F *__restrict__ v2) {
    F n1 = _const<F>(0), n2 = _const<F>(0), prod = _const<F>(0);
    #pragma unroll 4
    for (uint16_t f = 0; f < d_features_size; f++) {
      F f1 = v1[f];
      F f2 = v2[f];
      n1 = _fma(n1, f1, f1);
      n2 = _fma(n2, f2, f2);
      prod = _fma(prod, f1, f2);
    }
    return _float(distance(n1, n2, prod));
  }

  FPATTR static void normalize(uint32_t count __attribute__((unused)), F *vec) {
    F norm = _const<F>(0);
    #pragma unroll 4
    for (int f = 0; f < d_features_size; f++) {
      F v = vec[f];
      norm = _fma(norm, v, v);
    }

    norm = _fout(_reciprocal(_sqrt(_fin(norm))));
    #pragma unroll 4
    for (int f = 0; f < d_features_size; f++) {
      vec[f] = _mul(vec[f], norm);
    }
  }
};

#endif //KMCUDA_METRIC_ABSTRACTION_H
