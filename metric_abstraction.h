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
  FPATTR static F sum_squares(
      const F *__restrict__ vec, F *__restrict__ cache) {
    F ssqr = _const<F>(0), corr = _const<F>(0);
    #pragma unroll 4
    for (int f = 0; f < d_features_size; f++) {
      F v = vec[f];
      if (cache) {
        cache[f] = v;
      }
      F y = _fma(corr, v, v);
      F t = _add(ssqr, y);
      corr = _sub(y, _sub(t, ssqr));
      ssqr = t;
    }
    return ssqr;
  }

  FPATTR static typename HALF<F>::type distance(F sqr1, F sqr2, F prod) {
    return _fin(_fma(_add(sqr1, sqr2), _const<F>(-2), prod));
  }

  FPATTR static float distance(const F *__restrict__ v1, const F *__restrict__ v2) {
    // Kahan summation with inverted c
    F dist = _const<F>(0);
    F corr = _const<F>(0);
    #pragma unroll 4
    for (uint16_t f = 0; f < d_features_size; f++) {
      F d = _sub(v1[f], v2[f]);
      F y = _fma(corr, d, d);
      F t = _add(dist, y);
      corr = _sub(y, _sub(t, dist));
      dist = t;
    }
    return _sqrt(_float(_fin(dist)));
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
  FPATTR static F sum_squares(
      const F *__restrict__ vec, F *__restrict__ cache) {
    if (cache) {
      #pragma unroll 4
      for (uint16_t f = 0; f < d_features_size; f++) {
        cache[f] = vec[f];
      }
    }
    return _const<F>(1);
  }

  FPATTR static typename HALF<F>::type distance(
      F sqr1 __attribute__((unused)), F sqr2 __attribute__((unused)), F prod) {
    return _half<F>(acos(_float(_fin(prod))));
  }

  FPATTR static float distance(const F *__restrict__ v1, const F *__restrict__ v2) {
    // Kahan summation with inverted c
    F prod = _const<F>(0), corr = _const<F>(0);
    #pragma unroll 4
    for (uint16_t f = 0; f < d_features_size; f++) {
      F yprod = _fma(corr, v1[f], v2[f]);
      F tprod = _add(prod, yprod);
      corr = _sub(yprod, _sub(tprod, prod));
      prod = tprod;
    }
    return _float(distance(_const<F>(1), _const<F>(1), prod));
  }

  FPATTR static void normalize(uint32_t count __attribute__((unused)), F *vec) {
    // Kahan summation with inverted c
    F norm = _const<F>(0);
    F corr = _const<F>(0);
    #pragma unroll 4
    for (int f = 0; f < d_features_size; f++) {
      F v = vec[f];
      F y = _fma(corr, v, v);
      F t = _add(norm, y);
      corr = _sub(y, _sub(t, norm));
      norm = t;
    }

    norm = _fout(_reciprocal(_sqrt(_fin(norm))));
    #pragma unroll 4
    for (int f = 0; f < d_features_size; f++) {
      vec[f] = _mul(vec[f], norm);
    }
  }
};

#endif //KMCUDA_METRIC_ABSTRACTION_H
