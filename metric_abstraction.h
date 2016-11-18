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
  FPATTR static typename HALF<F>::type distance(F sqr1, F sqr2, F prod) {
    float fsqr1 = _float(_fin(sqr1)), fsqr2 = _float(_fin(sqr2)),
        fprod = _float(_fin(prod));
    float sqr_norm = fsqr1 * fsqr2;
    assert(sqr_norm > 0);
    return _half<F>(acos(fprod / _sqrt(sqr_norm)));
  }

  FPATTR static float distance(const F *__restrict__ v1, const F *__restrict__ v2) {
    // Kahan summation with inverted c
    F n1 = _const<F>(0), n2 = _const<F>(0), prod = _const<F>(0);
    F corr1 = _const<F>(0), corr2 = _const<F>(0), corrprod = _const<F>(0);
    #pragma unroll 4
    for (uint16_t f = 0; f < d_features_size; f++) {
      F f1 = v1[f];
      F f2 = v2[f];

      F y1 = _fma(corr1, f1, f1);
      F t1 = _add(n1, y1);
      corr1 = _sub(y1, _sub(t1, n1));
      n1 = t1;

      F y2 = _fma(corr2, f2, f2);
      F t2 = _add(n2, y2);
      corr2 = _sub(y2, _sub(t2, n2));
      n2 = t2;

      F yprod = _fma(corrprod, f1, f2);
      F tprod = _add(prod, yprod);
      corrprod = _sub(yprod, _sub(tprod, prod));
      prod = tprod;
    }
    return _float(distance(n1, n2, prod));
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
