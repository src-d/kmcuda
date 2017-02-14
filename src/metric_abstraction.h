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

  FPATTR static F sum_squares_t(
      const F *__restrict__ vec, F *__restrict__ cache, uint64_t size, uint64_t index) {
    F ssqr = _const<F>(0), corr = _const<F>(0);
    #pragma unroll 4
    for (uint64_t f = 0; f < d_features_size; f++) {
      F v = vec[f * size + index];
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
    F dist = _const<F>(0), corr = _const<F>(0);
    #pragma unroll 4
    for (int f = 0; f < d_features_size; f++) {
      F d = _sub(v1[f], v2[f]);
      F y = _fma(corr, d, d);
      F t = _add(dist, y);
      corr = _sub(y, _sub(t, dist));
      dist = t;
    }
    return _sqrt(_float(_fin(dist)));
  }

  FPATTR static float distance_t(const F *__restrict__ v1, const F *__restrict__ v2,
                                 uint64_t v1_size, uint64_t v1_index) {
    // Kahan summation with inverted c
    F dist = _const<F>(0), corr = _const<F>(0);
    #pragma unroll 4
    for (uint64_t f = 0; f < d_features_size; f++) {
      F d = _sub(v1[v1_size * f + v1_index], v2[f]);
      F y = _fma(corr, d, d);
      F t = _add(dist, y);
      corr = _sub(y, _sub(t, dist));
      dist = t;
    }
    return _sqrt(_float(_fin(dist)));
  }

  FPATTR static float distance_tt(const F *__restrict__ v, uint64_t size,
                                  uint64_t index1, uint64_t index2) {
    // Kahan summation with inverted c
    F dist = _const<F>(0), corr = _const<F>(0);
    #pragma unroll 4
    for (uint64_t f = 0; f < d_features_size; f++) {
      F d = _sub(v[size * f + index1], v[size * f + index2]);
      F y = _fma(corr, d, d);
      F t = _add(dist, y);
      corr = _sub(y, _sub(t, dist));
      dist = t;
    }
    return _sqrt(_float(_fin(dist)));
  }

  FPATTR static float partial(const F *__restrict__ v1, const F *__restrict__ v2,
                              uint16_t size) {
    // Kahan summation with inverted c
    F dist = _const<F>(0), corr = _const<F>(0);
    #pragma unroll 4
    for (int f = 0; f < size; f++) {
      F d = _sub(v1[f], v2[f]);
      F y = _fma(corr, d, d);
      F t = _add(dist, y);
      corr = _sub(y, _sub(t, dist));
      dist = t;
    }
    return _float(_fin(dist));
  }

  FPATTR static float partial_t(
      const F *__restrict__ v1, const F *__restrict__ v2, uint16_t f_size,
      uint64_t v1_size, uint64_t v1_offset, uint64_t v1_index) {
    // Kahan summation with inverted c
    F dist = _const<F>(0), corr = _const<F>(0);
    #pragma unroll 4
    for (int f = 0; f < f_size; f++) {
      F d = _sub(v1[v1_size * (f + v1_offset) + v1_index], v2[f]);
      F y = _fma(corr, d, d);
      F t = _add(dist, y);
      corr = _sub(y, _sub(t, dist));
      dist = t;
    }
    return _float(_fin(dist));
  }

  FPATTR static float finalize(float partial) {
    return _sqrt(partial);
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
      for (int f = 0; f < d_features_size; f++) {
        cache[f] = vec[f];
      }
    }
    return _const<F>(1);
  }

  FPATTR static F sum_squares_t(
      const F *__restrict__ vec, F *__restrict__ cache, uint64_t size, uint64_t index) {
    if (cache) {
      #pragma unroll 4
      for (uint64_t f = 0; f < d_features_size; f++) {
        cache[f] = vec[f * size + index];
      }
    }
    return _const<F>(1);
  }

  FPATTR static typename HALF<F>::type distance(
      F sqr1 __attribute__((unused)), F sqr2 __attribute__((unused)), F prod) {
    float fp = _float(_fin(prod));
    if (fp >= 1.f) return _half<F>(0.f);
    if (fp <= -1.f) return _half<F>(M_PI);
    return _half<F>(acos(fp));
  }

  FPATTR static float distance(const F *__restrict__ v1, const F *__restrict__ v2) {
    // Kahan summation with inverted c
    F prod = _const<F>(0), corr = _const<F>(0);
    #pragma unroll 4
    for (int f = 0; f < d_features_size; f++) {
      F yprod = _fma(corr, v1[f], v2[f]);
      F tprod = _add(prod, yprod);
      corr = _sub(yprod, _sub(tprod, prod));
      prod = tprod;
    }
    return _float(distance(_const<F>(1), _const<F>(1), prod));
  }

  FPATTR static float distance_t(const F *__restrict__ v1, const F *__restrict__ v2,
                                 uint64_t v1_size, uint64_t v1_index) {
    // Kahan summation with inverted c
    F prod = _const<F>(0), corr = _const<F>(0);
    #pragma unroll 4
    for (uint64_t f = 0; f < d_features_size; f++) {
      F yprod = _fma(corr, v1[v1_size * f + v1_index], v2[f]);
      F tprod = _add(prod, yprod);
      corr = _sub(yprod, _sub(tprod, prod));
      prod = tprod;
    }
    return _float(distance(_const<F>(1), _const<F>(1), prod));
  }

  FPATTR static float distance_tt(const F *__restrict__ v, uint64_t size,
                                  uint64_t index1, uint64_t index2) {
    // Kahan summation with inverted c
    F prod = _const<F>(0), corr = _const<F>(0);
    #pragma unroll 4
    for (uint64_t f = 0; f < d_features_size; f++) {
      F yprod = _fma(corr, v[size * f + index1], v[size * f + index2]);
      F tprod = _add(prod, yprod);
      corr = _sub(yprod, _sub(tprod, prod));
      prod = tprod;
    }
    return _float(distance(_const<F>(1), _const<F>(1), prod));
  }

  FPATTR static float partial(const F *__restrict__ v1, const F *__restrict__ v2,
                              uint16_t size) {
    // Kahan summation with inverted c
    F prod = _const<F>(0), corr = _const<F>(0);
    #pragma unroll 4
    for (int f = 0; f < size; f++) {
      F yprod = _fma(corr, v1[f], v2[f]);
      F tprod = _add(prod, yprod);
      corr = _sub(yprod, _sub(tprod, prod));
      prod = tprod;
    }
    return _float(_fin(prod));
  }

  FPATTR static float partial_t(
      const F *__restrict__ v1, const F *__restrict__ v2, uint16_t f_size,
      uint64_t v1_size, uint64_t v1_offset, uint64_t v1_index) {
    // Kahan summation with inverted c
    F prod = _const<F>(0), corr = _const<F>(0);
    #pragma unroll 4
    for (int f = 0; f < f_size; f++) {
      F yprod = _fma(corr, v1[v1_size * (f + v1_offset) + v1_index], v2[f]);
      F tprod = _add(prod, yprod);
      corr = _sub(yprod, _sub(tprod, prod));
      prod = tprod;
    }
    return _float(_fin(prod));
  }

  FPATTR static float finalize(float partial) {
    if (partial >= 1.f) return 0.f;
    if (partial <= -1.f) return M_PI;
    return acos(partial);
  }

  FPATTR static void normalize(uint32_t count __attribute__((unused)), float *vec) {
    // Kahan summation with inverted c
    float norm = 0, corr = 0;
    #pragma unroll 4
    for (int f = 0; f < d_features_size; f++) {
      float v = vec[f];
      float y = _fma(corr, v, v);
      float t = norm + y;
      corr = y - (t - norm);
      norm = t;
    }
    norm = _reciprocal(_sqrt(norm));

    #pragma unroll 4
    for (int f = 0; f < d_features_size; f++) {
      vec[f] = vec[f] * norm;
    }
  }

  #if CUDA_ARCH >= 60
  FPATTR static void normalize(uint32_t count __attribute__((unused)), half2 *vec) {
    // We really have to calculate norm in 32-bit floats because the maximum
    // value which 16-bit float may represent is 2^16.
    float norm = 0, corr = 0;
    #pragma unroll 4
    for (int f = 0; f < d_features_size; f++) {
      half2 v = vec[f];
      float v1 = _float(__high2half(v));
      float v2 = _float(__low2half(v));

      float y = _fma(corr, v1, v1);
      float t = norm + y;
      corr = y - (t - norm);
      norm = t;

      y = _fma(corr, v2, v2);
      t = norm + y;
      corr = y - (t - norm);
      norm = t;
    }
    norm = _reciprocal(_sqrt(norm));
    half2 norm2 = _fout(_half<half2>(norm));
    #pragma unroll 4
    for (int f = 0; f < d_features_size; f++) {
      vec[f] = _mul(vec[f], norm2);
    }
  }
  #endif
};

#endif //KMCUDA_METRIC_ABSTRACTION_H
