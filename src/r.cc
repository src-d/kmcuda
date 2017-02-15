#include <memory>
#include <string>
#include <unordered_map>

#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include "kmcuda.h"

namespace {
  std::unordered_map<std::string, SEXP> parse_args(
      std::initializer_list<std::string> required, SEXP args) {
    std::unordered_map<std::string, SEXP> result;
    args = CDR(args);
    bool pure = true;
    for (unsigned i = 0; args != R_NilValue; i++, args = CDR(args)) {
      SEXP arg = CAR(args);
      if (isNull(TAG(args))) {
        if (pure && i == result.size()) {
          result.emplace(required.begin()[i], arg);
        } else {
          error("positional argument follows keyword argument");
        }
      } else {
        pure = false;
        result.emplace(CHAR(PRINTNAME(TAG(args))), arg);
      }
    }
    return result;
  }

  template<typename R>
  R parse_dict(const std::unordered_map<std::string, R>& dict, const char *arg,
               SEXP name) {
    if (!isString(name)) {
      error("\"%s\" name must be a string", arg);
    }
    const char *init_name = CHAR(asChar(name));
    auto init_iter = dict.find(init_name);
    if (init_iter == dict.end()) {
      error("\"%s\" = \"%s\" is not supported", arg, init_name);
    }
    return init_iter->second;
  }
}

extern "C" {

static SEXP r_kmeans_cuda(SEXP args);
static SEXP r_knn_cuda(SEXP args);

static R_ExternalMethodDef externalMethods[] = {
   {"kmeans_cuda", (DL_FUNC) &r_kmeans_cuda, -1},
   {"knn_cuda", (DL_FUNC) &r_knn_cuda, -1},
   {NULL, NULL, 0}
};

void R_init_libKMCUDA(DllInfo *info) {
  R_registerRoutines(info, NULL, NULL, NULL, externalMethods);
}

static SEXP r_kmeans_cuda(SEXP args) {
  auto kwargs = parse_args({"samples", "clusters"}, args);
  std::unique_ptr<SEXP[]> samples_chunks;
  int chunks_size = 0;
  {
    SEXP samples_obj = kwargs["samples"];
    if (isVector(samples_obj)) {
      chunks_size = length(samples_obj);
      samples_chunks.reset(new SEXP[chunks_size]);
      for (unsigned i = 0; samples_obj != R_NilValue;
           i++, samples_obj = CDR(samples_obj)) {
        samples_chunks[i] = CAR(samples_obj);
      }
    } else {
      chunks_size = 1;
      samples_chunks.reset(new SEXP[1]);
      samples_chunks[0] = samples_obj;
    }
  }
  int samples_size = 0, features_size = 0;
  for (int i = 0; i < chunks_size; i++) {
    if (!isReal(samples_chunks[i])) {
      error("\"samples\" must be a 2D real matrix or a vector of 2D real matrices");
    }
    SEXP dims = getAttrib(samples_chunks[i], R_DimSymbol);
    if (length(dims) != 2) {
      error("\"samples\" must be a 2D real matrix or a vector of 2D real matrices");
    }
    int samples_size_i = INTEGER(dims)[0];
    if (static_cast<int64_t>(samples_size) + samples_size_i > INT32_MAX) {
      error("too many samples (>INT32_MAX)");
    }
    samples_size += samples_size_i;
	  int features_size_i = INTEGER(dims)[1];
    if (i == 0) {
      features_size = features_size_i;
    } else if (features_size != features_size_i) {
      error("\"samples\" vector contains matrices with different number of columns");
    }
  }
  auto samples = std::unique_ptr<float[]>(new float[
      static_cast<uint64_t>(samples_size) * features_size]);
  float *samples_float = samples.get();
  {
    int offset = 0;
    for (int i = 0; i < chunks_size; i++) {
      double *samples_double = REAL(samples_chunks[i]);
      SEXP dims = getAttrib(samples_chunks[i], R_DimSymbol);
      int samples_size_i = INTEGER(dims)[0] * features_size;
      #pragma omp parallel for simd
      for (int i = 0; i < samples_size_i; i++) {
        samples_float[offset + i] = samples_double[i];
      }
      offset += samples_size_i;
    }
  }
  SEXP clusters_obj = kwargs["clusters"];
  if (!isInteger(clusters_obj)) {
    error("\"clusters\" must be a positive integer");
  }
  int clusters_size = INTEGER(clusters_obj)[0];
  if (clusters_size <= 0) {
    error("\"clusters\" must be a positive integer");
  }
  if (static_cast<uint64_t>(clusters_size) * features_size > INT32_MAX
      || clusters_size >= samples_size) {
    error("\"clusters\" is too big");
  }
  KMCUDAInitMethod init = kmcudaInitMethodPlusPlus;
  int afkmc2_m = 0;
  auto init_iter = kwargs.find("init");
  if (init_iter != kwargs.end()) {
    if (isString(init_iter->second)) {
      init = parse_dict(kmcuda::init_methods, "init", init_iter->second);
    } else if (isList(init_iter->second)) {
      if (length(init_iter->second) == 0) {
        error("\"init\" may not be an empty list");
      }
      init = parse_dict(kmcuda::init_methods, "init", CAR(init_iter->second));
      if (init == kmcudaInitMethodAFKMC2 && length(init_iter->second) > 1) {
        SEXP afkmc2_m_obj = CAAR(init_iter->second);
        if (!isInteger(afkmc2_m_obj)) {
          error("\"init\" = %s: parameter must be a positive integer",
                CHAR(asChar(CAR(init_iter->second))));
        }
        afkmc2_m = INTEGER(afkmc2_m_obj)[0];
        if (afkmc2_m <= 0) {
          error("\"init\" = %s: parameter must be a positive integer",
                CHAR(asChar(CAR(init_iter->second))));
        }
      } else if (length(init_iter->second) != 1) {
        error("\"init\" has wrong number of parameters");
      }
    } else {
      error("\"init\" must be either a string or a list");
    }
  }
  float tolerance = 0.01;
  auto tolerance_iter = kwargs.find("tolerance");
  if (tolerance_iter != kwargs.end()) {
    if (!isReal(tolerance_iter->second)) {
      error("\"tolerance\" must be a real value");
    }
    tolerance = REAL(tolerance_iter->second)[0];
    if (tolerance < 0 || tolerance > 1) {
      error("\"tolerance\" must be in [0, 1]");
    }
  }
  float yinyang_t = 0.1;
  auto yinyang_t_iter = kwargs.find("yinyang_t");
  if (yinyang_t_iter != kwargs.end()) {
    if (!isReal(yinyang_t_iter->second)) {
      error("\"yinyang_t\" must be a real value");
    }
    yinyang_t = REAL(yinyang_t_iter->second)[0];
    if (yinyang_t < 0 || yinyang_t > 0.5) {
      error("\"tolerance\" must be in [0, 0.5]");
    }
  }
  KMCUDADistanceMetric metric = kmcudaDistanceMetricL2;
  auto metric_iter = kwargs.find("metric");
  if (metric_iter != kwargs.end()) {
    metric = parse_dict(kmcuda::metrics, "metric", metric_iter->second);
  }
  uint32_t seed = static_cast<uint32_t>(time(NULL));
  auto seed_iter = kwargs.find("seed");
  if (seed_iter != kwargs.end()) {
    seed = static_cast<uint32_t>(INTEGER(seed_iter->second)[0]);
  }
  int device = 0;
  auto device_iter = kwargs.find("device");
  if (device_iter != kwargs.end()) {
    device = INTEGER(device_iter->second)[0];
    if (device < 0) {
      error("\"device\" may not be negative");
    }
  }
  int verbosity = 0;
  auto verbosity_iter = kwargs.find("verbosity");
  if (verbosity_iter != kwargs.end()) {
    verbosity = INTEGER(verbosity_iter->second)[0];
  }
  float average_distance, *average_distance_ptr = nullptr;
  auto average_distance_iter = kwargs.find("average_distance");
  if (average_distance_iter != kwargs.end()) {
    if (LOGICAL(average_distance_iter->second)[0]) {
      average_distance_ptr = &average_distance;
    }
  }
  auto centroids = std::unique_ptr<float[]>(
      new float[clusters_size * features_size]);
  auto assignments = std::unique_ptr<uint32_t[]>(new uint32_t[samples_size]);
  auto result = kmeans_cuda(
    init, &afkmc2_m, tolerance, yinyang_t, metric, samples_size, features_size,
    clusters_size, seed, device, -1, 0, verbosity, samples_float,
    centroids.get(), assignments.get(), average_distance_ptr);
  if (result != kmcudaSuccess) {
    error("kmeans_cuda error %d %s%s", result,
          kmcuda::statuses.find(result)->second, (verbosity > 0)? "" : "; "
            "\"verbosity\" = 2 would reveal the details");
  }
  SEXP centroids2 = PROTECT(allocMatrix(REALSXP, clusters_size, features_size));
  double *centroids_double = REAL(centroids2);
  float *centroids_float = centroids.get();
  #pragma omp parallel for simd
  for (int i = 0; i < clusters_size * features_size; i++) {
    centroids_float[i] = centroids_double[i];
  }
  SEXP assignments2 = PROTECT(allocVector(INTSXP, samples_size));
  memcpy(INTEGER(assignments2), assignments.get(), samples_size * sizeof(int));
  SEXP tuple = PROTECT(allocVector(VECSXP, 2 + (average_distance_ptr != nullptr)));
  SET_VECTOR_ELT(tuple, 0, centroids2);
  SET_VECTOR_ELT(tuple, 1, assignments2);
  if (average_distance_ptr != nullptr) {
    SEXP average_distance2 = PROTECT(allocVector(REALSXP, 1));
    REAL(average_distance2)[0] = average_distance;
  }
  UNPROTECT(3 + (average_distance_ptr != nullptr));
  return tuple;
}

static SEXP r_knn_cuda(SEXP args) {
  Rprintf("%d arguments\n", length(args));
  return R_NilValue;
}

}  // extern "C"
