#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include "kmcuda.h"

extern "C" {

static SEXP r_kmeans_cuda(SEXP args);

static R_ExternalMethodDef externalMethods[] = {
   {"kmeans_cuda", (DL_FUNC) &r_kmeans_cuda, -1},
   {NULL, NULL, 0}
};

void R_init_libKMCUDA(DllInfo *info) {
  R_registerRoutines(info, NULL, NULL, NULL, externalMethods);
}

static SEXP r_kmeans_cuda(SEXP args) {
  Rprintf("%d arguments\n", length(args));
  return R_NilValue;
}

}  // extern "C"
