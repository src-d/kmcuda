/// avoid spurious trailing ‘%’ in format error
/// see https://stackoverflow.com/questions/8132399/how-to-printf-uint64-t-fails-with-spurious-trailing-in-format
#define __STDC_FORMAT_MACROS
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <cuda_runtime_api.h>
#include "kmcuda.h"

static char module_docstring[] =
    "This module provides fast K-means implementation which uses CUDA.";
static char kmeans_cuda_docstring[] =
    "Assigns cluster label to each sample and calculates cluster centers.";
static char knn_cuda_docstring[] =
    "Finds the nearest neighbors for every sample.";

static PyObject *py_kmeans_cuda(PyObject *self, PyObject *args, PyObject *kwargs);
static PyObject *py_knn_cuda(PyObject *self, PyObject *args, PyObject *kwargs);

static PyMethodDef module_functions[] = {
  {"kmeans_cuda", reinterpret_cast<PyCFunction>(py_kmeans_cuda),
   METH_VARARGS | METH_KEYWORDS, kmeans_cuda_docstring},
  {"knn_cuda", reinterpret_cast<PyCFunction>(py_knn_cuda),
   METH_VARARGS | METH_KEYWORDS, knn_cuda_docstring},
  {NULL, NULL, 0, NULL}
};

extern "C" {
PyMODINIT_FUNC PyInit_libKMCUDA(void) {
  static struct PyModuleDef moduledef = {
      PyModuleDef_HEAD_INIT,
      "libKMCUDA",         /* m_name */
      module_docstring,    /* m_doc */
      -1,                  /* m_size */
      module_functions,    /* m_methods */
      NULL,                /* m_reload */
      NULL,                /* m_traverse */
      NULL,                /* m_clear */
      NULL,                /* m_free */
  };
  PyObject *m = PyModule_Create(&moduledef);
  if (m == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "PyModule_Create() failed");
    return NULL;
  }
  // numpy
  import_array();
  PyObject_SetAttrString(m, "supports_fp16", CUDA_ARCH >= 60? Py_True : Py_False);
  return m;
}
}

template <typename O>
using pyobj_parent = std::unique_ptr<O, std::function<void(O*)>>;

template <typename O>
class _pyobj : public pyobj_parent<O> {
 public:
  _pyobj() : pyobj_parent<O>(
      nullptr, [](O *p){ if (p) Py_DECREF(p); }) {}
  explicit _pyobj(PyObject *ptr) : pyobj_parent<O>(
      reinterpret_cast<O *>(ptr), [](O *p){ if(p) Py_DECREF(p); }) {}
  void reset(PyObject *p) noexcept {
    pyobj_parent<O>::reset(reinterpret_cast<O*>(p));
  }
};

using pyobj = _pyobj<PyObject>;
using pyarray = _pyobj<PyArrayObject>;

static void set_cuda_malloc_error() {
  PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory on GPU");
}

static void set_cuda_device_error() {
  PyErr_SetString(PyExc_ValueError, "No such CUDA device exists");
}

static void set_cuda_memcpy_error() {
  PyErr_SetString(PyExc_RuntimeError, "cudaMemcpy failed");
}


static bool get_metric(PyObject *metric_obj, KMCUDADistanceMetric *metric) {
  if (metric_obj == Py_None) {
    *metric = kmcudaDistanceMetricL2;
  } else if (!PyUnicode_Check(metric_obj)) {
    PyErr_SetString(
        PyExc_TypeError, "\"metric\" must be either None or string.");
    return false;
  } else {
    pyobj bytes(PyUnicode_AsASCIIString(metric_obj));
    auto immetric = kmcuda::metrics.find(PyBytes_AsString(bytes.get()));
    if (immetric == kmcuda::metrics.end()) {
      PyErr_SetString(
          PyExc_ValueError,
          "Unknown metric. Supported values are \"L2\" and \"cos\".");
      return false;
    }
    *metric = immetric->second;
  }
  return true;
}

static bool validate_features_size(uint32_t features_size) {
  if (features_size > UINT16_MAX) {
    char msg[128];
    sprintf(msg, "\"samples\": more than %" PRIu32 " features is not supported",
            features_size);
    PyErr_SetString(PyExc_ValueError, msg);
    return false;
  }
  return true;
}

static bool get_samples(
    PyObject *samples_obj, pyarray *samples_array, float **samples,
    bool *fp16x2, uint32_t *samples_size, uint32_t *features_size) {
  samples_array->reset(PyArray_FROM_OTF(
      samples_obj, NPY_FLOAT16, NPY_ARRAY_IN_ARRAY));
  if (!*samples_array) {
    PyErr_Clear();
    samples_array->reset(PyArray_FROM_OTF(
        samples_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY));
    if (!*samples_array) {
      PyErr_SetString(PyExc_TypeError,
                      "\"samples\" must be a 2D float32 or float16 numpy array");
      return false;
    }
  } else {
    *fp16x2 = true;
  }
  auto ndims = PyArray_NDIM(samples_array->get());
  if (ndims != 2) {
    PyErr_SetString(PyExc_ValueError, "\"samples\" must be a 2D numpy array");
    return false;
  }
  auto dims = PyArray_DIMS(samples_array->get());
  *samples_size = static_cast<uint32_t>(dims[0]);
  *features_size = static_cast<uint32_t>(dims[1]);
  if (*fp16x2 && PyArray_TYPE(samples_array->get()) == NPY_FLOAT16) {
    if (*features_size % 2 != 0) {
      PyErr_SetString(PyExc_ValueError,
                      "the number of features must be even in fp16 mode");
      return false;
    }
    *features_size /= 2;
  }

  *samples = reinterpret_cast<float *>(PyArray_DATA(
      samples_array->get()));
  return true;
}

static PyObject *py_kmeans_cuda(PyObject *self, PyObject *args, PyObject *kwargs) {
  uint32_t clusters_size = 0,
           afkmc2_m = 0,
           seed = static_cast<uint32_t>(time(NULL)),
           device = 0;
  int32_t verbosity = 0;
  bool fp16x2 = false;
  int adflag = 0;
  float tolerance = .01, yinyang_t = .1;
  PyObject *samples_obj, *init_obj = Py_None, *metric_obj = Py_None;
  static const char *kwlist[] = {
      "samples", "clusters", "tolerance", "init", "yinyang_t", "metric",
      "average_distance", "seed", "device", "verbosity", NULL};

  /* Parse the input tuple */
  if (!PyArg_ParseTupleAndKeywords(
      args, kwargs, "OI|fOfOpIIi", const_cast<char**>(kwlist), &samples_obj,
      &clusters_size, &tolerance, &init_obj, &yinyang_t, &metric_obj, &adflag,
      &seed, &device, &verbosity)) {
    return NULL;
  }

  KMCUDAInitMethod init;
  auto set_init = [&init](PyObject *obj) {
    pyobj bytes(PyUnicode_AsASCIIString(obj));
    auto iminit = kmcuda::init_methods.find(PyBytes_AsString(bytes.get()));
    if (iminit == kmcuda::init_methods.end()) {
      PyErr_SetString(
          PyExc_ValueError,
          "Unknown centroids initialization method. Supported values are "
              "\"kmeans++\", \"random\" and <numpy array>.");
      return false;
    }
    init = iminit->second;
    return true;
  };

  if (init_obj == Py_None) {
    init = kmcudaInitMethodPlusPlus;
  } else if (PyUnicode_Check(init_obj)) {
    if (!set_init(init_obj)) {
      return NULL;
    }
  } else if PyTuple_Check(init_obj) {
    auto e1 = PyTuple_GetItem(init_obj, 0);
    if (e1 == nullptr || e1 == Py_None) {
      PyErr_SetString(
          PyExc_ValueError, "centroid initialization method may not be null.");
      return NULL;
    }
    if (!set_init(e1)) {
      return NULL;
    }
    if (PyTuple_Size(init_obj) > 1 && init == kmcudaInitMethodAFKMC2) {
      afkmc2_m = PyLong_AsUnsignedLong(PyTuple_GetItem(init_obj, 1));
    }
  } else {
    init = kmcudaInitMethodImport;
  }
  KMCUDADistanceMetric metric;
  if (!get_metric(metric_obj, &metric)) {
    return NULL;
  }
  if (clusters_size < 2 || clusters_size == UINT32_MAX) {
    PyErr_SetString(PyExc_ValueError, "\"clusters\" must be greater than 1 and "
                                      "less than (1 << 32) - 1");
    return NULL;
  }
  float *samples = nullptr, *centroids = nullptr;
  uint32_t *assignments = nullptr;
  uint32_t samples_size = 0, features_size = 0;
  int device_ptrs = -1;
  pyarray samples_array;
  if (PyTuple_Check(samples_obj)) {
    auto size = PyTuple_GET_SIZE(samples_obj);
    if (size != 3 && size != 5) {
      PyErr_SetString(PyExc_ValueError,
                      "len(\"samples\") must be either 3 or 5");
      return NULL;
    }
    auto member1 = PyTuple_GetItem(samples_obj, 0),
         member2 = PyTuple_GetItem(samples_obj, 1),
         member3 = PyTuple_GetItem(samples_obj, 2);
    if (!member1 || !member2 || !member3) {
      PyErr_SetString(PyExc_RuntimeError, "\"samples\" tuple contains nulls");
      return NULL;
    }
    auto ull_ptr = PyLong_AsUnsignedLongLong(member1);
    if (ull_ptr == NPY_MAX_ULONGLONG) {
      PyErr_SetString(PyExc_ValueError,
                      "\"samples\"[0] is not a pointer (integer)");
      return NULL;
    }
    samples = reinterpret_cast<float *>(static_cast<uintptr_t>(ull_ptr));
    if (samples == nullptr) {
      PyErr_SetString(PyExc_ValueError, "\"samples\"[0] is null");
      return NULL;
    }
    device_ptrs = PyLong_AsLong(member2);
    if (!PyTuple_Check(member3) || PyTuple_GET_SIZE(member3) != 2) {
      PyErr_SetString(PyExc_TypeError, "\"samples\"[2] must be a shape tuple");
      return NULL;
    }
    samples_size = PyLong_AsUnsignedLong(PyTuple_GetItem(member3, 0));
    features_size = PyLong_AsUnsignedLong(PyTuple_GetItem(member3, 1));
    if (PyTuple_Size(member3) == 3) {
      fp16x2 = PyObject_IsTrue(PyTuple_GetItem(member3, 2));
    }
    if (size == 5) {
      auto member4 = PyTuple_GetItem(samples_obj, 3),
           member5 = PyTuple_GetItem(samples_obj, 4);
      if (!member4 || !member5) {
        PyErr_SetString(PyExc_RuntimeError, "\"samples\" tuple contains nulls");
        return NULL;
      }
      centroids = reinterpret_cast<float *>(static_cast<uintptr_t>(
          PyLong_AsUnsignedLongLong(member4)));
      assignments = reinterpret_cast<uint32_t *>(static_cast<uintptr_t>(
          PyLong_AsUnsignedLongLong(member5)));
    }
  } else if (!get_samples(samples_obj, &samples_array, &samples,
                          &fp16x2, &samples_size, &features_size)) {
    return NULL;
  }
  if (!validate_features_size(features_size)) {
    return NULL;
  }
  pyarray centroids_array, assignments_array;
  if (device_ptrs < 0) {
    npy_intp centroid_dims[] = {
        clusters_size, fp16x2? features_size * 2 : features_size, 0};
    centroids_array.reset(PyArray_EMPTY(
        2, centroid_dims, fp16x2? NPY_FLOAT16 : NPY_FLOAT32, false));
    centroids = reinterpret_cast<float *>(PyArray_DATA(
        centroids_array.get()));
    npy_intp assignments_dims[] = {samples_size, 0};
    assignments_array.reset(PyArray_EMPTY(1, assignments_dims, NPY_UINT32, false));
    assignments = reinterpret_cast<uint32_t *>(PyArray_DATA(
        assignments_array.get()));
  } else if (centroids == nullptr) {
    if (cudaSetDevice(device_ptrs) != cudaSuccess) {
      set_cuda_device_error();
      return NULL;
    }
    if (cudaMalloc(reinterpret_cast<void **>(&centroids),
                   clusters_size * features_size * sizeof(float)) != cudaSuccess) {
      set_cuda_malloc_error();
      return NULL;
    }
    if (cudaMalloc(reinterpret_cast<void **>(&assignments),
                   static_cast<uint64_t>(samples_size) * sizeof(uint32_t)) != cudaSuccess) {
      set_cuda_malloc_error();
      return NULL;
    }
  }
  if (init == kmcudaInitMethodImport) {
    pyarray import_centroids_array(PyArray_FROM_OTF(
        init_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY));
    if (import_centroids_array == NULL) {
      PyErr_SetString(PyExc_TypeError, "\"init\" centroids must be a 2D numpy array");
      return NULL;
    }
    auto ndims = PyArray_NDIM(import_centroids_array.get());
    if (ndims != 2) {
      PyErr_SetString(PyExc_ValueError, "\"init\" centroids must be a 2D numpy array");
      return NULL;
    }
    auto dims = PyArray_DIMS(import_centroids_array.get());
    if (static_cast<uint32_t>(dims[0]) != clusters_size) {
      PyErr_SetString(PyExc_ValueError,
                      "\"init\" centroids shape[0] does not match "
                      "the number of clusters");
      return NULL;
    }
    if (static_cast<uint32_t>(dims[1]) != features_size) {
      PyErr_SetString(PyExc_ValueError,
                      "\"init\" centroids shape[1] does not match "
                          "the number of features");
      return NULL;
    }
    auto icd = reinterpret_cast<float *>(PyArray_DATA(
        import_centroids_array.get()));
    auto size = clusters_size * features_size * sizeof(float);
    if (device_ptrs < 0) {
      memcpy(centroids, icd, size);
    } else {
      if (cudaSetDevice(device_ptrs) != cudaSuccess) {
        set_cuda_device_error();
        return NULL;
      }
      if (cudaMemcpy(centroids, icd, size, cudaMemcpyHostToDevice) != cudaSuccess) {
        set_cuda_memcpy_error();
        return NULL;
      }
    }
  }
  float average_distance = 0;
  int result;
  Py_BEGIN_ALLOW_THREADS
  result = kmeans_cuda(
      init, &afkmc2_m, tolerance, yinyang_t, metric, samples_size,
      static_cast<uint16_t>(features_size), clusters_size, seed, device,
      device_ptrs, fp16x2, verbosity, samples, centroids, assignments,
      adflag? &average_distance : nullptr);
  Py_END_ALLOW_THREADS

  switch (result) {
    case kmcudaInvalidArguments:
      PyErr_SetString(PyExc_ValueError,
                      "Invalid arguments were passed to kmeans_cuda");
      return NULL;
    case kmcudaNoSuchDevice:
      set_cuda_device_error();
      return NULL;
    case kmcudaMemoryAllocationFailure:
      set_cuda_malloc_error();
      return NULL;
    case kmcudaMemoryCopyError:
      set_cuda_memcpy_error();
      return NULL;
    case kmcudaRuntimeError:
      PyErr_SetString(PyExc_AssertionError, "kmeans_cuda failure (bug?)");
      return NULL;
    case kmcudaSuccess:
      if (device_ptrs < 0) {
        if (!adflag) {
          return Py_BuildValue(
              "OO", centroids_array.get(), assignments_array.get());
        } else {
          return Py_BuildValue(
              "OOf", centroids_array.get(), assignments_array.get(),
              average_distance);
        }
      }
      if (!adflag) {
        return Py_BuildValue(
            "KK",
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(centroids)),
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(assignments)));
      } else {
        return Py_BuildValue(
            "KKf",
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(centroids)),
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(assignments)),
            average_distance);
      }
    default:
      PyErr_SetString(PyExc_AssertionError,
                      "Unknown error code returned from kmeans_cuda");
      return NULL;
  }
}

static PyObject *py_knn_cuda(PyObject *self, PyObject *args, PyObject *kwargs) {
  uint32_t device = 0, k = 0;
  int32_t verbosity = 0;
  bool fp16x2 = false;
  PyObject *samples_obj, *centroids_obj, *assignments_obj, *metric_obj = Py_None;
  static const char *kwlist[] = {
      "k", "samples", "centroids", "assignments", "metric", "device",
      "verbosity", NULL};

  /* Parse the input tuple */
  if (!PyArg_ParseTupleAndKeywords(
      args, kwargs, "IOOO|OIi", const_cast<char**>(kwlist), &k, &samples_obj,
      &centroids_obj, &assignments_obj, &metric_obj, &device, &verbosity)) {
    return NULL;
  }

  KMCUDADistanceMetric metric;
  if (!get_metric(metric_obj, &metric)) {
    return NULL;
  }
  if (k == 0 || k > UINT16_MAX) {
    PyErr_SetString(PyExc_ValueError, "\"k\" must be greater than 0 and "
        "less than (1 << 16)");
    return NULL;
  }
  float *samples = nullptr, *centroids = nullptr;
  uint32_t *assignments = nullptr, *neighbors = nullptr;
  uint32_t samples_size = 0, features_size = 0, clusters_size = 0;
  int device_ptrs = -1;
  pyarray samples_array, centroids_array, assignments_array;
  if (PyTuple_Check(samples_obj)) {
    auto size = PyTuple_GET_SIZE(samples_obj);
    if (size != 3 && size != 4) {
      PyErr_SetString(PyExc_ValueError, "len(\"samples\") must be either 3 or 4");
      return NULL;
    }
    auto member1 = PyTuple_GetItem(samples_obj, 0),
        member2 = PyTuple_GetItem(samples_obj, 1),
        member3 = PyTuple_GetItem(samples_obj, 2);
    if (!member1 || !member2 || !member3) {
      PyErr_SetString(PyExc_RuntimeError, "\"samples\" tuple contains nulls");
      return NULL;
    }
    auto ull_ptr = PyLong_AsUnsignedLongLong(member1);
    if (ull_ptr == NPY_MAX_ULONGLONG) {
      PyErr_SetString(PyExc_ValueError,
                      "\"samples\"[0] is not a pointer (integer)");
      return NULL;
    }
    samples = reinterpret_cast<float *>(static_cast<uintptr_t>(ull_ptr));
    if (samples == nullptr) {
      PyErr_SetString(PyExc_ValueError, "\"samples\"[0] is null");
      return NULL;
    }
    device_ptrs = PyLong_AsLong(member2);
    if (!PyTuple_Check(member3) || PyTuple_GET_SIZE(member3) != 2) {
      PyErr_SetString(PyExc_TypeError, "\"samples\"[2] must be a shape tuple");
      return NULL;
    }
    samples_size = PyLong_AsUnsignedLong(PyTuple_GetItem(member3, 0));
    features_size = PyLong_AsUnsignedLong(PyTuple_GetItem(member3, 1));
    if (PyTuple_Size(member3) == 3) {
      fp16x2 = PyObject_IsTrue(PyTuple_GetItem(member3, 2));
    }
    if (size == 4) {
      auto member4 = PyTuple_GetItem(samples_obj, 3);
      if (!member4) {
        PyErr_SetString(PyExc_RuntimeError, "\"samples\" tuple contains nulls");
        return NULL;
      }
      neighbors = reinterpret_cast<uint32_t *>(static_cast<uintptr_t>(
          PyLong_AsUnsignedLongLong(member4)));
    }
    if (!PyTuple_Check(centroids_obj)) {
      PyErr_SetString(PyExc_ValueError, "\"centroids\" must be a tuple of length 2");
      return NULL;
    }
    size = PyTuple_GET_SIZE(centroids_obj);
    if (size != 2) {
      PyErr_SetString(PyExc_ValueError, "len(\"centroids\") must be 2");
      return NULL;
    }
    member1 = PyTuple_GetItem(centroids_obj, 0);
    member2 = PyTuple_GetItem(centroids_obj, 1);
    if (!member1 || !member2) {
      PyErr_SetString(PyExc_RuntimeError, "\"centroids\" tuple contains nulls");
      return NULL;
    }
    ull_ptr = PyLong_AsUnsignedLongLong(member1);
    if (ull_ptr == NPY_MAX_ULONGLONG) {
      PyErr_SetString(PyExc_ValueError,
                      "\"centroids\"[0] is not a pointer (integer)");
      return NULL;
    }
    centroids = reinterpret_cast<float *>(static_cast<uintptr_t>(ull_ptr));
    if (centroids == nullptr) {
      PyErr_SetString(PyExc_ValueError, "\"centroids\"[0] is null");
      return NULL;
    }
    clusters_size = PyLong_AsUnsignedLong(member2);
    ull_ptr = PyLong_AsUnsignedLongLong(assignments_obj);
    if (ull_ptr == NPY_MAX_ULONGLONG) {
      PyErr_SetString(PyExc_ValueError,
                      "\"assignments\" is not a pointer (integer)");
      return NULL;
    }
    assignments = reinterpret_cast<uint32_t *>(static_cast<uintptr_t>(ull_ptr));
  } else {
    if (!get_samples(samples_obj, &samples_array, &samples,
                     &fp16x2, &samples_size, &features_size)) {
      return NULL;
    }
    if (fp16x2) {
      centroids_array.reset(PyArray_FROM_OTF(
          centroids_obj, NPY_FLOAT16, NPY_ARRAY_IN_ARRAY));
    } else {
      centroids_array.reset(PyArray_FROM_OTF(
          centroids_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY));
    }
    if (!centroids_array) {
      PyErr_SetString(PyExc_TypeError,
          "\"centroids\" must be a 2D float32 or float16 numpy array");
      return NULL;
    }
    auto ndims = PyArray_NDIM(centroids_array.get());
    if (ndims != 2) {
      PyErr_SetString(PyExc_ValueError, "\"centroids\" must be a 2D numpy array");
      return NULL;
    }
    auto dims = PyArray_DIMS(centroids_array.get());
    clusters_size = static_cast<uint32_t>(dims[0]);
    if (static_cast<uint32_t>(dims[1]) != features_size * (fp16x2? 2 : 1)) {
      PyErr_SetString(
          PyExc_ValueError, "\"centroids\" must have same number of features "
                            "as \"samples\" (shape[-1])");
      return NULL;
    }
    centroids = reinterpret_cast<float *>(PyArray_DATA(
        centroids_array.get()));
    assignments_array.reset(PyArray_FROM_OTF(
        assignments_obj, NPY_UINT32, NPY_ARRAY_IN_ARRAY));
    if (!assignments_array) {
      PyErr_SetString(PyExc_TypeError,
                      "\"assignments\" must be a 1D uint32 numpy array");
      return NULL;
    }
    ndims = PyArray_NDIM(assignments_array.get());
    if (ndims != 1) {
      PyErr_SetString(PyExc_ValueError, "\"assignments\" must be a 1D numpy array");
      return NULL;
    }
    dims = PyArray_DIMS(assignments_array.get());
    if (static_cast<uint32_t>(dims[0]) != samples_size) {
      PyErr_SetString(
          PyExc_ValueError, "\"assignments\" must be of the same length as "
                            "\"samples\"");
      return NULL;
    }
    assignments = reinterpret_cast<uint32_t *>(PyArray_DATA(
        assignments_array.get()));
  }
  if (!validate_features_size(features_size)) {
    return NULL;
  }
  pyarray neighbors_array;
  if (device_ptrs < 0) {
    npy_intp neighbors_dims[] = {samples_size, k, 0};
    neighbors_array.reset(PyArray_EMPTY(
        2, neighbors_dims, NPY_UINT32, false));
    neighbors = reinterpret_cast<uint32_t *>(PyArray_DATA(
        neighbors_array.get()));
  } else if (neighbors == nullptr) {
    if (cudaSetDevice(device_ptrs) != cudaSuccess) {
      set_cuda_device_error();
      return NULL;
    }
    if (cudaMalloc(reinterpret_cast<void **>(&neighbors),
                   static_cast<uint64_t>(samples_size) * k * sizeof(float)) != cudaSuccess) {
      set_cuda_malloc_error();
      return NULL;
    }
  }
  int result;
  Py_BEGIN_ALLOW_THREADS
    result = knn_cuda(k, metric, samples_size, features_size, clusters_size,
                      device, device_ptrs, fp16x2, verbosity,
                      samples, centroids, assignments, neighbors);
  Py_END_ALLOW_THREADS

  switch (result) {
    case kmcudaInvalidArguments:
      PyErr_SetString(PyExc_ValueError,
                      "Invalid arguments were passed to knn_cuda");
      return NULL;
    case kmcudaNoSuchDevice:
      set_cuda_device_error();
      return NULL;
    case kmcudaMemoryAllocationFailure:
      set_cuda_malloc_error();
      return NULL;
    case kmcudaMemoryCopyError:
      set_cuda_memcpy_error();
      return NULL;
    case kmcudaRuntimeError:
      PyErr_SetString(PyExc_AssertionError, "knn_cuda failure (bug?)");
      return NULL;
    case kmcudaSuccess:
      if (device_ptrs < 0) {
        return Py_BuildValue(
            "O",
            reinterpret_cast<PyObject*>(neighbors_array.get()));
      }
      return Py_BuildValue(
          "K",
          static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(neighbors)));
    default:
      PyErr_SetString(PyExc_AssertionError,
                      "Unknown error code returned from knn_cuda");
      return NULL;
  }
}
