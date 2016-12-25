#include <memory>
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

static PyObject *py_kmeans_cuda(PyObject *self, PyObject *args, PyObject *kwargs);

static PyMethodDef module_functions[] = {
  {"kmeans_cuda", reinterpret_cast<PyCFunction>(py_kmeans_cuda),
   METH_VARARGS | METH_KEYWORDS, kmeans_cuda_docstring},
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

static const std::unordered_map<std::string, KMCUDAInitMethod> init_methods {
    {"kmeans++", kmcudaInitMethodPlusPlus},
    {"k-means++", kmcudaInitMethodPlusPlus},
    {"random", kmcudaInitMethodRandom}
};

static const std::unordered_map<std::string, KMCUDADistanceMetric > metrics {
    {"L2", kmcudaDistanceMetricL2},
    {"l2", kmcudaDistanceMetricL2},
    {"cos", kmcudaDistanceMetricCosine},
    {"cosine", kmcudaDistanceMetricCosine},
    {"angular", kmcudaDistanceMetricCosine}
};

static void set_cuda_malloc_error() {
  PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory on GPU");
}

static void set_cuda_device_error() {
  PyErr_SetString(PyExc_ValueError, "No such CUDA device exists");
}

static void set_cuda_memcpy_error() {
  PyErr_SetString(PyExc_RuntimeError, "cudaMemcpy failed");
}

static PyObject *py_kmeans_cuda(PyObject *self, PyObject *args, PyObject *kwargs) {
  uint32_t clusters_size = 0, seed = static_cast<uint32_t>(time(NULL)), device = 0;
  int32_t verbosity = 0;
  bool fp16x2 = false;
  float tolerance = .01, yinyang_t = .1;
  PyObject *samples_obj, *init_obj = Py_None, *metric_obj = Py_None;
  static const char *kwlist[] = {
      "samples", "clusters", "tolerance", "init", "yinyang_t", "metric", "seed",
      "device", "verbosity", NULL};

  /* Parse the input tuple */
  if (!PyArg_ParseTupleAndKeywords(
      args, kwargs, "OI|fOfOIIi", const_cast<char**>(kwlist), &samples_obj,
      &clusters_size, &tolerance, &init_obj, &yinyang_t, &metric_obj, &seed,
      &device, &verbosity)) {
    return NULL;
  }

  KMCUDAInitMethod init;
  if (init_obj == Py_None) {
    init = kmcudaInitMethodPlusPlus;
  } else if (PyUnicode_Check(init_obj)) {
    pyobj bytes(PyUnicode_AsASCIIString(init_obj));
    auto iminit = init_methods.find(PyBytes_AsString(bytes.get()));
    if (iminit == init_methods.end()) {
      PyErr_SetString(
          PyExc_ValueError,
          "Unknown centroids initialization method. Supported values are "
              "\"kmeans++\", \"random\" and <numpy array>.");
      return NULL;
    }
    init = iminit->second;
  } else {
    init = kmcudaInitMethodImport;
  }
  KMCUDADistanceMetric metric;
  if (metric_obj == Py_None) {
    metric = kmcudaDistanceMetricL2;
  } else if (!PyUnicode_Check(metric_obj)) {
    PyErr_SetString(
        PyExc_TypeError, "\"metric\" must be either None or string.");
    return NULL;
  } else {
    pyobj bytes(PyUnicode_AsASCIIString(metric_obj));
    auto immetric = metrics.find(PyBytes_AsString(bytes.get()));
    if (immetric == metrics.end()) {
      PyErr_SetString(
          PyExc_ValueError,
          "Unknown metric. Supported values are \"L2\" and \"cos\".");
      return NULL;
    }
    metric = immetric->second;
  }
  if (device < 0) {
    PyErr_SetString(PyExc_ValueError, "\"device\" must be a binary device "
        "selection mask where 1 on n-th place activates n-th device; 0 "
        "activates all available devices.");
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
      PyErr_SetString(PyExc_RuntimeError, "\"samples\" tuple with nulls");
      return NULL;
    }
    samples = reinterpret_cast<float *>(static_cast<uintptr_t>(
        PyLong_AsUnsignedLongLong(member1)));
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
        PyErr_SetString(PyExc_RuntimeError, "\"samples\" tuple with nulls");
        return NULL;
      }
      centroids = reinterpret_cast<float *>(static_cast<uintptr_t>(
          PyLong_AsUnsignedLongLong(member4)));
      assignments = reinterpret_cast<uint32_t *>(static_cast<uintptr_t>(
          PyLong_AsUnsignedLongLong(member5)));
    }
  } else {
    samples_array.reset(PyArray_FROM_OTF(
        samples_obj, NPY_FLOAT16, NPY_ARRAY_IN_ARRAY));
    if (!samples_array) {
      PyErr_Clear();
      samples_array.reset(PyArray_FROM_OTF(
          samples_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY));
      if (!samples_array) {
        PyErr_SetString(PyExc_TypeError,
                        "\"samples\" must be a 2D float32 or float16 numpy array");
        return NULL;
      }
    } else {
      fp16x2 = true;
    }
    auto ndims = PyArray_NDIM(samples_array.get());
    if (ndims != 2) {
      PyErr_SetString(PyExc_ValueError, "\"samples\" must be a 2D numpy array");
      return NULL;
    }
    auto dims = PyArray_DIMS(samples_array.get());
    samples_size = static_cast<uint32_t>(dims[0]);
    features_size = static_cast<uint32_t>(dims[1]);
    if (fp16x2 && PyArray_TYPE(samples_array.get()) == NPY_FLOAT16) {
      if (features_size % 2 != 0) {
        PyErr_SetString(PyExc_ValueError,
                        "the number of features must be even in fp16 mode");
        return NULL;
      }
      features_size /= 2;
    }
    if (features_size > UINT16_MAX) {
      char msg[128];
      sprintf(msg, "\"samples\": more than %" PRIu32 " features is not supported",
              features_size);
      PyErr_SetString(PyExc_ValueError, msg);
      return NULL;
    }
    samples = reinterpret_cast<float *>(PyArray_DATA(
        samples_array.get()));
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
                   samples_size * sizeof(uint32_t)) != cudaSuccess) {
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
  int result;
  Py_BEGIN_ALLOW_THREADS
  result = kmeans_cuda(
      init, tolerance, yinyang_t, metric, samples_size,
      static_cast<uint16_t>(features_size), clusters_size, seed, device,
      device_ptrs, fp16x2, verbosity, samples, centroids, assignments);
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
        return Py_BuildValue("OO", centroids_array.get(), assignments_array.get());
      }
      return Py_BuildValue(
          "KK",
          static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(centroids)),
          static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(assignments)));
    default:
      PyErr_SetString(PyExc_AssertionError,
                      "Unknown error code returned from kmeans_cuda");
      return NULL;
  }
}
