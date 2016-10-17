[![Build Status](https://travis-ci.org/src-d/kmcuda.svg?branch=master)](https://travis-ci.org/src-d/kmcuda)

"Yinyang" K-means using NVIDIA CUDA
===================================

This implementation is based on ["Yinyang K-Means: A Drop-In Replacement
of the Classic K-Means with Consistent Speedup"](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ding15.pdf)
article. While it introduces some overhead and many conditional clauses
which are bad for CUDA, it still shows 1.6-2x speedup against the Lloyd
algorithm.

The major difference between this project and others is that it is
optimized for low memory consumption and large number of clusters. E.g.,
kmcuda can sort 4M samples in 480 dimensions into 40000 clusters (if you
have several days and 12 GB of GPU memory). 300K samples are grouped
into 5000 clusters in 4½ minutes on NVIDIA Titan X (15 iterations). Yinyang can be
turned off to save GPU memory but the slower Lloyd will be used then.
Three centroid initialization ways are supported: random, k-means++ and import.
If you've got several GPUs, they can be utilized together and it gives the
corresponding linear speedup either for Lloyd or Yinyang.

The code has been thoroughly tested to yield bit-to-bit identical
results from Yinyang and Lloyd.

Technically, this project is a library which exports the single function
defined in `wrappers.h`, `kmeans_cuda`. It has a built-in Python3 native
extension support, so you can `from libKMCUDA import kmeans_cuda`.

[Read the article](http://blog.sourced.tech/post/towards_kmeans_on_gpu/).

Notes
-----
Lloyd is tolerant to samples with NaN features while Yinyang is not.
It may happen that some of the resulting clusters contain zero elements.
In such cases, their features are set to NaN.

If you get OOM with the default parameters, set `yinyang_t` to 0 which
forces Lloyd. `verbosity` 2 will print the memory allocation statistics
(all GPU allocation happens at startup).

Data type is 32-bit float. Number of samples is limited by 1^32,
clusters by 1^32 and features by 1^16. Besides, the product of
clusters number and features number may not exceed 1^32.

Building
--------
```
cmake -DCMAKE_BUILD_TYPE=Release . && make
```
It requires cudart 7.5 / OpenMP 4.0 capable compiler.

Python users: if you are using Linux x86-64 and CUDA 7.5, then you can
install this easily:
```
pip install libKMCUDA
```
Otherwise, you'll have to install it from source:
```
pip install git+https://github.com/src-d/kmcuda.git
```

Python example
--------------
```
import numpy
from matplotlib import pyplot
from libKMCUDA import kmeans_cuda

numpy.random.seed(0)
arr = numpy.empty((10000, 2), dtype=numpy.float32)
arr[:2500] = numpy.random.rand(2500, 2) + [0, 2]
arr[2500:5000] = numpy.random.rand(2500, 2) - [0, 2]
arr[5000:7500] = numpy.random.rand(2500, 2) + [2, 0]
arr[7500:] = numpy.random.rand(2500, 2) - [2, 0]
centroids, assignments = kmeans_cuda(arr, 4, verbosity=1, seed=3)
print(centroids)
pyplot.scatter(arr[:, 0], arr[:, 1], c=assignments)
pyplot.scatter(centroids[:, 0], centroids[:, 1], c="white", s=150)
```
You should see something like this:
![Clustered dots](cls.png)

Python API
----------
```python
def kmeans_cuda(samples, clusters, tolerance=0.0, init="k-means++",
                yinyang_t=0.1, seed=time(), device=0, verbosity=0)
```
**samples** numpy array of shape \[number of samples, number of features\]
            or tuple(raw device pointer (int), device index (int), shape (tuple(number of samples, number of features))).
            In the latter case, negative device index means host pointer. Optionally,
            the tuple can be 2 items longer with preallocated device pointers for
            centroids and assignments. dtype must be float32.

**clusters** integer, the number of clusters.

**tolerance** float, if the relative number of reassignments drops below this value, stop.

**init** string or numpy array, sets the method for centroids initialization,
         may be "k=means++"/"kmeans++", "random" or numpy array of shape
         \[**clusters**, number of features\]. dtype must be float32.

**yinyang_t** float, the relative number of cluster groups, usually 0.1.

**seed** integer, random generator seed for reproducible results.

**device** integer, bitwise OR-ed CUDA device indices, e.g. 1 means first device, 2 means second device,
           3 means using first and second device. Special value 0 enables all available devices.
           Default value is 1.

**verbosity** integer, 0 means complete silence, 1 means mere progress logging,
              2 means lots of output.
              
 **return** tuple(centroids, assignments). If **samples** was a numpy array or
            a host pointer tuple, the types are numpy arrays, otherwise, raw pointers
            (integers) allocated on the same device.

C API
-----
```C
int kmeans_cuda(KMCUDAInitMethod init, float tolerance, float yinyang_t,
                uint32_t samples_size, uint16_t features_size, uint32_t clusters_size,
                uint32_t seed, uint32_t device, int device_ptrs, int32_t verbosity,
                const float *samples, float *centroids, uint32_t *assignments)
```
**init** specifies the centroids initialization method: k-means++, random or import
         (in the latter case, **centroids** is read).

**tolerance** if the number of reassignments drop below this ratio, stop.

**yinyang_t** the relative number of cluster groups, usually 0.1.

**samples_size** number of samples.

**features_size** number of features.

**clusters_size** number of clusters.

**seed** random generator seed passed to srand().

**device** CUDA device OR-ed indices - usually 1. For example, 1 means using first device,
           2 means second device, 3 means first and second device (2x speedup). Special
           value 0 enables all available devices.

**device_ptrs** configures the location of input and output. If it is negative,
                samples and returned arrays are on host, otherwise, they belong to the
                corresponding device. E.g., if device_ptrs is 0, **samples** is expected
                to be a pointer to device #0's memory and the resulting **centroids** and
                **assignments** are expected to be preallocated on device #0 as well.
                Usually the value is -1.

**verbosity** 0 - no output; 1 - progress output; >=2 - debug output.

**samples** input array of size samples_size x features_size in row major format.

**centroids** output array of centroids of size clusters_size x features_size
in row major format.

**assignments** output array of cluster indices for each sample of size
samples_size x 1.

Returns KMCUDAResult (see `kmcuda.h`);

License
-------
MIT license.
