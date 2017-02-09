[![Build Status](https://travis-ci.org/src-d/kmcuda.svg?branch=master)](https://travis-ci.org/src-d/kmcuda) [![PyPI](https://img.shields.io/pypi/v/libKMCUDA.svg)](https://pypi.python.org/pypi/libKMCUDA) [![10.5281/zenodo.286944](https://zenodo.org/badge/DOI/10.5281/zenodo.286944.svg)](https://doi.org/10.5281/zenodo.286944)

"Yinyang" K-means and K-nn using NVIDIA CUDA
============================================

K-means implementation is based on ["Yinyang K-Means: A Drop-In Replacement
of the Classic K-Means with Consistent Speedup"](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ding15.pdf)
article. While it introduces some overhead and many conditional clauses
which are bad for CUDA, it still shows 1.6-2x speedup against the Lloyd
algorithm. K-nearest neighbors employ the same triangle inequality idea and
require precalculated centroids and cluster assignments, similar to a flattened
ball tree.

Technically, this project is a library which exports the two functions
defined in `kmcuda.h`: `kmeans_cuda` and `knn_cuda`.
It has the built-in Python3 native extension support, so you can
`from libKMCUDA import kmeans_cuda`.

K-means
-------
The major difference between this project and others is that kmcuda is
optimized for low memory consumption and the large number of clusters. E.g.,
kmcuda can sort 4M samples in 480 dimensions into 40000 clusters (if you
have several days and 12 GB of GPU memory); 300K samples are grouped
into 5000 clusters in 4½ minutes on NVIDIA Titan X (15 iterations); 3M samples
and 1000 clusters take 20 minutes (33 iterations). Yinyang can be
turned off to save GPU memory but the slower Lloyd will be used then.
Four centroid initialization schemes are supported: random, k-means++,
[AFKMC2](http://olivierbachem.ch/files/afkmcmc-oral-pdf.pdf) and import.
Two distance metrics are supported: L2 (the usual one) and angular
(arccos of the scalar product). L1 is in development.
16-bit float support delivers 2x memory compression. If you've got several GPUs,
they can be utilized together and it gives the corresponding linear speedup
either for Lloyd or Yinyang.

The code has been thoroughly tested to yield bit-to-bit identical
results from Yinyang and Lloyd. "Fast and Provably Good Seedings for k-Means" was adapted from
[the reference code](https://github.com/obachem/kmc2).

Read the articles: [1](http://blog.sourced.tech/post/towards_kmeans_on_gpu/),
[2](https://blog.sourced.tech/post/kmcuda4/).

K-nn
----
Centroid distance matrix C<sub>ij</sub> is calculated together with clusters'
radiuses R<sub>i</sub> (the maximum distance from the centroid to the cluster's
members). Given sample S in cluster A, we avoid calculating the distances from S
to another cluster B's members if C<sub>AB</sub> - SA - R<sub>B</sub> is greater
than the current maximum K-nn distance. This resembles the [ball tree
algorithm](http://scikit-learn.org/stable/modules/neighbors.html#ball-tree).

The implemented algorithm is tolerant to NANs. There are two variants depending
on whether k is small enough to fit the sample's neighbors into CUDA shared memory.
Internally, the neighbors list is a [binary heap](https://en.wikipedia.org/wiki/Binary_heap) -
that reduces the complexity multiplier from O(k) to O(log k).

The implementation yields identical results to `sklearn.neighbors.NearestNeighbors`
except cases in which adjacent distances are equal and the order is undefined.
That is, the returned indices are sorted in the increasing order of the
corresponding distances.

Notes
-----
Lloyd is tolerant to samples with NaN features while Yinyang is not.
It may happen that some of the resulting clusters contain zero elements.
In such cases, their features are set to NaN.

Angular (cosine) distance metric effectively results in Spherical K-Means behavior.
The samples **must** be normalized to L2 norm equal to 1 before clustering,
it is not done automatically. The actual formula is:

![D(A, B)=\arccos\left(\frac{A\cdot B}{|A||B|}\right)](img/latex_angular.png)

If you get OOM with the default parameters, set `yinyang_t` to 0 which
forces Lloyd. `verbosity` 2 will print the memory allocation statistics
(all GPU allocation happens at startup).

Data type is either 32- or 16-bit float. Number of samples is limited by 1^32,
clusters by 1^32 and features by 1^16 (1^17 for fp16). Besides, the product of
clusters number and features number may not exceed 1^32.

In the case of 16-bit floats, the reduced precision often leads to a slightly
increased number of iterations, Yinyang is especially sensitive to that.
In some cases, there may be overflows and the clustering may fail completely.

Building
--------
```
cmake -DCMAKE_BUILD_TYPE=Release . && make
```
It requires cudart 8.0 / Pascal and OpenMP 4.0 capable compiler. The build has
been tested primarily on Linux but it works on macOS too with some blows and whistles
(see "macOS" subsection).
If you do not want to build the Python native module, add `-D DISABLE_PYTHON=y`.
If CUDA is not automatically found, add `-D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-8.0`
(change the path to the actual one). By default, CUDA kernels are compiled for
the architecture 60 (Pascal). It is possible to override it via `-D CUDA_ARCH=52`,
but fp16 support will be disabled then.

Python users: if you are using Linux x86-64 and CUDA 8.0, then you can
install libKMCUDA easily:
```
pip install libKMCUDA
```
Otherwise, you'll have to install it from source:
```
pip install git+https://github.com/src-d/kmcuda.git
```

#### macOS
Install [Homebrew](http://brew.sh/) and the [Command Line Developer Tools](https://developer.apple.com/download/more/)
which are compatible with your CUDA installation. E.g., CUDA 8.0 does not support
the latest 8.x and works with 7.3.1 and below. Install `clang` with OpenMP support
and Python with numpy:
```
brew install llvm --with-clang
brew install python3
pip3 install numpy
```
Execute this magic command which builds kmcuda afterwards:
```
CC=/usr/local/opt/llvm/bin/clang CXX=/usr/local/opt/llvm/bin/clang++ LDFLAGS=-L/usr/local/opt/llvm/lib/ cmake -DCMAKE_BUILD_TYPE=Release ..
```
And make the last important step - rename \*.dylib to \*.so so that Python is able to import the native extension:
```
mv libKMCUDA.{dylib,so}
```

Testing
-------
`test.py` contains the unit tests based on [unittest](https://docs.python.org/3/library/unittest.html).
They require either [cuda4py](https://github.com/ajkxyz/cuda4py) or [pycuda](https://github.com/inducer/pycuda) and
[scikit-learn](http://scikit-learn.org/stable/).

Python examples
---------------

#### K-means, L2 (Euclidean) distance

```python
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
![Clustered dots](img/cls_euclidean.png)

#### K-means, angular (cosine) distance + average

```python
import numpy
from matplotlib import pyplot
from libKMCUDA import kmeans_cuda

numpy.random.seed(0)
arr = numpy.empty((10000, 2), dtype=numpy.float32)
angs = numpy.random.rand(10000) * 2 * numpy.pi
for i in range(10000):
    arr[i] = numpy.sin(angs[i]), numpy.cos(angs[i])
centroids, assignments, avg_distance = kmeans_cuda(
    arr, 4, metric="cos", verbosity=1, seed=3, average_distance=True)
print("Average distance between centroids and members:", avg_distance)
print(centroids)
pyplot.scatter(arr[:, 0], arr[:, 1], c=assignments)
pyplot.scatter(centroids[:, 0], centroids[:, 1], c="white", s=150)
```
You should see something like this:
![Clustered dots](img/cls_angular.png)

#### K-nn

```python
import numpy
from libKMCUDA import kmeans_cuda, knn_cuda

numpy.random.seed(0)
arr = numpy.empty((10000, 2), dtype=numpy.float32)
angs = numpy.random.rand(10000) * 2 * numpy.pi
for i in range(10000):
    arr[i] = numpy.sin(angs[i]), numpy.cos(angs[i])
ca = kmeans_cuda(arr, 4, metric="cos", verbosity=1, seed=3)
neighbors = knn_cuda(10, arr, *ca, metric="cos", verbosity=1, device=1)
print(neighbors[0])
```
You should see
```
reassignments threshold: 100
performing kmeans++...
done
too few clusters for this yinyang_t => Lloyd
iteration 1: 10000 reassignments
iteration 2: 926 reassignments
iteration 3: 416 reassignments
iteration 4: 187 reassignments
iteration 5: 87 reassignments
initializing the inverse assignments...
calculating the cluster radiuses...
calculating the centroid distance matrix...
searching for the nearest neighbors...
calculated 0.276552 of all the distances
[1279 1206 9846 9886 9412 9823 7019 7075 6453 8933]
```

Python API
----------
```python
def kmeans_cuda(samples, clusters, tolerance=0.0, init="k-means++",
                yinyang_t=0.1, metric="L2", average_distance=False,
                seed=time(), device=0, verbosity=0)
```
**samples** numpy array of shape \[number of samples, number of features\]
            or tuple(raw device pointer (int), device index (int), shape (tuple(number of samples, number of features\[, fp16x2 marker\]))).
            In the latter case, negative device index means host pointer. Optionally,
            the tuple can be 2 items longer with preallocated device pointers for
            centroids and assignments. dtype must be either float16 or
            convertible to float32.

**clusters** integer, the number of clusters.

**tolerance** float, if the relative number of reassignments drops below this value, stop.

**init** string or numpy array, sets the method for centroids initialization,
         may be "k=means++"/"kmeans++", "random" or numpy array of shape
         \[**clusters**, number of features\]. dtype must be float32.

**yinyang_t** float, the relative number of cluster groups, usually 0.1.

**metric** str, the name of the distance metric to use. The default is Euclidean (L2),
           can be changed to "cos" to behave as Spherical K-means with the
           angular distance. Please note that samples *must* be normalized in that
           case.

**average_distance** boolean, the value indicating whether to calculate
                     the average distance between cluster elements and
                     the corresponding centroids. Useful for finding
                     the best K. Returned as the third tuple element.

**seed** integer, random generator seed for reproducible results.

**device** integer, bitwise OR-ed CUDA device indices, e.g. 1 means first device, 2 means second device,
           3 means using first and second device. Special value 0 enables all available devices.
           The default is 0.

**verbosity** integer, 0 means complete silence, 1 means mere progress logging,
              2 means lots of output.

**return** tuple(centroids, assignments). If **samples** was a numpy array or
           a host pointer tuple, the types are numpy arrays, otherwise, raw pointers
           (integers) allocated on the same device. If **samples** are float16,
           the returned centroids are float16 too.

```python
def knn_cuda(k, samples, centroids, assignments, metric="L2", device=0, verbosity=0)
```
**k** integer, the number of neighbors to search for each sample. Must be ≤ 1<sup>16</sup>.

**samples** numpy array of shape \[number of samples, number of features\]
            or tuple(raw device pointer (int), device index (int), shape (tuple(number of samples, number of features\[, fp16x2 marker\]))).
            In the latter case, negative device index means host pointer. Optionally,
            the tuple can be 1 item longer with the preallocated device pointer for
            neighbors. dtype must be either float16 or convertible to float32.

**centroids** numpy array with precalculated clusters' centroids (e.g., using
              K-means/kmcuda/kmeans_cuda()). dtype must match **samples**.
              If **samples** is a tuple then **centroids** must be a length-2
              tuple, the first element is the pointer and the second is the
              number of clusters. The shape is (number of clusters, number of features).

**assignments** numpy array with sample-cluster associations. dtype is expected
                to be compatible with uint32. If **samples** is a tuple then
                **assignments** is a pointer. The shape is (number of samples,).

**metric** str, the name of the distance metric to use. The default is Euclidean (L2),
                can be changed to "cos" to behave as Spherical K-means with the
                angular distance. Please note that samples *must* be normalized in that
                case.

**device** integer, bitwise OR-ed CUDA device indices, e.g. 1 means first device, 2 means second device,
           3 means using first and second device. Special value 0 enables all available devices.
           The default is 0.

**verbosity** integer, 0 means complete silence, 1 means mere progress logging,
              2 means lots of output.

**return** neighbor indices. If **samples** was a numpy array or
            a host pointer tuple, the return type is numpy array, otherwise, a
            raw pointer (integer) allocated on the same device. The shape is
            (number of samples, k).

C API
-----
```C
KMCUDAResult kmeans_cuda(
    KMCUDAInitMethod init, float tolerance, float yinyang_t,
    KMCUDADistanceMetric metric, uint32_t samples_size, uint16_t features_size,
    uint32_t clusters_size, uint32_t seed, uint32_t device, int32_t device_ptrs,
    int32_t fp16x2, int32_t verbosity, const float *samples, float *centroids,
    uint32_t *assignments, float *average_distance)
```
**init** specifies the centroids initialization method: k-means++, random or import
         (in the latter case, **centroids** is read).

**tolerance** if the number of reassignments drop below this ratio, stop.

**yinyang_t** the relative number of cluster groups, usually 0.1.

**metric** The distance metric to use. The default is Euclidean (L2), can be
           changed to cosine to behave as Spherical K-means with the angular
           distance. Please note that samples *must* be normalized in that case.

**samples_size** number of samples.

**features_size** number of features. if fp16x2 is set, one half of the number of features.

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
                Usually this value is -1.

**fp16x2** activates fp16 mode, two half-floats are packed into a single 32-bit float,
           features_size becomes effectively 2 times bigger, the returned
           centroids are fp16x2 too.

**verbosity** 0 - no output; 1 - progress output; >=2 - debug output.

**samples** input array of size samples_size x features_size in row major format.

**centroids** output array of centroids of size clusters_size x features_size
              in row major format.

**assignments** output array of cluster indices for each sample of size
                samples_size x 1.

**average_distance** output mean distance between cluster elements and
                     the corresponding centroids. If nullptr, not calculated.

Returns KMCUDAResult (see `kmcuda.h`);

```C
KMCUDAResult knn_cuda(
    uint16_t k, KMCUDADistanceMetric metric, uint32_t samples_size,
    uint16_t features_size, uint32_t clusters_size, uint32_t device,
    int32_t device_ptrs, int32_t fp16x2, int32_t verbosity,
    const float *samples, const float *centroids, const uint32_t *assignments,
    uint32_t *neighbors);
```
**k** integer, the number of neighbors to search for each sample.

**metric** The distance metric to use. The default is Euclidean (L2), can be
           changed to cosine to behave as Spherical K-means with the angular
           distance. Please note that samples *must* be normalized in that case.

**samples_size** number of samples.

**features_size** number of features. if fp16x2 is set, one half of the number of features.

**clusters_size** number of clusters.

**device** CUDA device OR-ed indices - usually 1. For example, 1 means using first device,
           2 means second device, 3 means first and second device (2x speedup). Special
           value 0 enables all available devices.

**device_ptrs** configures the location of input and output. If it is negative,
                samples, centroids, assignments and the returned array are on host,
                otherwise, they belong to the corresponding device.
                E.g., if device_ptrs is 0, **samples**, **centroids** and
                **assignments** are expected to be pointers to device #0's memory
                and the resulting **neighbors** is expected to be preallocated on
                device #0 as well. Usually this value is -1.

**fp16x2** activates fp16 mode, two half-floats are packed into a single 32-bit float,
           features_size becomes effectively 2 times bigger, affects **samples**
           and **centroids**.

**verbosity** 0 - no output; 1 - progress output; >=2 - debug output.

**samples** input array of size samples_size x features_size in row major format.

**centroids** input array of centroids of size clusters_size x features_size
              in row major format.

**assignments** input array of cluster indices for each sample of size
                samples_size x 1.

**neighbors** output array with the nearest neighbors of size
              samples_size x k in row major format.

Returns KMCUDAResult (see `kmcuda.h`);

License
-------
MIT license.
