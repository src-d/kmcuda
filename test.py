import unittest

import numpy
from libKMCUDA import kmeans_cuda


class KMCUDATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        numpy.random.seed(0)
        arr = numpy.empty((13000, 2), dtype=numpy.float32)
        arr[:2000] = numpy.random.rand(2000, 2) + [0, 0.5]
        arr[2000:4000] = numpy.random.rand(2000, 2) + [0, 1.5]
        arr[4000:6000] = numpy.random.rand(2000, 2) - [0, 0.5]
        arr[6000:8000] = numpy.random.rand(2000, 2) + [0.5, 0]
        arr[8000:10000] = numpy.random.rand(2000, 2) - [0.5, 0]
        arr[10000:] = numpy.random.rand(3000, 2) * 5 - [2, 2]
        cls.samples = arr

    def setUp(self):
        super(KMCUDATests, self).setUp()
        numpy.random.seed(0)

    def test_kmeanspp_lloyd(self):
        centroids, assignments = kmeans_cuda(
            self.samples, 50, init="kmeans++", device=1,
            verbosity=2, seed=3, tolerance=0.05, yinyang_t=0)

    def test_kmeanspp_yinyang(self):
        centroids, assignments = kmeans_cuda(
            self.samples, 50, init="kmeans++", device=1,
            verbosity=2, seed=3, tolerance=0.05, yinyang_t=0.1)

    def test_random_lloyd(self):
        centroids, assignments = kmeans_cuda(
            self.samples, 50, init="random", device=1,
            verbosity=2, seed=3, tolerance=0.05, yinyang_t=0)

    def test_import_lloyd(self):
        centroids, assignments = kmeans_cuda(
            self.samples, 50, init="random", device=1,
            verbosity=2, seed=3, tolerance=0.25, yinyang_t=0)
        centroids, assignments = kmeans_cuda(
            self.samples, 50, init=centroids, device=1,
            verbosity=2, seed=3, tolerance=0.05, yinyang_t=0)