import sys
import unittest

import numpy
from libKMCUDA import kmeans_cuda
from sklearn.cluster import KMeans


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

    @staticmethod
    def _reference(centroids):
        return KMeans(50, max_iter=1, init=centroids, n_init=1)

    def _validate(self, centroids, assignments, tolerance):
        ref = self._reference(centroids)
        next_asses = ref.fit_predict(self.samples)
        reasses = sum(assignments != next_asses)
        self.assertLess(reasses / len(self.samples), tolerance)

    def test_random_lloyd(self):
        centroids, assignments = kmeans_cuda(
            self.samples, 50, init="random", device=1,
            verbosity=2, seed=3, tolerance=0.05, yinyang_t=0)
        self.assertEqual(sys.getrefcount(centroids), 2)
        self.assertEqual(sys.getrefcount(assignments), 2)
        self.assertEqual(sys.getrefcount(self.samples), 2)
        self.assertEqual(centroids.shape, (50, 2))
        self.assertEqual(assignments.shape, (13000,))
        self._validate(centroids, assignments, 0.05)

    def test_kmeanspp_lloyd(self):
        centroids, assignments = kmeans_cuda(
            self.samples, 50, init="kmeans++", device=1,
            verbosity=2, seed=3, tolerance=0.05, yinyang_t=0)
        self._validate(centroids, assignments, 0.05)

    def test_kmeanspp_yinyang(self):
        centroids, assignments = kmeans_cuda(
            self.samples, 50, init="kmeans++", device=1,
            verbosity=2, seed=3, tolerance=0.01, yinyang_t=0.1)
        self._validate(centroids, assignments, 0.01)

    def test_import_lloyd(self):
        centroids, assignments = kmeans_cuda(
            self.samples, 50, init="random", device=1,
            verbosity=2, seed=3, tolerance=0.25, yinyang_t=0)
        centroids, assignments = kmeans_cuda(
            self.samples, 50, init=centroids, device=1,
            verbosity=2, seed=3, tolerance=0.05, yinyang_t=0)
        self._validate(centroids, assignments, 0.05)

if __name__ == "__main__":
    unittest.main()
