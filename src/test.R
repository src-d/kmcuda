library(testthat)

if (exists("testing")) {
  setwd(cwd)
  dyn.load("libKMCUDA.so")

  context("K-means")
  test_that("Random",{
    set.seed(42)
    samples <- replicate(4, runif(16000))
    result = .External("kmeans_cuda", samples, 50, tolerance=0.01,
                       init="random", seed=777, verbosity=2)
    kmcuda_asses = replicate(1, result$assignments)
    attach(kmeans(samples, result$centroids, iter.max=1))
    reasses = length(intersect(kmcuda_asses, cluster)) / length(cluster)
    print(sprintf("Reassignments: %f", reasses))
    expect_lt(reasses, 0.01)
  })
  test_that("SingleDeviceKmeans++Lloyd",{
    set.seed(42)
    samples <- replicate(4, runif(16000))
    result = .External("kmeans_cuda", samples, 50, yinyang_t=0, device=1,
                       init="kmeans++", seed=777, verbosity=2)
    kmcuda_asses = replicate(1, result$assignments)
    attach(kmeans(samples, result$centroids, iter.max=1))
    reasses = length(intersect(kmcuda_asses, cluster)) / length(cluster)
    print(sprintf("Reassignments: %f", reasses))
    expect_lt(reasses, 0.01)
  })
  test_that("MultiSamples",{
    set.seed(42)
    samples1 <- replicate(4, runif(16000))
    samples2 <- replicate(4, runif(16000))
    result = .External("kmeans_cuda", list(samples1, samples2), 50,
                       init="kmeans++", seed=777, verbosity=2)
    kmcuda_asses = replicate(1, result$assignments)
    expect_equal(length(kmcuda_asses), 32000)
    attach(kmeans(rbind(samples1, samples2), result$centroids, iter.max=1))
    reasses = length(intersect(kmcuda_asses, cluster)) / length(cluster)
    print(sprintf("Reassignments: %f", reasses))
    expect_lt(reasses, 0.01)
  })
  test_that("AFK-MC2",{
    set.seed(42)
    samples <- replicate(4, runif(16000))
    result = .External("kmeans_cuda", samples, 50, tolerance=0.01,
                       init=c("afkmc2", 100), seed=777, verbosity=2)
    kmcuda_asses = replicate(1, result$assignments)
    attach(kmeans(samples, result$centroids, iter.max=1))
    reasses = length(intersect(kmcuda_asses, cluster)) / length(cluster)
    print(sprintf("Reassignments: %f", reasses))
    expect_lt(reasses, 0.01)
  })
  test_that("ImportCentroids",{
    set.seed(42)
    samples <- replicate(4, runif(16000))
    centroids <- replicate(4, runif(50))
    result = .External("kmeans_cuda", samples, 50, tolerance=0.01,
                       init=centroids, seed=777, verbosity=2)
    kmcuda_asses = replicate(1, result$assignments)
    attach(kmeans(samples, result$centroids, iter.max=1))
    reasses = length(intersect(kmcuda_asses, cluster)) / length(cluster)
    print(sprintf("Reassignments: %f", reasses))
    expect_lt(reasses, 0.01)
  })
  test_that("RandomPlusAverageDistance",{
    set.seed(42)
    samples <- replicate(4, runif(16000))
    result = .External("kmeans_cuda", samples, 50, tolerance=0.01,
                       init="random", seed=777, verbosity=2,
                       average_distance=TRUE)
    print(result$average_distance)
    expect_equal(result$average_distance, 0.2124216, tolerance=0.0000001);
  })

  context("K-nn")
  test_that("Cosine",{
    set.seed(42)
    samples <- replicate(4, runif(16000))
    samples <- samples / sqrt(rowSums(samples^2))
    cls = .External("kmeans_cuda", samples, 50, tolerance=0.01, metric="cos",
                    seed=777, verbosity=2, yinyang_t=0)
    lapply(rowSums(cls$centroids^2), function(r) expect_equal(r, 1, 0.0001))
    result = .External("knn_cuda", 20, samples, cls$centroids,
                       cls$assignments, metric="cos", verbosity=2)
    # the result is properly validated in test.py
    expect_equal(dim(result), c(16000, 20))
    expect_equal(class(result), "matrix")
    expect_equal(sum(apply(result, 1, function(r) length(unique(r)))), 16000 * 20)
  })
} else {
  testing <- TRUE
  cwd <- getwd()
  thisFile <- function() {
    cmdArgs <- commandArgs(trailingOnly=FALSE)
    needle <- "--file="
    match <- grep(needle, cmdArgs)
    if (length(match) > 0) {
      return(normalizePath(sub(needle, "", cmdArgs[match])))
    } else {
      return(normalizePath(sys.frames()[[1]]$ofile))
    }
  }
  test_results <- test_dir(dirname(thisFile()), reporter="summary")
}
