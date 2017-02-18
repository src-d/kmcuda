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
