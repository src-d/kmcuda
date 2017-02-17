library(testthat)

if (exists("testing")) {
  setwd(cwd)
  dyn.load("libKMCUDA.so")

  context("K-means")
  test_that("Random Lloyd",{
    samples <- replicate(4, rnorm(16000))
    ret <- .External("kmeans_cuda", samples, 50, init="random", seed=777, verbosity=2)
    print(ret)
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
