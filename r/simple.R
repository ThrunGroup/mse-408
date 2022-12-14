#!/usr/bin/Rscript
library(cmdstanr)
check_cmdstan_toolchain(fix=TRUE, quiet=TRUE)
library(bayesplot)
library(posterior)
color_scheme_set("brightblue")

main <- function() {
  file <- file.path(cmdstan_path(), "examples", "bernoulli", "bernoulli.stan")
  mod <- cmdstan_model(file)
  # names correspond to the data block in the Stan program
  data_list <- list(N = 10, y = c(0,1,0,0,0,0,0,0,0,1))
  fit <- mod$sample(
    data = data_list, 
    seed = 123, 
    chains = 4, 
    parallel_chains = 4,
    refresh = 500 # print update every 500 iters
  )
}

if (!interactive()) {
  main()
}
