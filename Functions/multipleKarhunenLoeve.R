# Calculate Karhunen Loeve expansion for a matrix, where each row represents one observation
# - x: function to be approximated by Karhunen-Loeve expansion, must be in matrix form (i.e. 1st discretised function in 1st row)
# - t: domain of x(t), must have the same dimension as dim(x)[2]
# - proportion \in [0, 1] which determines how many principal components (i.e. k) to be preserved
multipleKarhunenLoeve = function(x, t, proportion, zeroMeanBool, nBasis = NA) {
  n = dim(x)[1]
  xApprox = matrix(rep(NA, n * length(t)), nrow = n)
  
  # Calculate covariance function, cov_x(t_1, t_2), i.e. covariance function of x at each point t
  ave = colMeans(x)
  if (zeroMeanBool == TRUE) {
    zeroMeanX = x
  } else{
    zeroMeanX = x - ave
  }
  sigma = var(zeroMeanX)
  
  # Need to perform Karhunen Loeve once to decide dimension of inner products and basis
  dummyRun = karhunenLoeve(x = as.vector(zeroMeanX[1, ], mode = 'numeric'), sigma = sigma, t = t, proportion = proportion, nBasis = nBasis)
  m = dummyRun$no_eigen
  innerProduct = matrix(rep(NA, n * m), nrow = n)
  innerProduct[1, ] = dummyRun$innerProduct
  xApprox[1, ] = dummyRun$xApprox
  basis = dummyRun$basis
  
  # Find Karhunen Loeve expansion for all rows of x
  for (i in 2:n) {
    kl = karhunenLoeve(x = as.vector(zeroMeanX[i, ], mode = 'numeric'), sigma = sigma, t = t, proportion = proportion, nBasis = nBasis)
    innerProduct[i, ] = kl$innerProduct
    xApprox[i, ] = kl$xApprox
  }
  
  # Recover original stochastic process x from Karhunen Loeve and its mean function because Karhunen Loeve expansion
  # is derived from the zero mean process
  if (zeroMeanBool == FALSE) {
    xApprox = xApprox + ave
  }
  
  out = list('innerProduct' = innerProduct, 
             'basis' = basis, 
             'no_eigen' = m, 
             'xApprox' = xApprox, 
             'zeroMeanBool' = zeroMeanBool)
  return(out)
}

