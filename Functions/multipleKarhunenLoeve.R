#########################################################################################################################################
#
#                                                       Author: Min Sun
#
#########################################################################################################################################

# Function to compute Karhunen-Loeve expansion a datafrmae, where each row represents one observation
#                                      x(t) = \sum_{i = 1}^k <x, phi_k> phi_k(t)
# Details: Section 3.2
# Input:
#   - x [dataframe] : functions to be approximated by Karhunen-Loeve expansion, must be in dataframe form (i.e. 1st discretised function in 1st row)
#   - t [array] : domain of x(t), must have the same dimension as dim(x)[2]
#   - proportion [double]: score of fPCA, which affects how many basis(eigenfunctions) to be preserved. Usually a high score is needed to
#                          fully recover the original function
#   - nBasis [int] : how many eigenfunctions to be preserved. It must be equal to the number of basis preserved in the model
#                    We usually use proportion instead (i.e. set nBasis = NA), but if nBasis is given, then proportion will NOT be effective!
#   - zeroMeanBool [Bool]: a boolean variable indicating wheter x has been centred (i.e. adjusted by the mean function)
# Output:
#   - innerProduct [dataframe] : Inner product <x, phi_k> for all k. i-th row is the inner products for i-th data
#   - Basis [dataframe] : truncated eigenvector of x. k-th column means the k-th basis
#   - no_eigen [int] : number of eigenvalues preserved
#   - xApprox [dataframe] : approximated value of x at each point t using basis expansion. i-th row is the inner products for i-th data
#   - zeroMeanBool [Bool]: same as input

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



# Example
# n = 20
# df = data.frame(matrix(rnorm(100, 0, 1), nrow = n))
# y = sample(0:1, n, replace = TRUE)
# t = 1:dim(df)[2]
# sigma = var(df)
# out = multipleKarhunenLoeve(x = df, t, proportion = 0.9, zeroMeanBool = TRUE, nBasis = NA)
