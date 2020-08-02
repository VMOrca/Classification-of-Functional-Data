#########################################################################################################################################
#
#                                                       Author: Min Sun
#
#########################################################################################################################################

# Function to compute Karhunen-Loeve expansion (for single function only!)
#                                      x(t) = \sum_{i = 1}^k <x, phi_k> phi_k(t)
# Details: Section 3.2
# Input:
#   - x [array] : function to be approximated by Karhunen-Loeve expansion, must be in vector form (i.e. discretised function)
#   - sigma [matrix] : cov_x(t_1, t_2), i.e. discretised covariance function of x at each point t => hence a covariance matrix
#   - t [array] : domain of x(t), must have the same dimension as x
#   - proportion [double]: score of fPCA, which affects how many basis(eigenfunctions) to be preserved. Usually a high score is needed to
#                          fully recover the original function
#   - nBasis [int] : how many eigenfunctions to be preserved. It must be equal to the number of basis preserved in the model
#                    We usually use proportion instead (i.e. set nBasis = NA), but if nBasis is given, then proportion will NOT be effective!
# Output:
#   - innerProduct [array] : Inner product <x, phi_k> for all k
#   - Basis [dataframe] : truncated eigenvector of x. k-th column means the k-th basis
#   - no_eigen [int] : number of eigenvalues preserved
#   - xApprox [array] : approximated value of x at each point t using basis expansion


karhunenLoeve = function(x, sigma, t, proportion, nBasis = NA) {
  pca = fpca(sigma, t, proportion, nBasis)
  eigenvector = pca$truncatedEigenvector
  eigenvalue = pca$truncatedEigenvalue
  
  # Compute inner product <x, phi_k>
  k = length(eigenvalue)
  trapezoidT = diff(t)
  innerProduct = rep(NA, k)
  for (j in 1:k) {
    f = x * eigenvector[, j]
    trapezoidX = c(f[1] + f[2], diff(cumsum(f), lag = 2))
    innerProduct[j] = sum(trapezoidT * trapezoidX/2)
  }
  xApprox = apply(t(eigenvector) * innerProduct, 2, sum, na.rm = TRUE)
  
  out = list('innerProduct' = innerProduct, 
             'basis' = eigenvector, 
             'no_eigen' = k, 
             'xApprox' = xApprox)
}




# Example
# n = 20
# df = data.frame(matrix(rnorm(100, 0, 1), nrow = n))
# y = sample(0:1, n, replace = TRUE)
# t = 1:dim(df)[2]
# sigma = var(df)
# out = karhunenLoeve(x = as.vector(df[1, ], mode = 'numeric'), sigma = sigma, t, proportion, nBasis = NA)
