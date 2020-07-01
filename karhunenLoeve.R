# Function to compute Karhunen-Loeve expansion
#                                      x(t) = \sum_{i = 1}^k <x, phi_k> phi_k(t)
# Input:
# - x: function to be approximated by Karhunen-Loeve expansion, must be in vector form (i.e. discretised function)
# - sigma: cov_x(t_1, t_2), i.e. covariance function of x at each point t
# - t: domain of x(t), must have the same dimension as x
# - proportion \in [0, 1] which determines how many principal components (i.e. k) to be preserved
# 
# Output: 
# - Inner product <x, phi_k> for all k
# - Basis: truncated eigenvector of x
# - no_eigen: number of eigenvalues preserved
# - xApprox: approximated value of x at each point t
karhunenLoeve = function(x, sigma, t, proportion) {
  pca = fpca(sigma, t, proportion)
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




x = select(dfSmoothNonTest, -idOriginal, -id, -label)
y = as.vector(x[1, ], mode = 'numeric')
b = karhunenLoeve(x = y, sigma = varNonTest, t = time, proportion = 0.999)
b$no_eigen

plot(t, y, type = 'l')
lines(t, b$xApprox, col = 2)

