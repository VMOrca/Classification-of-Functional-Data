# Function to perform functional principal component analysis
# - sigma: sample covariance matrix
# - k: truncated eigenvalues up to k-th one
fpca = function(x, t, proportion = 0.9, nBasis = NA) {
  eigenAnalysis = eigen(x)
  # gamma: eigenvector (matrix)
  # lambda: eigenvalue (matrix)
  gamma = eigenAnalysis$vectors
  lambda = eigenAnalysis$values
  
  scree = cumsum(lambda)/sum(lambda)
  # k = truncate eigenvalues and eigenvectors upto k-th one by using proportion
  if (is.na(nBasis)) {
    k = which(scree >= proportion)[1]
  } else {
    k = nBasis
  }
  
  J = dim(x)[2]
  w = (max(t) - min(t))/J
  
  # rho: eigenvalue (functional)
  # xi: eigenvalue (functional)
  rho = w * lambda
  xi = w^(-0.5) * gamma
  
  out = list('eigenvalue' = rho,
             'truncatedEigenvalue' = rho[1:k], 
             'eigenvector' = xi, 
             'truncatedEigenvector' = xi[, 1:k])
  return(out)
}
