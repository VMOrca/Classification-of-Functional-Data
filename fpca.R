# Function to perform functional principal component analysis
# - sigma: sample covariance matrix
# - k: truncated eigenvalues up to k-th one
fpca = function(x, t, proportion) {
  eigenAnalysis = eigen(x)
  # gamma: eigenvector (matrix)
  # lambda: eigenvalue (matrix)
  gamma = eigenAnalysis$vectors
  lambda = eigenAnalysis$values
  
  scree = cumsum(lambda)/sum(lambda)
  # k = truncate eigenvalues and eigenvectors upto k-th one by using proportion
  k = which(scree >= proportion)[1]
  
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
