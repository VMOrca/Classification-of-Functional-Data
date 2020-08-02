#########################################################################################################################################
#
#                                                       Author: Min Sun
#
#########################################################################################################################################

# Function to perform functional principal component analysis
# Details: Section 3.3
# Input:
#   - sigma [matrix] : sample covariance matrix
#   - t [array] : domain of observations x(t)
#   - proportion [double] : score of fPCA, which affects how many basis(eigenfunctions) to be preserved. Usually a high score is needed to 
#                           fully recover the original function
#   - nBasis [int] : how many eigenfunctions to be preserved. Usually set to NA as we use proportion instead.
#                    We usually use proportion instead (i.e. set nBasis = NA), but if nBasis is given, then proportion will NOT be effective!
# Output: [list]
#   - eigenvalue [array] : functional eigenvalue
#   - truncatedEigenvalue [array]: truncated functional eigenvalue
#   - eigenvector [dataframe]: discretised eigenfunction, due to sampling limitation there can be at most length(t) eigenfunctions though.
#                              Each column is one discretised eigenfunction
#   - truncatedEigenvector [dataframe]: truncated discretised eigenfunction



fpca = function(sigma, t, proportion = 0.9, nBasis = NA) {
  eigenAnalysis = eigen(sigma)
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
  
  J = dim(sigma)[2]
  w = (max(t) - min(t))/J
  
  # rho: eigenvalue (functional)
  # xi: discretised eigenfunction
  rho = w * lambda
  xi = w^(-0.5) * gamma
  
  out = list('eigenvalue' = rho,
             'truncatedEigenvalue' = rho[1:k], 
             'eigenvector' = xi, 
             'truncatedEigenvector' = xi[, 1:k])
  return(out)
}



# # Example
# sigma = matrix(rnorm(9, 0, 1), nrow = 3)
# t = 1:100
# out = fpca(sigma = sigma, t = t, proportion = 0.9, nBasis = NA)

