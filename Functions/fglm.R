#########################################################################################################################################
#
#                                                       Author: Min Sun
#
#########################################################################################################################################

# Function to
#   - assemble desgin matrix Z, where each entry (apart from 1st column) is an inner product derived from Karhunen-Loeve 
#    expansion of original data x
#   - run fGLM using inner products derived from KL expansion
#   - Details please see section 4.4.1 - 4.4.2, especially remark 4.4.13
# Details: Section 4.4.1, 4.4.2
# Input:
#   - x [dataframe]: a dataframe of observed functional data. Column = reading at each t. Row = each observation
#   - y [array]: an array of binary variable {0, 1}, e.g. c(0, 0, 0, 1, 1)
#   - t [array] : an array of domain of x, which is usually time. Must have the same dimension as dim(x)[2]
#   - proportion [double] : score of fPCA, which affects how many basis(eigenfunctions) to be preserved. Usually a high score is needed to 
#                           fully recover the original function
#   - expansion [char] : type of basis expansion to be performed. Currently only supports 'kl' (Karhunen-Loeve expansion)
#   - zeroMeanBool [Bool]: a boolean variable indicating wheter x has been centred (i.e. adjusted by the mean function)
# Output: [list]
#   - model [glm] : fGLM model
#   - zeroMeanBool [bool] : same as input zeroMeanBool
#   - basisType [char] : same asexpansion
#   - meanProcess [double] : mean function of x 
#   - intercept [double] : beta0 in regression 
#   - zeta [array] : coefficients for the regression based on inner products
#   - beta [array] : reconstructed coefficients for the regression based on x
#   - basis [dataframe] : basis vectors for specific choice of 'expansion'
#   - innerProductOfX [dataframe] : coordinates of 'basis' to x, which is the inner products of x and 'basis'
#   - xApprox [datafrmae] : approximation of x using truncated basis expansion,
#   - modelPar [list] : parameters for the fGLM - stores beta0 and beta only, not zeta




fglm = function(x, y, t, proportion = 0.999, expansion = 'kl', zeroMeanBool) {
  switch(expansion, 
         # Perform Karhunen Loeve expansion on all observations, i.e. to each row of x
         kl = {
           klOut = multipleKarhunenLoeve(x = x, t = t, proportion = proportion, zeroMeanBool = zeroMeanBool)
           innerProduct = klOut$innerProduct
           basis = klOut$basis
           m = klOut$no_eigen
           xApprox = klOut$xApprox
         })
  
  ave = colMeans(x, na.rm = TRUE)
  
  # Assemble matrix to run glm
  # Covariate matrix is now the matrix innerProduct
  df = cbind(y, data.frame(innerProduct))
  model = glm(y ~ ., data = df, family = binomial(link = 'logit'),
              control = list(maxit = 50))
  beta0 = model$coefficients[1]
  zeta = model$coefficients[-1]
  # Find beta from zeta
  beta = apply(t(basis) * zeta, 2, sum)

  out = list('model' = model,
             'zeroMeanBool' = zeroMeanBool, 
             'basisType' = expansion, 
             'meanProcess' = ave, 
             'intercept' = beta0,
             'zeta' = zeta,
             'beta' = beta,
             'basis' = basis,
             'innerProductOfX' = innerProduct,
             'xApprox' = xApprox,
             'modelPar' = c(beta0, beta))
  return(out)
}




# Example
# library(tidyverse)
# n = 10
# df = data.frame(matrix(rnorm(100, 0, 1), nrow = n))
# y = sample(0:1, n, replace = TRUE)
# t = 1:dim(df)[2]
# out = fglm(x =  df, 
#            y = y,
#            t = t, proportion = 0.99, expansion = 'kl', zeroMeanBool = TRUE)
