# Function to assemble desgin matrix, where each each entry (apart from 1st column) is derived from Karhunen-Loeve expansion of original data
# Input:
# - x: function to be approximated by Karhunen-Loeve expansion, must be in matrix form (i.e. 1st discretised function in 1st row)
# - y: label of x, must be a vector
# - t: domain of x(t), must have the same dimension as dim(x)[2]
# - proportion \in [0, 1] which determines how many principal components (i.e. k) to be preserved in Karhunen Loeve expansion
fglm = function(x, y, t, proportion = 0.999) {
  # Perform Karhunen Loeve expansion on all observations, i.e. to each row of x
  kl = multipleKarhunenLoeve(x = x, t = t, proportion = proportion)
  innerProduct = kl$innerProduct
  basis = kl$basis
  m = kl$no_eigen
  xApprox = kl$xApprox
  
  # Assemble matrix to run glm
  # Covariate matrix is now the matrix innerProduct
  df = cbind(y, data.frame(innerProduct))
  model = glm(y ~ ., data = df, family = binomial(link = 'logit'))
  beta0 = model$coefficients[1]
  zeta = model$coefficients[-1]
  # Find beta from zeta
  beta = apply(t(basis) * zeta, 2, sum)

  out = list('glm' = model,
             'intercept' = beta0,
             'zeta' = zeta,
             'beta' = beta,
             'basis' = basis,
             'innerProductOfX' = innerProduct,
             'xApprox' = xApprox,
             'modelPar' = c(beta0, beta))
  return(out)
}



# tic()
# myGlm = fglm(x = select(dfSmoothNonTest, -idOriginal, -id, -label), 
#              y = dfSmoothNonTest$label, t = time, proportion = 0.999)
# toc()