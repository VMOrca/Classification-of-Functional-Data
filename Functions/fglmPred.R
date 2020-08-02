#########################################################################################################################################
#
#                                                       Author: Min Sun
#
#########################################################################################################################################

# Function to predict new data using an existing functional GLM model
# Details: Section 4.4.1, 4.4.2
# Input:
#   - xNew [dataframe]: a matrix of new data, each row represents a new observation
#   - t [array] : an array of domain of xNew, which is usually time. Must have the same dimension as dim(xNew)[2]
#   - fglmModel [fglm]: must be a fglm object, i.e. z where z = fgml(...)
# Output: [array]
#   - an array of predicted value. Dimension will be equal to dim(xNew)[1]

fglmPred = function(xNew, t, fglmModel, ...) {
  beta0 = fglmModel$intercept
  beta = fglmModel$beta
  modelFamily = fglmModel$model$family
  
  xBeta = as.matrix(xNew) %*% diag(beta, length(beta), length(beta))
  xBetaInnerProduct = apply(xBeta, 1, auc, t = t)
  rhs = beta0 + xBetaInnerProduct
  yPred = modelFamily$linkinv(rhs)
  return(yPred)
}


# library(tidyverse)
# source('fglm.R')
# n = 10
# df = data.frame(matrix(rnorm(100, 0, 1), nrow = n))
# y = sample(0:1, n, replace = TRUE)
# t = 1:dim(df)[2]
# out = fglm(x =  df,
#            y = y,
#            t = t, proportion = 0.99, expansion = 'kl', zeroMeanBool = TRUE)
# out2 = fglmPred(xNew = df,
#                t = t,
#                fglmModel = out)