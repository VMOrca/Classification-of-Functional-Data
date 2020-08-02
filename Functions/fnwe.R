#########################################################################################################################################
#
#                                                       Author: Min Sun
#
#########################################################################################################################################

# Function to compute functional Nadaraya-Watson's estimator
# Details: Section 4.2.1, 4.2.2
# Input:
#   - kernelChoice [char] : asymmetric kernel function, current supports c('box', 'triangle', 'quadratic', 'gaussian')
#   - h [double] : bandwidth for kernel function
#   - metric [char] : metric to be used, currently only supports c('LpNorm', 'supNorm')
#   - y [array] : label for training data, must be vector
#   - x [dataframe] : training data, must be datafrmae
#   - xNew [dataframe] : validation or test data, must be datafrmae
# Output: [list]
#     - Label Prediction [array] : prediction of xNew. Dimension will be equal to dim(xNew)[1]
#     - Probability [dataframe] : probability of each predicted value of xNew


fnwe = function(kernelChoice, h, metric, y, x, xNew, ...) {
  # Check which kernel function to use
  switch(kernelChoice, 
         box = {
           k = function(u) {
             out = ifelse(u >= 0 & u <= 1, 1, 0)
             return(out)
           }
         }, 
         triangle = {
           k = function(u) {
             out = ifelse(u >= 0 & u <= 1, 1, 0) * (1 - u) * 2
             return(out)
           }
         }, 
         quadratic = {
           k = function(u) {
             out = ifelse(u >= 0 & u <= 1, 1, 0) * (1 - u^2) * 3/4 * 2
             return(out)
           }
         }, 
         gaussian = {
           k = function(u) {
             out = ifelse(u >= 0, 1, 0) * 1/sqrt(2 * pi) * exp(-u^2/2) * 2
             return(out)
           }
         })
  
  n = dim(x)[1]
  m = dim(xNew)[1]
  uniqueLabel = unique(y)
  
  # Calculate d(x, xnew)
  d = matrix(rep(NA, n * m), nrow = n)
  for (i in 1:n) {
    for (j in 1:m) {
      f = as.vector(x[i, ] - xNew[j, ], mode = 'numeric')
      switch(metric, 
             LpNorm = {
               d[i, j] = LpNorm(x = f, ...)
             }, 
             supNorm = {
               d[i, j] = supNorm(x = f)
             })
    }
  }
  
  # Calculate K(d(x, xnew)/h)
  kernelValue = k(d/h)
  
  # Calculate (\sum_{i = 1}^n K(d(x, xnew)/h) 1_{y_i = y})/(\sum_{i = 1}^n K(d(x, xnew)/h))
  probMatrix = matrix(rep(NA, m * length(uniqueLabel)), ncol = m)
  for (j in 1:m) {
    for (i in 1:length(uniqueLabel)){
        label = uniqueLabel[i]
        if (sum(kernelValue[, j]) == 0) {
          probMatrix[i, j] = 0
        } else{
          probMatrix[i, j] = (sum(kernelValue[which(y == rep(label, n)), j]))/sum(kernelValue[, j])
        }
    }
  }
  
  
  
  # Find which label gives max predictive probability
  yPred = uniqueLabel[apply(probMatrix, 2, which.max)]
  probMatrixOut = t(probMatrix)
  colnames(probMatrixOut) = uniqueLabel
  out = list('Label Prediction' = yPred, 
             'Probability' = probMatrixOut)
  return(out)
}



# library(tidyverse)
# n = 10
# df = data.frame(matrix(rnorm(100, 0, 1), nrow = n))
# y = sample(0:1, n, replace = TRUE)
# t = 1:dim(df)[2]
# out = fnwe(x = df, 
#            t = t,
#            y = y,
#            xNew = df,
#            h = 1,
#            metric = 'LpNorm',
#            kernelChoice = 'gaussian')