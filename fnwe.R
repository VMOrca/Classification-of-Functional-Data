# Function to compute functional Nadaraya-Watson's estimator
# kernel: asymmetric kernel function
# h: bandwidth
# metric: metric to be used, e.g. LpNorm
# y = label for training data, must be vector
# x = training data, must be matrix
# xNew = validation or test data, must be matrix

fnwe = function(kernel, h, metric, y, x, xNew, ...) {
  # Check which kernel function to use
  switch(kernel, 
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
      d[i, j] = metric(x = f, ...)
    }
  }
  
  # Calculate K(d(x, xnew)/h)
  kernelValue = k(d/h)
  
  # Calculate (\sum_{i = 1}^n K(d(x, xnew)/h) 1_{y_i = y})/(\sum_{i = 1}^n K(d(x, xnew)/h))
  probMatrix = matrix(rep(NA, m * length(uniqueLabel)), ncol = m)
  for (j in 1:m) {
    for (i in uniqueLabel){
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

# 
# out2 = fnwe(x = select(dfSmoothNonTest, -label, -idOriginal, -id), 
#           t = time, 
#           y = dfSmoothNonTest$label, 
#           xNew = select(dfSmoothTest, -label, -idOriginal, -id)[1, ], 
#           h = 20, 
#           metric = LpNorm, 
#           kernel = 'gaussian')
