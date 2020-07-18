# KNN with Lp norm as metric
# x: Training data, must be a matrix. row = i-th observation, column = x(t)
# y: Class label for observations in x. Length of y must be the same as number of rows of x.
# xNew: new observations - must be a matrix
# t: domain of x
# k: hyperparameter
knn = function(x, t, y, xNew, k, metric = 'LpNorm') {
  n = dim(x)[1]
  m = dim(xNew)[1]
  metricValue = matrix(rep(NA, n * m), nrow = n)
  neighbourOfXNew = matrix(rep(NA, k * m), nrow = k)
  yPred = rep(NA, m)
  yProbList = list()
  for (j in 1:m) {
    for (i in 1:n) {
      switch(metric, 
             LpNorm = {
               metricValue[i, j] = LpNorm(as.vector(x[i, ] - xNew[j, ], mode = 'numeric'), t, p = 2)
             }, 
             supNorm = {
               metricValue[i, j] = supNorm(as.vector(x[i, ] - xNew[j, ], mode = 'numeric'))
             })
    }
    neighbourOfXNew[, j] =  sort(metricValue[, j])[1:k]
    yInSmallBall = y[which(metricValue[, j] %in% neighbourOfXNew[, j])]
    yPred[j] = names(sort(-table(yInSmallBall)))[1]
    yProbList[[j]] = table(yInSmallBall)/length(yInSmallBall)
  }
  yProbMatrix = do.call(rbind, yProbList)
  out = list('Label Prediction' = yPred, 'Probability' = yProbMatrix)
  return(out)
}


# out = knn(x = select(dfSmoothNonTest, -label, -idOriginal, -id), 
#           t = time, 
#           y = dfSmoothNonTest$label, 
#           xNew = select(dfSmoothTest, -label, -idOriginal, -id)[1, ], 
#           k = 10, 
#           metric = LpNorm)

