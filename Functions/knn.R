#########################################################################################################################################
#
#                                                       Author: Min Sun
#
#########################################################################################################################################

# Function to compute KNN
# Detail: Section 4.1
# Input:
#   - x [dataframe] : training data, must be datafrmae
#   - t [array] : an array of domain of xNew, which is usually time. Must have the same dimension as dim(xNew)[2]
#   - xNew [dataframe] : validation or test data, must be datafrmae
#   - y [array] : label for training data, must be vector
#   - k [int] : hyperparameter, which controls the number of nearest points to be considered by the algorithm
#   - metric [char] : metric to be used, currently only supports c('LpNorm', 'supNorm')
# Output: [list]
#     - Label Prediction [array] : prediction of xNew. Dimension will be equal to dim(xNew)[1]
#     - Probability [dataframe] : probability of each predicted value of xNew

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



# Example
# library(tidyverse)
# n = 10
# df = data.frame(matrix(rnorm(100, 0, 1), nrow = n))
# y = sample(0:1, n, replace = TRUE)
# t = 1:dim(df)[2]
# out = knn(x =  df,
#           y = y,
#           t = t, 
#           xNew = df, 
#           k = 1, 
#           metric = 'LpNorm')
