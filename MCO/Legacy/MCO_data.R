options(stringsAsFactors = FALSE)

library(fda.usc)
library(tidyverse)
library(fda)
library(ggplot2)
library(parallel)
library(doSNOW)
library(tictoc)
library(mlr3)


data(MCO)
dfX = MCO$permea$data %>% 
  as.data.frame() %>% 
  rownames_to_column(var = 'id') %>% 
  cbind(idNumeric = 1:nrow(.)) %>% 
  select(id, idNumeric, everything())
time = MCO$permea$argvals
label = MCO$classpermea

xMax = max(dfX[, -(1:2)])
n = dim(dfX)[1]

# In manual it says that 'The structure of the curves during the initial period (first 180 seconds) of the experiment shows a
# erratic behavior (not very relevant in the experiment context) during this period.'
# Hence we do not need to plot and analyse them
df = data.frame(dfX) %>% 
  cbind(label = as.integer(label)) %>% 
  select(id, idNumeric, label, everything())
df = df[, -(4:(18 + 3))] # why 3: columns of id, idNumeric, label
time = time[-(1:18)]

dfControl = filter(df, label == 1)
nControl = dim(dfControl)[1]
dfTrt = filter(df, label == 2)
nTrt = dim(dfTrt)[1]

# Plot functional data by groups
par(mfrow = c(2, 1))
columnIsTimeIndex = 4:(dim(df)[2])
plot(time, dfControl[1, columnIsTimeIndex], type = 'l', col = 1, ylim = c(0, xMax), 
     xlab = 'Time', ylab = 'MCO')
for (i in 2:nControl) {
  lines(time, dfControl[i, columnIsTimeIndex], type = 'l', col = 1)
}

plot(time, dfTrt[1, columnIsTimeIndex], type = 'l', col = 2, ylim = c(0, xMax), 
     xlab = 'Time', ylab = 'MCO')
for (i in 2:nTrt) {
  lines(time, dfTrt[i, columnIsTimeIndex], type = 'l', col = 2)
}


# Observation wiggly -> smoothing needed
# Trial to see possible choices of bandwidth for kernel regression smoothing
# x = df[1, columnIsTimeIndex]
# t = time
# xNWE = ksmooth(t, x, kernel = 'normal', bandwidth = 50)
# plot(t, x)
# lines(xNWE, col = 2)


# Create parallel cluster
cl = makeCluster(10)
registerDoSNOW(cl)

bandwidthChoice = 50:150 # Choice of bandwidth
K = 10 # Iteration of cross validation
set.seed(1)

# Monitor bar for parallel computing
pb <- txtProgressBar(max = max(bw) - min(bw) + 1, style = 3)
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress = progress)


# Start parallel computing
# For each choice of bandwidth
#   For i = 1:n (n curves)
#     For k = 1:K (Number of iteration for cross validation)
#       - Calculate Nadaraya-Waston Estimator (NWE) from randomly choosed time points (~half of original time points) 
#       - Calculate residual sum of square (rss) using NWE as fitted value.
#       - Calculate the average of rss for all k
# Resulting matrix rss: row = i-th curve, column = j-th choice of bandwidth
tic()
rss = foreach (j = bandwidthChoice, .options.snow = opts, .combine = 'cbind') %dopar% {
  rssMatrix = matrix(rep(NA, n * K), nrow = n)
  for (i in 1:n) {
    dfIterFull = df[i, columnIsTimeIndex]
    for (k in 1:K) {
      columnsIndexCv = sort(sample(columnIsTimeIndex - 3, floor(length(columnIsTimeIndex - 3)/2)))
      dfIterHalf = dfIterFull[1, columnsIndexCv]
      xNWE = ksmooth(time[columnsIndexCv], dfIterHalf, kernel = 'normal', bandwidth = j, n.points = length(time))
      rssMatrix[i, k] = sum((as.vector(dfIterFull, mode = 'numeric') - xNWE$y)^2, na.rm = TRUE)
    }
  }
  rssSummary = rowMeans(rssMatrix)
  return(rssSummary)
}
toc()
stopCluster(cl)

# Choose bandwidth with the smallest rss for all i = 1:n
rssMatrix = data.frame(rss)
# write.csv(rssMatrix, 'D:/Academics/UNSW/Thesis/R/MCO/Data/rssKernelBandwidthSelection.csv')
# rssMatrix = read.csv('D:/Academics/UNSW/Thesis/R/MCO/Data/rssKernelBandwidthSelection.csv')
rssSummary = colMeans(rssMatrix)
bw = bandwidthChoice[which(rssSummary == min(rssSummary))]

# Construct smoothed curve for each observation
dfSmooth = matrix(rep(NA, n * length(time)), nrow = n)
for (i in 1:n) {
  dfIterFull = df[i, columnIsTimeIndex]
  xNWE = ksmooth(time, dfIterFull, kernel = 'normal', bandwidth = bw, n.points = length(time))
  dfSmooth[i, ] = xNWE$y
}

# Combine dfSmooth with the label of each data and subset to treatment/control group
dfSmooth = data.frame(dfSmooth) %>% 
  cbind(df[,  1:3], .)
colnames(dfSmooth)[4:dim(dfSmooth)[2]] = time
dfControlSmooth = filter(dfSmooth, label == 1)
dfTrtSmooth = filter(dfSmooth, label == 2)

# Plot SMOOTHED functional data by groups
par(mfrow = c(2, 1))
plot(time, dfControlSmooth[1, columnIsTimeIndex], type = 'l', col = 1, ylim = c(0, xMax), 
     xlab = 'Time', ylab = 'MCO')
for (i in 2:nControl) {
  lines(time, dfControlSmooth[i, columnIsTimeIndex], type = 'l', col = 1)
}

plot(time, dfTrtSmooth[1, columnIsTimeIndex], type = 'l', col = 2, ylim = c(0, xMax), 
     xlab = 'Time', ylab = 'MCO')
for (i in 2:nTrt) {
  lines(time, dfTrtSmooth[i, columnIsTimeIndex], type = 'l', col = 2)
}



# Divide all subjects to traing/validation/test sets
# Test: 20% fixed subjects. Training/Validation: 60%/20% which will vary between the remaining subjects for each round of cross validation
# Some methods such as KNN does not require parameter estimation - hence will not need training set. The hyperparameters will need to be
# determined by validation set though. In this case all remaining data (i.e. 80%) will be validation set.
set.seed(1)
idTest = sort(sample(1:n, n * 0.2))
dfSmoothTest = dfSmooth[idTest, ]
dfSmoothNonTest = dfSmooth[-idTest, ]
nTest = length(idTest)
nNonTest = n - nTest
nTraining = n * 0.6
nValidation = n * 0.2

# Create parallel cluster
cl = makeCluster(10)
registerDoSNOW(cl)

iter = 10
K = 1:10
set.seed(1)

# Monitor bar for parallel computing
pb <- txtProgressBar(max = max(bw) - min(bw) + 1, style = 3)
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress = progress)
# Start parallel computing
for (k in K) {
  accuracyAve = rep(NA, length(K))
  accuracyWithinIter = foreach (i = 1:iter, .combine = 'c', .options.snow = opts) %dopar% {
    idTraining = sort(sample(dfSmoothNonTest$idNumeric, nTraining))
    idValidation = dfSmoothNonTest$idNumeric[!dfSmoothNonTest$idNumeric %in% idTraining]
    dfSMoothTraining = dplyr::filter(dfSmooth, idNumeric %in% idTraining)
    dfSMoothValidation = dplyr::filter(dfSmooth, idNumeric %in% idValidation)
    out = knn(x = dplyr::select(dfSmoothTraining, -label, -id, -idNumeric),
              t = time,
              y = dfSmoothNonTest$label,
              xNew = dplyr::select(dfSMoothValidation, -label, -id, -idNumeric),
              k = k,
              metric = LpNorm)
    predLabel = as.integer(out[['Label Prediction']])
    validationLabel = dfSMoothValidation$label
    out = length(which(predLabel == validationLabel))/nValidation
    return(out)
  }
  accuracyAve[k] = mean(accuracyWithinIter)
}


stopCluster(cl)





idTest = sort(sample(1:n, n * 0.2))
dfSmoothTest = dfSmooth[idTest, ]
dfSmoothNonTest = dfSmooth[-idTest, ]
nTest = length(idTest)
nNonTest = n - nTest
nTraining = n * 0.6
nValidation = n * 0.2


method1 = mlFramework$new()
method1$setData(dfMeta = select(dfMeta, id, label, idNumeric), dfAll = dfSmooth, 
                dfNonTest = dfSmoothNonTest, dfTest = dfTest, 
                nNonTest = nNonTest, nTest = nTest, 
                idNonTest = id, idTest)













