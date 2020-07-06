options(stringsAsFactors = FALSE)

setwd('D:/Academics/UNSW/Thesis/R/MCO/')

source('Data_Preparation')


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
pb <- txtProgressBar(max = 1, style = 3)
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
dfControlSmooth = filter(dfSmooth, label == 0)
dfTrtSmooth = filter(dfSmooth, label == 1)

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
idTest = sort(sample(idAll, n * 0.2))
idNonTest = idAll[which(!idAll %in% idTest)]
dfSmoothTest = dfSmooth[idTest, ]
dfSmoothNonTest = dfSmooth[idNonTest, ]
nTest = length(idTest)
nNonTest = n - nTest
nTraining = n * 0.6
nValidation = n * 0.2


# Functional knn
set.seed(1)
mcoKnn = mlFramework$new()
mcoKnn$setData(dfMeta = dfMeta, 
               dfAll = select(dfSmooth, -idOriginal), 
               dfNonTest = select(dfSmoothNonTest, -idOriginal), 
               dfTest = select(dfSmoothTest, -idOriginal), 
               nNonTest = nNonTest, 
               nTest = nTest, 
               idNonTest = idNonTest, 
               idTest = idTest)
mcoKnn$setWd('D:/Academics/UNSW/Thesis/R/MCO/')
mcoKnn$setClassifier('knn')
tic()
mcoKnn$cvClassifier(iter = 500, hyperparChoice = 1:10, nCore = 10, trainingPct = 0.6, t = time, metric = LpNorm) 
toc()
mcoKnn$runOnTestSet(t = time, metric = LpNorm)

save(mcoKnn, file = 'mcoKnn.RData')




# Functional Nadaraya-Watson estimator
set.seed(1)
mcoFnwe = mlFramework$new()
mcoFnwe$setData(dfMeta = dfMeta, 
                dfAll = select(dfSmooth, -idOriginal), 
                dfNonTest = select(dfSmoothNonTest, -idOriginal), 
                dfTest = select(dfSmoothTest, -idOriginal), 
                nNonTest = nNonTest, 
                nTest = nTest, 
                idNonTest = idNonTest, 
                idTest = idTest)
mcoFnwe$setWd('D:/Academics/UNSW/Thesis/R/MCO/')
mcoFnwe$setClassifier('fnwe')
# tic()
# mcoFnwe$cvClassifier(iter = 80, hyperparChoice = c(18, 45, 56, 78, 111, 117, 134, 163, 175, 263), 
#                      nCore = 4, trainingPct = 0.6, t = time, metric = LpNorm, kernelChoice = 'gaussian') 
# toc()
tic()
mcoFnwe$cvClassifier(iter = 50, hyperparChoice = 1:150, 
                     nCore = 10, trainingPct = 0.6, t = time, metric = LpNorm, kernelChoice = 'gaussian') 
toc()
mcoFnwe$runOnTestSet(t = time, metric = LpNorm, kernelChoice = 'gaussian')

save(mcoFnwe, file = 'mcoFnwe.RData')





# Functional kernel rule
set.seed(1)
mcoKernelRule = mlFramework$new()
mcoKernelRule$setData(dfMeta = dfMeta, 
                      dfAll = select(dfSmooth, -idOriginal), 
                      dfNonTest = select(dfSmoothNonTest, -idOriginal), 
                      dfTest = select(dfSmoothTest, -idOriginal), 
                      nNonTest = nNonTest, 
                      nTest = nTest, 
                      idNonTest = idNonTest, 
                      idTest = idTest)
mcoKernelRule$setWd('D:/Academics/UNSW/Thesis/R/MCO/')
mcoKernelRule$setClassifier('kernelRule')
tic()
mcoKernelRule$cvClassifier(iter = 10, hyperparChoice = 1:100, nCore = 5, trainingPct = 0.6, t = time, metric = LpNorm, kernelChoice = 'gaussian') 
toc()
mcoKernelRule$runOnTestSet(t = time, metric = LpNorm, kernelChoice = 'gaussian')

save(mcoKernelRule, file = 'mcoKernelRule.RData')






# Functional GLM
set.seed(1)
mcoFglm = mlFramework$new()
mcoFglm$setData(dfMeta = dfMeta, 
                dfAll = select(dfSmooth, -idOriginal), 
                dfNonTest = select(dfSmoothNonTest, -idOriginal), 
                dfTest = select(dfSmoothTest, -idOriginal), 
                nNonTest = nNonTest, 
                nTest = nTest, 
                idNonTest = idNonTest, 
                idTest = idTest)
mcoFglm$setWd('D:/Academics/UNSW/Thesis/R/MCO/')
mcoFglm$setClassifier('fglm')
tic()
# Not sure why when zeroMeanBool = FALSE, the recovered function oscillate a lot => use zeroMeanBool = TRUE fow now
mcoFglm$trainClassifier(trainingPct = 0.6, t = time, proportion = 0.99, expansion = 'kl', zeroMeanBool = TRUE)
toc()
mcoFglm$runOnTestSet(t = time)

save(mcoFglm, file = 'mcoFglm.RData')





# Functional SVM
set.seed(1)
mcoFSvm = mlFramework$new()
mcoFSvm$setData(dfMeta = dfMeta, 
                dfAll = select(dfSmooth, -idOriginal), 
                dfNonTest = select(dfSmoothNonTest, -idOriginal), 
                dfTest = select(dfSmoothTest, -idOriginal), 
                nNonTest = nNonTest, 
                nTest = nTest, 
                idNonTest = idNonTest, 
                idTest = idTest)
mcoFSvm$setWd('D:/Academics/UNSW/Thesis/R/MCO/')
mcoFSvm$setClassifier('fSvm')
tic()
# Not sure why when zeroMeanBool = FALSE, the recovered function oscillate a lot => use zeroMeanBool = TRUE fow now
svmParChoice = list('gamma' = 1:5, 
                    'cost' = 1:20)
mcoFSvm$cvClassifier(iter = 10, hyperparChoice = svmParChoice, nCore = 5, trainingPct = 0.6, 
                     t = time, proportion = 0.99, expansion = 'kl', zeroMeanBool = TRUE, kernelChoice = 'radial')
toc()
mcoFSvm$runOnTestSet(t = time)
