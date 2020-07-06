options(stringsAsFactors = FALSE)

library(fda.usc)
library(tidyverse)
library(fda)
library(ggplot2)
library(parallel)
library(doSNOW)
library(tictoc)
library(e1071)
library(R6)

setwd('D:/Academics/UNSW/Thesis/R/MCO/')
source('Functions/fglm.R')
source('Functions/fglmPred.R')
source('Functions/fnwe.R')
source('Functions/fpca.R')
source('Functions/karhunenLoeve.R')
source('Functions/kernelRule.R')
source('Functions/knn.R')
source('Functions/L2InnerProduct.R')
source('Functions/LpNorm.R')
source('Functions/multipleKarhunenLoeve.R')
source('Functions/mlFramework.R')
source('Functions/cvFSvm.R')
source('Functions/fSvmPred.R')


#=============================================================================================================

data(MCO)
dfX = MCO$permea$data %>% 
  as.data.frame() %>% 
  rownames_to_column(var = 'idOriginal') %>% 
  cbind(id = 1:nrow(.)) %>% 
  select(idOriginal, id, everything())
time = MCO$permea$argvals
label = MCO$classpermea

xMax = max(dfX[, -(1:2)])
n = dim(dfX)[1]
idAll = 1:n

# In manual it says that 'The structure of the curves during the initial period (first 180 seconds) of the experiment shows a
# erratic behavior (not very relevant in the experiment context) during this period.'
# Hence we do not need to plot and analyse them
df = data.frame(dfX) %>% 
  cbind(label = as.integer(label)) %>% 
  select(idOriginal, id, label, everything()) %>% 
  mutate(label = ifelse(label == 1, 0, 1))
df = df[, -(4:(18 + 3))] # why 3: columns of idOriginal, id, label
time = time[-(1:18)]
dfMeta = select(df, id, label, idOriginal)

dfControl = filter(df, label == 0)
nControl = dim(dfControl)[1]
dfTrt = filter(df, label == 1)
nTrt = dim(dfTrt)[1]

# Plot functional data by groups
par(mfrow = c(2, 1))
columnIsTimeIndex = 4:(dim(df)[2])
plot(time, dfControl[1, columnIsTimeIndex], type = 'l', col = 1, ylim = c(0, xMax), 
     xlab = 'Time', ylab = 'MCO', main = paste('MCO(Label = ', as.character(unique(dfControl[, 'label'])), ')', sep = ''))
for (i in 2:nControl) {
  lines(time, dfControl[i, columnIsTimeIndex], type = 'l', col = 1)
}

plot(time, dfTrt[1, columnIsTimeIndex], type = 'l', col = 2, ylim = c(0, xMax), 
     xlab = 'Time', ylab = 'MCO', main = paste('MCO(Label = ', as.character(unique(dfTrt[, 'label'])), ')', sep = ''))
for (i in 2:nTrt) {
  lines(time, dfTrt[i, columnIsTimeIndex], type = 'l', col = 2)
}

#=============================================================================================================


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
     xlab = 'Time', ylab = 'MCO',
     main = paste('MCO(Label = ', 
                  as.character(unique(dfControlSmooth[, 'label'])), 
                  ', n = ', as.character(length(unique(dfControlSmooth$id))), 
                  ')', sep = ''))
for (i in 2:nControl) {
  lines(time, dfControlSmooth[i, columnIsTimeIndex], type = 'l', col = 1)
}

plot(time, dfTrtSmooth[1, columnIsTimeIndex], type = 'l', col = 2, ylim = c(0, xMax), 
     xlab = 'Time', ylab = 'MCO', 
     main = paste('MCO(Label = ', 
                  as.character(unique(dfTrtSmooth[, 'label'])), 
                  ', n = ', as.character(length(unique(dfTrtSmooth$id))), 
                  ')', sep = ''))
for (i in 2:nTrt) {
  lines(time, dfTrtSmooth[i, columnIsTimeIndex], type = 'l', col = 2)
}




#=============================================================================================================
# From smoothed data plot we can see that a small proportion of observations have MCO >= 2.0
# Try separate these observations and analyse them separately

# Transform dfSmooth to long format, then exclude the id whose maximum MCO reading is >= 2.0
dfSmoothLong = tidyr::gather(dfSmooth, time, value, '180':'3590', factor_key = FALSE) %>% 
  arrange(id, as.numeric(time))
dfSmoothLongBelow2 = dfSmoothLong %>% 
  group_by(id) %>% 
  filter(all(value < 2)) %>% 
  ungroup(id)
dfSmoothBelow2 = dfSmoothLongBelow2 %>% 
  tidyr::spread(time, value)
nameChar = names(select(dfSmoothBelow2, -idOriginal, -id, -label))
dfSmoothBelow2 = dfSmoothBelow2[, c(1:3, order(as.numeric(nameChar)) + 3)] %>% 
  as.data.frame(.)
dfControlSmoothBelow2 = filter(dfSmoothBelow2, label == 0)
dfTrtSmoothBelow2 = filter(dfSmoothBelow2, label == 1)

# Plot
par(mfrow = c(2, 1))
plot(time, dfControlSmoothBelow2[1, columnIsTimeIndex], type = 'l', col = 1, ylim = c(0, 2), 
     xlab = 'Time', ylab = 'MCO', 
     main = paste('MCO(Label = ', 
                  as.character(unique(dfControlSmoothBelow2[, 'label'])), 
                  ', n = ', as.character(length(unique(dfControlSmoothBelow2$id))), 
                  ')', sep = ''))
for (i in 2:nControl) {
  lines(time, dfControlSmoothBelow2[i, columnIsTimeIndex], type = 'l', col = 1)
}

plot(time, dfTrtSmoothBelow2[1, columnIsTimeIndex], type = 'l', col = 2, ylim = c(0, 2), 
     xlab = 'Time', ylab = 'MCO', 
     main = paste('MCO(Label = ', 
                  as.character(unique(dfTrtSmoothBelow2[, 'label'])), 
                  ', n = ', as.character(length(unique(dfTrtSmoothBelow2$id))), 
                  ')', sep = ''))
for (i in 2:nTrt) {
  lines(time, dfTrtSmoothBelow2[i, columnIsTimeIndex], type = 'l', col = 2)
}






#=============================================================================================================
# Exclude the id where at least one of its MCO reading is < 2.0 at any time t
dfSmoothLongAbove2 = dfSmoothLong %>% 
  group_by(id) %>% 
  filter(any(value >= 2)) %>% 
  ungroup(id)
dfSmoothAbove2 = dfSmoothLongAbove2 %>% 
  tidyr::spread(time, value)
nameChar = names(select(dfSmoothAbove2, -idOriginal, -id, -label))
dfSmoothAbove2 = dfSmoothAbove2[, c(1:3, order(as.numeric(nameChar)) + 3)] %>% 
  as.data.frame(.)
dfControlSmoothAbove2 = filter(dfSmoothAbove2, label == 0)
dfTrtSmoothAbove2 = filter(dfSmoothAbove2, label == 1)

# Plot
par(mfrow = c(2, 1))
plot(time, dfControlSmoothAbove2[1, columnIsTimeIndex], type = 'l', col = 1, ylim = c(0, xMax), 
     xlab = 'Time', ylab = 'MCO', 
     main = paste('MCO(Label = ', 
                  as.character(unique(dfControlSmoothAbove2[, 'label'])), 
                  ', n = ', as.character(length(unique(dfControlSmoothAbove2$id))), 
                  ')', sep = ''))
for (i in 2:nControl) {
  lines(time, dfControlSmoothAbove2[i, columnIsTimeIndex], type = 'l', col = 1)
}

plot(time, dfTrtSmoothAbove2[1, columnIsTimeIndex], type = 'l', col = 2, ylim = c(0, xMax), 
     xlab = 'Time', ylab = 'MCO', 
     main = paste('MCO(Label = ', 
                  as.character(unique(dfTrtSmoothAbove2[, 'label'])), 
                  ', n = ', as.character(length(unique(dfTrtSmoothAbove2$id))), 
                  ')', sep = ''))
for (i in 2:nTrt) {
  lines(time, dfTrtSmoothAbove2[i, columnIsTimeIndex], type = 'l', col = 2)
}


