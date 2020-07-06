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
source('Functions/cvFSvm')
source('Functions/fSvmPred')


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
     xlab = 'Time', ylab = 'MCO')
for (i in 2:nControl) {
  lines(time, dfControl[i, columnIsTimeIndex], type = 'l', col = 1)
}

plot(time, dfTrt[1, columnIsTimeIndex], type = 'l', col = 2, ylim = c(0, xMax), 
     xlab = 'Time', ylab = 'MCO')
for (i in 2:nTrt) {
  lines(time, dfTrt[i, columnIsTimeIndex], type = 'l', col = 2)
}


