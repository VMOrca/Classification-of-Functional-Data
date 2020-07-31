options(stringsAsFactors = FALSE)

library(R6)
library(fda.usc)
library(tidyverse)
library(fda)
library(ggplot2)
library(parallel)
library(doSNOW)
library(tictoc)
library(e1071)


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
source('Functions/supNorm.R')
source('Functions/multipleKarhunenLoeve.R')
source('Functions/mlFramework.R')
source('Functions/cvFSvm.R')
source('Functions/fSvmPred.R')




source('D:/Academics/UNSW/Thesis/R/Simulation/Simulation.R')
setwd('D:/Academics/UNSW/Thesis/R/Simulation/Hawkes/')


nCore = 5



# Derive 1st derivative from dfHawkes, store the result in dataframe dfHawkesDeriv
dfHawkesReading = select(dfHawkes, -id, -label)
dfMeta = select(dfHawkes, id, label)
t = as.numeric(colnames(select(dfHawkes, -id, -label)))

derivative = function(x, t) {
  out = diff(x)/diff(t)
  return(out)
}

dfHawkesDeriv = t(apply(dfHawkesReading, 1, derivative, t = t)) %>% 
  cbind(dfMeta, .)
# Need to recalculate t as number of time points are 1 less than the original t
t = as.numeric(colnames(select(dfHawkesDeriv, -id, -label)))


# # Plot derivative
# hawkes0 = filter(dfHawkesDeriv, label == 0) %>%
#   select(-id, -label)
# hawkes1 = filter(dfHawkesDeriv, label == 1) %>%
#   select(-id, -label)
# t = as.numeric(colnames(select(dfHawkesDeriv, -id, -label)))
# xMin = min(min(hawkes0, hawkes1))
# xMax = max(max(hawkes0, hawkes1))
# plotHawkes(t = t, HawkesSample = hawkes0, xMin = xMin, xMax = xMax, 
#            title = paste('1st Derivative of Intensity Process of a Hawkes Process\n (alpha = ', 2, ', beta = ', 20, ')', sep = ''))
# plotHawkes(t = t, HawkesSample = hawkes1, xMin = xMin, xMax = xMax, 
#            title = paste('1st Derivative of Intensity Process of a Hawkes Process\n (alpha = ', 2, ', beta = ', 25, ')', sep = ''))




nHawkesDeriv = dim(dfHawkesDeriv)[1]
idHawkesDerivAll = unique(dfHawkesDeriv$id)

# Divide all subjects to traing/validation/test sets
# Test: 20% fixed subjects. Training/Validation: 60%/20% which will vary between the remaining subjects for each round of cross validation
# Some methods such as KNN does not require parameter estimation - hence will not need training set. The hyperparameters will need to be
# determined by validation set though. In this case all remaining data (i.e. 80%) will be validation set.
set.seed(1)
idHawkesDerivTest = sort(sample(idHawkesDerivAll, nHawkesDeriv * 0.2))
idHawkesDerivNonTest = idHawkesDerivAll[which(!idHawkesDerivAll %in% idHawkesDerivTest)]
dfHawkesDerivTest = dfHawkesDeriv[idHawkesDerivTest, ]
dfHawkesDerivNonTest = dfHawkesDeriv[idHawkesDerivNonTest, ]
nHawkesDerivTest = length(idHawkesDerivTest)
nHawkesDerivNonTest = nHawkesDeriv - nHawkesDerivTest
nHawkesDerivTraining = nHawkesDeriv * 0.6
nHawkesDerivValidation = nHawkesDeriv * 0.2




# Functional knn
set.seed(1)
HawkesDerivKnn = mlFramework$new()
HawkesDerivKnn$setData(dfMeta = dfMeta, 
                       dfAll = dfHawkesDeriv, 
                       dfNonTest = dfHawkesDerivNonTest, 
                       dfTest = dfHawkesDerivTest, 
                       nNonTest = nHawkesDerivNonTest, 
                       nTest = nHawkesDerivTest, 
                       idNonTest = idHawkesDerivNonTest, 
                       idTest = idHawkesDerivTest)
HawkesDerivKnn$setWd('D:/Academics/UNSW/Thesis/R/Simulation/Hawkes/')
HawkesDerivKnn$setClassifier('knn')
tic()
HawkesDerivKnn$cvClassifier(iter = 10, hyperparChoice = 1:round((nHawkesDerivNonTest/10)), nCore = nCore, trainingPct = 0.6, t = t, metric = 'LpNorm')
# hyperparChoice = c(2, 13, 17, 20, 22, 24, 27, 30)
# HawkesDerivKnn$cvClassifier(iter = 10, hyperparChoice = hyperparChoice, nCore = nCore, trainingPct = 0.6, t = t, metric = 'LpNorm')
toc()
HawkesDerivKnn$runOnTestSet(t = t, metric = 'LpNorm')

save(HawkesDerivKnn, file = 'HawkesDerivKnn.RData')
# load('HawkesDerivKnn.RData')
# load('HawkesDerivSupNormKnn.RData')




# Functional Nadaraya-Watson estimator
set.seed(1)
HawkesDerivFnwe = mlFramework$new()
HawkesDerivFnwe$setData(dfMeta = dfMeta,
                        dfAll = dfHawkesDeriv,
                        dfNonTest = dfHawkesDerivNonTest,
                        dfTest = dfHawkesDerivTest,
                        nNonTest = nHawkesDerivNonTest,
                        nTest = nHawkesDerivTest,
                        idNonTest = idHawkesDerivNonTest,
                        idTest = idHawkesDerivTest)
HawkesDerivFnwe$setWd('D:/Academics/UNSW/Thesis/R/Simulation/Hawkes/')
HawkesDerivFnwe$setClassifier('fnwe')
# hyperparChoice = seq(0.4, 0.9, 0.1)
tic()
HawkesDerivFnwe$cvClassifier(iter = 10, hyperparChoice = seq(0.1, 1, 0.1),
                             nCore = nCore, trainingPct = 0.6, t = t, metric = 'LpNorm', kernelChoice = 'gaussian')
toc()
HawkesDerivFnwe$runOnTestSet(t = t, metric = 'LpNorm', kernelChoice = 'gaussian')

save(HawkesDerivFnwe, file = 'HawkesDerivFnwe.RData')
# load('HawkesDerivFnwe.RData')
# load('HawkesDerivSupNormFnwe.RData')





# Functional kernel rule
set.seed(1)
HawkesDerivKernelRule = mlFramework$new()
HawkesDerivKernelRule$setData(dfMeta = dfMeta,
                              dfAll = dfHawkesDeriv,
                              dfNonTest = dfHawkesDerivNonTest,
                              dfTest = dfHawkesDerivTest,
                              nNonTest = nHawkesDerivNonTest,
                              nTest = nHawkesDerivTest,
                              idNonTest = idHawkesDerivNonTest,
                              idTest = idHawkesDerivTest)
HawkesDerivKernelRule$setWd('D:/Academics/UNSW/Thesis/R/Simulation/Hawkes/')
HawkesDerivKernelRule$setClassifier('kernelRule')
tic()
# Try hyperparChoice = seq(0.55, 0.75, 0.01),
HawkesDerivKernelRule$cvClassifier(iter = 10, hyperparChoice = seq(0.1, 0.8, 0.1),
                                   nCore = nCore, trainingPct = 0.6, t = t, metric = 'LpNorm', kernelChoice = 'gaussian')
toc()
HawkesDerivKernelRule$runOnTestSet(t = t, metric = 'LpNorm', kernelChoice = 'gaussian')

save(HawkesDerivKernelRule, file = 'HawkesDerivKernelRule.RData')
# load('HawkesDerivKernelRule.RData')
# load('HawkesDerivSupNormKernelRule.RData')





# Functional GLM
set.seed(1)
HawkesDerivFglm = mlFramework$new()
HawkesDerivFglm$setData(dfMeta = dfMeta,
                        dfAll = dfHawkesDeriv,
                        dfNonTest = dfHawkesDerivNonTest,
                        dfTest = dfHawkesDerivTest,
                        nNonTest = nHawkesDerivNonTest,
                        nTest = nHawkesDerivTest,
                        idNonTest = idHawkesDerivNonTest,
                        idTest = idHawkesDerivTest)
HawkesDerivFglm$setWd('D:/Academics/UNSW/Thesis/R/Simulation/Hawkes/')
HawkesDerivFglm$setClassifier('fglm')
tic()
# Not sure why when zeroMeanBool = FALSE, the recovered function oscillate a lot => use zeroMeanBool = TRUE fow now
HawkesDerivFglm$trainClassifier(trainingPct = 0.6, t = t, proportion = 0.8, expansion = 'kl', zeroMeanBool = TRUE)
toc()
HawkesDerivFglm$runOnTestSet(t = t)

save(HawkesDerivFglm, file = 'HawkesDerivFglm.RData')
# load('HawkesDerivFglm.RData')





# Functional SVM
set.seed(1)
HawkesDerivFSvm = mlFramework$new()
HawkesDerivFSvm$setData(dfMeta = dfMeta,
                        dfAll = dfHawkesDeriv,
                        dfNonTest = dfHawkesDerivNonTest,
                        dfTest = dfHawkesDerivTest,
                        nNonTest = nHawkesDerivNonTest,
                        nTest = nHawkesDerivTest,
                        idNonTest = idHawkesDerivNonTest,
                        idTest = idHawkesDerivTest)
HawkesDerivFSvm$setWd('D:/Academics/UNSW/Thesis/R/Simulation/Hawkes/')
HawkesDerivFSvm$setClassifier('fSvm')
tic()
# Not sure why when zeroMeanBool = FALSE, the recovered function oscillate a lot => use zeroMeanBool = TRUE fow now
# Try Gamma very small, e.g. seq(0.001, 0.03, 0.005). cost is not important here
svmParChoice = list('gamma' = c(seq(0.005, 0.015, 0.0001)),
                    'cost' = 1)
# proportion = 0.85
HawkesDerivFSvm$cvClassifier(iter = 50, hyperparChoice = svmParChoice, nCore = nCore, trainingPct = 0.6,
                             t = t, proportion = 0.85, expansion = 'kl', zeroMeanBool = TRUE, kernelChoice = 'radial')
toc()
HawkesDerivFSvm$runOnTestSet(t = t)

save(HawkesDerivFSvm, file = 'HawkesDerivFSvm.RData')
# load('HawkesDerivFSvm.RData')



# Load RData
load('HawkesDerivKnn.RData')
load('HawkesDerivFnwe.RData')
load('HawkesDerivKernelRule.RData')
load('HawkesDerivFglm.RData')
load('HawkesDerivFSvm.RData')

load('HawkesDerivSupNormKnn.RData')
load('HawkesDerivSupNormFnwe.RData')
load('HawkesDerivSupNormKernelRule.RData')


# Collect results from all classification methods
accuracyValidation = c(HawkesDerivKnn$accuracyValidation,
                       HawkesDerivFnwe$accuracyValidation,
                       HawkesDerivKernelRule$accuracyValidation,
                       HawkesDerivFglm$accuracyValidation,
                       HawkesDerivFSvm$accuracyValidation)
accuracyPrediction = c(HawkesDerivKnn$accuracyPrediction,
                       HawkesDerivFnwe$accuracyPrediction,
                       HawkesDerivKernelRule$accuracyPrediction,
                       HawkesDerivFglm$accuracyPrediction,
                       HawkesDerivFSvm$accuracyPrediction)
classificationMethods = c('fKNN', 'fNWE', 'fKR', 'fGLM', 'fSVM')
performanceHawkesDeriv = data.frame('methods' = classificationMethods,
                                accuracyValidation = accuracyValidation,
                                accuracyTest = accuracyPrediction)

# library(xtable)
# xtable(performanceHawkesDeriv)
