#########################################################################################################################################
#
#                                                       Author: Min Sun
#
#########################################################################################################################################

# Script to run data analysis for the intensity process (of the Hawkes process)
# You will need to change the directories below to your local ones
options(stringsAsFactors = FALSE)
# How many cores for parallel computing
nCore = 5

library(R6)
library(fda.usc)
library(tidyverse)
library(fda)
library(ggplot2)
library(parallel)
library(doSNOW)
library(tictoc)
library(e1071)

setwd('D:/Academics/UNSW/Thesis/R/Git/')
source('Functions/fglm.R')
source('Functions/fglmPred.R')
source('Functions/fnwe.R')
source('Functions/fpca.R')
source('Functions/karhunenLoeve.R')
source('Functions/kernelRule.R')
source('Functions/knn.R')
source('Functions/auc.R')
source('Functions/LpNorm.R')
source('Functions/supNorm.R')
source('Functions/multipleKarhunenLoeve.R')
source('Functions/mlFramework.R')
source('Functions/cvFSvm.R')
source('Functions/fSvmPred.R')


source('D:/Academics/UNSW/Thesis/R/Git/Simulation/Simulation.R')
setwd('D:/Academics/UNSW/Thesis/R/Git/Simulation/Hawkes/')

#########################################################################################################################################
nHawkes = dim(dfHawkes)[1]
idHawkesAll = unique(dfHawkes$id)
dfMeta = select(dfHawkes, id, label)
t = as.numeric(colnames(select(dfHawkes, -id, -label)))

# Divide all subjects to traing/validation/test sets
# Test: 20% fixed subjects. Training/Validation: 60%/20% which will vary between the remaining subjects for each round of cross validation
# Some methods such as KNN does not require parameter estimation - hence will not need training set. The hyperparameters will need to be
# determined by validation set though. In this case all remaining data (i.e. 80%) will be validation set.
set.seed(1)
idHawkesTest = sort(sample(idHawkesAll, nHawkes * 0.2))
idHawkesNonTest = idHawkesAll[which(!idHawkesAll %in% idHawkesTest)]
dfHawkesTest = dfHawkes[idHawkesTest, ]
dfHawkesNonTest = dfHawkes[idHawkesNonTest, ]
nHawkesTest = length(idHawkesTest)
nHawkesNonTest = nHawkes - nHawkesTest
nHawkesTraining = nHawkes * 0.6
nHawkesValidation = nHawkes * 0.2




# Functional knn
set.seed(1)
HawkesKnn = mlFramework$new()
HawkesKnn$setData(dfMeta = dfMeta, 
                  dfAll = dfHawkes, 
                  dfNonTest = dfHawkesNonTest, 
                  dfTest = dfHawkesTest, 
                  nNonTest = nHawkesNonTest, 
                  nTest = nHawkesTest, 
                  idNonTest = idHawkesNonTest, 
                  idTest = idHawkesTest)
HawkesKnn$setClassifier('knn')
tic()
# HawkesKnn$cvClassifier(iter = 100, hyperparChoice = 1:round((n/2)), nCore = nCore, trainingPct = 0.6, t = time, metric = LpNorm)
# hyperparChoice = 7:17
HawkesKnn$cvClassifier(iter = 5, hyperparChoice = 1:round((nHawkesNonTest/10)), nCore = nCore, trainingPct = 0.6, t = t, metric = 'supNorm')
toc()
HawkesKnn$runOnTestSet(t = t, metric = 'supNorm')

save(HawkesKnn, file = 'HawkesSupNormKnn.RData')
# load('HawkesKnn.RData')
# load('HawkesSupNormKnn.RData')




# Functional Nadaraya-Watson estimator
set.seed(1)
HawkesFnwe = mlFramework$new()
HawkesFnwe$setData(dfMeta = dfMeta, 
                   dfAll = dfHawkes, 
                   dfNonTest = dfHawkesNonTest, 
                   dfTest = dfHawkesTest, 
                   nNonTest = nHawkesNonTest, 
                   nTest = nHawkesTest, 
                   idNonTest = idHawkesNonTest, 
                   idTest = idHawkesTest)
HawkesFnwe$setClassifier('fnwe')
# hyperparChoice = seq(0.4, 0.9, 0.1)
tic()
HawkesFnwe$cvClassifier(iter = 5, hyperparChoice = c(0.1, 0.5, 1, 5, 10, 50, 100), 
                        nCore = nCore, trainingPct = 0.6, t = t, metric = 'supNorm', kernelChoice = 'gaussian') 
toc()
HawkesFnwe$runOnTestSet(t = t, metric = 'supNorm', kernelChoice = 'gaussian')

save(HawkesFnwe, file = 'HawkesSupNormFnwe.RData')
# load('HawkesFnwe.RData')
# load('HawkesSupNormFnwe.RData')






# Functional kernel rule
set.seed(1)
HawkesKernelRule = mlFramework$new()
HawkesKernelRule$setData(dfMeta = dfMeta,
                         dfAll = dfHawkes,
                         dfNonTest = dfHawkesNonTest,
                         dfTest = dfHawkesTest,
                         nNonTest = nHawkesNonTest,
                         nTest = nHawkesTest,
                         idNonTest = idHawkesNonTest,
                         idTest = idHawkesTest)
HawkesKernelRule$setClassifier('kernelRule')
tic()
# Try hyperparChoice = seq(0.55, 0.75, 0.01),
HawkesKernelRule$cvClassifier(iter = 5, hyperparChoice = seq(0.2, 0.6, 0.1),
                              nCore = nCore, trainingPct = 0.6, t = t, metric = 'supNorm', kernelChoice = 'gaussian')
toc()
HawkesKernelRule$runOnTestSet(t = t, metric = 'supNorm', kernelChoice = 'gaussian')

save(HawkesKernelRule, file = 'HawkesSupNormKernelRule.RData')
# load('HawkesKernelRule.RData')
# load('HawkesSupNormKernelRule.RData')






# Functional GLM
set.seed(1)
HawkesFglm = mlFramework$new()
HawkesFglm$setData(dfMeta = dfMeta,
                   dfAll = dfHawkes,
                   dfNonTest = dfHawkesNonTest,
                   dfTest = dfHawkesTest,
                   nNonTest = nHawkesNonTest,
                   nTest = nHawkesTest,
                   idNonTest = idHawkesNonTest,
                   idTest = idHawkesTest)
HawkesFglm$setClassifier('fglm')
tic()
# Not sure why when zeroMeanBool = FALSE, the recovered function oscillate a lot => use zeroMeanBool = TRUE fow now
HawkesFglm$trainClassifier(trainingPct = 0.6, t = t, proportion = 0.95, expansion = 'kl', zeroMeanBool = TRUE)
toc()
HawkesFglm$runOnTestSet(t = t)

save(HawkesFglm, file = 'HawkesFglm.RData')
# load('HawkesFglm.RData')





# Functional SVM
set.seed(1)
HawkesFSvm = mlFramework$new()
HawkesFSvm$setData(dfMeta = dfMeta,
                   dfAll = dfHawkes,
                   dfNonTest = dfHawkesNonTest,
                   dfTest = dfHawkesTest,
                   nNonTest = nHawkesNonTest,
                   nTest = nHawkesTest,
                   idNonTest = idHawkesNonTest,
                   idTest = idHawkesTest)
HawkesFSvm$setClassifier('fSvm')
tic()
# Not sure why when zeroMeanBool = FALSE, the recovered function oscillate a lot => use zeroMeanBool = TRUE fow now
# Try Gamma very small, e.g. seq(0.001, 0.03, 0.005). cost is not important here
# svmParChoice = list('gamma' = c(seq(0.001, 0.1, 0.01)),
#                     'cost' = 1:10)
svmParChoice = list('gamma' = c(seq(0.005, 0.015, 0.001)),
                    'cost' = 1)
# proportion = 0.85
HawkesFSvm$cvClassifier(iter = 5, hyperparChoice = svmParChoice, nCore = nCore, trainingPct = 0.6,
                        t = t, proportion = 0.85, expansion = 'kl', zeroMeanBool = TRUE, kernelChoice = 'radial')
toc()
HawkesFSvm$runOnTestSet(t = t)

save(HawkesFSvm, file = 'HawkesFSvm.RData')
# load('HawkesFSvm.RData')




# Load RData after completing all analysis above
# load('HawkesKnn.RData')
# load('HawkesFnwe.RData')
# load('HawkesKernelRule.RData')
# load('HawkesFglm.RData')
# load('HawkesFSvm.RData')
# 
# load('HawkesSupNormKnn.RData')
# load('HawkesSupNormFnwe.RData')
# load('HawkesSupNormKernelRule.RData')



# Collect results from all classification methods
accuracyValidation = c(HawkesKnn$accuracyValidation,
                       HawkesFnwe$accuracyValidation,
                       HawkesKernelRule$accuracyValidation,
                       HawkesFglm$accuracyValidation,
                       HawkesFSvm$accuracyValidation)
accuracyPrediction = c(HawkesKnn$accuracyPrediction,
                       HawkesFnwe$accuracyPrediction,
                       HawkesKernelRule$accuracyPrediction,
                       HawkesFglm$accuracyPrediction,
                       HawkesFSvm$accuracyPrediction)
classificationMethods = c('fKNN', 'fNWE', 'fKR', 'fGLM', 'fSVM')
performanceHawkes = data.frame('methods' = classificationMethods,
                                accuracyValidation = accuracyValidation,
                                accuracyTest = accuracyPrediction)

# Convert to LaTeX table
# library(xtable)
# xtable(performanceHawkes)