#########################################################################################################################################
#
#                                                       Author: Min Sun
#
#########################################################################################################################################

# Script to run data analysis for OU processes with different \sigma
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
setwd('D:/Academics/UNSW/Thesis/R/Git/Simulation/OUDifSigma/')

#########################################################################################################################################
nOUDifSigma = dim(dfOUDifSigma)[1]
idOUDifSigmaAll = unique(dfOUDifSigma$id)
dfMeta = select(dfOUDifSigma, id, label)
t = as.numeric(colnames(select(dfOUDifSigma, -id, -label)))

# Divide all subjects to traing/validation/test sets
# Test: 20% fixed subjects. Training/Validation: 60%/20% which will vary between the remaining subjects for each round of cross validation
# Some methods such as KNN does not require parameter estimation - hence will not need training set. The hyperparameters will need to be
# determined by validation set though. In this case all remaining data (i.e. 80%) will be validation set.
set.seed(1)
idOUDifSigmaTest = sort(sample(idOUDifSigmaAll, nOUDifSigma * 0.2))
idOUDifSigmaNonTest = idOUDifSigmaAll[which(!idOUDifSigmaAll %in% idOUDifSigmaTest)]
dfOUDifSigmaTest = dfOUDifSigma[idOUDifSigmaTest, ]
dfOUDifSigmaNonTest = dfOUDifSigma[idOUDifSigmaNonTest, ]
nOUDifSigmaTest = length(idOUDifSigmaTest)
nOUDifSigmaNonTest = nOUDifSigma - nOUDifSigmaTest
nOUDifSigmaTraining = nOUDifSigma * 0.6
nOUDifSigmaValidation = nOUDifSigma * 0.2




# Functional knn
set.seed(1)
OUDifSigmaKnn = mlFramework$new()
OUDifSigmaKnn$setData(dfMeta = dfMeta, 
                      dfAll = dfOUDifSigma, 
                      dfNonTest = dfOUDifSigmaNonTest, 
                      dfTest = dfOUDifSigmaTest, 
                      nNonTest = nOUDifSigmaNonTest, 
                      nTest = nOUDifSigmaTest, 
                      idNonTest = idOUDifSigmaNonTest, 
                      idTest = idOUDifSigmaTest)
OUDifSigmaKnn$setClassifier('knn')
tic()
# OUDifSigmaKnn$cvClassifier(iter = 100, hyperparChoice = 1:round((n/2)), nCore = nCore, trainingPct = 0.6, t = time, metric = LpNorm)
# hyperparChoice = 7:17
OUDifSigmaKnn$cvClassifier(iter = 10, hyperparChoice = 1:30, nCore = nCore, trainingPct = 0.6, t = t, metric = 'supNorm')
toc()
OUDifSigmaKnn$runOnTestSet(t = t, metric = 'supNorm')

save(OUDifSigmaKnn, file = 'OUDifSigmaSupNormKnn.RData')
# load('OUDifSigmaKnn.RData')
# load('OUDifSigmaSupNormKnn.RData')




# Functional Nadaraya-Watson estimator
set.seed(1)
OUDifSigmaFnwe = mlFramework$new()
OUDifSigmaFnwe$setData(dfMeta = dfMeta,
                       dfAll = dfOUDifSigma,
                       dfNonTest = dfOUDifSigmaNonTest,
                       dfTest = dfOUDifSigmaTest,
                       nNonTest = nOUDifSigmaNonTest,
                       nTest = nOUDifSigmaTest,
                       idNonTest = idOUDifSigmaNonTest,
                       idTest = idOUDifSigmaTest)
OUDifSigmaFnwe$setClassifier('fnwe')
# hyperparChoice = seq(0.4, 0.9, 0.1)
tic()
OUDifSigmaFnwe$cvClassifier(iter = 10, hyperparChoice = seq(0.1, 1.5, 0.1),
                            nCore = nCore, trainingPct = 0.6, t = t, metric = 'supNorm', kernelChoice = 'gaussian')
toc()
OUDifSigmaFnwe$runOnTestSet(t = t, metric = 'supNorm', kernelChoice = 'gaussian')

save(OUDifSigmaFnwe, file = 'OUDifSigmaSupNormFnwe.RData')
# load('OUDifSigmaFnwe.RData')
# load('OUDifSigmaSupNormFnwe.RData')





# Functional kernel rule
set.seed(1)
OUDifSigmaKernelRule = mlFramework$new()
OUDifSigmaKernelRule$setData(dfMeta = dfMeta, 
                             dfAll = dfOUDifSigma, 
                             dfNonTest = dfOUDifSigmaNonTest, 
                             dfTest = dfOUDifSigmaTest, 
                             nNonTest = nOUDifSigmaNonTest, 
                             nTest = nOUDifSigmaTest, 
                             idNonTest = idOUDifSigmaNonTest, 
                             idTest = idOUDifSigmaTest)
OUDifSigmaKernelRule$setClassifier('kernelRule')
tic()
# Try hyperparChoice = seq(0.55, 0.75, 0.01),
OUDifSigmaKernelRule$cvClassifier(iter = 10, hyperparChoice = seq(0.3, 1, 0.1), 
                                  nCore = nCore, trainingPct = 0.6, t = t, metric = 'supNorm', kernelChoice = 'gaussian') 
toc()
OUDifSigmaKernelRule$runOnTestSet(t = t, metric = 'supNorm', kernelChoice = 'gaussian')

save(OUDifSigmaKernelRule, file = 'OUDifSigmaSupNormKernelRule.RData')
# load('OUDifSigmaKernelRule.RData')
# load('OUDifSigmaSupNormKernelRule.RData')






# Functional GLM
set.seed(1)
OUDifSigmaFglm = mlFramework$new()
OUDifSigmaFglm$setData(dfMeta = dfMeta,
                       dfAll = dfOUDifSigma,
                       dfNonTest = dfOUDifSigmaNonTest,
                       dfTest = dfOUDifSigmaTest,
                       nNonTest = nOUDifSigmaNonTest,
                       nTest = nOUDifSigmaTest,
                       idNonTest = idOUDifSigmaNonTest,
                       idTest = idOUDifSigmaTest)
OUDifSigmaFglm$setClassifier('fglm')
tic()
# Not sure why when zeroMeanBool = FALSE, the recovered function oscillate a lot => use zeroMeanBool = TRUE fow now
OUDifSigmaFglm$trainClassifier(trainingPct = 0.6, t = t, proportion = 0.9, expansion = 'kl', zeroMeanBool = TRUE)
toc()
OUDifSigmaFglm$runOnTestSet(t = t)

save(OUDifSigmaFglm, file = 'OUDifSigmaFglm.RData')
# load('OUDifSigmaFglm.RData')





# Functional SVM
set.seed(1)
OUDifSigmaFSvm = mlFramework$new()
OUDifSigmaFSvm$setData(dfMeta = dfMeta,
                       dfAll = dfOUDifSigma,
                       dfNonTest = dfOUDifSigmaNonTest,
                       dfTest = dfOUDifSigmaTest,
                       nNonTest = nOUDifSigmaNonTest,
                       nTest = nOUDifSigmaTest,
                       idNonTest = idOUDifSigmaNonTest,
                       idTest = idOUDifSigmaTest)
OUDifSigmaFSvm$setClassifier('fSvm')
tic()
# Not sure why when zeroMeanBool = FALSE, the recovered function oscillate a lot => use zeroMeanBool = TRUE fow now
# Try Gamma very small, e.g. seq(0.001, 0.03, 0.005). cost is not important here
svmParChoice = list('gamma' = c(seq(0.001, 0.1, 0.01)),
                    'cost' = c(seq(0.5, 1.5, 0.1)))
OUDifSigmaFSvm$cvClassifier(iter = 10, hyperparChoice = svmParChoice, nCore = nCore, trainingPct = 0.6,
                            t = t, proportion = 0.9, expansion = 'kl', zeroMeanBool = TRUE, kernelChoice = 'radial')
toc()
OUDifSigmaFSvm$runOnTestSet(t = t)

save(OUDifSigmaFSvm, file = 'OUDifSigmaFSvm.RData')
# load('OUDifSigmaFSvm.RData')






# Collect results from all classification methods
accuracyValidation = c(OUDifSigmaKnn$accuracyValidation,
                       OUDifSigmaFnwe$accuracyValidation,
                       OUDifSigmaKernelRule$accuracyValidation,
                       OUDifSigmaFglm$accuracyValidation,
                       OUDifSigmaFSvm$accuracyValidation)
accuracyPrediction = c(OUDifSigmaKnn$accuracyPrediction,
                       OUDifSigmaFnwe$accuracyPrediction,
                       OUDifSigmaKernelRule$accuracyPrediction,
                       OUDifSigmaFglm$accuracyPrediction,
                       OUDifSigmaFSvm$accuracyPrediction)
classificationMethods = c('fKNN', 'fNWE', 'fKR', 'fGLM', 'fSVM')
performanceOUDifSigma = data.frame('methods' = classificationMethods,
                                   accuracyValidation = accuracyValidation,
                                   accuracyTest = accuracyPrediction)

# Convert to LaTeX table
# library(xtable)
# xtable(performanceOUDifSigma)
