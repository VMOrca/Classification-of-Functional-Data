#########################################################################################################################################
#
#                                                       Author: Min Sun
#
#########################################################################################################################################

# Script to run data analysis for OU processes with different \mu
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
setwd('D:/Academics/UNSW/Thesis/R/Git/Simulation/OUDifMu/')

#########################################################################################################################################
nOUDifMu = dim(dfOUDifMu)[1]
idOUDifMuAll = unique(dfOUDifMu$id)
dfMeta = select(dfOUDifMu, id, label)
t = as.numeric(colnames(select(dfOUDifMu, -id, -label)))

# Divide all subjects to traing/validation/test sets
# Test: 20% fixed subjects. Training/Validation: 60%/20% which will vary between the remaining subjects for each round of cross validation
# Some methods such as KNN does not require parameter estimation - hence will not need training set. The hyperparameters will need to be
# determined by validation set though. In this case all remaining data (i.e. 80%) will be validation set.
set.seed(1)
idOUDifMuTest = sort(sample(idOUDifMuAll, nOUDifMu * 0.2))
idOUDifMuNonTest = idOUDifMuAll[which(!idOUDifMuAll %in% idOUDifMuTest)]
dfOUDifMuTest = dfOUDifMu[idOUDifMuTest, ]
dfOUDifMuNonTest = dfOUDifMu[idOUDifMuNonTest, ]
nOUDifMuTest = length(idOUDifMuTest)
nOUDifMuNonTest = nOUDifMu - nOUDifMuTest
nOUDifMuTraining = nOUDifMu * 0.6
nOUDifMuValidation = nOUDifMu * 0.2




# Functional knn
set.seed(1)
OUDifMuKnn = mlFramework$new()
OUDifMuKnn$setData(dfMeta = dfMeta, 
                   dfAll = dfOUDifMu, 
                   dfNonTest = dfOUDifMuNonTest, 
                   dfTest = dfOUDifMuTest, 
                   nNonTest = nOUDifMuNonTest, 
                   nTest = nOUDifMuTest, 
                   idNonTest = idOUDifMuNonTest, 
                   idTest = idOUDifMuTest)
OUDifMuKnn$setClassifier('knn')
tic()
# OUDifMuKnn$cvClassifier(iter = 100, hyperparChoice = 1:round((n/2)), nCore = nCore, trainingPct = 0.6, t = time, metric = LpNorm)
# hyperparChoice = 7:17
OUDifMuKnn$cvClassifier(iter = 100, hyperparChoice = 7:17, nCore = nCore, trainingPct = 0.6, t = t, metric = 'LpNorm')
toc()
OUDifMuKnn$runOnTestSet(t = t, metric = 'LpNorm')

# save(OUDifMuKnn, file = 'OUDifMuKnn.RData')
load('OUDifMuKnn.RData')




# Functional Nadaraya-Watson estimator
set.seed(1)
OUDifMuFnwe = mlFramework$new()
OUDifMuFnwe$setData(dfMeta = dfMeta, 
                    dfAll = dfOUDifMu, 
                    dfNonTest = dfOUDifMuNonTest, 
                    dfTest = dfOUDifMuTest, 
                    nNonTest = nOUDifMuNonTest, 
                    nTest = nOUDifMuTest, 
                    idNonTest = idOUDifMuNonTest, 
                    idTest = idOUDifMuTest)
OUDifMuFnwe$setClassifier('fnwe')
# hyperparChoice = seq(0.4, 0.9, 0.1)
tic()
OUDifMuFnwe$cvClassifier(iter = 100, hyperparChoice = seq(0.4, 0.9, 0.1), 
                     nCore = nCore, trainingPct = 0.6, t = t, metric = 'LpNorm', kernelChoice = 'gaussian') 
toc()
OUDifMuFnwe$runOnTestSet(t = t, metric = 'LpNorm', kernelChoice = 'gaussian')

# save(OUDifMuFnwe, file = 'OUDifMuFnwe.RData')
load('OUDifMuFnwe.RData')





# Functional kernel rule
set.seed(1)
OUDifMuKernelRule = mlFramework$new()
OUDifMuKernelRule$setData(dfMeta = dfMeta, 
                          dfAll = dfOUDifMu, 
                          dfNonTest = dfOUDifMuNonTest, 
                          dfTest = dfOUDifMuTest, 
                          nNonTest = nOUDifMuNonTest, 
                          nTest = nOUDifMuTest, 
                          idNonTest = idOUDifMuNonTest, 
                          idTest = idOUDifMuTest)
OUDifMuKernelRule$setClassifier('kernelRule')
tic()
# Try hyperparChoice = seq(0.55, 0.75, 0.01),
OUDifMuKernelRule$cvClassifier(iter = 100, hyperparChoice = seq(0.55, 0.75, 0.01), 
                               nCore = nCore, trainingPct = 0.6, t = t, metric = 'LpNorm', kernelChoice = 'gaussian') 
toc()
OUDifMuKernelRule$runOnTestSet(t = t, metric = 'LpNorm', kernelChoice = 'gaussian')

# save(OUDifMuKernelRule, file = 'OUDifMuKernelRule.RData')
load('OUDifMuKernelRule.RData')






# Functional GLM
set.seed(1)
OUDifMuFglm = mlFramework$new()
OUDifMuFglm$setData(dfMeta = dfMeta, 
                dfAll = dfOUDifMu, 
                dfNonTest = dfOUDifMuNonTest, 
                dfTest = dfOUDifMuTest, 
                nNonTest = nOUDifMuNonTest, 
                nTest = nOUDifMuTest, 
                idNonTest = idOUDifMuNonTest, 
                idTest = idOUDifMuTest)
OUDifMuFglm$setClassifier('fglm')
tic()
# Not sure why when zeroMeanBool = FALSE, the recovered function oscillate a lot => use zeroMeanBool = TRUE fow now
OUDifMuFglm$trainClassifier(trainingPct = 0.6, t = t, proportion = 0.95, expansion = 'kl', zeroMeanBool = TRUE)
toc()
OUDifMuFglm$runOnTestSet(t = t)

# save(OUDifMuFglm, file = 'OUDifMuFglm.RData')
load('OUDifMuFglm.RData')





# Functional SVM
set.seed(1)
OUDifMuFSvm = mlFramework$new()
OUDifMuFSvm$setData(dfMeta = dfMeta, 
                    dfAll = dfOUDifMu, 
                    dfNonTest = dfOUDifMuNonTest, 
                    dfTest = dfOUDifMuTest, 
                    nNonTest = nOUDifMuNonTest, 
                    nTest = nOUDifMuTest, 
                    idNonTest = idOUDifMuNonTest, 
                    idTest = idOUDifMuTest)
OUDifMuFSvm$setClassifier('fSvm')
tic()
# Not sure why when zeroMeanBool = FALSE, the recovered function oscillate a lot => use zeroMeanBool = TRUE fow now
# Try Gamma very small, e.g. seq(0.001, 0.03, 0.005). cost is not important here
svmParChoice = list('gamma' = c(seq(0.001, 0.03, 0.005)),
                    'cost' = 1:5)
OUDifMuFSvm$cvClassifier(iter = 100, hyperparChoice = svmParChoice, nCore = nCore, trainingPct = 0.6, 
                     t = t, proportion = 0.99, expansion = 'kl', zeroMeanBool = TRUE, kernelChoice = 'radial')
toc()
OUDifMuFSvm$runOnTestSet(t = t)

# save(OUDifMuFSvm, file = 'OUDifMuFSvm.RData')
load('OUDifMuFSvm.RData')






# Collect results from all classification methods
accuracyValidation = c(OUDifMuKnn$accuracyValidation, 
                       OUDifMuFnwe$accuracyValidation, 
                       OUDifMuKernelRule$accuracyValidation, 
                       OUDifMuFglm$accuracyValidation, 
                       OUDifMuFSvm$accuracyValidation)
accuracyPrediction = c(OUDifMuKnn$accuracyPrediction, 
                       OUDifMuFnwe$accuracyPrediction, 
                       OUDifMuKernelRule$accuracyPrediction, 
                       OUDifMuFglm$accuracyPrediction, 
                       OUDifMuFSvm$accuracyPrediction)
classificationMethods = c('fKNN', 'fNWE', 'fKR', 'fGLM', 'fSVM')
performanceOUDifMu = data.frame('methods' = classificationMethods, 
                                 accuracyValidation = accuracyValidation, 
                                 accuracyTest = accuracyPrediction)

# Convert to LaTeX table
# library(xtable)
# xtable(performanceOUDifMu)
