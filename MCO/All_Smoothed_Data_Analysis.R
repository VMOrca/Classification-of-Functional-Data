#########################################################################################################################################
#
#                                                       Author: Min Sun
#
#########################################################################################################################################
# R script to analyse smoothed MCO data
options(stringsAsFactors = FALSE)

#########################################################################################################################################
# Change hyperparameters here!
#########################################################################################################################################
setwd('D:/Academics/UNSW/Thesis/R/Git/MCO/')
source('Data_Preparation.R')
# How many cores for parallel computing
nCore = 5





#########################################################################################################################################
n = dim(dfSmooth)[1]
idAll = unique(dfSmooth$id)


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
mcoKnn$setClassifier('knn')
# hyperparChoice = 3:9
tic()
mcoKnn$cvClassifier(iter = 100, hyperparChoice = 1:round((n/2)), nCore = nCore, trainingPct = 0.6, t = time, metric = 'LpNorm')
toc()
mcoKnn$runOnTestSet(t = time, metric = 'LpNorm')

# save(mcoKnn, file = 'mcoKnn.RData')
load('mcoKnn.RData')




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
mcoFnwe$setClassifier('fnwe')
# hyperparChoice = 1:5
tic()
mcoFnwe$cvClassifier(iter = 100, hyperparChoice = 1:5,
                     nCore = nCore, trainingPct = 0.6, t = time, metric = 'LpNorm', kernelChoice = 'gaussian')
toc()
mcoFnwe$runOnTestSet(t = time, metric = 'LpNorm', kernelChoice = 'gaussian')

# save(mcoFnwe, file = 'mcoFnwe.RData')
load('mcoFnwe.RData')




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
mcoKernelRule$setClassifier('kernelRule')
# hyperparChoice = 1:5
tic()
mcoKernelRule$cvClassifier(iter = 100, hyperparChoice = 1:100, 
                           nCore = nCore, trainingPct = 0.6, t = time, metric = 'LpNorm', kernelChoice = 'gaussian')
toc()
mcoKernelRule$runOnTestSet(t = time, metric = 'LpNorm', kernelChoice = 'gaussian')

# save(mcoKernelRule, file = 'mcoKernelRule.RData')
load('mcoKernelRule.RData')





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
mcoFglm$setClassifier('fglm')
tic()
# Not sure why when zeroMeanBool = FALSE, the recovered function oscillate a lot => use zeroMeanBool = TRUE fow now
mcoFglm$trainClassifier(trainingPct = 0.6, t = time, proportion = 0.99, expansion = 'kl', zeroMeanBool = TRUE)
toc()
mcoFglm$runOnTestSet(t = time)

# save(mcoFglm, file = 'mcoFglm.RData')
load('mcoFglm.RData')




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
mcoFSvm$setClassifier('fSvm')
tic()
# Not sure why when zeroMeanBool = FALSE, the recovered function oscillate a lot => use zeroMeanBool = TRUE fow now
# hyperparChoice: gamma = c(6, 0.6, 2, 4, 5), cost not very significant
svmParChoice = list('gamma' = c(seq(0.1, 0.9, 0.1), 1:50),
                    'cost' = 1:50)
mcoFSvm$cvClassifier(iter = 100, hyperparChoice = svmParChoice, nCore = nCore, trainingPct = 0.6,
                     t = time, proportion = 0.99, expansion = 'kl', zeroMeanBool = TRUE, kernelChoice = 'radial')
toc()
mcoFSvm$runOnTestSet(t = time)

# save(mcoFSvm, file = 'mcoFSvm.RData')
load('mcoFSvm.RData')





# Collect results from all classification methods
accuracyValidation = c(mcoKnn$accuracyValidation, 
                       mcoFnwe$accuracyValidation, 
                       mcoKernelRule$accuracyValidation, 
                       mcoFglm$accuracyValidation, 
                       mcoFSvm$accuracyValidation)
accuracyPrediction = c(mcoKnn$accuracyPrediction, 
                       mcoFnwe$accuracyPrediction, 
                       mcoKernelRule$accuracyPrediction, 
                       mcoFglm$accuracyPrediction, 
                       mcoFSvm$accuracyPrediction)
classificationMethods = c('fKNN', 'fNWE', 'fKR', 'fGLM', 'fSVM')
performanceSmoothed = data.frame('methods' = classificationMethods, 
                                 accuracyValidation = accuracyValidation, 
                                 accuracyTest = accuracyPrediction)


# Convert to LaTeX table
# library(xtable)
# xtable(performanceSmoothed)
