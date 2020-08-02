#########################################################################################################################################
#
#                                                       Author: Min Sun
#
#########################################################################################################################################
# R script to analyse unsmoothed MCO data
options(stringsAsFactors = FALSE)

#########################################################################################################################################
# Change hyperparameters here!
#########################################################################################################################################
setwd('D:/Academics/UNSW/Thesis/R/Git/MCO/')
source('Data_Preparation.R')
# How many cores for parallel computing
nCore = 5





#########################################################################################################################################
dfUnSmooth = df
names(dfUnSmooth) = names(dfSmooth)
n = dim(dfUnSmooth)[1]
idAll = unique(dfUnSmooth$id)

# Divide all subjects to traing/validation/test sets
# Test: 20% fixed subjects. Training/Validation: 60%/20% which will vary between the remaining subjects for each round of cross validation
# Some methods such as KNN does not require parameter estimation - hence will not need training set. The hyperparameters will need to be
# determined by validation set though. In this case all remaining data (i.e. 80%) will be validation set.
set.seed(1)
idTest = sort(sample(idAll, n * 0.2))
idNonTest = idAll[which(!idAll %in% idTest)]
dfUnSmoothTest = dfUnSmooth[idTest, ]
dfUnSmoothNonTest = dfUnSmooth[idNonTest, ]
nTest = length(idTest)
nNonTest = n - nTest
nTraining = n * 0.6
nValidation = n * 0.2



# Functional knn
set.seed(1)
mcoUnSmoothKnn = mlFramework$new()
mcoUnSmoothKnn$setData(dfMeta = dfMeta, 
               dfAll = select(dfUnSmooth, -idOriginal), 
               dfNonTest = select(dfUnSmoothNonTest, -idOriginal), 
               dfTest = select(dfUnSmoothTest, -idOriginal), 
               nNonTest = nNonTest, 
               nTest = nTest, 
               idNonTest = idNonTest, 
               idTest = idTest)
mcoUnSmoothKnn$setClassifier('knn')
tic()
mcoUnSmoothKnn$cvClassifier(iter = 100, hyperparChoice = 3:9, nCore = ncore, trainingPct = 0.6, t = time, metric = 'LpNorm') 
toc()
mcoUnSmoothKnn$runOnTestSet(t = time, metric = 'LpNorm')

# save(mcoUnSmoothKnn, file = 'mcoUnSmoothKnn.RData')
load('mcoUnSmoothKnn.RData')



# Functional Nadaraya-Watson estimator
set.seed(1)
mcoUnSmoothFnwe = mlFramework$new()
mcoUnSmoothFnwe$setData(dfMeta = dfMeta, 
                dfAll = select(dfUnSmooth, -idOriginal), 
                dfNonTest = select(dfUnSmoothNonTest, -idOriginal), 
                dfTest = select(dfUnSmoothTest, -idOriginal), 
                nNonTest = nNonTest, 
                nTest = nTest, 
                idNonTest = idNonTest, 
                idTest = idTest)
mcoUnSmoothFnwe$setClassifier('fnwe')
tic()
mcoUnSmoothFnwe$cvClassifier(iter = 100, hyperparChoice = 1:5, 
                             nCore = ncore, trainingPct = 0.6, t = time, metric = 'LpNorm', kernelChoice = 'gaussian') 
toc()
mcoUnSmoothFnwe$runOnTestSet(t = time, metric = 'LpNorm', kernelChoice = 'gaussian')

# save(mcoUnSmoothFnwe, file = 'mcoUnSmoothFnwe.RData')
load('mcoUnSmoothFnwe.RData')




# Functional kernel rule
set.seed(1)
mcoUnSmoothKernelRule = mlFramework$new()
mcoUnSmoothKernelRule$setData(dfMeta = dfMeta, 
                      dfAll = select(dfUnSmooth, -idOriginal), 
                      dfNonTest = select(dfUnSmoothNonTest, -idOriginal), 
                      dfTest = select(dfUnSmoothTest, -idOriginal), 
                      nNonTest = nNonTest, 
                      nTest = nTest, 
                      idNonTest = idNonTest, 
                      idTest = idTest)
mcoUnSmoothKernelRule$setClassifier('kernelRule')
tic()
mcoUnSmoothKernelRule$cvClassifier(iter = 100, hyperparChoice = 1:5, 
                                   nCore = ncore, trainingPct = 0.6, t = time, metric = 'LpNorm', kernelChoice = 'gaussian') 
toc()
mcoUnSmoothKernelRule$runOnTestSet(t = time, metric = 'LpNorm', kernelChoice = 'gaussian')

# save(mcoUnSmoothKernelRule, file = 'mcoUnSmoothKernelRule.RData')
load('mcoUnSmoothKernelRule.RData')





# Functional GLM
set.seed(1)
mcoUnSmoothFglm = mlFramework$new()
mcoUnSmoothFglm$setData(dfMeta = dfMeta, 
                dfAll = select(dfUnSmooth, -idOriginal), 
                dfNonTest = select(dfUnSmoothNonTest, -idOriginal), 
                dfTest = select(dfUnSmoothTest, -idOriginal), 
                nNonTest = nNonTest, 
                nTest = nTest, 
                idNonTest = idNonTest, 
                idTest = idTest)
mcoUnSmoothFglm$setClassifier('fglm')
tic()
# Not sure why when zeroMeanBool = FALSE, the recovered function oscillate a lot => use zeroMeanBool = TRUE fow now
mcoUnSmoothFglm$trainClassifier(trainingPct = 0.6, t = time, proportion = 0.99, expansion = 'kl', zeroMeanBool = TRUE)
toc()
mcoUnSmoothFglm$runOnTestSet(t = time)

# save(mcoUnSmoothFglm, file = 'mcoUnSmoothFglm.RData')
load('mcoUnSmoothFglm.RData')




# Functional SVM
set.seed(1)
mcoUnSmoothFSvm = mlFramework$new()
mcoUnSmoothFSvm$setData(dfMeta = dfMeta, 
                dfAll = select(dfUnSmooth, -idOriginal), 
                dfNonTest = select(dfUnSmoothNonTest, -idOriginal), 
                dfTest = select(dfUnSmoothTest, -idOriginal), 
                nNonTest = nNonTest, 
                nTest = nTest, 
                idNonTest = idNonTest, 
                idTest = idTest)
mcoUnSmoothFSvm$setClassifier('fSvm')
tic()
# Not sure why when zeroMeanBool = FALSE, the recovered function oscillate a lot => use zeroMeanBool = TRUE fow now
svmParChoice = list('gamma' = c(6, 0.6, 2, 4, 5), 
                    'cost' = 1:5)
mcoUnSmoothFSvm$cvClassifier(iter = 100, hyperparChoice = svmParChoice, nCore = ncore, trainingPct = 0.6, 
                     t = time, proportion = 0.99, expansion = 'kl', zeroMeanBool = TRUE, kernelChoice = 'radial')
toc()
mcoUnSmoothFSvm$runOnTestSet(t = time)

# save(mcoUnSmoothFSvm, file = 'mcoUnSmoothFSvm.RData')
load('mcoUnSmoothFSvm.RData')






# Collect results from all classification methods
accuracyValidation = c(mcoUnSmoothKnn$accuracyValidation, 
                       mcoUnSmoothFnwe$accuracyValidation, 
                       mcoUnSmoothKernelRule$accuracyValidation, 
                       mcoUnSmoothFglm$accuracyValidation, 
                       mcoUnSmoothFSvm$accuracyValidation)
accuracyPrediction = c(mcoUnSmoothKnn$accuracyPrediction, 
                       mcoUnSmoothFnwe$accuracyPrediction, 
                       mcoUnSmoothKernelRule$accuracyPrediction, 
                       mcoUnSmoothFglm$accuracyPrediction, 
                       mcoUnSmoothFSvm$accuracyPrediction)
classificationMethods = c('fKNN', 'fNWE', 'fKR', 'fGLM', 'fSVM')
performanceUnSmoothed = data.frame('methods' = classificationMethods, 
                                   accuracyValidation = accuracyValidation, 
                                   accuracyTest = accuracyPrediction)


# Convert to LaTeX table
# library(xtable)
# xtable(performanceUnSmoothed)
