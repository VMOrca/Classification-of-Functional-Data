#########################################################################################################################################
#
#                                                       Author: Min Sun
#
#########################################################################################################################################
# R script to analyse smoothed data where MCO < 2
options(stringsAsFactors = FALSE)

#########################################################################################################################################
# Change hyperparameters here!
#########################################################################################################################################
setwd('D:/Academics/UNSW/Thesis/R/Git/MCO/')
source('Data_Preparation.R')
# How many cores for parallel computing
nCore = 5





#########################################################################################################################################
idAllBelow2 = unique(dfSmoothBelow2$id)
nBelow2 = length(idAllBelow2)
dfBelow2Meta = select(dfSmoothBelow2, id, label, idOriginal)


# Divide all subjects to traing/validation/test sets
# Test: 20% fixed subjects. Training/Validation: 60%/20% which will vary between the remaining subjects for each round of cross validation
# Some methods such as KNN does not require parameter estimation - hence will not need training set. The hyperparameters will need to be
# determined by validation set though. In this case all remaining data (i.e. 80%) will be validation set.
set.seed(1)
idBelow2Test = sort(sample(idAllBelow2, round(nBelow2 * 0.2)))
idBelow2NonTest = idAllBelow2[which(!idAllBelow2 %in% idBelow2Test)]
dfSmoothBelow2Test = filter(dfSmoothBelow2, id %in% idBelow2Test)
dfSmoothBelow2NonTest = filter(dfSmoothBelow2, id %in% idBelow2NonTest)
nBelow2Test = length(idBelow2Test)
nBelow2NonTest = nBelow2 - nBelow2Test
nBelow2Training = round(nBelow2 * 0.6)
nBelow2Validation = round(nBelow2 * 0.2)





# Functional knn
set.seed(1)
mcoKnnBelow2 = mlFramework$new()
mcoKnnBelow2$setData(dfMeta = dfBelow2Meta,
                     dfAll = select(dfSmoothBelow2, -idOriginal),
                     dfNonTest = select(dfSmoothBelow2NonTest, -idOriginal),
                     dfTest = select(dfSmoothBelow2Test, -idOriginal),
                     nNonTest = nBelow2NonTest,
                     nTest = nBelow2Test,
                     idNonTest = idBelow2NonTest,
                     idTest = idBelow2Test)
mcoKnnBelow2$setClassifier('knn')
# hyperparChoice = 1:6
tic()
mcoKnnBelow2$cvClassifier(iter = 100, hyperparChoice = 1:round((nBelow2/2)), nCore = nCore, trainingPct = 0.6, t = time, metric = 'LpNorm')
toc()
mcoKnnBelow2$runOnTestSet(t = time, metric = 'LpNorm')

# save(mcoKnnBelow2, file = 'mcoKnnBelow2.RData')
load('mcoKnnBelow2.RData')



# Functional Nadaraya-Watson estimator
set.seed(1)
mcoFnweBelow2 = mlFramework$new()
mcoFnweBelow2$setData(dfMeta = dfBelow2Meta, 
                      dfAll = select(dfSmoothBelow2, -idOriginal), 
                      dfNonTest = select(dfSmoothBelow2NonTest, -idOriginal), 
                      dfTest = select(dfSmoothBelow2Test, -idOriginal), 
                      nNonTest = nBelow2NonTest, 
                      nTest = nBelow2Test, 
                      idNonTest = idBelow2NonTest, 
                      idTest = idBelow2Test)
mcoFnweBelow2$setClassifier('fnwe')
tic()
# hyperparChoice = 1:5
mcoFnweBelow2$cvClassifier(iter = 100, hyperparChoice = 1:100, 
                           nCore = nCore, trainingPct = 0.6, t = time, metric = 'LpNorm', kernelChoice = 'gaussian') 
toc()
mcoFnweBelow2$runOnTestSet(t = time, metric = 'LpNorm', kernelChoice = 'gaussian')

save(mcoFnweBelow2, file = 'mcoFnweBelow2.RData')
# load('mcoFnweBelow2.RData')




# Functional kernel rule
set.seed(1)
mcoKernelRuleBelow2 = mlFramework$new()
mcoKernelRuleBelow2$setData(dfMeta = dfBelow2Meta,
                            dfAll = select(dfSmoothBelow2, -idOriginal),
                            dfNonTest = select(dfSmoothBelow2NonTest, -idOriginal),
                            dfTest = select(dfSmoothBelow2Test, -idOriginal),
                            nNonTest = nBelow2NonTest,
                            nTest = nBelow2Test,
                            idNonTest = idBelow2NonTest,
                            idTest = idBelow2Test)
mcoKernelRuleBelow2$setClassifier('kernelRule')
# hyperparChoice = 1:3
tic()
mcoKernelRuleBelow2$cvClassifier(iter = 100, hyperparChoice = 1:100, 
                                 nCore = nCore, trainingPct = 0.6, t = time, metric = 'LpNorm', kernelChoice = 'gaussian')
toc()
mcoKernelRuleBelow2$runOnTestSet(t = time, metric = 'LpNorm', kernelChoice = 'gaussian')

# save(mcoKernelRuleBelow2, file = 'mcoKernelRuleBelow2.RData')
load('mcoKernelRuleBelow2.RData')





# Functional GLM
set.seed(1)
mcoFglmBelow2 = mlFramework$new()
mcoFglmBelow2$setData(dfMeta = dfBelow2Meta,
                      dfAll = select(dfSmoothBelow2, -idOriginal),
                      dfNonTest = select(dfSmoothBelow2NonTest, -idOriginal),
                      dfTest = select(dfSmoothBelow2Test, -idOriginal),
                      nNonTest = nBelow2NonTest,
                      nTest = nBelow2Test,
                      idNonTest = idBelow2NonTest,
                      idTest = idBelow2Test)
mcoFglmBelow2$setClassifier('fglm')
tic()
# Not sure why when zeroMeanBool = FALSE, the recovered function oscillate a lot => use zeroMeanBool = TRUE fow now
mcoFglmBelow2$trainClassifier(trainingPct = 0.6, t = time, proportion = 0.99, expansion = 'kl', zeroMeanBool = TRUE)
toc()
mcoFglmBelow2$runOnTestSet(t = time)

# save(mcoFglmBelow2, file = 'mcoFglmBelow2.RData')
load('mcoFglmBelow2.RData')



# Functional SVM
set.seed(1)
mcoFSvmBelow2 = mlFramework$new()
mcoFSvmBelow2$setData(dfMeta = dfBelow2Meta,
                      dfAll = select(dfSmoothBelow2, -idOriginal),
                      dfNonTest = select(dfSmoothBelow2NonTest, -idOriginal),
                      dfTest = select(dfSmoothBelow2Test, -idOriginal),
                      nNonTest = nBelow2NonTest,
                      nTest = nBelow2Test,
                      idNonTest = idBelow2NonTest,
                      idTest = idBelow2Test)
mcoFSvmBelow2$setClassifier('fSvm')
tic()
# Not sure why when zeroMeanBool = FALSE, the recovered function oscillate a lot => use zeroMeanBool = TRUE fow now
# hyperparChoice: gamma = c(0.2, 0.3, 0.5, 0.9), cost not significant here
svmParChoice = list('gamma' = c(seq(0.1, 0.9, 0.1), 1:10),
                    'cost' = 1:5)
mcoFSvmBelow2$cvClassifier(iter = 100, hyperparChoice = svmParChoice, nCore = nCore, trainingPct = 0.6,
                           t = time, proportion = 0.99, expansion = 'kl', zeroMeanBool = TRUE, kernelChoice = 'radial')
toc()
mcoFSvmBelow2$runOnTestSet(t = time)

# save(mcoFSvmBelow2, file = 'mcoFSvmBelow2.RData')
load('mcoFSvmBelow2.RData')





# Collect results from all classification methods
accuracyValidation = c(mcoKnnBelow2$accuracyValidation, 
                       mcoFnweBelow2$accuracyValidation, 
                       mcoKernelRuleBelow2$accuracyValidation, 
                       mcoFglmBelow2$accuracyValidation, 
                       mcoFSvmBelow2$accuracyValidation)
accuracyPrediction = c(mcoKnnBelow2$accuracyPrediction, 
                       mcoFnweBelow2$accuracyPrediction, 
                       mcoKernelRuleBelow2$accuracyPrediction, 
                       mcoFglmBelow2$accuracyPrediction, 
                       mcoFSvmBelow2$accuracyPrediction)
classificationMethods = c('fKNN', 'fNWE', 'fKR', 'fGLM', 'fSVM')
performanceSmoothedBelow2 = data.frame('methods' = classificationMethods, 
                                       accuracyValidation = accuracyValidation, 
                                       accuracyTest = accuracyPrediction)

# Convert to LaTeX table
# library(xtable)
# xtable(performanceSmoothedBelow2)
