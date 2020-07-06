options(stringsAsFactors = FALSE)

setwd('D:/Academics/UNSW/Thesis/R/MCO/')

source('Data_Preparation.R')



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
mcoKnn$cvClassifier(iter = 100, hyperparChoice = 1:round((n/2)), nCore = 10, trainingPct = 0.6, t = time, metric = LpNorm) 
toc()
tic()
mcoKnn$cvClassifier(iter = 5, hyperparChoice = 1:2, nCore = 5, trainingPct = 0.6, t = time, metric = LpNorm) 
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

save(mcoFSvm, file = 'mcoFSvm.R')