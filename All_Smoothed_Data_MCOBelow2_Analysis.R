options(stringsAsFactors = FALSE)

setwd('D:/Academics/UNSW/Thesis/R/MCO/')

source('Data_Preparation.R')


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
mcoFSvmBelow2$setWd('D:/Academics/UNSW/Thesis/R/MCO/')
mcoFSvmBelow2$setClassifier('fSvm')
tic()
# Not sure why when zeroMeanBool = FALSE, the recovered function oscillate a lot => use zeroMeanBool = TRUE fow now
svmParChoice = list('gamma' = 1:5, 
                    'cost' = 1:20)
mcoFSvmBelow2$cvClassifier(iter = 10, hyperparChoice = svmParChoice, nCore = 5, trainingPct = 0.6, 
                     t = time, proportion = 0.99, expansion = 'kl', zeroMeanBool = TRUE, kernelChoice = 'radial')
toc()
mcoFSvmBelow2$runOnTestSet(t = time)
