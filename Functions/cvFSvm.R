#########################################################################################################################################
#
#                                                       Author: Min Sun
#
#########################################################################################################################################

# Function to do cross validation for functional support vector machine (fSVM)
# Details: Section 4.3.1
# Input:
#   - x [dataframe]: a dataframe of observed functional data. Column = reading at each t. Row = each observation
#   - y [array]: an array of binary variable {0, 1}, e.g. c(0, 0, 0, 1, 1)
#   - t [array]: an array of domain of x, which is usually time. Must have the same dimension as dim(x)[2]
#   - proportion [double]: score of fPCA, which affects how many basis(eigenfunctions) to be preserved. Usually a high score is needed to 
#                          fully recover the original function
#   - expansion [char]: type of basis expansion. Currently only supports 'kl', which means Karhunen-Loeve
#   - zeroMeanBool [Bool]: a boolean variable indicating whether x has been centred (i.e. adjusted by the mean function)
#   - kernelChoice [char]: kernel of svm. Must be one of {'linear'. 'polynomial', 'radial', 'sigmoid'}
#   - hyperparChoice [list]: hyperparameters to be cross validated. Must have the following form for example
#                            svmParChoice = list('gamma' = 1:5, 'cost' = 1:5)
#   - iter [int]: number of cross validation iteration
#   - nCore [int]: number of cores to be used for parallel computing
#   - nNonTest [int]: number of observations in non test set
#   - nAll [int]: number of total observations
#   - trainingPct [double]: training percentage, must be in (0, 1)
# Output: [list]
#   - model [svm] : the best svm model chosen by cross validation
#   - accuracyValidationAve [dataframe] : a dataframe stores performance of svm for each choice of hyperparChoice 
#   - accuracyValidation [double]: classification accuracy on the validation set for the best svm model
#   - zeroMeanBool [bool] : same as input zeroMeanBool
#   - basisType [char] : same asexpansion
#   - meanProcess [double] : mean function of x 
#   - basis [dataframe] : basis vectors for specific choice of 'expansion'
#   - innerProductOfX [dataframe] : coordinates of 'basis' to x, which is the inner products of x and 'basis'
#   - xApprox [datafrmae] : approximation of x using truncated basis expansion,
#   - modelPar [list] : parameters for the best svm model chosen 
#   - proportion [double] : same as proportion



                         
cvFSvm = function(x, y, t, proportion = 0.99, expansion = 'kl', zeroMeanBool, kernelChoice, hyperparChoice, iter, nCore, 
                 nNonTest, nAll, trainingPct, ...){
  # Create parallel cluster
  cl = makeCluster(nCore, outfile="")
  registerDoSNOW(cl)
  
  gamma = hyperparChoice[['gamma']]
  cost = hyperparChoice[['cost']]
  nNonTest = dim(x)[1]
  
  pb <- txtProgressBar(min = 0, max = length(gamma), style = 3)
  progress <- function(n) setTxtProgressBar(pb, n)
  opts <- list(progress = progress)
  
  switch(expansion, 
         # Perform Karhunen Loeve expansion on all observations, i.e. to each row of x
         kl = {
           klOut = multipleKarhunenLoeve(x = x, t = t, proportion = proportion, zeroMeanBool = zeroMeanBool)
           innerProduct = klOut$innerProduct
           basis = klOut$basis
           m = klOut$no_eigen
           xApprox = klOut$xApprox
         })
  
  # Store mean function as a class attribute (which may be needed for prediction)
  ave = colMeans(x, na.rm = TRUE)
  meanProcessTraining = ave
  
  # Assemble matrix to run svm
  # Covariate matrix is now the matrix innerProduct
  df = cbind(yFactor = as.factor(y), data.frame(innerProduct))
  
  # Fine-tune parameters for SVM using inner products (from Karhunen Loeve expansion) as input features
  svmTune = foreach(i = 1:length(gamma),  .combine = 'rbind', .options.snow = opts, .packages = c('R6', 'dplyr', 'e1071'),
                    .export = c('tune', 'svm')) %dopar% {
                      svmFit = tune(svm, yFactor ~ ., data = df, kernel = kernelChoice, ranges = list('gamma' = gamma[i], 'cost' = cost),
                                    tunecontrol = tune.control(nrepeat = iter, sampling = "cross",
                                                               cross = round((nNonTest/nAll)/(nNonTest/nAll - trainingPct))))
                      out = svmFit$performances
                      return(out)
                    }
  stopCluster(cl)
  
  # Rearrange results from cross validation
  svmBestModel = filter(svmTune, error == min(error))[1, ]
  hyperpar = list('gamma' = svmBestModel[, 'gamma'],
                       'cost' = svmBestModel[, 'cost'])
  accuracyValidation = 1 - svmBestModel[, 'error']
  # Save best model
  model = svm(yFactor ~ ., data = df, kernel = kernelChoice, gamma = hyperpar$gamma, cost = hyperpar$cost)
  
  out = list('model' = model,
             'accuracyValidationAve' = mutate(svmTune, accuracy = 1 - error), 
             'accuracyValidation' = 1 - svmBestModel[, 'error'], 
             'zeroMeanBool' = zeroMeanBool, 
             'basisType' = expansion, 
             'meanProcess' = ave, 
             'basis' = basis,
             'innerProductOfX' = innerProduct,
             'xApprox' = xApprox,
             'modelPar' = hyperpar, 
             'proportion' = proportion)
  return(out)
}



# Example
# library(tidyverse)
# library(parallel)
# library(doSNOW)
# library(e1071)
# n = 10
# df = data.frame(matrix(rnorm(100, 0, 1), nrow = n))
# y = sample(0:1, n, replace = TRUE)
# t = 1:dim(df)[2]
# svmParChoice = list('gamma' = 1:5,
#                    'cost' = 1:5)
# out = cvFSvm(x =  df,
#       y = y,
#       t = t, proportion = 0.99, expansion = 'kl', zeroMeanBool = TRUE, kernelChoice = 'radial', hyperparChoice = svmParChoice,
#       iter = 10, nCore = 5,
#       nNonTest = 10, nAll = 12, trainingPct = 0.6)
