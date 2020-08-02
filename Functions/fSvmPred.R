#########################################################################################################################################
#
#                                                       Author: Min Sun
#
#########################################################################################################################################

# Function to predict new data using an existing functional SVM model
# Details: Section 4.3.1
# Input:
#   - model [svm] : a svm model
#   - xNew [dataframe]: a matrix of new data, each row represents a new observation
#   - t [array] : an array of domain of xNew, which is usually time. Must have the same dimension as dim(xNew)[2]
#   - expansion [char]: type of basis expansion. Currently only supports 'kl', which means Karhunen-Loeve
#   - proportion [double]: score of fPCA, which affects how many basis(eigenfunctions) to be preserved. Usually a high score is needed to
#                          fully recover the original function
#   - zeroMeanBool [Bool]: a boolean variable indicating wheter x has been centred (i.e. adjusted by the mean function)
#   - nBasis [int] : how many eigenfunctions to be preserved. It must be equal to the number of basis preserved in the model
#                    We usually use proportion instead (i.e. set nBasis = NA), but if nBasis is given, then proportion will NOT be effective!
# Output: [array]
#   - an array of predicted value. Dimension will be equal to dim(xNew)[1]

fSvmPred = function(model, xNew, t, expansion, proportion, zeroMeanBool, nBasis) {
  switch(expansion, 
         # Perform Karhunen Loeve expansion on all observations, i.e. to each row of xNew
         kl = {
           klOut = multipleKarhunenLoeve(x = xNew, t = t, proportion = proportion, zeroMeanBool = zeroMeanBool, nBasis = nBasis)
           innerProduct = klOut$innerProduct
           basis = klOut$basis
           m = klOut$no_eigen
           xApprox = klOut$xApprox
         })
  
  df = data.frame(innerProduct)
  
  yLabel = predict(model, df)
  return(yLabel)
}



# library(tidyverse)
# library(e1071)
# source('cvFSvm.R')
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
#       nNonTest = n, nAll = n * 1.2, trainingPct = 0.6)
# out2 = fSvmPred(model = out$model, xNew = df, t = t, expansion = 'kl', proportion = 0.99, zeroMeanBool = TRUE, nBasis = dim(out$basis)[2])
