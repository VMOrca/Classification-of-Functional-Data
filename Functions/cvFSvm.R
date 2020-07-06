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





# svmParChoice = list('gamma' = 1:5,
#                     'cost' = 1:5)
# a = cvFSvm(x =  select(dfSmoothNonTest, -idOriginal, -id, -label),
#       y = dfSmoothNonTest$label,
#       t = time, proportion = 0.99, expansion = 'kl', zeroMeanBool = TRUE, kernelChoice = 'radial', hyperparChoice = svmParChoice,
#       iter = 10, nCore = 5,
#       nAll = 90, trainingPct = 0.6)
