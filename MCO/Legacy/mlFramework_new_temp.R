# library(R6)

mlFramework = R6Class(
  'mlFramework', 
  public = list(
    # Class attributes
    # Working directory
    wd = NA, 
    # dfMeta to contain id and other attributes to each id
    dfMeta = NA, 
    
    #dfNonTest, dfTest, dfAll: contain 1st column = id, 2nd column = label (response), then all other columns as features
    dfNonTest = NA, 
    dfTraining = NA, 
    dfTest = NA, 
    dfAll = NA, 
    
    nNonTest = NA, 
    nTraining = NA, 
    nTest = NA, 
    nAll = NA, 
    
    idNonTest = NA, 
    idTraining = NA, 
    idTest = NA, 
    idAll = NA, 
    
    classifier = NA,
    model = NA, 
    par = NA, 
    
    meanProcessTraining = NA, 
    basisType = NA, 
    # A boolean variable keeping record of whether observed data x are zero mean processes
    zeroMeanBool = NA, 
    
    # Hyperparameter to be fine-tuned in cross validation
    hyperpar = NA,
    hyperParChoice = NA, 
    accuracyValidation = NA, 
    accuracyPrediction = NA, 
    
    # Class methods
    # Set working directory
    setWd = function(wd) {
      self$wd = wd
    }, 
    
    setData = function(dfMeta, dfAll, 
                       dfNonTest, dfTest, 
                       nNonTest, nTest, 
                       idNonTest, idTest) {
      self$dfMeta = dfMeta
      self$dfAll = dfAll
      
      self$dfNonTest = dfNonTest
      self$dfTest = dfTest
      
      self$nNonTest = nNonTest
      self$nTest = nTest
      self$nAll = nNonTest + nTest
      
      self$idNonTest = idNonTest
      self$idTest = idTest
      self$idAll = c(idNonTest, idTest)
    }, 
    
    # Classifier must be a function
    setClassifier = function(classifier) {
      self$classifier = classifier
    }, 
    
    trainClassifier = function(trainingPct = 0.6, ...) {
      dfNonTest = self$dfNonTest
      dfTest = self$dfTest
      nAll = self$nAll
      classifier = self$classifier
      
      idTraining = sort(sample(dfNonTest$id, nAll * trainingPct))
      idValidation = dfNonTest$id[!dfNonTest$id %in% idTraining]
      dfTraining = dplyr::filter(dfNonTest, id %in% idTraining)
      dfValidation = dplyr::filter(dfNonTest, id %in% idValidation)
      
      self$dfTraining = dfTraining
      self$idTraining = idTraining
      self$nTraining = length(idTraining)
      
      switch(classifier, 
             fglm = {
               # Use dfNonTest for fGLM because we don't need to do cross validation for this method
               fglmModel = fglm(x = dplyr::select(dfNonTest, -label, -id),
                                y = dfNonTest$label,
                                ...)
               self$model = fglmModel
               self$par = fglmModel$modelPar
               self$meanProcessTraining = fglmModel$meanProcess
               self$basisType = fglmModel$basisType
               self$zeroMeanBool = fglmModel$zeroMeanBool
             })
    }, 
    
    # Cross validation
    cvClassifier = function(iter = 100, hyperparChoice = 1, nCore = 2, trainingPct = 0.6, expansion = 'kl', kernelChoice = 'radial', ...) {
      dfNonTest = self$dfNonTest
      dfTest = self$dfTest
      idNonTest = self$idNonTest
      nAll = self$nAll
      nNonTest = self$nNonTest
      classifier = self$classifier
      model = self$model
      par = self$par
      
      # Create parallel cluster
      cl = makeCluster(nCore, outfile="")
      registerDoSNOW(cl)
      
      
      # This group of classifier has more than 1 hyperparameter to be tuned
      if (class(hyperparChoice) == 'data.frame' | class(hyperparChoice) == 'list') {
        switch(classifier, 
               # Function SVM
               fSvm = {
                 gamma = hyperparChoice[['gamma']]
                 cost = hyperparChoice[['cost']]
                 
                 pb <- txtProgressBar(min = 0, max = length(gamma), style = 3)
                 progress <- function(n) setTxtProgressBar(pb, n)
                 opts <- list(progress = progress)
                 
                 switch(expansion, 
                        # Perform Karhunen Loeve expansion on all observations, i.e. to each row of x
                        kl = {
                          klOut = multipleKarhunenLoeve(x = dplyr::select(dfNonTest, -label, -id), ...)
                          innerProduct = klOut$innerProduct
                          basis = klOut$basis
                          m = klOut$no_eigen
                          xApprox = klOut$xApprox
                        })
                 
                 # Store mean function as a class attribute (which may be needed for prediction)
                 ave = colMeans(dplyr::select(dfNonTest, -label, -id), na.rm = TRUE)
                 self$meanProcessTraining = ave
                 
                 # Assemble matrix to run svm
                 # Covariate matrix is now the matrix innerProduct
                 df = cbind(yFactor = as.factor(dfNonTest$label), data.frame(innerProduct))
                 
                 # Fine-tune parameters for SVM using inner products (from Karhunen Loeve expansion) as input features
                 svmTune = foreach(i = 1:length(gamma),  .combine = 'rbind', .options.snow = opts, .packages = c('R6', 'dplyr', 'e1071'),
                                   .export = c('tune', 'svm')) %dopar% {
                     svmFit = tune(svm, yFactor ~ ., data = df, kernel = kernelChoice, ranges = list('gamma' = gamma[i], 'cost' = cost),
                                   tunecontrol = tune.control(nrepeat = iter, sampling = "cross",
                                                              cross = round((nNonTest/nAll)/(nNonTest/nAll - trainingPct))))
                     out = svmFit$performances
                     return(svmFit$performances)
                   }
                 
                 # Rearrange results from cross validation
                 svmBestModel = filter(svmTune, error == min(error))[1, ]
                 self$hyperpar = list('gamma' = svmBestModel[, 'gamma'],
                                      'cost' = svmBestModel[, 'cost'])
                 self$accuracyValidation = 1 - svmBestModel[, 'error']
                 # Save best model
                 self$model = svm(yFactor ~ ., data = df, kernel = kernelChoice, gamma = self$hyperpar$gamma, cost = self$hyperpar$cost)
               })
      } else{
        # This group of classifier has only one hyperparameter to be tuned
        # Monitor bar for parallel computing
        pb <- txtProgressBar(min = 0, max = iter, style = 3)
        progress <- function(n) setTxtProgressBar(pb, n)
        opts <- list(progress = progress)
        
        # Start parallel computing
        accuracyAve = rep(NA, length(hyperparChoice))
        
        for (k in 1:length(hyperparChoice)) {
          accuracyWithinIter = foreach(i = 1:iter, .combine = 'c',  .options.snow = opts, 
                                       .export = c('LpNorm', 'knn', 'fnwe', 'kernelRule'), .packages = c('R6', 'dplyr')) %dopar% {
             idTraining = sort(sample(dfNonTest$id, nAll * trainingPct))
             idValidation = dfNonTest$id[!dfNonTest$id %in% idTraining]
             dfTraining = dplyr::filter(dfNonTest, id %in% idTraining)
             dfValidation = dplyr::filter(dfNonTest, id %in% idValidation)
             switch(classifier, 
                    knn = {
                      out = knn(x = dplyr::select(dfTraining, -label, -id),
                                y = dfTraining$label,
                                xNew = dplyr::select(dfValidation, -label, -id),
                                k = hyperparChoice[k],
                                ...)
                      # Sys.sleep(0.1)
                      # out = list()
                      # out[['Label Prediction']] = 1
                    }, 
                    fnwe = {
                      out = fnwe(x = dplyr::select(dfTraining, -label, -id),
                                 y = dfTraining$label,
                                 xNew = dplyr::select(dfValidation, -label, -id),
                                 h = hyperparChoice[k],
                                 kernelChoice = kernelChoice, 
                                 ...)
                      # Sys.sleep(0.1)
                      # out = list()
                      # out[['Label Prediction']] = 1
                    }, 
                    kernelRule = {
                      out = kernelRule(x = dplyr::select(dfTraining, -label, -id),
                                       y = dfTraining$label,
                                       xNew = dplyr::select(dfValidation, -label, -id),
                                       h = hyperparChoice[k], 
                                       kernelChoice = kernelChoice, 
                                       ...)
                    }, 
                    fglm = {
                      stop('Do you really need to do cross validation on a GLM? There is no hyperparameter to tune!')
                    })
             predLabel = as.integer(out[['Label Prediction']])
             validationLabel = dfValidation$label
             out = length(which(predLabel == validationLabel))/length(idValidation)
             return(out)
           }
          accuracyAve[k] = mean(accuracyWithinIter)
          # Log progress of the outer loop to a local file
          sink(paste(self$wd, "hyperparChoice_Round.txt", sep = ''))
          cat(sprintf('Hyperparameter: %i/%i', k, length(hyperparChoice)))
          sink()
        }
        
        accuracy = max(accuracyAve)
        hyperpar = hyperparChoice[which(accuracyAve == max(accuracyAve))]
        self$accuracyValidation = accuracy
        self$hyperpar = hyperpar
      }
      
      stopCluster(cl)
    }, 
    
    runOnTestSet = function(...) {
      dfNonTest = self$dfNonTest
      dfTest = self$dfTest
      idTest = self$idTest
      classifier = self$classifier
      hyperpar = self$hyperpar
      model = self$model
      par = self$par
      basisType = self$basisType
      ave = self$meanProcessTraining
      zeroMeanBool = self$zeroMeanBool
      
      switch(classifier, 
             knn = {
               out = knn(x = dplyr::select(dfNonTest, -label, -id),
                         y = dfNonTest$label,
                         xNew = dplyr::select(dfTest, -label, -id),
                         k = hyperpar,
                         ...)
               # Sys.sleep(0.1)
               # out = list()
               # out[['Label Prediction']] = 1
             }, 
             fnwe = {
               out = fnwe(x = dplyr::select(dfNonTest, -label, -id),
                          y = dfNonTest$label,
                          xNew = dplyr::select(dfTest, -label, -id),
                          h = hyperpar,
                          ...)
             }, 
             kernelRule = {
               out = kernelRule(x = dplyr::select(dfNonTest, -label, -id),
                                y = dfNonTest$label,
                                xNew = dplyr::select(dfTest, -label, -id),
                                h = hyperpar,
                                ...)
             }, 
             fglm = {
               fglmModel = model
               if (zeroMeanBool == TRUE) {
                 yPred = fglmPred(xNew = dplyr::select(dfTest, -label, -id),
                                  fglmModel = fglmModel, ...)
               } else {
                 yPred = fglmPred(xNew = dplyr::select(dfTest, -label, -id) + ave,
                                  fglmModel = fglmModel, ...)
               }
               yLabel = ifelse(yPred >= 0.5, 1, 0)
               yProbMatrix = round(matrix(c(1 - yPred, yPred), ncol = 2, byrow = FALSE), 4)
               colnames(yProbMatrix) = c('0', '1')
               out = list('Label Prediction' = as.character(yLabel),
                          'Probability' = yProbMatrix)
             }, 
             fSvm = {
               switch(expansion, 
                      # Perform Karhunen Loeve expansion on all observations, i.e. to each row of x
                      kl = {
                        klOut = multipleKarhunenLoeve(x = dplyr::select(dfTest, -label, -id), 
                                                      t = t, proportion = proportion, zeroMeanBool = zeroMeanBool)
                        innerProduct = klOut$innerProduct
                        basis = klOut$basis
                        m = klOut$no_eigen
                        xApprox = klOut$xApprox
                      })
               
               # Assemble matrix to run svm
               # Covariate matrix is now the matrix innerProduct
               df = cbind(yFactor = as.factor(dfTest$label), data.frame(innerProduct))
               
               yLabel = predict(model, select(df, -yFactor))
               out = list('Label Prediction' = as.character(yLabel), 
                          'Probability' = NULL)
             })
      predLabel = as.integer(out[['Label Prediction']])
      testLabel = dfTest$label
      self$accuracyPrediction = length(which(predLabel == testLabel))/length(idTest)
    }
  )
)
