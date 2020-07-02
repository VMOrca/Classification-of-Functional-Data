library(R6)

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
    dfTest = NA, 
    dfAll = NA, 
    
    nNonTest = NA, 
    nTest = NA, 
    nAll = NA, 
    
    idNonTest = NA, 
    idTest = NA, 
    idAll = NA, 
    
    classifier = NA,
    model = NA, 
    par = NA, 
    
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
      
      switch(classifier, 
             fglm = {
               fglmModel = fglm(x = dplyr::select(dfTraining, -label, -id),
                                y = dfTraining$label,
                                ...)
               self$model = fglmModel
               self$par = fglmModel$modelPar
             })
    }, 
    
    # Cross validation
    cvClassifier = function(iter = 100, hyperparChoice = 1, nCore = 2, trainingPct = 0.6, ...) {
      dfNonTest = self$dfNonTest
      dfTest = self$dfTest
      nAll = self$nAll
      classifier = self$classifier
      model = self$model
      par = self$par
      
      # Create parallel cluster
      cl = makeCluster(nCore, outfile="")
      registerDoSNOW(cl)
      
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
                                                                 ...)
                                              }, 
                                              fglm = {
                                                stop('Do you really need to do cross validation on a GLM? There is no hyperparameter to tune!')
                                                # I still put code here as one may extend fGLM to penalised fGLM, for which we do need cross validation
                                                fglmModel = model
                                                yPred = fglmPred(xNew = dplyr::select(dfValidation, -label, -id),
                                                                 fglmModel = fglmModel, ...)
                                                yLabel = ifelse(yPred >= 0.5, 1, 0)
                                                yProbMatrix = round(matrix(c(1 - yPred, yPred), ncol = 2, byrow = FALSE), 4)
                                                colnames(yProbMatrix) = c('0', '1')
                                                out = list('Label Prediction' = as.character(yLabel),
                                                           'Probability' = yProbMatrix)
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
      stopCluster(cl)
      
      accuracy = max(accuracyAve)
      hyperpar = hyperparChoice[which(accuracyAve == max(accuracyAve))]
      self$accuracyValidation = accuracy
      self$hyperpar = hyperpar
    }, 
    
    runOnTestSet = function(...) {
      dfNonTest = self$dfNonTest
      dfTest = self$dfTest
      idTest = self$idTest
      classifier = self$classifier
      hyperpar = self$hyperpar
      model = self$model
      par = self$par
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
               yPred = fglmPred(xNew = dplyr::select(dfTest, -label, -id),
                                fglmModel = fglmModel, ...)
               yLabel = ifelse(yPred >= 0.5, 1, 0)
               yProbMatrix = round(matrix(c(1 - yPred, yPred), ncol = 2, byrow = FALSE), 4)
               colnames(yProbMatrix) = c('0', '1')
               out = list('Label Prediction' = as.character(yLabel),
                          'Probability' = yProbMatrix)
             })
      predLabel = as.integer(out[['Label Prediction']])
      testLabel = dfTest$label
      self$accuracyPrediction = length(which(predLabel == testLabel))/length(idTest)
    }
  )
)
