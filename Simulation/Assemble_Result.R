#########################################################################################################################################
#
#                                                       Author: Min Sun
#
#########################################################################################################################################

# Script to Assemble all results
# Needs to run all analysis beforehands and set your working directory before loading and .RData
classificationMethods = c('fKNN', 'fNWE', 'fKR', 'fGLM', 'fSVM')

# OU processes
OUDifMu = c(OUDifMuKnn$accuracyPrediction, 
                       OUDifMuFnwe$accuracyPrediction, 
                       OUDifMuKernelRule$accuracyPrediction, 
                       OUDifMuFglm$accuracyPrediction, 
                       OUDifMuFSvm$accuracyPrediction)


OUDifSigma = c(OUDifSigmaKnn$accuracyPrediction,
                       OUDifSigmaFnwe$accuracyPrediction,
                       OUDifSigmaKernelRule$accuracyPrediction,
                       OUDifSigmaFglm$accuracyPrediction,
                       OUDifSigmaFSvm$accuracyPrediction)

# Hawkes Processes
load('HawkesKnn.RData')
load('HawkesFnwe.RData')
load('HawkesKernelRule.RData')
load('HawkesFglm.RData')
load('HawkesFSvm.RData')
HawkesL2 = c(HawkesKnn$accuracyPrediction,
                       HawkesFnwe$accuracyPrediction,
                       HawkesKernelRule$accuracyPrediction,
                       HawkesFglm$accuracyPrediction,
                       HawkesFSvm$accuracyPrediction)


load('HawkesSupNormKnn.RData')
load('HawkesSupNormFnwe.RData')
load('HawkesSupNormKernelRule.RData')
HawkesSup = c(HawkesKnn$accuracyPrediction,
             HawkesFnwe$accuracyPrediction,
             HawkesKernelRule$accuracyPrediction,
             HawkesFglm$accuracyPrediction,
             HawkesFSvm$accuracyPrediction)




load('HawkesDerivKnn.RData')
load('HawkesDerivFnwe.RData')
load('HawkesDerivKernelRule.RData')
load('HawkesDerivFglm.RData')
load('HawkesDerivFSvm.RData')
HawkesDerivL2 = c(HawkesDerivKnn$accuracyPrediction,
                       HawkesDerivFnwe$accuracyPrediction,
                       HawkesDerivKernelRule$accuracyPrediction,
                       HawkesDerivFglm$accuracyPrediction,
                       HawkesDerivFSvm$accuracyPrediction)



load('HawkesDerivSupNormKnn.RData')
load('HawkesDerivSupNormFnwe.RData')
load('HawkesDerivSupNormKernelRule.RData')
HawkesDerivSup = c(HawkesDerivKnn$accuracyPrediction,
                  HawkesDerivFnwe$accuracyPrediction,
                  HawkesDerivKernelRule$accuracyPrediction,
                  HawkesDerivFglm$accuracyPrediction,
                  HawkesDerivFSvm$accuracyPrediction)


# MCO data analysis
mcoUnSmooth = c(mcoUnSmoothKnn$accuracyPrediction, 
                       mcoUnSmoothFnwe$accuracyPrediction, 
                       mcoUnSmoothKernelRule$accuracyPrediction, 
                       mcoUnSmoothFglm$accuracyPrediction, 
                       mcoUnSmoothFSvm$accuracyPrediction)

mcoSmooth = c(mcoKnn$accuracyPrediction, 
                       mcoFnwe$accuracyPrediction, 
                       mcoKernelRule$accuracyPrediction, 
                       mcoFglm$accuracyPrediction, 
                       mcoFSvm$accuracyPrediction)

mcoSmoothBelow2 = c(mcoKnnBelow2$accuracyPrediction, 
                       mcoFnweBelow2$accuracyPrediction, 
                       mcoKernelRuleBelow2$accuracyPrediction, 
                       mcoFglmBelow2$accuracyPrediction, 
                       mcoFSvmBelow2$accuracyPrediction)


# Assemble all results to a dataframe
dfPerformance = data.frame(OUDifMu, 
                           OUDifSigma, 
                           HawkesL2, 
                           HawkesSup, 
                           HawkesDerivL2, 
                           HawkesDerivSup, 
                           mcoSmooth, 
                           mcoUnSmooth, 
                           mcoSmoothBelow2)
dfPerformance = t(dfPerformance)
colnames(dfPerformance) = classificationMethods

# Convert to LaTeX table
# library(xtable)
# xtable(dfPerformance)
