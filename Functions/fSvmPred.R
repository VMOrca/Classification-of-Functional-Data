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
