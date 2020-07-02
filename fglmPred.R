# Function to predict new data using an existing functional GLM model
# Input:
# - xNew: a matrix of new data, each row represents a new observation
# - t: domain of xNew(t)
# - fglmModel: must be a fglm object, i.e. z where z = fgml(...)
fglmPred = function(xNew, t, fglmModel, ...) {
  beta0 = fglmModel$intercept
  beta = fglmModel$beta
  modelFamily = fglmModel$glm$family
  
  xBeta = as.matrix(xNew) %*% diag(beta, length(beta), length(beta))
  xBetaInnerProduct = apply(xBeta, 1, L2InnerProduct, t = t)
  rhs = beta0 + xBetaInnerProduct
  yPred = modelFamily$linkinv(rhs)
  return(yPred)
}



# 
# 
# tic()
# myGlm = fglm(x = select(dfSmoothNonTest, -idOriginal, -id, -label), 
#              y = dfSmoothNonTest$label, t = time, proportion = 0.999)
# toc()
# 
# tic()
# pred = fglmPred(xNew = select(dfSmoothNonTest, -idOriginal, -id, -label), 
#                 t = time, fglmModel = myGlm)
# toc()