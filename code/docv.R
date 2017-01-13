# DESCRIPTION:
# Lets us run a k-fold cross validation on a k-nearest neighbor analysis. Or in technical terms: “docvknn is for kNN, it calls docv handing in a wrapper of kknn.” (not quite sure what that means).
#  
# Importantly, the function returns data in the Sum of Squares format, so if you want to plot it, you’ll want to correct for this (see “Usage” below)
# 
# 
# USAGE:
# 
# To do 10-fold cross validation on how our dataset Xdata predicts for Ydata, across the range of possible k-nearest neighbors of 1-100:
# 
# Krange = 1:100
# newcv = docvkkn(Xdata, Ydata, Krange, nfold=10)
# 
# Or to see an example from the course notes (For more details on this usage example, see slide 78 of 01_knn.pdf):
# cvtemp = docvknn(xs, y, kv, nfold=10)
#  
# In order to optimally plot this data you will want to get the data out of Sum of Squares format:
# cvtemp = sqrt(cvtemp/(length(y))
#   
# ARGUMENTS:
# x = the variables that are your regressors/covariates (takes dataframe, matrix or vector object)
# y = the variable what we want to predict (takes dataframe, matrix or vector object)
# k = a range of possible values for how many neighbors that we want to test (our “tuning knob”). (It definitely takes a vector, I don't know about anything else)
# nfold = number of folds in your cross validation (usually 5 or 10, but you can plug in whatever you want)
#  
# Optional arguments
# doran = ??? (If omitted from the function when it is called, docvkkn works just find, so probably just leave this one alone)
# verbose = indicates the amount of description included in the output (Defaults to verbose mode if omitted from the function when it is called)
# 

############################################################
############################################################
## Function to do cross validation.
## docv is a general method that takes a prediction function as an argument.
## docvknn is for kNN, it calls docv handing in a wrapper of kknn.
############################################################
############################################################
#--------------------------------------------------
mse=function(y,yhat) {return(sum((y-yhat)^2))}
doknn=function(x,y,xp,k) {
  kdo=k[1]
  train = data.frame(x,y=y)
  test = data.frame(xp); names(test) = names(train)[1:(ncol(train)-1)]
  near  = kknn(y~.,train,test,k=kdo,kernel='rectangular')
  return(near$fitted)
}
#--------------------------------------------------
docv = function(x,y,set,predfun,loss,nfold=10,doran=TRUE,verbose=TRUE,...)
{
  #a little error checking
  if(!(is.matrix(x) | is.data.frame(x))) {cat('error in docv: x is not a matrix or data frame\n'); return(0)}
  if(!(is.vector(y))) {cat('error in docv: y is not a vector\n'); return(0)}
  if(!(length(y)==nrow(x))) {cat('error in docv: length(y) != nrow(x)\n'); return(0)}
  
  nset = nrow(set); n=length(y) #get dimensions
  if(n==nfold) doran=FALSE #no need to shuffle if you are doing them all.
  cat('in docv: nset,n,nfold: ',nset,n,nfold,'\n')
  lossv = rep(0,nset) #return values
  if(doran) {ii = sample(1:n,n); y=y[ii]; x=x[ii,,drop=FALSE]} #shuffle rows
  
  fs = round(n/nfold) # fold size
  for(i in 1:nfold) { #fold loop
    bot=(i-1)*fs+1; top=ifelse(i==nfold,n,i*fs); ii =bot:top
    if(verbose) cat('on fold: ',i,', range: ',bot,':',top,'\n')
    xin = x[-ii,,drop=FALSE]; yin=y[-ii]; xout=x[ii,,drop=FALSE]; yout=y[ii]
    for(k in 1:nset) { #setting loop
      yhat = predfun(xin,yin,xout,set[k,],...)
      lossv[k]=lossv[k]+loss(yout,yhat)
    } 
  } 
  
  return(lossv)
}
#--------------------------------------------------
#cv version for knn
docvknn = function(x,y,k,nfold=10,doran=TRUE,verbose=TRUE) {
  return(docv(x,y,matrix(k,ncol=1),doknn,mse,nfold=nfold,doran=doran,verbose=verbose))
}
