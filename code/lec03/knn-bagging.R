## load libraries
library(kknn) ## knn library
library(MASS) ## a library of example datasets
attach(Boston)

n = dim(Boston)[1]

ddf = data.frame(lstat,medv)

thek=5
B=500
n=nrow(ddf)
nn = rep(0,B)
fmat=matrix(0,n,B)
set.seed(1)

oo = order(lstat)

par(mfrow=c(3,3)) 
for(i in 1:B) {
   if((i%%100)==0) cat('i: ',i,'\n')
   ii = sample(1:n,n,replace=TRUE)
   nn[i] = length(unique(ii))
   near = kknn(medv~lstat, ddf[ii,], ddf, k=thek, kernel='rectangular')
   fmat[,i]=near$fitted
   if (i <= 9){
     plot(lstat, medv)
     lines(lstat[oo],fmat[oo,i],col="red")
     # readline()
   }
}

#--------------------------------------------------
efit = apply(fmat,1,mean)
par(mfrow=c(1,1)) 
plot(lstat,medv)
points(lstat,efit,col='red')

# plot 100 different fits
for(i in 1:100) {
 lines(lstat[oo],fmat[oo,i],col=i)
}

# plot the final predictor
lines(lstat[oo],efit[oo],col='red',lwd=4)
 
