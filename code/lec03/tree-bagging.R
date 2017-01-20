## load library
library(tree)
library(MASS) ## a library of example datasets
attach(Boston)
n = dim(Boston)[1]
ddf = data.frame(x=lstat,y=medv)
oo=order(ddf$x)
ddf = ddf[oo,]

## bagging
B=400
n=nrow(ddf)
nn = rep(0,B)
fmat=matrix(0,n,B)
set.seed(1)

par(mfrow=c(1,2))
for(i in 1:B) {
   if((i%%100)==0) cat('i: ',i,'\n')
   ii = sample(1:n,n,replace=TRUE)
   nn[i] = length(unique(ii))
   bigtree = tree(y~x,ddf[ii,],mindev=.0002)
   #print(length(unique(bigtree$where)))
   temptree = prune.tree(bigtree,best=30)
   #print(length(unique(temptree$where)))
   fmat[,i]=predict(temptree,ddf)

   plot(ddf$x,ddf$y)
   lines(ddf$x,fmat[,i],col=i,lwd=2)
   plot(temptree,type="uniform")
   Sys.sleep(.1)
}


par(mfrow=c(1,1))
plot(ddf$x,ddf$y)
efit = apply(fmat,1,mean)
lines(ddf$x,efit,col='blue',lwd=4)

