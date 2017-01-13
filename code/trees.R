library(rpart)      # package for trees
# install.packages("rpart.plot")
library(rpart.plot) # package that enhances plotting capabilities for rpart
library(MASS)  # contains boston housing data


# fit a tree to boston data just using lstat.
# first get a big tree by appropriately setting rpart.control
#
# see help by typing ?rpart.control for more details
# you will notice that you can set
#   - minsplit - which sets the minimum number of observations needed in a node to consider a plit
#   - cp - splits are considered only if the fit is improved by amount controled by cp
#
#
temp = rpart(medv~lstat, data=Boston, 
             control=rpart.control(minsplit=5,  # we want to have at least 5 observations in a node
                                   cp=0.001,    # we will consider even small improvements in a fit 
                                   xval=0)      # do not run cross-validation now
             )
rpart.plot(temp)
#if the tree is too small, make cp smaller!!

# then prune it down to one with 7 leaves
# again, we control the size of the tree through cp 
# which is not the most intuitive of measures
boston.tree = prune(temp, cp=0.01)
rpart.plot(boston.tree)

## plot data with fit
boston.fit = predict(boston.tree) #get training fitted values
plot(Boston$lstat, Boston$medv, cex=.5, pch=16) #plot data
oo=order(Boston$lstat)
lines(Boston$lstat[oo],boston.fit[oo],col="red",lwd=3) #step function fit

# predict at lstat = 15 and 25.
preddf = data.frame(lstat=c(15,25))
yhat = predict(boston.tree,preddf)
points(preddf$lstat,yhat,col="blue",pch="*",cex=3)

###################################
## Fit the tree using lstat and dis
###################################

# to simplify things, we will create a new data frame
# consisting of columns dis, lstat and medv

df2=Boston[,c(8,13,14)] # pick off dis,lstat,medv
print(names(df2))

# create a big tree
temp = rpart(medv~., data=df2, 
             control=rpart.control(minsplit=5,  
                                   cp=0.001,
                                   xval=0)   
)
rpart.plot(temp)

# then prune it down to one with 7 leaves
boston.tree = prune(temp, cp=0.017)
rpart.plot(boston.tree)

## create a perspective plot
pv=seq(from=.01,to=.99,by=.05)                # quantiles to use in making 2D grid
x1q = quantile(df2$lstat, probs=pv)
x2q = quantile(df2$dis, probs=pv)
xx = expand.grid(x1q,x2q)                     # matrix with two columns using all combinations of x1q and x2q
dfpred = data.frame(dis=xx[,2], lstat=xx[,1])
lmedpred = predict(boston.tree, dfpred)

persp(x1q,x2q,matrix(lmedpred,ncol=length(x2q),byrow=T),
      theta=150,xlab='dis',ylab='lstat',zlab='medv',
      zlim=c(min(df2$medv),1.1*max(df2$medv)))


# let’s compare in-sample fits from our two trees with each other and y
boston.fit2 = predict(boston.tree)
fmat = cbind(Boston$medv, boston.fit, boston.fit2)
colnames(fmat)=c("y=medv","treel","treeld")
pairs(fmat)
print(cor(fmat))


###########################################################
## Bostong Housing with p = 4 (nox, rm, ptratio, and lstat)
###########################################################

df4=Boston[,c(5,6,11,13,14)] # pick off variables
print(names(df4))


# create a big tree
temp = rpart(medv~., data=df4, 
             control=rpart.control(minsplit=5,  
                                   cp=0.001,
                                   xval=0)   
)
rpart.plot(temp)

# then prune it down to one with 12 leaves (picked 12 arbitrarily)
boston.tree4 = prune(temp, cp=0.014)
rpart.plot(boston.tree4)

# compare fits
fmat4 = cbind(fmat, predict(boston.tree4))
colnames(fmat4)[4]="tree4"
pairs(fmat4)
print(cor(fmat4))


###########################################################
## Bostong Housing with cross validation
###########################################################



big.tree = rpart(medv~lstat, data=Boston, 
                         control=rpart.control(minsplit=5,  
                                               cp=0.0001,
                                               xval=10)   
)

nbig = length(unique(big.tree$where))
cat('size of big tree: ',nbig,'\n')

# let us look at the cross-validation results
#
# the following prints out a table summarizing the output of cross-validation
# you want to find cp corrsponding to the smallest value of xerror 
(cptable = printcp(big.tree))
(bestcp = cptable[ which.min(cptable[,"xerror"]), "CP" ])   # this is the optimal cp parameter

plotcp(big.tree) # plot results

# show fit from some trees
oo = order(bdf$lstat)
cpvec = c(.0157,bestcp,.004)

par(mfrow=c(3,2))
for(i in 1:3) {
  plot(Boston$lstat,Boston$medv,pch=16,col='blue',cex=.5)
  ptree = prune(big.tree,cp=cpvec[i])
  pfit = predict(ptree)
  lines(Boston$lstat[oo],pfit[oo],col='red',lwd=2)
  title(paste('alpha = ',round(cpvec[i],3)))
  rpart.plot(ptree)
}

par(mfrow=c(1,1))
best.tree = prune(big.tree,cp=bestcp)
rpart.plot(best.tree)

