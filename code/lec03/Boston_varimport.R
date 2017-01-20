################################################################################
## Variable Importance:
## Fit boosting and random forests and plot variable importance.
################################################################################

library(gbm) #boost package
library(randomForest) 
library(rpart)
library(MASS)

data(Boston)
attach(Boston)

#--------------------------------------------------
#fit boost and plot  variable importance
boostfit = gbm(medv~.,data=Boston,distribution='gaussian',
                interaction.depth=2,n.trees=100,shrinkage=.2)

par(mfrow=c(1,1))
p=ncol(Boston)-1
vsum=summary(boostfit,plotit=FALSE) #this will have the variable importance info
print(vsum)
plot(vsum)


#--------------------------------------------------
#fit random forest and plot variable importance

rffit = randomForest(medv~.,data=Boston,mtry=3,ntree=500)
varImpPlot(rffit)

#--------------------------------------------------
#fit a single tree and plot variable importance
#fit a big tree using rpart.control
big.tree = rpart(medv~.,method="anova",data=Boston,
                        control=rpart.control(minsplit=5,cp=.0005))
nbig = length(unique(big.tree$where))
cat('size of big tree: ',nbig,'\n')

#plotcp(big.tree)
bestcp=big.tree$cptable[which.min(big.tree$cptable[,"xerror"]),"CP"]
best.tree = prune(big.tree,cp=bestcp)
tvimp = best.tree$variable.importance

plot(tvimp,axes=F,pch=16,col='red',ylab="variable importance, rpart",cex=2,cex.lab=1.5)
axis(1,labels=names(tvimp),at=1:length(tvimp))
axis(2)



