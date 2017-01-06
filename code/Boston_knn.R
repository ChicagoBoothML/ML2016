#load knn library (need to have installed this with install.packages("kknn"))
library(kknn)

#get Boston data
library(MASS) ## a library of example datasets
#try:
names(Boston)
dim(Boston)
Boston[1:5,]
summary(Boston)
ls()
mean(Boston$lstat)
mean(Boston[,13])

#make the variables in Boston directly accessible
attach(Boston)
ls(pos=1)
ls(pos=2)

#plot the data
plot(lstat,medv,xlab="% lower status",ylab="median value")

#run regression, print summary, add line to plot
lmB = lm(medv~lstat,Boston)
print(summary(lmB))
abline(lmB$coef,col="red",lwd=4) #lwd: line width=4
#try:
names(lmB)
cor(lmB$fitted.values,lmB$residuals) #cor is 0!

#fit knn with k=50
train = data.frame(lstat,medv) #data frame with variables of interest
#test is data frame with x you want f(x) at, sort lstat to make plots nice.
test = data.frame(lstat = sort(lstat))
kf50 = kknn(medv~lstat,train,test,k=50,kernel = "rectangular")

#add knn50 fit to plot
lines(test$lstat,kf50$fitted.values,col="blue",lwd=2)

#add k=200
kf200 = kknn(medv~lstat,train,test,k=200,kernel = "rectangular")
lines(test$lstat,kf200$fitted.values,col="magenta",lwd=2,lty=2) #line type 2

#add legend to plot
legend("topright",legend=c("lin","knn50","knn200"),
           col=c("red","blue","magenta"),lty=c(1,1,2))

#get price prediction at lstat = 30 using k=50
dfp = data.frame(lstat=30)
k50 = kknn(medv~lstat,train,dfp,k=50,kernel = "rectangular")
cat("kNN50: predicted house price at lstat=30 is ",k50$fitted,"\n")
points(30,k50$fitted,pch=4,cex=2,col="black")

#get prediction from linear fit
p30L = predict(lmB,dfp)
cat("Linear: predicted house price at lstat=30 is ",p30L,"\n")
text(30,p30L,"LIN",cex=2,col="black")


###########################
# Out-of-Sample Predictions
###########################

#load libraries and get Boston data
library(kknn) ## knn library
library(MASS) ## a library of example datasets
n = nrow(Boston)
# get in-sample and out-of-sample data frames
df = data.frame(lstat=Boston$lstat,medv=Boston$medv) #simple data frame for conveniance
ntrain=400 #number of observations for training data
set.seed(99) #set seed for random sampling of training data
tr = sample(1:nrow(df),ntrain)
train = df[tr,] #training data
test = df[-tr,] #test data
#loop over values of k, fit on train, predict on test
kvec=2:350; nk=length(kvec)
outMSE = rep(0,nk) #will will put the out-of-sample MSE here
for(i in 1:nk) {
    near = kknn(medv~lstat,train,test,k=kvec[i],kernel = "rectangular")
    MSE = mean((test$medv-near$fitted)^2)
    outMSE[i] = MSE
}

#plot
par(mfrow=c(1,2))
plot(kvec,sqrt(outMSE))
plot(log(1/kvec),sqrt(outMSE))
imin = which.min(outMSE)
cat("best k is ",kvec[imin],"\n")
#fit with all data and best k and plot
test = data.frame(lstat=sort(df$lstat))
near = kknn(medv~lstat,df,test,k=kvec[imin],kernel = "rectangular")
par(mfrow=c(1,1))
plot(df)
lines(test$lstat,near$fitted,col="red",type="b")


############################
# CV
############################

#load libraries and docv.R
library(MASS)
library(kknn)
download.file("https://raw.githubusercontent.com/ChicagoBoothML/HelpR/master/docv.R", "docv.R")
source("docv.R") #this has docvknn used below
#do k-fold cross validation, 5 twice, 10 once
set.seed(99) #always set the seed! 
kv = 2:100 #these are the k values (k as in kNN) we will try
#docvknn(matrix x, vector y,vector of k values, number of folds),
#does cross-validation for training data (x,y).
cv1 = docvknn(matrix(Boston$lstat,ncol=1),Boston$medv,kv,nfold=5)
cv2 = docvknn(matrix(Boston$lstat,ncol=1),Boston$medv,kv,nfold=5)
cv3 = docvknn(matrix(Boston$lstat,ncol=1),Boston$medv,kv,nfold=10)
#docvknn returns error sum of squares, want RMSE
cv1 = sqrt(cv1/length(Boston$medv))
cv2 = sqrt(cv2/length(Boston$medv))
cv3 = sqrt(cv3/length(Boston$medv))

#plot
rgy = range(c(cv1,cv2,cv3))
plot(log(1/kv),cv1,type="l",col="red",ylim=rgy,lwd=2,cex.lab=2.0, xlab="log(1/k)", ylab="RMSE")
lines(log(1/kv),cv2,col="blue",lwd=2)
lines(log(1/kv),cv3,col="green",lwd=2)
legend("topleft",legend=c("5-fold 1","5-fold 2","10 fold"),
       col=c("red","blue","green"),lwd=2,cex=1.5)
#get the min
cv = (cv1+cv2+cv3)/3 #use average
kbest = kv[which.min(cv)]
cat("the best k is: ",kbest,"\n")
#fit kNN with best k and plot the fit.
kfbest = kknn(medv~lstat,Boston,data.frame(lstat=sort(Boston$lstat)),
              k=kbest,kernel = "rectangular")
plot(Boston$lstat,Boston$medv,cex.lab=1.2)
lines(sort(lstat),kfbest$fitted,col="red",lwd=2,cex.lab=2)


################
# x1 and x2 vs y
################

#get the Boston data and needed libraries
library(kknn)
library(MASS)
download.file("https://raw.githubusercontent.com/ChicagoBoothML/HelpR/master/docv.R", "docv.R")
source("docv.R") #this has docvknn used below
#get variables we want
x = cbind(Boston$lstat,Boston$indus)
colnames(x) = c("lstat","indus")
y = Boston$medv
mmsc=function(x) {return((x-min(x))/(max(x)-min(x)))}
xs = apply(x,2,mmsc) #apply scaling function to each column of x

#plot y vs each x
par(mfrow=c(1,2)) #two plot frames
plot(x[,1],y,xlab="lstat",ylab="medv")
plot(x[,2],y,xlab="indus",ylab="medv")
#run cross val once
par(mfrow=c(1,1))
set.seed(99)
kv = 2:20 #k values to try
n = length(y)
cvtemp = docvknn(xs,y,kv,nfold=10)
cvtemp = sqrt(cvtemp/n) #docvknn returns sum of squares
plot(kv,cvtemp)

#run cross val several times
set.seed(99)
cvmean = rep(0,length(kv)) #will keep average rmse here
ndocv = 50 #number of CV splits to try
n=length(y)
cvmat = matrix(0,length(kv),ndocv) #keep results for each split
for(i in 1:ndocv) {
    cvtemp = docvknn(xs,y,kv,nfold=10)
    cvmean = cvmean + cvtemp
    cvmat[,i] = sqrt(cvtemp/n)
}
cvmean = cvmean/ndocv
cvmean = sqrt(cvmean/n)
plot(kv,cvmean,type="n",ylim=range(cvmat),xlab="k",cex.lab=1.5)
for(i in 1:ndocv) lines(kv,cvmat[,i],col=i,lty=3) #plot each result
lines(kv,cvmean,type="b",col="black",lwd=5) #plot average result

#refit using all the data and k=5
ddf = data.frame(y,xs)
near5 = kknn(y~.,ddf,ddf,k=5,kernel = "rectangular")
lmf = lm(y~.,ddf)
fmat = cbind(y,near5$fitted,lmf$fitted)
colnames(fmat)=c("y","kNN5","linear")
pairs(fmat)
print(cor(fmat))

#predict price of house in place with lstat=10, indus=11.
x1=10; x2=11
x1s = (x1-min(x[,1]))/(max(x[,1])-min(x[,1]))
x2s = (x2-min(x[,2]))/(max(x[,2])-min(x[,2]))
near = kknn(y~.,ddf,data.frame(lstat=x1s,indus=x2s),k=5,kernel = "rectangular")
cat("knn predicted value: ",near$fitted,"\n")

#what does a linear model predict?
print(predict(lmf,data.frame(lstat=x1s,indus=x2s)))

#let’s check we did the scaling right
lmtemp = lm(medv~lstat+indus,Boston)
print(predict(lmtemp,data.frame(lstat=10,indus=11)))


