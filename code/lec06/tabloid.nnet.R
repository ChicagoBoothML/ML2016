library(nnet)
source("https://raw.githubusercontent.com/ChicagoBoothML/HelpR/master/lift.R")
source("plot.nnet.R")

### read in tabloid data from web
download.file(
    'https://github.com/ChicagoBoothML/MLClassData/raw/master/Tabloid/Tabloid_test.csv',
    'Tabloid_test.csv')
download.file(
    'https://github.com/ChicagoBoothML/MLClassData/raw/master/Tabloid/Tabloid_train.csv',
    'Tabloid_train.csv')
trainDf = read.csv("Tabloid_train.csv")
testDf = read.csv("Tabloid_test.csv")

trainDf$purchase = as.factor(trainDf$purchase)
testDf$purchase = as.factor(testDf$purchase)
names(trainDf)[1]="y"
names(testDf)[1]="y"

### setup storage for results
phatL = list() #store the test phat for the different methods here

### fit logit
lgfit = glm(y~.,trainDf,family=binomial)
print(summary(lgfit))
phat = predict(lgfit,testDf,type="response")
phatL$logit = matrix(phat, ncol=1) #logit phat

### fit nnet
nnetfit = nnet(y~., data=trainDf, size=10, decay=0.5, maxit=10000)
print(summary(nnetfit))
phat = predict(nnetfit,testDf,type="raw")
phatL$nnet = matrix(phat, ncol=1) #nnet phat

plot(phatL$logit, phatL$nnet)

plot.nnet(nnetfit)

lift.many.plot(phatL, testDf$y)
legend("topleft",legend=names(phatL),col=1:2,lty=rep(1,2))

