# install.packages("FNN")
library(FNN)

# download and read all the data 
download.file(
  'https://raw.githubusercontent.com/ChicagoBoothML/DATA___UsedCars/master/UsedCars.csv',
  'UsedCars.csv')
df = read.csv('UsedCars.csv') 
n = nrow(df)

# create train and validation set
set.seed(1)
perm = sample(n, n)
nTrain = 1000
nValid = 5000
train_index = perm[1:nTrain]
validation_index = perm[(nTrain+1):(nTrain+nValid)]

train_df = df[train_index, ]
validation_df = df[validation_index, ]

# 
mse = function(y, yhat) mean((y-yhat)^2)

# run simple kNN 
# choose k that minimizes the error on validation set

# here are values of k that we will try
kArray = c(
  seq(60, 150, by = 10)
)

(numK = length(kArray))
validErr = rep(0, numK)
fhat_kNN = matrix(0, nValid, numK)

for (indK in 1:numK) {
  k = kArray[indK]
  knn_fit = knn.reg(train = as.matrix( train_df[,"mileage"] ), 
                    test = as.matrix( validation_df[,"mileage"] ),
                    y = train_df$price,
                    k = k)
  fhat_kNN[,indK] = knn_fit$pred
  validErr[indK] = mse(fhat_kNN[,indK], validation_df$price)
}
# plot validation error
plot(kArray, validErr, type="l")

# the best value of K
(ind_bestK = which.min(validErr))
(bestK = kArray[ind_bestK])

# plot best kNN fit
plot(train_df$mileage, train_df$price, type="p", cex=0.4)   
oo = order(validation_df$mileage)
lines(validation_df$mileage[oo], fhat_kNN[oo,ind_bestK], col="red", lw=2)


## illustration of bagging for few falues of k
# we will fit kNN 1000 times 

kArrayBag = c(60, 80, 90, 100)
numK = length(kArrayBag)
validErrBag = rep(0, numK)
B = 1000

fhat = array(rep(0, nValid*B*numK), c(nValid, B, numK)) # we will store results of estimation here

for (indK in 1:numK) {           # for every value of k run bagging
  k = kArrayBag[indK]
  cat('k: ',k,'\n')
  
  for(i in 1:B) {
    if((i%%10)==0) cat('k: ', k, '   i: ', i, '\n')
    
    # get index of observations sampled in this iteration
    ii = sample(1:nTrain, nTrain, replace=TRUE)
    knn_fit = knn.reg(train = as.matrix( train_df[ii,"mileage"] ), 
                      test = as.matrix( validation_df[,"mileage"] ),
                      y = train_df$price[ii],
                      k = k)
    fhat[,i,indK] = knn_fit$pred    
  }    
  
  # our prediction is an average of all fits
  yhat = rowMeans(fhat[,,indK])
  validErrBag[indK] = mse(yhat, validation_df$price)
}

# plot different fits
plot(train_df$mileage, train_df$price, type="p", cex=0.4)   
oo = order(validation_df$mileage)
lines(validation_df$mileage[oo], fhat_kNN[oo,ind_bestK], col="red", lw=2)   # original curve
lines(validation_df$mileage[oo], rowMeans(fhat[oo,,1]), col="green", lw=2)
lines(validation_df$mileage[oo], rowMeans(fhat[oo,,2]), col="blue", lw=2)
lines(validation_df$mileage[oo], rowMeans(fhat[oo,,3]), col="yellow", lw=2)
lines(validation_df$mileage[oo], rowMeans(fhat[oo,,4]), col="cyan", lw=2)

