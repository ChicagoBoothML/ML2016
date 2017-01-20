# install.packages("FNN")
library(FNN)
# library(kknn)

# download and read all the data 
download.file(
  'https://raw.githubusercontent.com/ChicagoBoothML/DATA___UsedCars/master/UsedCars.csv',
  'UsedCars.csv')
df = read.csv('UsedCars.csv') 
n = nrow(df)

# split data into train and validation
# we will use 4000 observations for train
set.seed(1)
perm = sample(n, n)
nTrain = 750
nValid = 1000
train_index = perm[1:nTrain]
validation_index = perm[(nTrain+1):(nTrain+nValid)]

train_df = df[train_index, ]
validation_df = df[validation_index, ]
nValid = nrow(validation_df)

# 
mse = function(y, yhat) mean((y-yhat)^2)

# run simple kNN 
# choose k that minimizes the error on validation set

# here are values of k that we will try
kArray = c(
  seq(60, 200, by = 20),
  seq(250, 500, by = 50)
)

kArray = c(
  seq(5, 100, by = 5)
)

(numK = length(kArray))
validErr = rep(0, numK)
fhat_kNN = matrix(0, nValid, numK)

for (indK in 1:numK) {
  k = kArray[indK]
  # knn_fit = kknn(price~mileage,
  #                train_df, 
  #                validation_df,
  #                k=k,
  #                kernel='rectangular') 
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


## illustration of bagging

validErrBag = rep(0, numK)

# we will fit kNN 500 times for different values of k
B = 1000
# we will store results of estimation here
fhat = array(rep(0, nValid*B*numK), c(nValid, B, numK))
  
for (indK in 1:numK) {           # for every value of k run bagging
  k = kArray[indK]
  cat('k: ',k,'\n')

  for(i in 1:B) {
    if((i%%10)==0) cat('k: ', k, '   i: ', i, '\n')
    
    # get index of observations sampled in this iteration
    ii = sample(1:nTrain, nTrain, replace=TRUE)
    # knn_fit = kknn(price~mileage,
    #                train_df[ii,], 
    #                validation_df,
    #                k=k,
    #                kernel='rectangular')     
    # fhat[,i,indK] = fitted(knn_fit) 
    
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
# plot validation error
plot(kArray, validErr, type="l")
lines(kArray, validErrBag, col="red")

# the best value of K for bagging
(ind_bestK_bag = which.min(validErrBag))
(bestK_bag = kArray[ind_bestK_bag])


# plot different fits
plot(validation_df$mileage, validation_df$price, type="p", cex=0.4)   
oo = order(validation_df$mileage)
lines(validation_df$mileage[oo], fhat_kNN[oo,ind_bestK], col="red", lw=2)
lines(validation_df$mileage[oo], rowMeans(fhat[oo,,ind_bestK_bag]), col="green", lw=2)

