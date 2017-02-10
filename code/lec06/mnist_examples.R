library(randomForest)
library(e1071)
library(h2o)
library(glmnet)

# we will not train many models in the classroom on this example
# as they will take a long time
inClass = TRUE

MNIST_DIR = "."

# we will download files into the MNIST directory
# by default, this is the current directory
fileNames = c("train-images.idx3-ubyte",
              "t10k-images.idx3-ubyte",
              "train-labels.idx1-ubyte",
              "t10k-labels.idx1-ubyte")
GIT_REPO = "https://github.com/ChicagoBoothML/DATA___LeCun___MNISTDigits/raw/master/"
for (fName in fileNames) {
    download.file(paste(GIT_REPO, fName, sep = ""), 
                  destfile = file.path(MNIST_DIR, fName))
}


# code to load digits and show one digit
source("https://raw.githubusercontent.com/ChicagoBoothML/MachineLearning_Fall2015/master/Programming%20Scripts/Lecture06/mnist.helper.R")

# load all digits
digit.data = load_mnist(MNIST_DIR)

# training sample size and number of pixels
dim(digit.data$train$x)

#   Pixels are organized into images like this:
#  
#   001 002 003 ... 026 027 028
#   029 030 031 ... 054 055 056
#   057 058 059 ... 082 083 084
#    |   |   |  ...  |   |   |
#   729 730 731 ... 754 755 756
#   757 758 759 ... 782 783 784
#

show_digit(digit.data$train$x[1, ])
show_digit(digit.data$train$x[2100, ])


####################################
### 
### Logistic regression
###
####################################

if (file.exists("mnist.glmnet.RData")) {
  load("mnist.glmnet.RData")
} else {
  glm_fit = cv.glmnet(x=digit.data$train$x, y=as.factor(digit.data$train$y), 
                      family="multinomial",
                      type.logistic="modified.Newton")
  save(glm_fit, file = "glmnet.mnist.RData")
}

phat = predict(glm_fit, digit.data$test$x, s=glm_fit$lambda.1se, type = "response")
yhat = apply(phat,1,which.max) - 1
ot = table(yhat, digit.data$test$y)
print(ot)
sum(diag(ot)) / 10000 # accuracy 

plot(glm_fit)

plot(glm_fit$glmnet.fit, xvar = "lambda")


####################################
### 
### Random forest
###
####################################

if (file.exists("mnist.rf.mtry_28.RData")) {
  load("mnist.rf.mtry_28.RData")
} else {
  num_trees = 1000
  
  rf_28 = randomForest(
    x=digit.data$train$x, 
    y=as.factor(digit.data$train$y), 
    xtests=digit.data$test$x,
    sampsize=6000,   # sample about 10% of data
    ntree=num_trees, 
    mtry=28,         # try 28 = sqrt(784) features at each split
    importance=TRUE, 
    nodesize=100     # need this many observations in the leaf
  )
  
  save(rf_28, file = "fName")
}
rf_28

varImpPlot(rf_28, type=2, n.var=20, main="Variable importance")
predicted.test = predict(rf_28, digit.data$test$x)
temp=table(predicted.test,digit.data$test$y)




####################################
### 
### Neural networks
###
####################################

# start or connect to h2o server
h2oServer <- h2o.init(ip="localhost", port=54321, max_mem_size="4g", nthreads=-1)

# we need to load data into h2o format
train_hex = as.h2o(data.frame(x=digit.data$train$x, y=digit.data$train$y))
test_hex = as.h2o(data.frame(x=digit.data$test$x, y=digit.data$test$y))

predictors = 1:784
response = 785

train_hex[,response] <- as.factor(train_hex[,response])
test_hex[,response] <- as.factor(test_hex[,response])

# create frames with input features only
# we will need these later for unsupervised training
trainX = train_hex[,-response]
testX = test_hex[,-response]


####################################################################

if (inClass == FALSE) {
  dl_model <- h2o.deeplearning(x=predictors, y=response,
                             training_frame=train_hex,
                             activation="RectifierWithDropout",
                             input_dropout_ratio=0.2,
                             classification_stop=-1,  # Turn off early stopping
                             l1=1e-5,
                             hidden=c(128,128,256), epochs=10,
                             model_id = "DL_FirstMNIST"
  )
  h2o.saveModel(dl_model, path = "mnist" )  
} else {
  dl_model = h2o.loadModel(file.path("mnist", "DL_FirstMNIST"))
}

# performance on test
ptest = h2o.performance(dl_model, test_hex )
h2o.confusionMatrix(ptest)

# performance on train
ptrain = h2o.performance(dl_model, train_hex)
h2o.confusionMatrix(ptrain)

####################################################################
# training many models to see which may do well
# 

if (inClass == FALSE) {
  # it will take some time to train all
  
  EPOCHS = 2
  args <- list(
    list(epochs=EPOCHS),
    list(epochs=EPOCHS, activation="Tanh"),
    list(epochs=EPOCHS, hidden=c(512,512)),
    list(epochs=5*EPOCHS, hidden=c(64,128,128)),
    list(epochs=5*EPOCHS, hidden=c(512,512), 
         activation="RectifierWithDropout", input_dropout_ratio=0.2, l1=1e-5),
    list(epochs=5*EPOCHS, hidden=c(256,256,256), 
         activation="RectifierWithDropout", input_dropout_ratio=0.2, l1=1e-5),
    list(epochs=5*EPOCHS, hidden=c(200,200), 
         activation="RectifierWithDropout", input_dropout_ratio=0.2, l1=1e-5),
    list(epochs=5*EPOCHS, hidden=c(100,100,100), 
         activation="RectifierWithDropout", input_dropout_ratio=0.2, l1=1e-5)
  )

  run <- function(extra_params) {
    str(extra_params)
    print("Training.")
    model <- do.call(h2o.deeplearning, modifyList(list(x=predictors, y=response,
                                                       training_frame=train_hex), extra_params))
    sampleshist <- model@model$scoring_history$samples
    samples <- sampleshist[length(sampleshist)]
    time <- model@model$run_time/1000
    print(paste0("training samples: ", samples))
    print(paste0("training time   : ", time, " seconds"))
    print(paste0("training speed  : ", samples/time, " samples/second"))
    
    print("Scoring on test set.")
    p <- h2o.performance(model, test_hex)
    cm <- h2o.confusionMatrix(p)
    test_error <- cm$Error[length(cm$Error)]
    print(paste0("test set error  : ", test_error))
    
    c(paste(names(extra_params), extra_params, sep = "=", collapse=" "), 
      samples, sprintf("%.3f", time), 
      sprintf("%.3f", samples/time), sprintf("%.3f", test_error))
  }

  writecsv <- function(results) {
    table <- matrix(unlist(results), ncol = 5, byrow = TRUE)
    colnames(table) <- c("parameters", "training samples",
                         "training time", "training speed", "test set error")
    table
  }

  table = writecsv(lapply(args, run))
  save(table, file="mnist.h2o.table_results.RData")
  
} else {
  load("mnist.h2o.table_results.RData")
  table
}

##############################################################
# model trained for 1000 epochs
# The code below took long time to run
# (about 15h on my old desktop)

if (inClass == FALSE) {
  mdl = h2o.deeplearning(x=predictors, y=response,
                            training_frame=train_hex,
                            activation="RectifierWithDropout",
                            input_dropout_ratio=0.2,
                            classification_stop=-1,  # Turn off early stopping
                            l1=1e-5,
                            hidden=c(1024,1024,2048), epochs=1000,
                            model_id = "goodModel.epoch1000"
  )
  h2o.saveModel(mdl, path = "mnist")  
} else {
  mdl = h2o.loadModel( path = file.path("mnist", "goodModel.epoch1000") )
}

# performance on test
ptest = h2o.performance(mdl, test_hex )
h2o.confusionMatrix(ptest)

# performance on train
ptrain = h2o.performance(mdl, train_hex)
h2o.confusionMatrix(ptrain)

# extract features using the model

trainX.deep.features = h2o.deepfeatures(mdl, trainX, layer = 3)
testX.deep.features = h2o.deepfeatures(mdl, testX, layer = 3)

dim(trainX.deep.features)

if (inClass == FALSE) {
  deep.rf = h2o.randomForest(x=1:2048, y=2049, 
                           training_frame = h2o.cbind(trainX.deep.features, train_hex[,response]),
                           ntrees = 1000,
                           min_rows = 100,
                           model_id = "DRF_features.2048"
                           )
  h2o.saveModel(deep.rf, path="mnist")
} else {
  deep.rf = h2o.loadModel( path = file.path("mnist", "DRF_features.2048") )
}

phat = h2o.predict(deep.rf, testX.deep.features)
head(phat)

h2o.confusionMatrix(deep.rf, h2o.cbind(testX.deep.features, test_hex[,785]))

