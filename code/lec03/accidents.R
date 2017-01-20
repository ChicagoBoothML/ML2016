download.file(
    'https://raw.githubusercontent.com/ChicagoBoothML/DATA___TransportAccidents/master/Accidents.csv',
    'Accidents.csv')

accidents_df = read.csv("Accidents.csv")
n = nrow(accidents_df)

accidents_df$INJURY = rep(1, n)
accidents_df$INJURY[accidents_df$MAX_SEV_IR == 0] = 0
drops = c("MAX_SEV_IR", "FATALITIES", "PRPTYDMG_CRASH", "NO_INJ_I", "INJURY_CRASH", "VEH_INVL")
accidents_df = accidents_df[, !(names(accidents_df) %in% drops)]

set.seed(1)
train_ind = sample.int(n, floor(0.8*n))
accidents_df_train = accidents_df[train_ind,]
accidents_df_test = accidents_df[-train_ind,]

# every variable is categorical variable, tell R that
accidents_df_train[] = lapply(accidents_df_train, factor)
accidents_df_test[] = lapply(accidents_df_test, factor)

####
library(randomForest)
library(gbm)
library(rpart)
library(rpart.plot)

# Without any information, we can think of accidents with injuries and without injuries as i.i.d. Bernoulli(p). 
# Our best guess for p is fraction of accidents with injuries out of all injuries.

(tb_INJURY = table(accidents_df_train$INJURY))
# estimated probability of injury
p_INJURY = tb_INJURY["1"] / (sum(tb_INJURY)); print(p_INJURY)

temp = rpart(INJURY~., data=accidents_df_train, 
             control=rpart.control(minsplit=10,  # we want to have at least 5 observations in a node
                                   cp=0.0001,    # we will consider even small improvements in a fit 
                                   xval=10)      
)

(cptable = printcp(temp))
(bestcp = cptable[ which.min(cptable[,"xerror"]), "CP" ])   # this is the optimal cp parameter

optimal.tree = prune(temp, cp=bestcp)
rpart.plot(optimal.tree)
length(unique(optimal.tree$where))   # number of leaves

optimal_tree_predictions = predict(optimal.tree, accidents_df_test, type="class")
1 - mean(optimal_tree_predictions == accidents_df_test$INJURY)  # error rate using all variables
table(actual = accidents_df_test$INJURY, predictions = optimal_tree_predictions)  # confusion table

#####
library(e1071)   # for naive bayes
nb_model = naiveBayes(accidents_df_train[,-19], accidents_df_train$INJURY)

# report error on test data
nb_test_predictions = predict(nb_model, accidents_df_test) 
1 - mean(nb_test_predictions == accidents_df_test$INJURY)  # error rate using all variables
table(actual = accidents_df_test$INJURY, predictions = nb_test_predictions)  # confusion table


#### 
# fit logistic regression
lr_model = glm(INJURY~., family=binomial, data=accidents_df_train)

# report error on test data
lr_test_predictions = predict(lr_model, accidents_df_test, type="response")
class0_ind =  lr_test_predictions < 0.5 
class1_ind =  lr_test_predictions >= 0.5 
lr_test_predictions[class0_ind] = levels(accidents_df_train$INJURY)[1]
lr_test_predictions[class1_ind] = levels(accidents_df_train$INJURY)[2]
1 - mean(lr_test_predictions == accidents_df_test$INJURY)  # error rate using all variables
table(actual = accidents_df_test$INJURY, predictions = lr_test_predictions)  # confusion table

####
# random forest

rffit = randomForest(INJURY~.,data=accidents_df_train,mtry=3,ntree=1000)
rf_test_predictions = predict(rffit, accidents_df_test) 
1 - mean(rf_test_predictions == accidents_df_test$INJURY)  # error rate using all variables
table(actual = accidents_df_test$INJURY, predictions = rf_test_predictions)  # confusion table

varImpPlot(rffit)

####
# boosting

accidents_df_train$y = as.numeric(accidents_df_train$INJURY)-1
accidents_df_test$y = as.numeric(accidents_df_test$INJURY)-1

boostfit = gbm(y~.,data=accidents_df_train,
               distribution='bernoulli',
               interaction.depth=4,
               n.trees=10,
               shrinkage=.02)
b_test_predictions = predict(boostfit, accidents_df_test, n.trees = 10, type = "response") 
class0_ind =  b_test_predictions < 0.5 
class1_ind =  b_test_predictions >= 0.5 
b_test_predictions[class0_ind] = levels(accidents_df_train$INJURY)[1]
b_test_predictions[class1_ind] = levels(accidents_df_train$INJURY)[2]
1 - mean(rf_test_predictions == accidents_df_test$INJURY)  # error rate using all variables
table(actual = accidents_df_test$INJURY, predictions = rf_test_predictions)  # confusion table

