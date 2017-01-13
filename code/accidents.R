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
