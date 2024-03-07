library(ggplot2)
library(class)
library(tidyverse)
library(caret)
library(ISLR)
library(car)
library(tree)

train<- read.csv("FlightTrainNew1.csv")
test<- read.csv("FlightTestNoYNew1.csv")

#Turn necessary variables into factors
train$Destination_airport<- as.factor(train$Destination_airport)
train$O.City<-as.factor(train$O.City)
train$O.State<-as.factor(train$O.State)
train$Origin_city<-as.factor(train$Origin_city)
train$Destination_city<-as.factor(train$Destination_city)
train$AIRLINE<-as.factor(train$AIRLINE)
train$Cancelled<-as.factor(train$Cancelled)
test$Destination_airport<- as.factor(test$Destination_airport)
test$O.City<-as.factor(test$O.City)
test$Origin_airport<-as.factor(test$Origin_airport)
test$AIRLINE<-as.factor(test$AIRLINE)

train$MONTH<- as.factor(train$MONTH)
test$MONTH<- as.factor(test$MONTH)
train$DAY<- as.factor(train$DAY)
test$DAY<- as.factor(test$DAY)
train$DAY_OF_WEEK<-as.factor(train$DAY_OF_WEEK)
test$DAY_OF_WEEK<-as.factor(test$DAY_OF_WEEK)


View(train)

attach(train)
attach(test)




back <- sapply(train[,c(27:46)], function(x) sum(is.na(x)))

backt <- transpose(as.data.frame(back),  )

summary(train$DIVERTED)

sapply(train[,c(0:26)], function(x) sum(is.na(x)))

sapply(newtrain, function(x) sum(is.na(x)))


rownames(as.data.frame(back)) 
backt


write.csv(a,"C:\\Users\\adamk\\Documents\\R\\Stats 101C\\Kaggle Project\\a.csv", row.names = F)

as.data.frame(back, rownames(as.data.frame(back)))
a<-data.frame(back,rownames(as.data.frame(back) ))
a <- transpose(a,)

a

#na.omit(train) was removing too much of the data so
#to deal with missing data, I used variables that had less missing values as the predictors.
#make a new training set of variables with no missing data 
newtrain<- train[,c(8:20,38)]
newtrain<- na.omit(newtrain)
newtest<- test[,c(8:20)]
newtest<-na.omit(newtest)
View(newtrain)





#I tried to set up knn first, ran into some issues I didnt have time to debug
# Ytraining<- newtrain$Cancelled
# Xtraining<- newtrain[,-c(4,5)]
# 
# summary(Xtraining)
# summary(newtest)
# 
# knn(Xtraining[,-5], newtest, Ytraining, k = 1)


#Logistic regression

m1<- glm(Cancelled~Passengers+Seats+Flights+Distance+Origin_population+Destination_population+Org_airport_lat+Org_airport_long+Dest_airport_lat+Dest_airport_long+MONTH+DAY_OF_WEEK, data = newtrain, family = binomial)


p1<-predict(m1,test, type = "response")  #make predictions
preds<- ifelse(p1>.5, "YES", "NO")     # encode predictions to fit kaggle format 

ptraining<-predict(m1, type = "response")  #predictions to test training 

trainingpreds <-ifelse(ptraining>.5, "YES", "NO")   #predictions 

mean(trainingpreds == train$Cancelled)   #Training Accuracy BEST ONE 




#Tree 
mtree<- tree(Cancelled~Passengers+Seats+Flights+Distance+Origin_population+Destination_population+Org_airport_lat+Org_airport_long+Dest_airport_lat+Dest_airport_long+MONTH+DAY_OF_WEEK, data = newtrain)

summary(mtree)
plot(mtree)
text(mtree,pretty = 0)

p2train <- predict(mtree, type = "class")          #predict training data
p2test<- predict(mtree, test, type = "class")      #predict test data

table(p2,newtrain$Cancelled)            

mean(p2train == newtrain$Cancelled)               #training accuracy 


#Prune the tree

cvbc<- cv.tree(mtree, FUN = prune.misclass)
plot(cvbc$size, cvbc$dev,type = "b")



mprune<- prune.misclass(mtree, best = 5)

summary(mprune)
plot(mprune)
text(mprune,pretty = 0)


pprune<- predict(mprune,type = "class") #make predictionsusing pruned tree
mean(pprune == newtrain$Cancelled)      #tree accuracy



#format data to submit to kaggle
Solution<- data.frame(1:nrow(test), preds)
View(Solution)
Solution

names(Solution)[1] <- "Ob"
names(Solution)[2]<- "Cancelled"

write.csv(Solution,"C:\\Users\\adamk\\Documents\\R\\Stats 101C\\Kaggle Project\\Solution.csv", row.names = F)




#random test stuff

sapply(train, function(x)all(is.na(x)))
  
na.test <-  function (x) {
    w <- sapply(x, function(x)all(is.na(x)))
    if (any(w)) {
      stop(paste("All NA in columns", paste(which(w), collapse=", ")))
    }
  }
na.test(train)



library(randomForest)

rf.air <- randomForest(Cancelled~Passengers+Seats+Flights+Distance+Origin_population+Destination_population+Org_airport_lat+Org_airport_long+Dest_airport_lat+Dest_airport_long+MONTH+DAY_OF_WEEK, data = newtrain, ntree = 500, mtry = 5)

summary(rf.air)

rf.pred = predict(rf.air, type = "response")

preds2 <- predict(rf.air, test, type = "response")

ptraining<-predict(rf.air, type = "response")  #predictions to test training 

mean(rf.pred == train$Cancelled)   #Training Accuracy BEST ONE 

table(rf.pred,newtrain$Cancelled)

control <- trainControl(method="repeatedcv", number=10, repeats=3, search="random")

mtry <- sqrt(14)
rf_random <- train(Cancelled~Passengers+Seats+Flights+Distance+Origin_population+Destination_population+Org_airport_lat+Org_airport_long+Dest_airport_lat+Dest_airport_long+MONTH+DAY_OF_WEEK, data = newtrain, method="rf", metric="Accuracy", tuneLength=15, trControl=control)
print(rf_random)
plot(rf_random)



#Ensemble This is the best now!

library(h2o)       # for fitting stacked models
# initialize the h2o
h2o.init()

# create the train and test h2o data frames

train_df<-as.h2o(newtrain)
test_df<-as.h2o(newtest)

# Identify predictors and response
y <- "Cancelled"
x <- setdiff(names(train_df), y)

# Number of CV folds (to generate level-one data for stacking)
nfolds <- 5

# 1. Generate a 3-model ensemble (GBM + RF + Logistic)

# Train &amp; Cross-validate a GBM
my_gbm <- h2o.gbm(x = x,
                  y = y,
                  training_frame = train_df,
                  nfolds = nfolds,
                  keep_cross_validation_predictions = TRUE,
                  seed = 5)

# Train &amp; Cross-validate a RF
my_rf <- h2o.randomForest(x = x,
                          y = y,
                          training_frame = train_df,
                          nfolds = nfolds,
                          keep_cross_validation_predictions = TRUE,
                          seed = 5)


# Train &amp; Cross-validate a LR
my_lr <- h2o.glm(x = x,
                 y = y,
                 training_frame = train_df,
                 family = c("binomial"),
                 nfolds = nfolds,
                 keep_cross_validation_predictions = TRUE,
                 seed = 5)



# Train a stacked random forest ensemble using the GBM, RF and LR above
ensemble <- h2o.stackedEnsemble(x = x,
                                y = y,
                                metalearner_algorithm="drf",
                                training_frame = train_df,
                                base_models = list(my_gbm, my_rf, my_lr))


# Eval ensemble performance on a test set
perf <- h2o.performance(ensemble, newdata = train_df)


# Compare to base learner performance on the test set
perf_gbm <- h2o.performance(my_gbm, newdata = train_df)
perf_rf <- h2o.performance(my_rf, newdata = train_df)
perf_lr <- h2o.performance(my_lr, newdata = train_df)
bestof_base_models <- max(h2o.auc(perf_gbm), h2o.auc(perf_rf), h2o.auc(perf_lr))
stacked <- h2o.auc(perf)
h2o.auc(perf_rf) #this is the best!! for adam kaba

preds3updated<- h2o.predict(my_rf, test_df)
preds3updated<- as.data.frame(preds3updated)
preds3updated<- preds3updated[,1]

mean(preds3updated== preds3)

preds4<-h2o.predict(my_rf, test_df)
preds4<-as.data.frame(preds4)
preds4<-preds4[,1]

pred<-h2o.predict(ensemble, train_df)
rf_h2o_pred<-as.data.frame(pred)
rf_h2o_pred<-rf_h2o_pred[,1]
train_df<-as.data.frame(train_df)
table(rf_h2o_pred,train_df$Cancelled)
mean(rf_h2o_pred == train_df$Cancelled) 


##########################################################################################################






library(h2o)       # for fitting stacked models
# initialize the h2o
h2o.init()

# create the train and test h2o data frames

train_df<-as.h2o(newtrain)
test_df<-as.h2o(newtest)

# Identify predictors and response
y <- "Cancelled"

x <- setdiff(names(train_df), y)

# Number of CV folds (to generate level-one data for stacking)
nfolds <- 10

# 1. Generate a 3-model ensemble (GBM + RF + Logistic)
#h2o.removeAll()
# Train &amp; Cross-validate a GBM
my_ml <- h2o.automl(x = x,
                    y = y,
                    training_frame = train_df,
                    nfolds = nfolds,
                    max_models=10,
                    keep_cross_validation_predictions = TRUE,
                    seed = 5)
my_ml@leader
my_ml
# Train &amp; Cross-validate a RF
my_rf <- h2o.randomForest(x = x,
                          y = y,
                          training_frame = train_df,
                          nfolds = nfolds,
                          keep_cross_validation_predictions = TRUE,
                          seed = 5)
my_rf

# Train &amp; Cross-validate a LR
my_dl <- h2o.deeplearning(x = x,
                     y = y,
                     training_frame = train_df,
                     nfolds = nfolds,
                     keep_cross_validation_predictions = TRUE,
                     seed = 5)

my_dl 
# Train a stacked random forest ensemble using the GBM, RF and LR above
ensemble <- h2o.stackedEnsemble(x = x,
                                y = y,
                                metalearner_algorithm="drf",
                                training_frame = train_df,
                                base_models = list(my_ml, my_rf, my_dl))




# Compare to base learner performance on the test set
perf_gbm <- h2o.performance(my_gbm, newdata = train_df)
perf_rf <- h2o.performance(my_rf, newdata = train_df)
perf_lr <- h2o.performance(my_lr, newdata = train_df)
bestof_base_models <- max(h2o.auc(perf_gbm), h2o.auc(perf_rf), h2o.auc(perf_lr))
stacked <- h2o.performance(ensemble,newdata = train_df )

#Preparing it for Kaggle/adam
pred<-h2o.predict(my_ml@leader, train_df)
rf_h2o_pred<-as.data.frame(pred)
rf_h2o_pred<-rf_h2o_pred[,1]

# train_df<-as.data.frame(train_df)
# 
# length(rf_h2o_pred)
# length(train_df$Cancelled)
# table(rf_h2o_pred,train_df$Cancelled)
 mean(rf_h2o_pred == train$Cancelled) 


preds<-h2o.predict(my_ml@leader, test_df)
preds6<-as.data.frame(preds)
preds6<-preds6[,1]

preds<-h2o.predict(ensemble, test_df)
preds6<-as.data.frame(preds)
preds6<-preds6[,1]


mean(preds5 == preds6)



h2o.init()

# create the train and test h2o data frames

train_df<-as.h2o(newtrain)
test_df<-as.h2o(newtest)

# Identify predictors and response
y <- "Cancelled"
x <- setdiff(names(train_df), y)



ml <- h2o.automl(x = x,
                    y = y,
                    training_frame = train_df,
                    nfolds = 100,
                    max_models=30,
                    keep_cross_validation_predictions = TRUE,
                    seed = 5)
ml@leader



Solution<- data.frame(1:nrow(test), preds6)
View(Solution)
Solution

names(Solution)[1] <- "Ob"
names(Solution)[2]<- "Cancelled"

write.csv(newtest,"C:\\Users\\adamk\\Documents\\R\\Stats 101C\\Kaggle Project\\newtest.csv", row.names = F)
