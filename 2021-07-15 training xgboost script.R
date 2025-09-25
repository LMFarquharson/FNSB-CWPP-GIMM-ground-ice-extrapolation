# Use XGBoost and train on ground ice extent in Fairbanks
# to predict ground_ice_3 labels in the rest of Fairbanks CWPP region.
#
# There will be two models overall:
#     thaw_xgb_fit is the model that has "ice_wedge" as a predictor, and it will only apply in thaw extent
#     CWPP_xgb_fit is the model for outside of thaw feature (in "other CWPP" extent)

#---------------------------------------------------------------

# Quick notes:
# General example of multiclass xgboost process: https://rpubs.com/mharris/multiclass_xgboost
# XGBoost parameters: https://xgboost.readthedocs.io/en/latest/parameter.html
# About logloss: https://towardsdatascience.com/intuition-behind-log-loss-score-4e0c9979680a



library(tidyverse)
library(xgboost)
library(caret)
library(ROCR)

# I don't want to see scientific notation
options(scipen=8) 

# Set random seed for reproducibility
set.seed(42)

#############################################################

if (!dir.exists("./model stats")) {dir.create("./model stats")}
if (!dir.exists("./process")) {dir.create("./process")}

#############################################################
# My functions
#############################################################

# FUNCTION create_dummy_vars
# PARAMETERS:
#   df : a data frame
#   factor_name : a character string that specified the factor-type
#         variable in df to be converted into dummy variables
# RETURNS:  modified df, now containing the dummy variables
create_dummy_vars <- function(df, factor_name) {
  factor_levels <- levels(df[[factor_name]])
  for (n in 1:length(factor_levels)) {
    newvar_name <- paste0(factor_name, as.character(n))
    df[[newvar_name]] = ifelse(is.na(df[[factor_name]]), 
                               NA,
                               ifelse(df[[factor_name]]==factor_levels[n],1,0))
    # remove the variable has no variance
    if (var(df[[newvar_name]], na.rm = TRUE)==0){
      df[[newvar_name]] <- NULL
    }
  }
  return(df)
}


###################################################################

#...........................................................
# Prepare the data for XGBoost 

load(file="train test folds df.RData")

# The goal is to predict ground_ice_3 labels.

# To use xgboost, I will need to do the following:
# (1) Make sure my labels in ground_ice_3 are as integers 0, 1, and 2

# (2) Change all the predictive factors to numeric variables,
# either into numeric equivalents or into hotcoding.
#


# > str(df)
# 'data.frame':	853870 obs. of  14 variables:
# $ XCoord           : num  282586 282586 282586 282586 282586 ...
# $ YCoord           : num  1683508 1683538 1683568 1683598 1683628 ...
# $ ground_ice_3     : Factor w/ 3 levels "Low","Medium",..: 1 1 1 1 1 1 1 1 1 1 ...
# $ ice_wedge        : Factor w/ 2 levels "0","1": 1 1 1 1 1 1 1 1 1 1 ...
# $ Tanana_floodplain: Factor w/ 2 levels "0","1": 1 1 1 1 1 1 1 1 1 1 ...
# $ ELEV             : num  393 385 378 371 363 ...
# $ Local_Elev       : num  175 168 163 157 151 ...
# $ SLOPE            : num  19.3 18.9 18.5 18 15.9 ...
# $ temp_summer      : num  14.6 14.6 14.6 14.6 14.6 ...
# $ temp_winter      : num  -14.5 -14.5 -14.5 -14.5 -14.5 ...
# $ isWetland        : Factor w/ 2 levels "0","1": 2 1 1 2 1 1 1 1 1 1 ...
# $ ABOVE_1984_adj   : Factor w/ 15 levels "Shallows","Bog",..: 11 15 15 11 11 15 15 15 15 15 ...
# $ folds            : int  8 8 8 8 8 8 8 8 8 8 ...
# $ isTestData       : num  0 0 0 0 0 0 0 0 0 0 ...

# Drop the "Water" level from ABOVE_1984_adj
df <- df %>% 
  filter(ABOVE_1984_adj != "Water") %>%
  mutate(ABOVE_1984_adj = droplevels(ABOVE_1984_adj)) 

# Change ground_ice_3 and binary labels.  (Note: I checked that these results are good.)
df <- df %>% 
  mutate(ground_ice_3 = as.integer(ground_ice_3)-1 ) %>%
  mutate(ice_wedge = as.integer(ice_wedge)-1 ) %>%
  mutate(Tanana_floodplain = as.integer(Tanana_floodplain)-1 ) %>%
  mutate(isWetland = as.integer(isWetland)-1 ) 


# Hotcode the vegetation data
df <- create_dummy_vars(df, "ABOVE_1984_adj")
df <- select(df, -ABOVE_1984_adj)


save(df, file="./process/prepared df.RData")

#############################################################################
# This first training is for the model that includes "ice_wedge", 
# which will apply to thaw feature extent.
#############################################################################

# Split into train and test
train_df <- df %>%
  filter(isTestData == 0) %>%
  select(ground_ice_3, everything()) %>%
  select(-isTestData)

test_df <- df %>%
  filter(isTestData == 1) %>%
  select(ground_ice_3, everything()) %>%
  select(-folds, -isTestData)


# Now that the data sets are prepared, create xgb.DMatrix data types
train_data <- train_df %>%
  select(-ground_ice_3, -XCoord, -YCoord, -folds)

dtrain <- xgb.DMatrix(data = as.matrix(train_data), 
                      label=train_df$ground_ice_3)

test_data <- test_df %>%
  select(-ground_ice_3, -XCoord, -YCoord)

dtest <- xgb.DMatrix(data = as.matrix(test_data), 
                     label=test_df$ground_ice_3)

save(train_df, dtrain, test_df, dtest, train_data, test_data, 
     file="./process/data prepared for xgboost.RData")


#---------------------------------------------------------------

#starting parameters for XGBoost
xgb_params <- list(booster = "gbtree", 
               objective = "multi:softprob",
               eval_metric = "mlogloss",
               num_class = 3, # three categories in ground_ice_3
               eta=0.3, 
               gamma=0, 
               max_depth=5, 
               min_child_weight=1, 
               subsample=0.1, 
               colsample_bytree=0.5)

# Custom folds
# From xgboost documentation:
# The parameter "folds" is a list which provides a possibility 
# to use a list of pre-defined CV folds (each element
# must be a vector of test fold's indices). When folds are supplied, 
# the nfold and stratified parameters are ignored.
#
# My pre-determined folds are in train_df$folds as numbers 1,2,3,4 ... 9.
# What I need is a list of four vectors, each vector having
# the index of the observation where folds==k.

custom_folds <- list()
for (k in 1:9) { custom_folds[[k]] <- which(train_df$folds==k)}
save(custom_folds, file="./process/custom_folds.RData")

# estimate reasonable number of rounds, using cross-validation
xgbcv <- xgb.cv( params = xgb_params, 
                 data = dtrain, 
                 # nrounds = 100, 
                 nrounds = 10, 
                 folds = custom_folds,
                 showsd = T, 
                 print_every_n = 10, 
                 early_stopping_rounds = 20, 
                 metrics = "mlogloss",
                 maximize = FALSE)

# [81]	train-mlogloss:0.246920+0.004195	test-mlogloss:0.310987+0.036671 
# [91]	train-mlogloss:0.242794+0.004141	test-mlogloss:0.310450+0.036829 
# [100]	train-mlogloss:0.239632+0.004044	test-mlogloss:0.310454+0.037574 

save(xgbcv, file="./process/xgb first model with standard params.RData")


#----------------------------------------------------------------
# Tune parameters

# Set the parameter grid
xgbGrid <- expand.grid( nrounds = 200,
                        max_depth = c(5, 10),
                        colsample_bytree = c(0.5, 0.75),
                        eta = c(0.1, 0.3),
                        gamma = c(0, 2),
                        alpha = c(0, 2),
                        lambda = 1,
                        min_child_weight = 1,
                        subsample = 1)

# Get the names of parameters
param_names <- names(xgbGrid)
# Get the number of iterations
num_iterations <- dim(xgbGrid)[1]
# Initialize new columns in xgbGrid for info from evaluation_log
log_names <- names(xgbcv$evaluation_log)



for (name in log_names) {
  xgbGrid[[name]] <- 0
}
xgbGrid



save(train_data, train_df, custom_folds, param_names, num_iterations, log_names, 
     file="./process/For building xgbGrid, not including latest xgbGrid.RData")

# ################################################################################
# # To prepare for R crashing, I am saving specifically what is necessary to run the iterations
# # that build xgbGrid.  This save includes everything except the latest xgbGrid, which needs to
# # be loaded separately.  Because of some bugs in the xgboost package, I need to re-create dtrain
# # after each reboot.
# 
# library(xgboost)
# options(scipen=8) 
# set.seed(42)
# 
# load(file="./process/For building xgbGrid, not including latest xgbGrid.RData")
# load(file = "./process/Latest xgbGrid.RData")
# 
# dtrain <- xgb.DMatrix(data = as.matrix(train_data), 
#                       label=train_df$ground_ice_3)
# rm(train_data, train_df)
# ################################################################################

for (n in 1:num_iterations) {
 
  # build a list of parameters
  params <- list()  # Start with these
  for (name in param_names) {
    params[[name]] <- xgbGrid[n, name]   # add all other parameters from the grid
  }
  
  print(paste0("Iteration: ", n, " out of ", num_iterations))
  print(xgbGrid[n, 1:9])
  
  # Run xgb.cv with these parameters
  xgbcv <- xgb.cv( params = params, 
                   data = dtrain, 
                   booster = "gbtree", 
                   objective = "multi:softprob",
                   num_class = 3, # three categories in ground_ice_3
                   nrounds = 200, 
                   folds = custom_folds,
                   showsd = T, 
                   print_every_n = 10, 
                   early_stopping_rounds = 10, 
                   metrics = "mlogloss", #multicategory log loss function
                   maximize = FALSE)
  
  for (name in log_names) {
    xgbGrid[[name]][n] <- xgbcv$evaluation_log[[name]][xgbcv$niter]
  }

  save(xgbGrid, file = "./process/Latest xgbGrid.RData")
}


write.csv(xgbGrid, file="./process/Parameter tuning results 1.csv", row.names = FALSE)

#................
# Check which hyperparameters seem to lower mlogloss.

if (!dir.exists("./process/params")) {dir.create("./process/params")}
attach(xgbGrid)

png("./process/params/tmm by max_depth.png")
plot(test_mlogloss_mean ~ max_depth)
abline(lm(test_mlogloss_mean ~ max_depth))
dev.off()


png("./process/params/tmm by colsample_bytree.png")
plot(test_mlogloss_mean ~ colsample_bytree)
abline(lm(test_mlogloss_mean ~ colsample_bytree))
dev.off()


png("./process/params/tmm by eta.png")
plot(test_mlogloss_mean ~ eta)
abline(lm(test_mlogloss_mean ~ eta))
dev.off()


png("./process/params/tmm by gamma.png")
plot(test_mlogloss_mean ~ gamma)
abline(lm(test_mlogloss_mean ~ gamma))
dev.off()

png("./process/params/tmm by alpha.png")
plot(test_mlogloss_mean ~ alpha)
abline(lm(test_mlogloss_mean ~ alpha))
dev.off()

detach()

# Observations:
#   alpha doesn't matter.  gamma doesn't matter.
#   higher colsample_bytree lowers test_mlogloss_mean.  Stick with 0.75 for decorrelation.
#   lower eta lowers average test_mlogloss, but it's quite scattered.  Possible interaction.
#   hither maxdepth lowers average test_mlogloss_mean.

# Let's try a few more combinations with more max_depth and lower eta.


# Set the parameter grid
xgbGrid <- expand.grid( nrounds = 200,
                        max_depth = c(10, 15),
                        colsample_bytree = 0.75,
                        eta = c(0.01, 0.1),
                        gamma = 0,
                        alpha = 0,
                        lambda = 1,
                        min_child_weight = 1,
                        subsample = 1)

# Get the names of parameters
param_names <- names(xgbGrid)
# Get the number of iterations
num_iterations <- dim(xgbGrid)[1]
# Initialize new columns in xgbGrid for info from evaluation_log
log_names <- names(xgbcv$evaluation_log)

for (name in log_names) {
  xgbGrid[[name]] <- 0
}
xgbGrid
save(xgbGrid, file = "./process/Latest xgbGrid.RData")

# ################################################################################
# # To prepare for R crashing, I am saving specifically what is necessary to run the iterations
# # that build xgbGrid.  This save includes everything except the latest xgbGrid, which needs to
# # be loaded separately.  Because of some bugs in the xgboost package, I need to re-create dtrain
# # after each reboot.
# 
# library(xgboost)
# options(scipen=8) 
# set.seed(42)
# 
# load(file="./process/For building xgbGrid, not including latest xgbGrid.RData")
# load(file = "./process/Latest xgbGrid.RData")
# 
# dtrain <- xgb.DMatrix(data = as.matrix(train_data), 
#                       label=train_df$ground_ice_3)
# rm(train_data, train_df)
# ################################################################################


for (n in 1:num_iterations) {
  # build a list of parameters
  params <- list()  # Start with these
  for (name in param_names) {
    params[[name]] <- xgbGrid[n, name]   # add all other parameters from the grid
  }
  
  print(paste0("Iteration: ", n, " out of ", num_iterations))
  print(xgbGrid[n, 1:9])
  
  # Run xgb.cv with these parameters
  xgbcv <- xgb.cv( params = params, 
                   data = dtrain, 
                   booster = "gbtree", 
                   objective = "multi:softprob",
                   num_class = 3, # three categories in ground_ice_3
                   nrounds = 200, 
                   folds = custom_folds,
                   showsd = T, 
                   print_every_n = 10, 
                   early_stopping_rounds = 10, 
                   metrics = "mlogloss", #multicategory log loss function
                   maximize = FALSE)
  
  for (name in log_names) {
    xgbGrid[[name]][n] <- xgbcv$evaluation_log[[name]][xgbcv$niter]
  }
  save(xgbGrid, file = "./process/Latest xgbGrid.RData")
}

write.csv(xgbGrid, file="./process/Parameter tuning results 2.csv", row.names = FALSE)

xgbGrid2 <- xgbGrid
xgbGrid <- read.csv("./process/Parameter tuning results 1.csv")
xgbGrid <- rbind(xgbGrid, xgbGrid2)

write.csv(xgbGrid, file = "./process/Parameter tuning results combined.csv", row.names = FALSE)
rm(xgbGrid2)


# Looking at combined results:
# Clear best is max_depth = 10 and eta = 0.1.
# colsample_bytree should be 0.75.
# Gamma and lapha don't matter much, go back to default of 0.
# The parameter tuning is taking a lot of time, so I will stop here.


#-----------------------------------------------------------------
# Train a model on best-tuned parameters


params <- list(booster = "gbtree", 
               objective = "multi:softprob",
               eval_metric = "mlogloss", #multicategory log loss function
               num_class = 3, # three categories in ground_ice_3
               max_depth	= 10,
               colsample_bytree =	0.75,
               eta =	0.1,
               alpha = 0,
               gamma	= 0,
               min_child_weight =	1,
               subsample =	1, 
               lambda = 1
)

thaw_xgb_fit  <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 200,
  nthreads=1,
  early_stopping_rounds = 10,
  watchlist = list(val1=dtrain, val2=dtest), 
  verbose = 1,
  print_every_n = 1
)



# Stopping. Best iteration:
#   [108]	val1-mlogloss:0.177585	val2-mlogloss:0.272134


save(thaw_xgb_fit, file = "xgboost FNSB-thaw extent.RData")


#-----------------------------------------------------------------
# Find threshhold for probability of ground_ice where the
# precision and recall balance -- that is, where the F statistic
# from the confusion matrix is maximal for the "High" value.

# Use training data.
ground_ice_3_prob_output <- predict(thaw_xgb_fit, newdata = dtrain)

# The output is an array that is three times the length of the training data.
# Sort out the output by type of ground_ice_3 labels
ground_ice_3_probs <- matrix(ground_ice_3_prob_output, 
                             nrow=3, # three classes 
                             ncol = length(ground_ice_3_prob_output)/3)
rm(ground_ice_3_prob_output)

ground_ice_3_probs <- ground_ice_3_probs %>%
  t() %>%  # transpose
  as.data.frame()
  
names(ground_ice_3_probs) <- c("Low_prob", "Medium_prob", "High_prob")
  
ground_ice_3_probs <- ground_ice_3_probs %>%
  mutate(max_prob = max.col(.,"last")) %>%
  mutate(ground_ice_pred = ifelse(max_prob == 1, "Low", 
                                  ifelse(max_prob == 2, "Medium", "High"))) %>%
  mutate(ground_ice_pred = as.factor(ground_ice_pred)) %>%
  mutate(ground_ice_pred = factor(ground_ice_pred, levels = c("Low", "Medium", "High")))%>%
  select(-max_prob)



# combine with actual labels
thaw_train_results <- train_df %>%
  select(XCoord, YCoord, ground_ice_3) %>%
  mutate(ground_ice_labels = ifelse(ground_ice_3==0, "Low", 
                                    ifelse(ground_ice_3==1, "Medium", "High"))) %>%
  mutate(ground_ice_labels = as.factor(ground_ice_labels)) %>%
  mutate(ground_ice_labels = factor(ground_ice_labels, levels = c("Low", "Medium", "High")))

thaw_train_results <- cbind(thaw_train_results, ground_ice_3_probs)

# look at confusion matrix
cm <- confusionMatrix(thaw_train_results$ground_ice_pred, thaw_train_results$ground_ice_labels, mode = "everything")

# > cm
# Confusion Matrix and Statistics
# 
#                     Reference
# Prediction    Low   Medium   High
#     Low    182468       47   3512
#     Medium     91   227379  30827
#     High     2837     4962 189191
# 
# Overall Statistics
# 
# Accuracy : 0.9341          
# 95% CI : (0.9335, 0.9347)
# No Information Rate : 0.3624          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.9006          
# 
# Mcnemar's Test P-Value : < 2.2e-16       
# 
# Statistics by Class:
# 
#                      Class: Low Class: Medium Class: High
# Sensitivity              0.9842        0.9784      0.8464
# Specificity              0.9922        0.9244      0.9813
# Pos Pred Value           0.9809        0.8803      0.9604
# Neg Pred Value           0.9936        0.9869      0.9227
# Precision                0.9809        0.8803      0.9604
# Recall                   0.9842        0.9784      0.8464
# F1                       0.9825        0.9268      0.8998
# Prevalence               0.2891        0.3624      0.3486
# Detection Rate           0.2845        0.3546      0.2950
# Detection Prevalence     0.2901        0.4028      0.3072
# Balanced Accuracy        0.9882        0.9514      0.9139

# This is actually not bad at all!  Of course, this is performance on training data set, not the test data set.
# So prepare to be more concerned.
# Without considering different cutoffs,
# the model performs very well on "Low" category, and ok on "Medium" and "High" categories.
# "High" has great precision (if the model says it's "High", it very likely is),
# but could be better on recall (if the observation really is "High", will it get characterised as "High",
# or "Medium").
# So improving F1 measure for the "High" category may be worthwhile. 


# prepare the dataset for searching a cutoff for "High" label that works better for the F1 measure
thaw_train_results <- thaw_train_results %>%
  rename(ground_ice_pred_simple = ground_ice_pred) %>%
  mutate(High_pred = ifelse(High_prob<0.01, 0, 1)) %>%
  mutate(High_pred = as.factor(High_pred)) %>%
  mutate(High_actual = ifelse(ground_ice_labels == "High", 1, 0)) %>%
  mutate(High_actual = as.factor(High_actual))



cm <- confusionMatrix(thaw_train_results$High_pred, thaw_train_results$High_actual, positive = "1")

cm_by_cutoff <- c(0.01, cm$byClass)
names(cm_by_cutoff)[1] <- "Cutoff"

for (cutoff in (2:99)/100) {
  thaw_train_results <- thaw_train_results %>%
    mutate(High_pred = ifelse(High_prob<cutoff, 0, 1)) %>%
    mutate(High_pred = as.factor(High_pred)) 
  cm <- confusionMatrix(thaw_train_results$High_pred, thaw_train_results$High_actual, positive = "1")
  cm_by_cutoff <- rbind(cm_by_cutoff, c(cutoff, cm$byClass))
}

cm_by_cutoff <- as.data.frame(cm_by_cutoff)
rownames(cm_by_cutoff) <- c()
cm_by_cutoff <- cm_by_cutoff %>%
  filter(!is.na(F1))
names(cm_by_cutoff)[1] <- "cutoff"

write.csv(cm_by_cutoff, file="./model stats/Thaw Confusion matrix High stats by cutoff.csv", row.names = FALSE)


cutoff <- cm_by_cutoff$cutoff[which(cm_by_cutoff$F1==max(cm_by_cutoff$F1))]
# > cutoff
# [1] 0.41
save(cutoff, file = "./model stats/thaw_xgb_fit cutoff maximizing F1 for High.RData")

thaw_train_results <- thaw_train_results %>%
  select(-High_pred, -High_actual) %>%
  mutate(ground_ice_pred = ifelse(High_prob >= cutoff, "High",  # call this observation "High", even if something else has higher prob
                                  ifelse(High_prob >= Medium_prob & High_prob >= Low_prob, "High",  # if High_prob is the highest of the three
                                         ifelse(Low_prob >= High_prob & Low_prob >= Medium_prob, "Low", "Medium")))) %>%
  mutate(ground_ice_pred = as.factor(ground_ice_pred)) %>%
  mutate(ground_ice_pred = factor(ground_ice_pred, levels = c("Low", "Medium", "High")))

# look at confusion matrix
cm <- confusionMatrix(thaw_train_results$ground_ice_pred, thaw_train_results$ground_ice_labels, mode = "everything")

# > cm
# Confusion Matrix and Statistics
# 
#                 Reference
# Prediction    Low Medium   High
#     Low    180903     46   2237
#     Medium     91 224162  26834
#     High     4402   8180 194459
# 
# Overall Statistics
# 
# Accuracy : 0.9348          
# 95% CI : (0.9342, 0.9354)
# No Information Rate : 0.3624          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.9017          
# 
# Mcnemar's Test P-Value : < 2.2e-16       
# 
# Statistics by Class:
# 
#                      Class: Low Class: Medium Class: High
# Sensitivity              0.9758        0.9646      0.8699
# Specificity              0.9950        0.9342      0.9699
# Pos Pred Value           0.9875        0.8928      0.9392
# Neg Pred Value           0.9902        0.9789      0.9331
# Precision                0.9875        0.8928      0.9392
# Recall                   0.9758        0.9646      0.8699
# F1                       0.9816        0.9273      0.9033
# Prevalence               0.2891        0.3624      0.3486
# Detection Rate           0.2821        0.3495      0.3032
# Detection Prevalence     0.2856        0.3915      0.3228
# Balanced Accuracy        0.9854        0.9494      0.9199

# All that work barely improved F1 for High class.  Oh well.


#----------------------------------------------------------------
# Feature importance

# feature importance
imp <- xgb.importance(model=thaw_xgb_fit)
write.csv(imp, file="./model stats/Feature importance thaw_xgb_fit.csv", row.names = FALSE)

# cleanup
rm(imp, cm_by_cutoff)

###############################################################################################################

#-----------------------------------------------------------------
# Thaw extent model performance on test data

# Use training data.
ground_ice_3_prob_output <- predict(thaw_xgb_fit, newdata = dtest)

# The output is an array that is three times the length of the training data.
# Sort out the output by type of ground_ice_3 labels
ground_ice_3_probs <- matrix(ground_ice_3_prob_output, 
                             nrow=3, # three classes 
                             ncol = length(ground_ice_3_prob_output)/3)
rm(ground_ice_3_prob_output)

ground_ice_3_probs <- ground_ice_3_probs %>% 
  t() %>%  # transpose
  as.data.frame()
  
names(ground_ice_3_probs) <- c("Low_prob", "Medium_prob", "High_prob")

ground_ice_3_probs <- ground_ice_3_probs %>%
  mutate(ground_ice_pred = ifelse(High_prob >= cutoff, "High",  # call this observation "High", even if something else has higher prob
                                  ifelse(High_prob >= Medium_prob & High_prob >= Low_prob, "High",  # if High_prob is the highest of the three
                                         ifelse(Low_prob >= High_prob & Low_prob >= Medium_prob, "Low", "Medium")))) %>%
  mutate(ground_ice_pred = as.factor(ground_ice_pred)) %>%
  mutate(ground_ice_pred = factor(ground_ice_pred, levels = c("Low", "Medium", "High")))


# combine with actual labels
thaw_test_results <- test_df %>%
  select(XCoord, YCoord, ground_ice_3) %>%
  mutate(ground_ice_labels = ifelse(ground_ice_3==0, "Low", 
                                    ifelse(ground_ice_3==1, "Medium", "High"))) %>%
  mutate(ground_ice_labels = as.factor(ground_ice_labels)) %>%
  mutate(ground_ice_labels = factor(ground_ice_labels, levels = c("Low", "Medium", "High")))

thaw_test_results <- cbind(thaw_test_results, ground_ice_3_probs)
rm(ground_ice_3_probs)

# look at confusion matrix
cm <- confusionMatrix(thaw_test_results$ground_ice_pred, thaw_test_results$ground_ice_labels, mode = "everything")

# > cm
# Confusion Matrix and Statistics
# 
#                 Reference
# Prediction   Low Medium  High
#     Low    67361    178  3188
#     Medium   120  64916 10277
#     High    4529   3871 58116
# 
# Overall Statistics
# 
# Accuracy : 0.8957
# 95% CI : (0.8944, 0.897)
# No Information Rate : 0.3388
# P-Value [Acc > NIR] : < 2.2e-16
# 
# Kappa : 0.8437
# 
# Mcnemar's Test P-Value : < 2.2e-16
# 
# Statistics by Class:
# 
#                      Class: Low Class: Medium Class: High
# Sensitivity              0.9354        0.9413      0.8119
# Specificity              0.9761        0.9276      0.9404
# Pos Pred Value           0.9524        0.8619      0.8737
# Neg Pred Value           0.9672        0.9705      0.9078
# Precision                0.9524        0.8619      0.8737
# Recall                   0.9354        0.9413      0.8119
# F1                       0.9438        0.8999      0.8417
# Prevalence               0.3388        0.3245      0.3368
# Detection Rate           0.3169        0.3054      0.2734
# Detection Prevalence     0.3327        0.3543      0.3129
# Balanced Accuracy        0.9557        0.9344      0.8762



#..........................
# Let's get the AUC-ROC curves for the three categories

thaw_test_results <- thaw_test_results %>%
  mutate(isLow = ifelse(ground_ice_labels == "Low", 1, 0)) %>%
  mutate(isLow = as.factor(isLow)) %>%
  mutate(isMedium = ifelse(ground_ice_labels == "Medium", 1, 0)) %>%
  mutate(isMedium = as.factor(isMedium)) %>%
  mutate(isHigh = ifelse(ground_ice_labels == "High", 1, 0)) %>%
  mutate(isHigh = as.factor(isHigh)) 

# Initialize an array to keep track of AUC-ROC values.  First three are for
# thaw_xgb_fit (low, medium, and high), last three are for CWPP_xgb_fit.
aucrocr = rep(0,6)

#..........
# ROC curves and stats for "Low"

predROC <- prediction(thaw_test_results$Low_prob, thaw_test_results$isLow)

perfROC <- performance(predROC, measure = "prec", x.measure = "rec")
png("./model stats/thaw_xgb_fit Low ROC precision-recall.png")
plot(perfROC, colorize = TRUE)
dev.off()

perfROC <- performance(predROC, measure = "tpr", x.measure = "fpr")
png("./model stats/thaw_xgb_fit Low ROC true positive-false positive.png")
plot(perfROC, colorize = TRUE)
dev.off()

auc_ROCR <- performance(predROC, measure = "auc")
auc_ROCR <- auc_ROCR@y.values[[1]]
aucrocr[1] <- auc_ROCR

# > auc_ROCR
# [1] 0.9940777


#..........
# ROC curves and stats for "Medium"

predROC <- prediction(thaw_test_results$Medium_prob, thaw_test_results$isMedium)

perfROC <- performance(predROC, measure = "prec", x.measure = "rec")
png("./model stats/thaw_xgb_fit Medium ROC precision-recall.png")
plot(perfROC, colorize = TRUE)
dev.off()

perfROC <- performance(predROC, measure = "tpr", x.measure = "fpr")
png("./model stats/thaw_xgb_fit Medium ROC true positive-false positive.png")
plot(perfROC, colorize = TRUE)
dev.off()

auc_ROCR <- performance(predROC, measure = "auc")
auc_ROCR <- auc_ROCR@y.values[[1]]
aucrocr[2] <- auc_ROCR

# > auc_ROCR
# [1] 0.9699598


#..........
# ROC curves and stats for "High"

predROC <- prediction(thaw_test_results$High_prob, thaw_test_results$isHigh)

perfROC <- performance(predROC, measure = "prec", x.measure = "rec")
png("./model stats/thaw_xgb_fit High ROC precision-recall.png")
plot(perfROC, colorize = TRUE)
dev.off()

perfROC <- performance(predROC, measure = "tpr", x.measure = "fpr")
png("./model stats/thaw_xgb_fit High ROC true positive-false positive.png")
plot(perfROC, colorize = TRUE)
dev.off()

auc_ROCR <- performance(predROC, measure = "auc")
auc_ROCR <- auc_ROCR@y.values[[1]]
aucrocr[3] <- auc_ROCR

# > auc_ROCR
# [1] 0.9428774



# cleanup
rm(perfROC, predROC, auc_ROCR)

rm(cm, thaw_xgb_fit, thaw_train_results, thaw_test_results)

rm(dtest, dtrain)


###############################################################################################################
# Now do all that again, but this time for the model that excludes "ice_wedge",
# so this model would be used to predict ground_ice labels outside of thaw expent in the rest
# of the CWPP region.


#.....................................................................

# I am going to use the same hyperparameters on "other CWPP"
# extent, because I don't expect that the lack of "ice wedge" is
# going to make much of a difference on those.

# Now that the data sets are prepared, create xgb.DMatrix data types
train_data_other <- train_df %>%
  select(-ground_ice_3, -XCoord, -YCoord, -folds, -ice_wedge)

dtrain_other <- xgb.DMatrix(data = as.matrix(train_data_other), 
                            label=train_df$ground_ice_3)

test_data_other <- test_df %>%
  select(-ground_ice_3, -XCoord, -YCoord, -ice_wedge)

dtest_other <- xgb.DMatrix(data = as.matrix(test_data_other), 
                           label=test_df$ground_ice_3)

save(train_data_other, test_data_other, 
     file="./process/data prepared for xgboost for other CWPP.RData")

CWPP_xgb_fit  <- xgb.train(
  params = params,
  data = dtrain_other,
  nrounds = 200,
  nthreads=1,
  early_stopping_rounds = 10,
  watchlist = list(val1=dtrain_other, val2=dtest_other), 
  verbose = 1,
  print_every_n = 1
)



# Stopping. Best iteration:
#  [95]	val1-mlogloss:0.186683	val2-mlogloss:0.274082


save(CWPP_xgb_fit, file = "xgboost FNSB-Other CWPP.RData")

rm(train_data_other, test_data_other)


#-----------------------------------------------------------------
# Find threshhold for probability of ground_ice where the
# precision and recall balance -- that is, where the F statistic
# from the confusion matrix is maximal for the "High" value.

dtest <- dtest_other
dtrain <- dtrain_other
rm(dtest_other, dtrain_other)

# Use training data.
ground_ice_3_prob_output <- predict(CWPP_xgb_fit, newdata = dtrain)

# The output is an array that is three times the length of the training data.
# Sort out the output by type of ground_ice_3 labels
ground_ice_3_probs <- matrix(ground_ice_3_prob_output, 
                             nrow=3, # three classes 
                             ncol = length(ground_ice_3_prob_output)/3)
rm(ground_ice_3_prob_output)

ground_ice_3_probs <- ground_ice_3_probs %>%
  t() %>%  # transpose
  as.data.frame()

names(ground_ice_3_probs) <- c("Low_prob", "Medium_prob", "High_prob")

ground_ice_3_probs <- ground_ice_3_probs %>%
  mutate(max_prob = max.col(.,"last")) %>%
  mutate(ground_ice_pred = ifelse(max_prob == 1, "Low", 
                                  ifelse(max_prob == 2, "Medium", "High"))) %>%
  mutate(ground_ice_pred = as.factor(ground_ice_pred)) %>%
  mutate(ground_ice_pred = factor(ground_ice_pred, levels = c("Low", "Medium", "High")))%>%
  select(-max_prob)



# combine with actual labels
CWPP_train_results <- train_df %>%
  select(XCoord, YCoord, ground_ice_3) %>%
  mutate(ground_ice_labels = ifelse(ground_ice_3==0, "Low", 
                                    ifelse(ground_ice_3==1, "Medium", "High"))) %>%
  mutate(ground_ice_labels = as.factor(ground_ice_labels)) %>%
  mutate(ground_ice_labels = factor(ground_ice_labels, levels = c("Low", "Medium", "High")))

CWPP_train_results <- cbind(CWPP_train_results, ground_ice_3_probs)

# look at confusion matrix
cm <- confusionMatrix(CWPP_train_results$ground_ice_pred, CWPP_train_results$ground_ice_labels, mode = "everything")

# > cm
# Confusion Matrix and Statistics
# 
# Reference
# Prediction    Low Medium   High
# Low    181973     56   4031
# Medium    103 226993  31702
# High     3320   5339 187797
# 
# Overall Statistics
# 
# Accuracy : 0.9305          
# 95% CI : (0.9299, 0.9312)
# No Information Rate : 0.3624          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.8952          
# 
# Mcnemar's Test P-Value : < 2.2e-16       
# 
# Statistics by Class:
# 
#                      Class: Low Class: Medium Class: High
# Sensitivity              0.9815        0.9768      0.8401
# Specificity              0.9910        0.9222      0.9793
# Pos Pred Value           0.9780        0.8771      0.9559
# Neg Pred Value           0.9925        0.9859      0.9197
# Precision                0.9780        0.8771      0.9559
# Recall                   0.9815        0.9768      0.8401
# F1                       0.9798        0.9243      0.8943
# Prevalence               0.2891        0.3624      0.3486
# Detection Rate           0.2838        0.3539      0.2928
# Detection Prevalence     0.2901        0.4035      0.3063
# Balanced Accuracy        0.9863        0.9495      0.9097

# Again: not bad, for training data.


# prepare the dataset for searching a cutoff for "High" label that works better for the F1 measure
CWPP_train_results <- CWPP_train_results %>%
  rename(ground_ice_pred_simple = ground_ice_pred) %>%
  mutate(High_pred = ifelse(High_prob<0.01, 0, 1)) %>%
  mutate(High_pred = as.factor(High_pred)) %>%
  mutate(High_actual = ifelse(ground_ice_labels == "High", 1, 0)) %>%
  mutate(High_actual = as.factor(High_actual))



cm <- confusionMatrix(CWPP_train_results$High_pred, CWPP_train_results$High_actual, positive = "1")

cm_by_cutoff <- c(0.01, cm$byClass)
names(cm_by_cutoff)[1] <- "Cutoff"

for (cutoff in (2:99)/100) {
  CWPP_train_results <- CWPP_train_results %>%
    mutate(High_pred = ifelse(High_prob<cutoff, 0, 1)) %>%
    mutate(High_pred = as.factor(High_pred)) 
  cm <- confusionMatrix(CWPP_train_results$High_pred, CWPP_train_results$High_actual, positive = "1")
  cm_by_cutoff <- rbind(cm_by_cutoff, c(cutoff, cm$byClass))
}

cm_by_cutoff <- as.data.frame(cm_by_cutoff)
rownames(cm_by_cutoff) <- c()
cm_by_cutoff <- cm_by_cutoff %>%
  filter(!is.na(F1))
names(cm_by_cutoff)[1] <- "cutoff"

write.csv(cm_by_cutoff, file="./model stats/CWPP Confusion matrix High stats by cutoff.csv", row.names = FALSE)


cutoff <- cm_by_cutoff$cutoff[which(cm_by_cutoff$F1==max(cm_by_cutoff$F1))]
# > cutoff
# [1] 0.41
save(cutoff, file = "./model stats/CWPP_xgb_fit cutoff maximizing F1 for High.RData")

CWPP_train_results <- CWPP_train_results %>%
  select(-High_pred, -High_actual) %>%
  mutate(ground_ice_pred = ifelse(High_prob >= cutoff, "High",  # call this observation "High", even if something else has higher prob
                                  ifelse(High_prob >= Medium_prob & High_prob >= Low_prob, "High",  # if High_prob is the highest of the three
                                         ifelse(Low_prob >= High_prob & Low_prob >= Medium_prob, "Low", "Medium")))) %>%
  mutate(ground_ice_pred = as.factor(ground_ice_pred)) %>%
  mutate(ground_ice_pred = factor(ground_ice_pred, levels = c("Low", "Medium", "High")))

# look at confusion matrix
cm <- confusionMatrix(CWPP_train_results$ground_ice_pred, CWPP_train_results$ground_ice_labels, mode = "everything")

# > cm
# Confusion Matrix and Statistics
# 
#                Reference
# Prediction    Low Medium   High
#     Low    180194     56   2629
#     Medium    103 223648  27764
#     High     5099   8684 193137
# 
# Overall Statistics
# 
# Accuracy : 0.9309          
# 95% CI : (0.9302, 0.9315)
# No Information Rate : 0.3624          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.8957          
# 
# Mcnemar's Test P-Value : < 2.2e-16       
# 
# Statistics by Class:
# 
#                      Class: Low Class: Medium Class: High
# Sensitivity              0.9719        0.9624      0.8640
# Specificity              0.9941        0.9319      0.9670
# Pos Pred Value           0.9853        0.8892      0.9334
# Neg Pred Value           0.9887        0.9776      0.9300
# Precision                0.9853        0.8892      0.9334
# Recall                   0.9719        0.9624      0.8640
# F1                       0.9786        0.9244      0.8974
# Prevalence               0.2891        0.3624      0.3486
# Detection Rate           0.2810        0.3487      0.3012
# Detection Prevalence     0.2852        0.3922      0.3227
# Balanced Accuracy        0.9830        0.9471      0.9155

# No practical difference, and in fact got a little worse, but not substantially so.  Will stick with this method
# for consistency.


#----------------------------------------------------------------
# Feature importance

imp <- xgb.importance(model=CWPP_xgb_fit)
write.csv(imp, file="./model stats/Feature importance CWPP_xgb_fit.csv", row.names = FALSE)

# cleanup
rm(imp, cm_by_cutoff)

###############################################################################################################

#-----------------------------------------------------------------
# CWPP extent model performance on test data

# Use training data.
ground_ice_3_prob_output <- predict(CWPP_xgb_fit, newdata = dtest)

# The output is an array that is three times the length of the training data.
# Sort out the output by type of ground_ice_3 labels
ground_ice_3_probs <- matrix(ground_ice_3_prob_output, 
                             nrow=3, # three classes 
                             ncol = length(ground_ice_3_prob_output)/3)
rm(ground_ice_3_prob_output)

ground_ice_3_probs <- ground_ice_3_probs %>% 
  t() %>%  # transpose
  as.data.frame()

  
names(ground_ice_3_probs) <- c("Low_prob", "Medium_prob", "High_prob")

ground_ice_3_probs <- ground_ice_3_probs %>%
  mutate(ground_ice_pred = ifelse(High_prob >= cutoff, "High",  # call this observation "High", even if something else has higher prob
                                  ifelse(High_prob >= Medium_prob & High_prob >= Low_prob, "High",  # if High_prob is the highest of the three
                                         ifelse(Low_prob >= High_prob & Low_prob >= Medium_prob, "Low", "Medium")))) %>%
  mutate(ground_ice_pred = as.factor(ground_ice_pred)) %>%
  mutate(ground_ice_pred = factor(ground_ice_pred, levels = c("Low", "Medium", "High")))


# combine with actual labels
CWPP_test_results <- test_df %>%
  select(XCoord, YCoord, ground_ice_3) %>%
  mutate(ground_ice_labels = ifelse(ground_ice_3==0, "Low", 
                                    ifelse(ground_ice_3==1, "Medium", "High"))) %>%
  mutate(ground_ice_labels = as.factor(ground_ice_labels)) %>%
  mutate(ground_ice_labels = factor(ground_ice_labels, levels = c("Low", "Medium", "High")))

CWPP_test_results <- cbind(CWPP_test_results, ground_ice_3_probs)
rm(ground_ice_3_probs)

# look at confusion matrix
cm <- confusionMatrix(CWPP_test_results$ground_ice_pred, CWPP_test_results$ground_ice_labels, mode = "everything")

# > cm
# Confusion Matrix and Statistics
# 
# Reference
# Prediction   Low Medium  High
# Low    67012    193  3299
# Medium   124  64919 10244
# High    4874   3853 58038
# 
# Overall Statistics
# 
# Accuracy : 0.8937         
# 95% CI : (0.8924, 0.895)
# No Information Rate : 0.3388         
# P-Value [Acc > NIR] : < 2.2e-16      
# 
# Kappa : 0.8407         
# 
# Mcnemar's Test P-Value : < 2.2e-16      
# 
# Statistics by Class:
# 
#                      Class: Low Class: Medium Class: High
# Sensitivity              0.9306        0.9413      0.8108
# Specificity              0.9752        0.9278      0.9381
# Pos Pred Value           0.9505        0.8623      0.8693
# Neg Pred Value           0.9648        0.9705      0.9071
# Precision                0.9505        0.8623      0.8693
# Recall                   0.9306        0.9413      0.8108
# F1                       0.9404        0.9001      0.8390
# Prevalence               0.3388        0.3245      0.3368
# Detection Rate           0.3153        0.3054      0.2730
# Detection Prevalence     0.3317        0.3542      0.3141
# Balanced Accuracy        0.9529        0.9346      0.8744


#..........................
# Let's get the AUC-ROC curves for the three categories

CWPP_test_results <- CWPP_test_results %>%
  mutate(isLow = ifelse(ground_ice_labels == "Low", 1, 0)) %>%
  mutate(isLow = as.factor(isLow)) %>%
  mutate(isMedium = ifelse(ground_ice_labels == "Medium", 1, 0)) %>%
  mutate(isMedium = as.factor(isMedium)) %>%
  mutate(isHigh = ifelse(ground_ice_labels == "High", 1, 0)) %>%
  mutate(isHigh = as.factor(isHigh)) 


#..........
# ROC curves and stats for "Low"

predROC <- prediction(CWPP_test_results$Low_prob, CWPP_test_results$isLow)

perfROC <- performance(predROC, measure = "prec", x.measure = "rec")
png("./model stats/CWPP_xgb_fit Low ROC precision-recall.png")
plot(perfROC, colorize = TRUE)
dev.off()

perfROC <- performance(predROC, measure = "tpr", x.measure = "fpr")
png("./model stats/CWPP_xgb_fit Low ROC true positive-false positive.png")
plot(perfROC, colorize = TRUE)
dev.off()

auc_ROCR <- performance(predROC, measure = "auc")
auc_ROCR <- auc_ROCR@y.values[[1]]
aucrocr[4] <- auc_ROCR

# > auc_ROCR
# [1] 0.9937379


#..........
# ROC curves and stats for "Medium"

predROC <- prediction(CWPP_test_results$Medium_prob, CWPP_test_results$isMedium)

perfROC <- performance(predROC, measure = "prec", x.measure = "rec")
png("./model stats/CWPP_xgb_fit Medium ROC precision-recall.png")
plot(perfROC, colorize = TRUE)
dev.off()

perfROC <- performance(predROC, measure = "tpr", x.measure = "fpr")
png("./model stats/CWPP_xgb_fit Medium ROC true positive-false positive.png")
plot(perfROC, colorize = TRUE)
dev.off()

auc_ROCR <- performance(predROC, measure = "auc")
auc_ROCR <- auc_ROCR@y.values[[1]]
aucrocr[5] <- auc_ROCR

# > auc_ROCR
# [1] 0.970397


#..........
# ROC curves and stats for "High"

predROC <- prediction(CWPP_test_results$High_prob, CWPP_test_results$isHigh)

perfROC <- performance(predROC, measure = "prec", x.measure = "rec")
png("./model stats/CWPP_xgb_fit High ROC precision-recall.png")
plot(perfROC, colorize = TRUE)
dev.off()

perfROC <- performance(predROC, measure = "tpr", x.measure = "fpr")
png("./model stats/CWPP_xgb_fit High ROC true positive-false positive.png")
plot(perfROC, colorize = TRUE)
dev.off()

auc_ROCR <- performance(predROC, measure = "auc")
auc_ROCR <- auc_ROCR@y.values[[1]]
aucrocr[6] <- auc_ROCR

# > auc_ROCR
# [1] 0.9425134



names(aucrocr) <- c("thaw Low", "thaw Medium", "thaw High", "CWPP Low", "CWPP Medium", "CWPP High")

# > aucrocr
# thaw Low thaw Medium   thaw High    CWPP Low CWPP Medium   CWPP High 
# 0.9940777   0.9699598   0.9428774   0.9937379   0.9703970   0.9425134 


save(aucrocr, file = "./model stats/AUC-ROCR values for both models.RData")

# cleanup
rm(aucrocr)
rm(perfROC, predROC, auc_ROCR)

rm(cm, CWPP_xgb_fit, CWPP_train_results, CWPP_test_results)

rm(dtest, dtrain)

####################################################################################



