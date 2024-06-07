library(rpart)
library(rpart.plot)
library(dplyr)
library(caret)
library(Metrics)
library(gbm)

#read in the data
us <- read.csv('C:\\Users\\books\\OneDrive\\Documents\\DS Projects\\Environmental Abuses on Human Health\\FinalCleanedUSData.csv')
summary(us)
str(us)
dim(us)

#scale the data
us <- us[, -1]
str(us)
scale(us)


#Model Analysis Using Cross Validation
#training and testing
#death rate
set.seed(123)

trainIndex_dr <- createDataPartition(us$Death.rate, p = 0.7, list = FALSE)
train_data_dr <- us[trainIndex_dr, ]
test_data_dr <- us[-trainIndex_dr, ]


#linear regression using cross validation
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

mod_dr <- train(Death.rate ~ ., data = train_data_dr, method = "lm", trControl = train_control)

# Print the model to see the results
print(mod_dr)

# Check the detailed results of the cross-validation
mod_dr$results

pred_dr <- predict(mod_dr, newdata = test_data_dr)

# Calculate performance metrics
actuals <- test_data_dr$Death.rate
mae_value <- mean(abs(pred_dr - actuals))
mse_value <- mean((pred_dr - actuals)^2)
rmse_value <- sqrt(mse_value)
r_squared <- cor(pred_dr, actuals)^2

# Print the evaluation metrics
cat("MAE: ", mae_value, "\n")
cat("MSE: ", mse_value, "\n")
cat("RMSE: ", rmse_value, "\n")
cat("R-squared: ", r_squared, "\n")

#regression trees using cross validation
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

mod_dr2 <- train(Death.rate ~ ., data = train_data_dr, method = "rpart", trControl = train_control)

#plot the final tree
plot(mod_dr2$finalModel)
text(mod_dr2$finalModel, use.n = TRUE)

pred_dr2 <- predict(mod_dr2, newdata = test_data_dr)

# Calculate performance metrics
actuals <- test_data_dr$Death.rate
mae_value <- mean(abs(pred_dr2 - actuals))
mse_value <- mean((pred_dr2 - actuals)^2)
rmse_value <- sqrt(mse_value)
r_squared <- cor(pred_dr2, actuals)^2

# Print the evaluation metrics
cat("MAE: ", mae_value, "\n")
cat("MSE: ", mse_value, "\n")
cat("RMSE: ", rmse_value, "\n")
cat("R-squared: ", r_squared, "\n")

#random forests using cross validation
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

mod_dr3 <- train(Death.rate ~ ., data = train_data_dr, method = "rf", trControl = train_control)

#plot variable importance
varImp(mod_dr3)

pred_dr3 <- predict(mod_dr3, newdata = test_data_dr)

# Calculate performance metrics
actuals <- test_data_dr$Death.rate
mae_value <- mean(abs(pred_dr3 - actuals))
mse_value <- mean((pred_dr3 - actuals)^2)
rmse_value <- sqrt(mse_value)
r_squared <- cor(pred_dr3, actuals)^2

# Print the evaluation metrics
cat("MAE: ", mae_value, "\n")
cat("MSE: ", mse_value, "\n")
cat("RMSE: ", rmse_value, "\n")
cat("R-squared: ", r_squared, "\n")

#gradient boosting regression using cross validation
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

mod_dr4 <- train(Death.rate ~ ., data = train_data_dr, method = "gbm", trControl = train_control)

#plot variable importance
varImp(mod_dr4)

pred_dr4 <- predict(mod_dr4, newdata = test_data_dr)

# Calculate performance metrics
actuals <- test_data_dr$Death.rate
mae_value <- mean(abs(pred_dr4 - actuals))
mse_value <- mean((pred_dr4 - actuals)^2)
rmse_value <- sqrt(mse_value)
r_squared <- cor(pred_dr4, actuals)^2

# Print the evaluation metrics
cat("MAE: ", mae_value, "\n")
cat("MSE: ", mse_value, "\n")
cat("RMSE: ", rmse_value, "\n")
cat("R-squared: ", r_squared, "\n")





#Model Analysis Using Cross Validation
#training and testing
#life expectancy
set.seed(345)

trainIndex_le <- createDataPartition(us$Life.expectancy, p = 0.7, list = FALSE)
train_data_le <- us[trainIndex_le, ]
test_data_le <- us[-trainIndex_le, ]


#linear regression using cross validation
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

mod_le <- train(Life.expectancy ~ ., data = train_data_le, method = "lm", trControl = train_control)

# Check the detailed results of the cross-validation
mod_le$results
summary(mod_le)

pred_le <- predict(mod_le, newdata = test_data_le)

# Calculate performance metrics
actuals <- test_data_le$Life.expectancy
mae_value <- mean(abs(pred_le - actuals))
mse_value <- mean((pred_le - actuals)^2)
rmse_value <- sqrt(mse_value)
r_squared <- cor(pred_le, actuals)^2

# Print the evaluation metrics
cat("MAE: ", mae_value, "\n")
cat("MSE: ", mse_value, "\n")
cat("RMSE: ", rmse_value, "\n")
cat("R-squared: ", r_squared, "\n")

#regression trees using cross validation
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

mod_le2 <- train(Life.expectancy ~ ., data = train_data_le, method = "rpart", trControl = train_control)

#plot the final tree
plot(mod_le2$finalModel)
text(mod_le2$finalModel, use.n = TRUE)

pred_le2 <- predict(mod_le2, newdata = test_data_le)

# Calculate performance metrics
actuals <- test_data_le$Life.expectancy
mae_value <- mean(abs(pred_le2 - actuals))
mse_value <- mean((pred_le2 - actuals)^2)
rmse_value <- sqrt(mse_value)
r_squared <- cor(pred_le2, actuals)^2

# Print the evaluation metrics
cat("MAE: ", mae_value, "\n")
cat("MSE: ", mse_value, "\n")
cat("RMSE: ", rmse_value, "\n")
cat("R-squared: ", r_squared, "\n")

#random forests using cross validation
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

mod_le3 <- train(Life.expectancy ~ ., data = train_data_le, method = "rf", trControl = train_control)

#plot variable importance
varImp(mod_le3)

pred_le3 <- predict(mod_le3, newdata = test_data_le)

# Calculate performance metrics
actuals <- test_data_le$Life.expectancy
mae_value <- mean(abs(pred_le3 - actuals))
mse_value <- mean((pred_le3 - actuals)^2)
rmse_value <- sqrt(mse_value)
r_squared <- cor(pred_le3, actuals)^2

# Print the evaluation metrics
cat("MAE: ", mae_value, "\n")
cat("MSE: ", mse_value, "\n")
cat("RMSE: ", rmse_value, "\n")
cat("R-squared: ", r_squared, "\n")

#gradient boosting regression using cross validation
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

mod_le4 <- train(Life.expectancy ~ ., data = train_data_le, method = "gbm", trControl = train_control)

#plot variable importance
varImp(mod_le4)

pred_le4 <- predict(mod_le4, newdata = test_data_le)

# Calculate performance metrics
actuals <- test_data_le$Life.expectancy
mae_value <- mean(abs(pred_le4 - actuals))
mse_value <- mean((pred_le4 - actuals)^2)
rmse_value <- sqrt(mse_value)
r_squared <- cor(pred_le4, actuals)^2

# Print the evaluation metrics
cat("MAE: ", mae_value, "\n")
cat("MSE: ", mse_value, "\n")
cat("RMSE: ", rmse_value, "\n")
cat("R-squared: ", r_squared, "\n")






#Model Analysis Using Cross Validation
#training and testing
#fertility rate
set.seed(456)

trainIndex_fr <- createDataPartition(us$Fertility.rate, p = 0.7, list = FALSE)
train_data_fr <- us[trainIndex_fr, ]
test_data_fr <- us[-trainIndex_fr, ]


#linear regression using cross validation
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

mod_fr <- train(Fertility.rate ~ ., data = train_data_fr, method = "lm", trControl = train_control)

# Check the detailed results of the cross-validation
mod_fr$results

pred_fr <- predict(mod_fr, newdata = test_data_fr)

# Calculate performance metrics
actuals <- test_data_fr$Fertility.rate
mae_value <- mean(abs(pred_fr - actuals))
mse_value <- mean((pred_fr - actuals)^2)
rmse_value <- sqrt(mse_value)
r_squared <- cor(pred_fr, actuals)^2

# Print the evaluation metrics
cat("MAE: ", mae_value, "\n")
cat("MSE: ", mse_value, "\n")
cat("RMSE: ", rmse_value, "\n")
cat("R-squared: ", r_squared, "\n")

#regression trees using cross validation
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

mod_fr2 <- train(Fertility.rate ~ ., data = train_data_fr, method = "rpart", trControl = train_control)

#plot the final tree
plot(mod_fr2$finalModel)
text(mod_fr2$finalModel, use.n = TRUE)

pred_fr2 <- predict(mod_fr2, newdata = test_data_fr)

# Calculate performance metrics
actuals <- test_data_fr$Fertility.rate
mae_value <- mean(abs(pred_fr2 - actuals))
mse_value <- mean((pred_fr2 - actuals)^2)
rmse_value <- sqrt(mse_value)
r_squared <- cor(pred_fr2, actuals)^2

# Print the evaluation metrics
cat("MAE: ", mae_value, "\n")
cat("MSE: ", mse_value, "\n")
cat("RMSE: ", rmse_value, "\n")
cat("R-squared: ", r_squared, "\n")

#random forests using cross validation
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

mod_fr3 <- train(Fertility.rate ~ ., data = train_data_fr, method = "rf", trControl = train_control)

#plot variable importance
varImp(mod_fr3)

pred_fr3 <- predict(mod_fr3, newdata = test_data_fr)

# Calculate performance metrics
actuals <- test_data_fr$Fertility.rate
mae_value <- mean(abs(pred_fr3 - actuals))
mse_value <- mean((pred_fr3 - actuals)^2)
rmse_value <- sqrt(mse_value)
r_squared <- cor(pred_fr3, actuals)^2

# Print the evaluation metrics
cat("MAE: ", mae_value, "\n")
cat("MSE: ", mse_value, "\n")
cat("RMSE: ", rmse_value, "\n")
cat("R-squared: ", r_squared, "\n")

#gradient boosting regression using cross validation
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

mod_fr4 <- train(Fertility.rate ~ ., data = train_data_fr, method = "gbm", trControl = train_control)

#plot variable importance
varImp(mod_fr4)

pred_fr4 <- predict(mod_fr4, newdata = test_data_fr)

# Calculate performance metrics
actuals <- test_data_fr$Fertility.rate
mae_value <- mean(abs(pred_fr4 - actuals))
mse_value <- mean((pred_fr4 - actuals)^2)
rmse_value <- sqrt(mse_value)
r_squared <- cor(pred_fr4, actuals)^2

# Print the evaluation metrics
cat("MAE: ", mae_value, "\n")
cat("MSE: ", mse_value, "\n")
cat("RMSE: ", rmse_value, "\n")
cat("R-squared: ", r_squared, "\n")







#Model Analysis Using Cross Validation
#training and testing
#mortality from cvd, cancer, diabetes, or crd
set.seed(678)

trainIndex_mort <- createDataPartition(us$Diabetes.prevalence, p = 0.7, list = FALSE)
train_data_mort <- us[trainIndex_mort, ]
test_data_mort <- us[-trainIndex_mort, ]


#linear regression using cross validation
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

mod_mort <- train(Diabetes.prevalence ~ ., data = train_data_mort, method = "lm", trControl = train_control)

# Check the detailed results of the cross-validation
mod_mort$results

pred_mort <- predict(mod_mort, newdata = test_data_mort)

# Calculate performance metrics
actuals <- test_data_mort$Diabetes.prevalence
mae_value <- mean(abs(pred_mort - actuals))
mse_value <- mean((pred_mort - actuals)^2)
rmse_value <- sqrt(mse_value)
r_squared <- cor(pred_mort, actuals)^2

# Print the evaluation metrics
cat("MAE: ", mae_value, "\n")
cat("MSE: ", mse_value, "\n")
cat("RMSE: ", rmse_value, "\n")
cat("R-squared: ", r_squared, "\n")

#regression trees using cross validation
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

mod_mort2 <- train(Diabetes.prevalence ~ ., data = train_data_mort, method = "rpart", trControl = train_control)

#plot the final tree
plot(mod_mort2$finalModel)
text(mod_mort2$finalModel, use.n = TRUE)

pred_mort2 <- predict(mod_mort2, newdata = test_data_mort)

# Calculate performance metrics
actuals <- test_data_mort$Diabetes.prevalence
mae_value <- mean(abs(pred_mort2 - actuals))
mse_value <- mean((pred_mort2 - actuals)^2)
rmse_value <- sqrt(mse_value)
r_squared <- cor(pred_mort2, actuals)^2

# Print the evaluation metrics
cat("MAE: ", mae_value, "\n")
cat("MSE: ", mse_value, "\n")
cat("RMSE: ", rmse_value, "\n")
cat("R-squared: ", r_squared, "\n")

#random forests using cross validation
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

mod_mort3 <- train(Diabetes.prevalence ~ ., data = train_data_mort, method = "rf", trControl = train_control)

#plot variable importance
varImp(mod_mort3)

pred_mort3 <- predict(mod_mort3, newdata = test_data_mort)

# Calculate performance metrics
actuals <- test_data_mort$Diabetes.prevalence
mae_value <- mean(abs(pred_mort3 - actuals))
mse_value <- mean((pred_mort3 - actuals)^2)
rmse_value <- sqrt(mse_value)
r_squared <- cor(pred_mort3, actuals)^2

# Print the evaluation metrics
cat("MAE: ", mae_value, "\n")
cat("MSE: ", mse_value, "\n")
cat("RMSE: ", rmse_value, "\n")
cat("R-squared: ", r_squared, "\n")

#gradient boosting regression using cross validation
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

mod_mort4 <- train(Diabetes.prevalence ~ ., data = train_data_mort, method = "gbm", trControl = train_control)

#plot variable importance
varImp(mod_mort4)

pred_mort4 <- predict(mod_mort4, newdata = test_data_mort)

# Calculate performance metrics
actuals <- test_data_mort$Diabetes.prevalence
mae_value <- mean(abs(pred_mort4 - actuals))
mse_value <- mean((pred_mort4 - actuals)^2)
rmse_value <- sqrt(mse_value)
r_squared <- cor(pred_mort4, actuals)^2

# Print the evaluation metrics
cat("MAE: ", mae_value, "\n")
cat("MSE: ", mse_value, "\n")
cat("RMSE: ", rmse_value, "\n")
cat("R-squared: ", r_squared, "\n")

