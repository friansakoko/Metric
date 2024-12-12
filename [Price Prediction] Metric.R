# Load the data from the URL
url <- "https://raw.githubusercontent.com/friansakoko/Regression/refs/heads/main/housing_price_dataset.csv"
model_data <- read.csv(url)
View(model_data)

library(ggplot2)
library(caret)
library(dplyr)
library(reshape2)
library(glmnet)

# Handle missing values
model_data <- na.omit(model_data) 

# Convert Neighborhood to factor number
model_data$Neighborhood <- as.factor(model_data$Neighborhood)
model_data$Neighborhood <- as.numeric(model_data$Neighborhood) 

# Prepare predictor variables
Data_subset <- model_data[, c("SquareFeet", "Bedrooms", "Bathrooms", "Neighborhood", "YearBuilt")]

# Create dummy variables for categorical predictors
dummies <- dummyVars(~ ., data = Data_subset)
Data_dummies <- predict(dummies, newdata = Data_subset)
View(Data_dummies)

# Normalize predictors
preProcValues <- preProcess(Data_dummies, method = c("range")) 
Data_normalized <- as.data.frame(predict(preProcValues, Data_dummies))

# Add target variable
Data_normalized$Price <- model_data$Price/1000

# Train-test split
set.seed(123)
trainIndex <- createDataPartition(Data_normalized$Price, p = 0.7, list = FALSE)
train_data <- Data_normalized[trainIndex, ]
test_data <- Data_normalized[-trainIndex, ]

#A. Build linear regression model, Predictors 1-----
lm_model <- lm(Price ~ SquareFeet + Bedrooms + Bathrooms + Neighborhood + YearBuilt, data = train_data) 

# Display the model summary
summary(lm_model)

# Testing
# Make predictions on the testing dataset
test_data$Predicted <- predict(lm_model, newdata = test_data)

# Check the first few rows of the test data with predictions
head(test_data)

# Model Evaluation
# Compare actual vs. predicted values
results <- data.frame(
  Actual = test_data$Price,
  Predicted = test_data$Predicted
)
head(results)

# Calculate Mean Absolute Error (MAE)
mae_a1 <- mean(abs(results$Actual - results$Predicted))
print(paste("MAE of Linear Regression - Predictor 1:", round(mae_a1, 4)))

# Calculate Mean Squared Error (MSE)
mse_a1 <- mean((results$Actual - results$Predicted)^2)
print(paste("MSE of Linear Regression - Predictor 1:", round(mse_a1, 4)))

# Calculate Root Mean Squared Error (RMSE)
rmse_a1 <- sqrt(mean((results$Actual - results$Predicted)^2))
print(paste("RMSE of Linear Regression - Predictor 1:", round(rmse_a1, 4)))

# Model Equation
coefficients <- coef(lm_model)
intercept <- coefficients[1]
slopes <- coefficients[-1]

# Format the equation
equation <- paste0("Price = ", round(intercept, 4), 
                   " + ", paste0(round(slopes, 4), " * ", names(slopes), collapse = " + "))
cat("Regression Equation of Linear Regression:\n")
cat(equation, "\n")

#A. Build linear regression model, Predictors 2-----
lm_model <- lm(Price ~ SquareFeet + Neighborhood, data = train_data)

# Display the model summary
summary(lm_model)

# Testing
# Make predictions on the testing dataset
test_data$Predicted <- predict(lm_model, newdata = test_data)

# Check the first few rows of the test data with predictions
head(test_data)

# Model Evaluation
# Compare actual vs. predicted values
results <- data.frame(
  Actual = test_data$Price,
  Predicted = test_data$Predicted
)
head(results)

# Calculate Mean Absolute Error (MAE)
mae_a2 <- mean(abs(results$Actual - results$Predicted))
print(paste("MAE of Linear Regression - Predictor 2:", round(mae_a2, 4)))

# Calculate Mean Squared Error (MSE)
mse_a2 <- mean((results$Actual - results$Predicted)^2)
print(paste("MSE of Linear Regression - Predictor 2:", round(mse_a2, 4)))

# Calculate Root Mean Squared Error (RMSE)
rmse_a2 <- sqrt(mean((results$Actual - results$Predicted)^2))
print(paste("RMSE of Linear Regression - Predictor 2:", round(rmse_a2, 4)))

# Model Equation
coefficients <- coef(lm_model)
intercept <- coefficients[1]
slopes <- coefficients[-1]

# Format the equation
equation <- paste0("Price = ", round(intercept, 4), 
                   " + ", paste0(round(slopes, 4), " * ", names(slopes), collapse = " + "))
cat("Regression Equation of Linear Regression:\n")
cat(equation, "\n")

#A. Build linear regression model, Predictors 3-----
lm_model <- lm(Price ~ SquareFeet + YearBuilt, data = train_data)

# Display the model summary
summary(lm_model)

# Testing
# Make predictions on the testing dataset
test_data$Predicted <- predict(lm_model, newdata = test_data)

# Check the first few rows of the test data with predictions
head(test_data)

# Model Evaluation
# Compare actual vs. predicted values
results <- data.frame(
  Actual = test_data$Price,
  Predicted = test_data$Predicted
)
head(results)

# Calculate Mean Absolute Error (MAE)
mae_a3 <- mean(abs(results$Actual - results$Predicted))
print(paste("MAE of Linear Regression - Predictor 3:", round(mae_a3, 4)))

# Calculate Mean Squared Error (MSE)
mse_a3 <- mean((results$Actual - results$Predicted)^2)
print(paste("MSE of Linear Regression - Predictor 3:", round(mse_a3, 4)))

# Calculate Root Mean Squared Error (RMSE)
rmse_a3 <- sqrt(mean((results$Actual - results$Predicted)^2))
print(paste("RMSE of Linear Regression - Predictor 3:", round(rmse_a3, 4)))

# Model Equation
coefficients <- coef(lm_model)
intercept <- coefficients[1]
slopes <- coefficients[-1]

# Format the equation
equation <- paste0("Price = ", round(intercept, 4), 
                   " + ", paste0(round(slopes, 4), " * ", names(slopes), collapse = " + "))
cat("Regression Equation of Linear Regression:\n")
cat(equation, "\n")

#B. Build Polynomial regression model, Predictors 1-----
lm_model <- lm(Price ~ poly(SquareFeet, 2) + Bedrooms + Bathrooms + poly(Neighborhood, 2) + poly(YearBuilt, 2), data = train_data)  # poly(Predictor1, 2) + poly(Predictor2, 2) + Predictor3

# Display the model summary
summary(lm_model)

# Testing
# Make predictions on the testing dataset
test_data$Predicted <- predict(lm_model, newdata = test_data)

# Check the first few rows of the test data with predictions
head(test_data)

# Model Evaluation
# Compare actual vs. predicted values
results <- data.frame(
  Actual = test_data$Price,
  Predicted = test_data$Predicted
)
head(results)

# Calculate Mean Absolute Error (MAE)
mae_b1 <- mean(abs(results$Actual - results$Predicted))
print(paste("MAE of Polynomial Regression 1- Predictor 1:", round(mae_b1, 4)))

# Calculate Mean Squared Error (MSE)
mse_b1 <- mean((results$Actual - results$Predicted)^2)
print(paste("MSE of Polynomial Regression 1 - Predictor 1:", round(mse_b1, 4)))

# Calculate Root Mean Squared Error (RMSE)
rmse_b1 <- sqrt(mean((results$Actual - results$Predicted)^2))
print(paste("RMSE of Polynomial Regression 1 - Predictor 1:", round(rmse_b1, 4)))

# Model Equation
coefficients <- coef(lm_model)
intercept <- coefficients[1]
slopes <- coefficients[-1]

# Format the equation
equation <- paste0("Price = ", round(intercept, 4), 
                   " + ", paste0(round(slopes, 4), " * ", names(slopes), collapse = " + "))
cat("Regression Equation of Linear Regression:\n")
cat(equation, "\n")

#B. Build Polynomial regression model, Predictors 2-----
lm_model <- lm(Price ~ poly(SquareFeet, 2) +  poly(Neighborhood, 2), data = train_data)  # poly(Predictor1, 2) + poly(Predictor2, 2) + Predictor3

# Display the model summary
summary(lm_model)

# Testing
# Make predictions on the testing dataset
test_data$Predicted <- predict(lm_model, newdata = test_data)

# Check the first few rows of the test data with predictions
head(test_data)

# Model Evaluation
# Compare actual vs. predicted values
results <- data.frame(
  Actual = test_data$Price,
  Predicted = test_data$Predicted
)
head(results)

# Calculate Mean Absolute Error (MAE)
mae_b2 <- mean(abs(results$Actual - results$Predicted))
print(paste("MAE of Polynomial Regression 1- Predictor 2:", round(mae_b2, 4)))

# Calculate Mean Squared Error (MSE)
mse_b2 <- mean((results$Actual - results$Predicted)^2)
print(paste("MSE of Polynomial Regression 1 - Predictor 2:", round(mse_b2, 4)))

# Calculate Root Mean Squared Error (RMSE)
rmse_b2 <- sqrt(mean((results$Actual - results$Predicted)^2))
print(paste("RMSE of Polynomial Regression 1 - Predictor 2:", round(rmse_b2, 4)))

# Model Equation
coefficients <- coef(lm_model)
intercept <- coefficients[1]
slopes <- coefficients[-1]

# Format the equation
equation <- paste0("Price = ", round(intercept, 4), 
                   " + ", paste0(round(slopes, 4), " * ", names(slopes), collapse = " + "))
cat("Regression Equation of Linear Regression:\n")
cat(equation, "\n")



#B. Build Polynomial regression model, Predictors 3-----
lm_model <- lm(Price ~ poly(SquareFeet, 2) +  poly(YearBuilt, 2), data = train_data)  # poly(Predictor1, 2) + poly(Predictor2, 2) + Predictor3

# Display the model summary
summary(lm_model)

# Testing
# Make predictions on the testing dataset
test_data$Predicted <- predict(lm_model, newdata = test_data)

# Check the first few rows of the test data with predictions
head(test_data)

# Model Evaluation
# Compare actual vs. predicted values
results <- data.frame(
  Actual = test_data$Price,
  Predicted = test_data$Predicted
)
head(results)

# Calculate Mean Absolute Error (MAE)
mae_b3 <- mean(abs(results$Actual - results$Predicted))
print(paste("MAE of Polynomial Regression 1- Predictor 3:", round(mae_b3, 4)))

# Calculate Mean Squared Error (MSE)
mse_b3 <- mean((results$Actual - results$Predicted)^2)
print(paste("MSE of Polynomial Regression 1 - Predictor 3:", round(mse_b3, 4)))

# Calculate Root Mean Squared Error (RMSE)
rmse_b3 <- sqrt(mean((results$Actual - results$Predicted)^2))
print(paste("RMSE of Polynomial Regression 1 - Predictor 3:", round(rmse_b3, 4)))

# Model Equation
coefficients <- coef(lm_model)
intercept <- coefficients[1]
slopes <- coefficients[-1]

# Format the equation
equation <- paste0("Price = ", round(intercept, 4), 
                   " + ", paste0(round(slopes, 4), " * ", names(slopes), collapse = " + "))
cat("Regression Equation of Linear Regression:\n")
cat(equation, "\n")


#C. Build Decision tree model, Predictors 1----
library(rpart)
library(rpart.plot)  # For visualizing the tree

# Decision tree model
# Formula(Target ~ Predictor1 + Predictor2 + ...)
dt_model <- rpart(Price ~ SquareFeet + Bedrooms + Bathrooms + Neighborhood + YearBuilt, 
                  data = train_data, 
                  method = "anova")  # Use "anova" for regression, "class" for classification

# Visualize the Decision Tree
rpart.plot(dt_model, type = 2, extra = 101, fallen.leaves = TRUE, main = "Decision Tree")

# Testing
# Make predictions on the testing dataset
test_data$Predicted <- predict(dt_model, newdata = test_data)

# Model Evaluation
# Compare actual vs predicted values for Decision Tree
results <- data.frame(
  Actual = test_data$Price,
  Predicted = test_data$Predicted
)

# Calculate Mean Absolute Error (MAE)
mae_c1 <- mean(abs(results$Actual - results$Predicted))
print(paste("MAE of Decision tree - Predictor 1:", round(mae_c1, 4)))

# Calculate Mean Squared Error (MSE)
mse_c1 <- mean((results$Actual - results$Predicted)^2)
print(paste("MSE of Decision tree - Predictor 1:", round(mse_c1, 4)))

# Calculate Root Mean Squared Error (RMSE)
rmse_c1 <- sqrt(mean((results$Actual - results$Predicted)^2))
print(paste("RMSE of Decision tree - Predictor 1:", round(rmse_c1, 4)))

#C. Build Decision tree model, Predictors 2----
library(rpart)
library(rpart.plot)  # For visualizing the tree

# Decision tree model
# Formula(Target ~ Predictor1 + Predictor2 + ...)
dt_model <- rpart(Price ~ SquareFeet + Neighborhood, 
                  data = train_data, 
                  method = "anova")  # Use "anova" for regression, "class" for classification

# Visualize the Decision Tree
rpart.plot(dt_model, type = 2, extra = 101, fallen.leaves = TRUE, main = "Decision Tree")

# Testing
# Make predictions on the testing dataset
test_data$Predicted <- predict(dt_model, newdata = test_data)

# Model Evaluation
# Compare actual vs predicted values for Decision Tree
results <- data.frame(
  Actual = test_data$Price,
  Predicted = test_data$Predicted
)

# Calculate Mean Absolute Error (MAE)
mae_c2 <- mean(abs(results$Actual - results$Predicted))
print(paste("MAE of Decision tree - Predictor 2:", round(mae_c2, 4)))

# Calculate Mean Squared Error (MSE)
mse_c2 <- mean((results$Actual - results$Predicted)^2)
print(paste("MSE of Decision tree - Predictor 2:", round(mse_c2, 4)))

# Calculate Root Mean Squared Error (RMSE)
rmse_c2 <- sqrt(mean((results$Actual - results$Predicted)^2))
print(paste("RMSE of Decision tree - Predictor 2:", round(rmse_c2, 4)))


#C. Build Decision tree model, Predictors 3----
library(rpart)
library(rpart.plot)  # For visualizing the tree

# Decision tree model
# Formula(Target ~ Predictor1 + Predictor2 + ...)
dt_model <- rpart(Price ~ SquareFeet + YearBuilt, 
                  data = train_data, 
                  method = "anova")  # Use "anova" for regression, "class" for classification

# Visualize the Decision Tree
rpart.plot(dt_model, type = 2, extra = 101, fallen.leaves = TRUE, main = "Decision Tree")

# Testing
# Make predictions on the testing dataset
test_data$Predicted <- predict(dt_model, newdata = test_data)

# Model Evaluation
# Compare actual vs predicted values for Decision Tree
results <- data.frame(
  Actual = test_data$Price,
  Predicted = test_data$Predicted
)

# Calculate Mean Absolute Error (MAE)
mae_c3 <- mean(abs(results$Actual - results$Predicted))
print(paste("MAE of Decision tree - Predictor 3:", round(mae_c3, 4)))

# Calculate Mean Squared Error (MSE)
mse_c3 <- mean((results$Actual - results$Predicted)^2)
print(paste("MSE of Decision tree - Predictor 3:", round(mse_c3, 4)))

# Calculate Root Mean Squared Error (RMSE)
rmse_c3 <- sqrt(mean((results$Actual - results$Predicted)^2))
print(paste("RMSE of Decision tree - Predictor 3:", round(rmse_c3, 4)))



#D. Build Ridge regression model, Predictors 1----
# Prepare predictors and target for training and testing
x_train <- as.matrix(train_data[, c("SquareFeet", "Bedrooms", "Bathrooms", "Neighborhood", "YearBuilt")]) 
y_train <- train_data$Price
x_test <- as.matrix(test_data[, c("SquareFeet", "Bedrooms", "Bathrooms", "Neighborhood", "YearBuilt")])
y_test <- test_data$Price
# Fit Ridge Regression (alpha = 0 for Ridge)
ridge_model <- glmnet(x_train, y_train, alpha = 0)

# Display the model summary
summary(ridge_model)

# Perform cross-validation to find the best lambda
cv_ridge <- cv.glmnet(x_train, y_train, alpha = 0)

# Extract the optimal lambda
optimal_lambda <- cv_ridge$lambda.min
print(paste("Optimal Lambda:", optimal_lambda))

# Testing
# Predict on test data using the optimal lambda
test_data$Predicted_Ridge <- predict(ridge_model, s = optimal_lambda, newx = x_test)

# Check the first few rows with predictions
head(test_data)

# Model Evaluation
# Compare actual vs. predicted values
results <- data.frame(
  Actual = test_data$Price,
  Predicted = test_data$Predicted_Ridge
)
head(results)

# Calculate Mean Absolute Error (MAE)
mae_d1 <- mean(abs(results$Actual - results$s1))
print(paste("MAE of Ridge Regression - Predictor 1:", round(mae_d1, 4)))

# Calculate Mean Squared Error (MSE)
mse_d1 <- mean((results$Actual - results$s1)^2)
print(paste("MSE of Ridge Regression - Predictor 1:", round(mse_d1, 4)))

# Calculate Root Mean Squared Error (RMSE)
rmse_d1 <- sqrt(mean((results$Actual - results$s1)^2))
print(paste("RMSE of Ridge Regression - Predictor 1:", round(rmse_d1, 4)))


# Model Equation
# Extract coefficients at the optimal lambda
coefficients <- coef(ridge_model, s = optimal_lambda)

# Convert to a named vector for easier access
coefficients <- as.vector(coefficients)
names(coefficients) <- rownames(coef(ridge_model, s = optimal_lambda))

# Intercept
intercept <- coefficients[1]

# Coefficients for predictors
slopes <- coefficients[-1]

# Format the equation
equation <- paste0(
  "Price = ", round(intercept, 4), " + ",
  paste0(round(slopes, 4), " * ", names(slopes), collapse = " + ")
)

# Display the equation
cat("Ridge Regression Model Equation:\n")
cat(equation, "\n")

#D. Build Ridge regression model, Predictors 2----
# Prepare predictors and target for training and testing
x_train <- as.matrix(train_data[, c("SquareFeet", "Neighborhood")]) 
y_train <- train_data$Price
x_test <- as.matrix(test_data[, c("SquareFeet", "Neighborhood")])
y_test <- test_data$Price
# Fit Ridge Regression (alpha = 0 for Ridge)
ridge_model <- glmnet(x_train, y_train, alpha = 0)

# Display the model summary
summary(ridge_model)

# Perform cross-validation to find the best lambda
cv_ridge <- cv.glmnet(x_train, y_train, alpha = 0)

# Extract the optimal lambda
optimal_lambda <- cv_ridge$lambda.min
print(paste("Optimal Lambda:", optimal_lambda))

# Testing
# Predict on test data using the optimal lambda
test_data$Predicted_Ridge <- predict(ridge_model, s = optimal_lambda, newx = x_test)

# Check the first few rows with predictions
head(test_data)

# Model Evaluation
# Compare actual vs. predicted values
results <- data.frame(
  Actual = test_data$Price,
  Predicted = test_data$Predicted_Ridge
)
head(results)

# Calculate Mean Absolute Error (MAE)
mae_d2 <- mean(abs(results$Actual - results$s1))
print(paste("MAE of Ridge Regression - Predictor 2:", round(mae_d2, 4)))

# Calculate Mean Squared Error (MSE)
mse_d2 <- mean((results$Actual - results$s1)^2)
print(paste("MSE of Ridge Regression - Predictor 2:", round(mse_d2, 4)))

# Calculate Root Mean Squared Error (RMSE)
rmse_d2 <- sqrt(mean((results$Actual - results$s1)^2))
print(paste("RMSE of Ridge Regression - Predictor 2:", round(rmse_d2, 4)))


# Model Equation
# Extract coefficients at the optimal lambda
coefficients <- coef(ridge_model, s = optimal_lambda)

# Convert to a named vector for easier access
coefficients <- as.vector(coefficients)
names(coefficients) <- rownames(coef(ridge_model, s = optimal_lambda))

# Intercept
intercept <- coefficients[1]

# Coefficients for predictors
slopes <- coefficients[-1]

# Format the equation
equation <- paste0(
  "Price = ", round(intercept, 4), " + ",
  paste0(round(slopes, 4), " * ", names(slopes), collapse = " + ")
)

# Display the equation
cat("Ridge Regression Model Equation:\n")
cat(equation, "\n")


#D. Build Ridge regression model, Predictors 3----
# Prepare predictors and target for training and testing
x_train <- as.matrix(train_data[, c("SquareFeet", "YearBuilt")]) 
y_train <- train_data$Price
x_test <- as.matrix(test_data[, c("SquareFeet", "YearBuilt")])
y_test <- test_data$Price
# Fit Ridge Regression (alpha = 0 for Ridge)
ridge_model <- glmnet(x_train, y_train, alpha = 0)

# Display the model summary
summary(ridge_model)

# Perform cross-validation to find the best lambda
cv_ridge <- cv.glmnet(x_train, y_train, alpha = 0)

# Extract the optimal lambda
optimal_lambda <- cv_ridge$lambda.min
print(paste("Optimal Lambda:", optimal_lambda))

# Testing
# Predict on test data using the optimal lambda
test_data$Predicted_Ridge <- predict(ridge_model, s = optimal_lambda, newx = x_test)

# Check the first few rows with predictions
head(test_data)

# Model Evaluation
# Compare actual vs. predicted values
results <- data.frame(
  Actual = test_data$Price,
  Predicted = test_data$Predicted_Ridge
)
head(results)

# Calculate Mean Absolute Error (MAE)
mae_d3 <- mean(abs(results$Actual - results$s1))
print(paste("MAE of Ridge Regression - Predictor 3:", round(mae_d3, 4)))

# Calculate Mean Squared Error (MSE)
mse_d3 <- mean((results$Actual - results$s1)^2)
print(paste("MSE of Ridge Regression - Predictor 3:", round(mse_d3, 4)))

# Calculate Root Mean Squared Error (RMSE)
rmse_d3 <- sqrt(mean((results$Actual - results$s1)^2))
print(paste("RMSE of Ridge Regression - Predictor 3:", round(rmse_d3, 4)))


# Model Equation
# Extract coefficients at the optimal lambda
coefficients <- coef(ridge_model, s = optimal_lambda)

# Convert to a named vector for easier access
coefficients <- as.vector(coefficients)
names(coefficients) <- rownames(coef(ridge_model, s = optimal_lambda))

# Intercept
intercept <- coefficients[1]

# Coefficients for predictors
slopes <- coefficients[-1]

# Format the equation
equation <- paste0(
  "Price = ", round(intercept, 4), " + ",
  paste0(round(slopes, 4), " * ", names(slopes), collapse = " + ")
)

# Display the equation
cat("Ridge Regression Model Equation:\n")
cat(equation, "\n")

#E. Build LASSO regression model, Predictors 1----
# Prepare predictors and target for training and testing
x_train <- as.matrix(train_data[, c("SquareFeet", "Bedrooms", "Bathrooms", "Neighborhood", "YearBuilt")]) 
y_train <- train_data$Price
x_test <- as.matrix(test_data[, c("SquareFeet", "Bedrooms", "Bathrooms", "Neighborhood", "YearBuilt")])
y_test <- test_data$Price

# Fit LASSO regression model
# alpha = 1 for LASSO (default), lambda is the regularization parameter
lasso_model <- glmnet(
  x = x_train,
  y = y_train,
  alpha = 1  # 1 = LASSO regression
)

# Cross-validation to find the optimal lambda
cv_lasso <- cv.glmnet(
  x = x_train,
  y = y_train,
  alpha = 1,
  nfolds = 10  # Number of folds for cross-validation
)

# Best lambda from cross-validation
best_lambda <- cv_lasso$lambda.min
cat("Best lambda:", best_lambda, "\n")

# Fit the final LASSO model with the optimal lambda
final_lasso_model <- glmnet(
  x = x_train,
  y = y_train,
  alpha = 1,
  lambda = best_lambda
)

# Display the model summary
summary(final_lasso_model)

# Testing
# Predict on test data using the optimal lambda
# Make predictions on the test set
test_data$Predicted_lasso <- predict(final_lasso_model, s = best_lambda, newx = x_test)

# Explicitly assign the prediction results to a new column with the desired name
test_data$Prediction <- as.numeric(predict(final_lasso_model, s = best_lambda, newx = x_test))

# Check the first few rows to confirm the column name change
head(test_data)


# Model Evaluation
# Compare actual vs. predicted values
results <- data.frame(
  Actual = test_data$Price,
  Predicted = test_data$Predicted_lasso
)
head(results)

# Calculate Mean Absolute Error (MAE)
mae_e1 <- mean(abs(y_test - test_data$Predicted_lasso))
print(paste("MAE of LASSO Regression - Predictor 1:", round(mae_e1, 4)))

# Calculate Mean Squared Error (MSE)
mse_e1 <- mean((y_test - test_data$Predicted_lasso)^2)
print(paste("MSE of LASSO Regression - Predictor 1:", round(mse_e1, 4)))

# Calculate Root Mean Squared Error (RMSE)
rmse_e1 <- sqrt(mean((y_test - test_data$Predicted_lasso)^2))
print(paste("RMSE of LASSO Regression - Predictor 1:", round(rmse_e1, 4)))


# Model Equation
# Extract coefficients at the optimal lambda
coefficients <- coef(final_lasso_model, s = best_lambda)

# Convert to a named vector for easier access
coefficients <- as.vector(coefficients)
names(coefficients) <- rownames(coef(lasso_model, s = best_lambda))

# Intercept
intercept <- coefficients[1]

# Coefficients for predictors
slopes <- coefficients[-1]

# Format the equation
equation <- paste0(
  "Price = ", round(intercept, 4), " + ",
  paste0(round(slopes, 4), " * ", names(slopes), collapse = " + ")
)

# Display the equation
cat("Lasso Regression Model Equation:\n")
cat(equation, "\n")

#E. Build LASSO regression model, Predictors 2----
# Prepare predictors and target for training and testing
x_train <- as.matrix(train_data[, c("SquareFeet", "Neighborhood")]) 
y_train <- train_data$Price
x_test <- as.matrix(test_data[, c("SquareFeet", "Neighborhood")])
y_test <- test_data$Price

# Fit LASSO regression model
# alpha = 1 for LASSO (default), lambda is the regularization parameter
lasso_model <- glmnet(
  x = x_train,
  y = y_train,
  alpha = 1  # 1 = LASSO regression
)

# Cross-validation to find the optimal lambda
cv_lasso <- cv.glmnet(
  x = x_train,
  y = y_train,
  alpha = 1,
  nfolds = 10  # Number of folds for cross-validation
)

# Best lambda from cross-validation
best_lambda <- cv_lasso$lambda.min
cat("Best lambda:", best_lambda, "\n")

# Fit the final LASSO model with the optimal lambda
final_lasso_model <- glmnet(
  x = x_train,
  y = y_train,
  alpha = 1,
  lambda = best_lambda
)

# Display the model summary
summary(final_lasso_model)

# Testing
# Predict on test data using the optimal lambda
# Make predictions on the test set
test_data$Predicted_lasso <- predict(final_lasso_model, s = best_lambda, newx = x_test)

# Explicitly assign the prediction results to a new column with the desired name
test_data$Prediction <- as.numeric(predict(final_lasso_model, s = best_lambda, newx = x_test))

# Check the first few rows to confirm the column name change
head(test_data)

# Check the first few rows with predictions
head(test_data)

# Model Evaluation
# Compare actual vs. predicted values
results <- data.frame(
  Actual = test_data$Price,
  Predicted = test_data$Predicted_lasso
)
head(results)

# Calculate Mean Absolute Error (MAE)
mae_e2 <- mean(abs(y_test - test_data$Predicted_lasso))
print(paste("MAE of LASSO Regression - Predictor 2:", round(mae_e2, 4)))

# Calculate Mean Squared Error (MSE)
mse_e2 <- mean((y_test - test_data$Predicted_lasso)^2)
print(paste("MSE of LASSO Regression - Predictor 2:", round(mse_e2, 4)))

# Calculate Root Mean Squared Error (RMSE)
rmse_e2 <- sqrt(mean((y_test - test_data$Predicted_lasso)^2))
print(paste("RMSE of LASSO Regression - Predictor 2:", round(rmse_e2, 4)))

# Model Equation
# Extract coefficients at the optimal lambda
coefficients <- coef(final_lasso_model, s = best_lambda)

# Convert to a named vector for easier access
coefficients <- as.vector(coefficients)
names(coefficients) <- rownames(coef(lasso_model, s = best_lambda))

# Intercept
intercept <- coefficients[1]

# Coefficients for predictors
slopes <- coefficients[-1]

# Format the equation
equation <- paste0(
  "Price = ", round(intercept, 4), " + ",
  paste0(round(slopes, 4), " * ", names(slopes), collapse = " + ")
)

# Display the equation
cat("Lasso Regression Model Equation:\n")
cat(equation, "\n")

#E. Build LASSO regression model, Predictors 3----
# Prepare predictors and target for training and testing
x_train <- as.matrix(train_data[, c("SquareFeet", "YearBuilt")]) 
y_train <- train_data$Price
x_test <- as.matrix(test_data[, c("SquareFeet", "YearBuilt")])
y_test <- test_data$Price

# Fit LASSO regression model
# alpha = 1 for LASSO (default), lambda is the regularization parameter
lasso_model <- glmnet(
  x = x_train,
  y = y_train,
  alpha = 1  # 1 = LASSO regression
)

# Cross-validation to find the optimal lambda
cv_lasso <- cv.glmnet(
  x = x_train,
  y = y_train,
  alpha = 1,
  nfolds = 10  # Number of folds for cross-validation
)

# Best lambda from cross-validation
best_lambda <- cv_lasso$lambda.min
cat("Best lambda:", best_lambda, "\n")

# Fit the final LASSO model with the optimal lambda
final_lasso_model <- glmnet(
  x = x_train,
  y = y_train,
  alpha = 1,
  lambda = best_lambda
)

# Display the model summary
summary(final_lasso_model)

# Testing
# Predict on test data using the optimal lambda
# Make predictions on the test set
test_data$Predicted_lasso <- predict(final_lasso_model, s = best_lambda, newx = x_test)

# Explicitly assign the prediction results to a new column with the desired name
test_data$Prediction <- as.numeric(predict(final_lasso_model, s = best_lambda, newx = x_test))

# Check the first few rows to confirm the column name change
head(test_data)

# Check the first few rows with predictions
head(test_data)


# Model Evaluation
# Compare actual vs. predicted values
results <- data.frame(
  Actual = test_data$Price,
  Predicted = test_data$Predicted_lasso
)
head(results)

# Calculate Mean Absolute Error (MAE)
mae_e3 <- mean(abs(y_test - test_data$Predicted_lasso))
print(paste("MAE of LASSO Regression - Predictor 3:", round(mae_e3, 4)))

# Calculate Mean Squared Error (MSE)
mse_e3 <- mean((y_test - test_data$Predicted_lasso)^2)
print(paste("MSE of LASSO Regression - Predictor 3:", round(mse_e3, 4)))

# Calculate Root Mean Squared Error (RMSE)
rmse_e3 <- sqrt(mean((y_test - test_data$Predicted_lasso)^2))
print(paste("RMSE of LASSO Regression - Predictor 3:", round(rmse_e3, 4)))


# Model Equation
# Extract coefficients at the optimal lambda
coefficients <- coef(final_lasso_model, s = best_lambda)

# Convert to a named vector for easier access
coefficients <- as.vector(coefficients)
names(coefficients) <- rownames(coef(lasso_model, s = best_lambda))

# Intercept
intercept <- coefficients[1]

# Coefficients for predictors
slopes <- coefficients[-1]

# Format the equation
equation <- paste0(
  "Price = ", round(intercept, 4), " + ",
  paste0(round(slopes, 4), " * ", names(slopes), collapse = " + ")
)

# Display the equation
cat("Lasso Regression Model Equation:\n")
cat(equation, "\n")

# Model Comparison----
# Create the dataframe
data_eval <- data.frame(
  Model = c(rep("LR", 3), rep("Polynomial", 3), rep("DT", 3), rep("Ridge", 3), rep("Lasso", 3)),
  Metric = rep(c("MAE", "MSE", "RMSE"), 5),
  Predictor_1 = c(mae_a1, mse_a1, rmse_a1, mae_b1, mse_b1, rmse_b1, mae_c1, mse_c1, rmse_c1, mae_d1, mse_d1, rmse_d1, mae_e1, mse_e1, rmse_e1),
  Predictor_2 = c(mae_a2, mse_a2, rmse_a2, mae_b2, mse_b2, rmse_b2, mae_c2, mse_c2, rmse_c2, mae_d2, mse_d2, rmse_d2, mae_e2, mse_e2, rmse_e2),
  Predictor_3 = c(mae_a3, mse_a3, rmse_a3, mae_b3, mse_b3, rmse_b3, mae_c3, mse_c3, rmse_c3, mae_d3, mse_d3, rmse_d3, mae_e3, mse_e3, rmse_e3)
)

# Display the dataframe
print(data_eval)

# Reshape the dataframe for ggplot2
data_long <- melt(data_eval, id.vars = c("Model", "Metric"), 
                  variable.name = "Feature", 
                  value.name = "Value")

# Define custom limits for each metric
limits <- list(
  MSE = c(2400, 2800),
  MAE = c(39, 42),
  RMSE = c(49, 53)
)

# Plot function for each metric
plot_metric <- function(metric_name) {
  ggplot(data_long %>% filter(Metric == metric_name), aes(x = Feature, y = Value, fill = Model)) +
    geom_bar(stat = "identity", position = "dodge") +
    coord_cartesian(ylim = limits[[metric_name]]) +  # Set specific limits
    labs(title = paste("Performance Metrics -", metric_name),
         x = "Predictors",
         y = "Metric Value") +
    theme_minimal() +
    theme(strip.text = element_text(size = 12)) +
    scale_fill_brewer(palette = "Set2")
}

# Create individual plots
plot_MSE <- plot_metric("MSE")
plot_MAE <- plot_metric("MAE")
plot_RMSE <- plot_metric("RMSE")

# Display plots (use patchwork or gridExtra if you want them combined)
library(patchwork)
plot_MSE / plot_MAE / plot_RMSE

