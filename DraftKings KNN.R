# The section through line 42 is where we train our model and find our "ideal k".
# First, we need to set our working directory so that we can import our files easily.
setwd("~/Draftkings")
# Next, bring in the "data.table" library so that we can use the "fread" function to efficiently load our data.
library(data.table)
# Use the "fread" function to load in the data.
train <- fread("RB.csv")
# Scale the "projection" and "salary" columns so they will have equal weight when we calculate our distance for KNN.
train$ProjectionScaled <- train$Projection/mean(train$Projection)
train$SalaryScaled <- train$Salary/mean(train$Salary)
# Create empty "fitted_value" and "error_value" vectors.
fitted_value <- NULL
error_value <- NULL
# Start a for loop that will be used to evaluate every k-value. Can be modified if the dataset is too large for this function to run quickly (e.g., take every 2nd value)
for (j in (1:nrow(train)-1)){
  # Set squaredError to 0, so we can aggregate the total for each k-value.
  squaredError <- 0
  # This is our leave-one-out for loop that will cycle through every selection as being the testing value while we train on the remainder of the dataset. This is a high variance cross-validation method, but
  # we will use it here due to the small size of our training data.
  for(i in 1:nrow(train)){
    # Our testing "set" is the ith row.
    validation <- train[i,]
    # Our training set is every other row.
    training <- train[-i,]
    # Set our k-value (the number of values we will average to get our projection).
    k <- j
    # Calculate our "Distance". We'll be using the euclidean distance method since it's the most geometrically accurate if we're trying to get a "mean" value.
    training$Distance <- sqrt((validation$ProjectionScaled - training$ProjectionScaled)^2 + (validation$SalaryScaled - training$SalaryScaled)^2)
    # Sort by distance
    training <- training[order(Distance),]
    # Pick the k closest values
    training <- training[1:k,]
    # Take the average of the k closest values. This is our prediction.
    fitted_value[i] <- mean(training$Actual)
    # Take the root mean squared error between our prediction and our validation point.
    squaredError <- squaredError + (fitted_value[i] - validation$Actual)^2
  }
  # Calculate the total error for one k value. This will iterate until all of the k values are tried.
error_value[j] <- sqrt(squaredError/nrow(train))
}
# Return the value of k with the lowest root mean squared error.
idealK <- which.min(error_value)
# The below code is the section where we actually use our model to predict on new data.First we bring in our testing data.
test <- fread("Test.csv")
# Scale the testing data similarly to how we scaled the training data above.
test$ProjectionScaled <- test$Projection/mean(train$Projection)
test$SalaryScaled <- test$Salary/mean(train$Salary)
# Create an empty vector so we can append our projections.
fitted_value <- NULL
# For loop to predict for all values within our testing set.
for (i in 1:nrow(test)){
  # Set our testing value to be the ith row of our testing data
  testing <- test[i]
  # Set our training dataset as our original train set. This is done so that we are free to manipulate the training set on every iteration.
  training <- train
  # Calculate the distance from the testing value for each training set value.
  training$Distance <- sqrt((testing$ProjectionScaled - training$ProjectionScaled)^2 + (testing$SalaryScaled - training$SalaryScaled)^2)
  # Sort by distance.
  training <- training[order(Distance),]
  # Take the idealK closest values
  training <- training[1:idealK,]
  # Set our projection for the ith value of the testing set.
  fitted_value[i] <- mean(training$Actual)
}
# Append all of our predictions to our original dataset.
test$Pred <- fitted_value
# Write the final dataset as a csv file.
write.csv(test, "testResults.csv")