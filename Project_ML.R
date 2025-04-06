# Install the caret package if you don't have it already
install.packages("caret")

# Load the caret package
library(caret)

library(class)
# Install the MASS package if you don't have it
install.packages("MASS")

# Load the MASS package
library(MASS)

library(ISLR)
accident_data <- read.csv("cleaned.csv")
splitIndex <- createDataPartition(accident_data$Accident_severity, p = 0.7, list = FALSE)
train_data <- accident_data[splitIndex, ]
test_data <- accident_data[-splitIndex, ]
head(train_data)

# Perform Linear Discriminant Analysis (LDA) on the training data
lda_model <- lda(Accident_severity ~ ., data = train_data)

# Use the model to predict on the test data
lda_pred <- predict(lda_model, test_data)
# Ensure that both are factors with the same levels
lda_pred$class <- as.factor(lda_pred$class)
test_data$Accident_severity <- as.factor(test_data$Accident_severity)

# Align the factor levels of predicted values with the actual values
levels(lda_pred$class) <- levels(test_data$Accident_severity)

# Now generate the confusion matrix
lda_conf_matrix <- confusionMatrix(lda_pred$class, test_data$Accident_severity)

# Print the confusion matrix and performance metrics
print(lda_conf_matrix)


#QDA
# Check correlation matrix
# Check class distribution in training data
table(train_data$Accident_severity)

# Check correlation matrix
cor(train_data[, sapply(train_data, is.numeric)])

# Perform Quadratic Discriminant Analysis (QDA) on the training data
qda_model <- qda(Accident_severity ~ ., data = train_data)

#QDA unable to use due to high correlation so using RDA

# Install and load klaR package for Regularized QDA
install.packages("klaR")
library(klaR)

# Perform Regularized Discriminant Analysis (Regularized QDA)
rda_model <- rda(Accident_severity ~ ., data = train_data)

# Predict on test data
rda_pred <- predict(rda_model, test_data)

# Confusion matrix for RDA
rda_conf_matrix <- confusionMatrix(rda_pred$class, test_data$Accident_severity)

# Print the confusion matrix and performance metrics
print(rda_conf_matrix)

