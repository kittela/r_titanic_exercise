library(titanic)    # loads titanic_train data frame
library(caret)
library(tidyverse)
library(rpart)
library(dplyr)

# 3 significant digits
options(digits = 3)

# clean the data - `titanic_train` is loaded with the titanic package
titanic_clean <- titanic_train %>%
  mutate(Survived = factor(Survived),
         Embarked = factor(Embarked),
         Age = ifelse(is.na(Age), median(Age, na.rm = TRUE), Age), # NA age to median age
         FamilySize = SibSp + Parch + 1) %>%    # count family members
  select(Survived,  Sex, Pclass, Age, Fare, SibSp, Parch, FamilySize, Embarked)

set.seed(42, sample.kind = "Rounding")

# Split the 'titanic_clean' into training (80%) and test (20%) sets
index <- createDataPartition(y = titanic_clean$Survived, p = 0.2, list = FALSE)
train_titanic_clean <- titanic_clean[-index,]
test_titanic_clean <- titanic_clean[index,]

# How many observations are in the training set?
nrow(train_titanic_clean)

# How many observations are in the test set?
nrow(test_titanic_clean)

# What proportion of individuals in the training set survived?
mean(as.numeric(as.character(train_titanic_clean$Survived)))

# How accurate of a prediction can we make by randomly guessing if someone survives?
set.seed(3, sample.kind = "Rounding")
y_hat <- sample(c(0,1), nrow(test_titanic_clean), replace = TRUE) %>% factor()
confusionMatrix(y_hat, test_titanic_clean$Survived)$overall['Accuracy']

# What proportion of training set females survived?
train_titanic_clean %>%
  group_by(Sex) %>%
  summarize(Survived = mean(Survived == 1)) %>%
  filter(Sex == "female") %>%
  pull(Survived)

# What proportion of training set males survived?
train_titanic_clean %>%
  group_by(Sex) %>%
  summarize(Survived = mean(Survived == 1)) %>%
  filter(Sex == "male") %>%
  pull(Survived)

# Based on these proportions, how accurate is a prediction of predicting survival
# for all females and death for all males?
y_hat <- ifelse(test_titanic_clean$Sex == "female", 1, 0) %>% factor()
confusionMatrix(y_hat, test_titanic_clean$Survived)
F_meas(y_hat, reference = test_titanic_clean$Survived)

# In the training set, which classes (Pclass) were more likely to survive than die?
train_titanic_clean %>%
  group_by(Pclass) %>%
  summarize(Survived = mean(Survived == 1)) %>% 
  filter(Survived > 0.5)

# How accurate is a prediction of predicting survival based on class
y_hat <- ifelse(test_titanic_clean$Pclass == 1, 1, 0) %>% factor()
confusionMatrix(y_hat, test_titanic_clean$Survived)
F_meas(y_hat, reference = test_titanic_clean$Survived)

# Now grouping by both sex and class, which combinations were more likely to survive
# than die?
train_titanic_clean %>%
  group_by(Pclass, Sex) %>%
  summarize(Survived = mean(Survived == 1)) %>% 
  filter(Survived > 0.5)

# How accurate is a prediction of predicting survival based on both sex and class?
y_hat <- ifelse(test_titanic_clean$Sex == 'female' & test_titanic_clean$Pclass < 3, 1, 0) %>%
  factor()
confusionMatrix(y_hat, test_titanic_clean$Survived)
F_meas(y_hat, reference = test_titanic_clean$Survived)

set.seed(1, sample.kind = "Rounding")

# How accurate is a linear discriminant analysis (LDA) model trained on the data using
# only fare as a predictor?
lda_fit <- train(Survived ~ Fare, data = train_titanic_clean, method = "lda")
lda_y_hat <- predict(lda_fit, test_titanic_clean) %>% factor()
confusionMatrix(lda_y_hat, test_titanic_clean$Survived)$overall["Accuracy"]

# How accurate is a quadratic discriminant analysis (QDA) model trained on the data using
# only fare as a predictor?
qda_fit <- train(Survived ~ Fare, data = train_titanic_clean, method = "qda")
qda_y_hat <- predict(qda_fit, test_titanic_clean) %>% factor()
confusionMatrix(qda_y_hat, test_titanic_clean$Survived)$overall["Accuracy"]

set.seed(1, sample.kind = "Rounding")

# What is the accuracy of training a logistic regression model (GLM) using age as
# the only predictor?
glm_fit <- train(Survived ~ Age, data = train_titanic_clean, method = "glm")
glm_y_hat <- predict(glm_fit, test_titanic_clean) %>% factor()
confusionMatrix(glm_y_hat, test_titanic_clean$Survived)$overall["Accuracy"]

set.seed(1, sample.kind = "Rounding")

# What is the accuracy of training a logistic regression model (GLM) using sex, class
# fare, and age as predictors?
glm_fit <- train(Survived ~ Sex + Pclass + Age + Fare, data = train_titanic_clean, method = "glm")
glm_y_hat <- predict(glm_fit, test_titanic_clean) %>% factor()
confusionMatrix(glm_y_hat, test_titanic_clean$Survived)$overall["Accuracy"]

set.seed(1, sample.kind = "Rounding")

# What is the accuracy of training a logistic regression model (GLM) using all predictors?
glm_fit <- train(Survived ~ ., data = train_titanic_clean, method = "glm")
glm_y_hat <- predict(glm_fit, test_titanic_clean) %>% factor()
confusionMatrix(glm_y_hat, test_titanic_clean$Survived)$overall["Accuracy"]

set.seed(6, sample.kind = "Rounding")

# After training a kNN model on the data, tuning k with k = seq(3, 51, 2), what is
# the optimal value of k?
knn_fit <- train(Survived ~ ., data = train_titanic_clean, method = "knn", 
                 tuneGrid = data.frame(k = seq(3, 51, 2)))
knn_fit$results$k[which.max(knn_fit$results$Accuracy)]
ggplot(knn_fit, highlight = TRUE)

# What is the accuracy of this kNN model on the test set?
knn_y_hat <- predict(knn_fit, test_titanic_clean) %>% factor()
confusionMatrix(knn_y_hat, test_titanic_clean$Survived)$overall["Accuracy"]

set.seed(8, sample.kind = "Rounding")

# How does the optimal k-value and accuracy change when using 10-fold cross-validation
# instead of the default training control?
control <- trainControl(method = "cv", number = 10, p = .9)
knn_cv_fit <- train(Survived ~ ., data = train_titanic_clean, method = "knn", 
                    tuneGrid = data.frame(k = seq(3, 51, 2)), trControl = control)
knn_cv_fit$results$k[which.max(knn_cv_fit$results$Accuracy)]
knn_cv_y_hat <- predict(knn_cv_fit, test_titanic_clean) %>% factor()
confusionMatrix(knn_cv_y_hat, test_titanic_clean$Survived)$overall["Accuracy"]
ggplot(knn_cv_fit, highlight = TRUE)

set.seed(10, sample.kind = "Rounding")

# Training a decision tree (rpart) model by tuning the complexity parameter (cp)
# with cp = seq(0, 0.05, 0.002), what is the optimal cp-value and what is that model's
# accuracy on the test set?
rpart_fit <- train(Survived ~ ., data = train_titanic_clean, method = "rpart", 
                   tuneGrid = data.frame(cp = seq(0, 0.05, 0.002)))
rpart_fit$results$cp[which.max(rpart_fit$results$Accuracy)]
ggplot(rpart_fit, highlight = TRUE)
rpart_y_hat <- predict(rpart_fit, test_titanic_clean) %>% factor()
confusionMatrix(rpart_y_hat, test_titanic_clean$Survived)$overall["Accuracy"]

# In the decision tree above, which variables are ultimately used?
rpart_fit$finalModel
plot(rpart_fit$finalModel, margin = 0.1)
text(rpart_fit$finalModel)

set.seed(14, sample.kind = "Rounding")

# Training a random forest (rf) model by tuning the mtry-value with mtry = seq(1:7),
# which mtry-value maximizes accuracy and what is that accuracy?
rf_fit <- train(Survived ~ ., data = train_titanic_clean, method = "rf", 
                tuneGrid = data.frame(mtry = seq(1:7)), ntree = 100)
rf_fit$results$mtry[which.max(rf_fit$results$Accuracy)]
ggplot(rf_fit, highlight = TRUE)
rf_y_hat <- predict(rf_fit, test_titanic_clean) %>% factor()
confusionMatrix(rf_y_hat, test_titanic_clean$Survived)$overall["Accuracy"]

# Which variable is the most important in this rf model?
varImp(rf_fit)