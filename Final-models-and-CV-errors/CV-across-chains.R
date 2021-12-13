library(tidyverse)
library(glmnet)
library(stringr)
library(bnstruct)
library(randomForest)
library(e1071)

heart.full <- read.csv("Final-models-and-CV-errors/heart_full.csv")[,-1]%>% 
  mutate(Sex = as.factor(Sex)) %>% 
  mutate(ChestPainType= as.factor(ChestPainType)) %>% 
  mutate(RestingECG= as.factor(RestingECG)) %>%
  mutate(ExerciseAngina = as.factor(ExerciseAngina)) %>%
  mutate(ST_Slope = as.factor(ST_Slope)) %>%
  mutate(HeartDisease = as.factor(HeartDisease)) 

# set threshold for glm
threshold <- 0.5

set.seed(1)
shuffled_indexes <- sample(nrow(heart.full))
folds <- cut(seq(1,length(shuffled_indexes)),breaks=5,labels=FALSE)

# object to save errors
misclassification.glm <- rep(0, 5)
misclassification.rf <- rep(0, 5)
misclassification.svm <- rep(0, 5)
misclassification.aggregate <- rep(0, 5)

helper.f <- function(prediction, y_test) {
  tab <- table(y_test, prediction)
  misclassification_rate <- 1 - sum(diag(tab))/sum(tab)
  return(misclassification_rate)
}

for(i in 1:5){
  #Segment your data by fold using the which() function 
  testIndexes <- shuffled_indexes[which(folds==i,arr.ind=TRUE)]
  # get y.test from the unimputed data-set
  testData <- heart.full[testIndexes, ]
  y.test <- testData$HeartDisease
  
  overall.prediction.glm <- matrix(nrow = length(y.test), ncol = 5)
  overall.prediction.rf <- matrix(nrow = length(y.test), ncol = 5)
  overall.prediction.svm <- matrix(nrow = length(y.test), ncol = 5)
  
  for(j in 1:5){
    heart <- read.csv(str_c("Final-models-and-CV-errors/Chain ", j, 
                            "/heart.imp.", j, ".csv"))[,-1]%>% 
      mutate(Sex = as.factor(Sex)) %>% 
      mutate(ChestPainType= as.factor(ChestPainType)) %>% 
      mutate(RestingECG= as.factor(RestingECG)) %>%
      mutate(ExerciseAngina = as.factor(ExerciseAngina)) %>%
      mutate(ST_Slope = as.factor(ST_Slope)) %>%
      mutate(HeartDisease = as.factor(HeartDisease)) 
    
    testData <- heart[testIndexes, ]
    X.test <- testData[, -12]
    trainData <- heart[-testIndexes, ]
    # get the saved specification on SVM
    svm.specs <- readRDS(str_c("Final-models-and-CV-errors/Chain ", j, 
                               "/SVM-specs.RDS"))
    svm.name <- svm.specs[[1]]
    svm.cost <- svm.specs[[2]][1,1]
    # get the saved formula for glm
    glm.formula <- readRDS(str_c("Final-models-and-CV-errors/Chain ", j,
                                 "/glm.RDS"))$formula
    # fit the models using training data
    set.seed(1)
    svm.fit <- svm(HeartDisease ~ ., kernel = svm.name, data = trainData, 
                      cost = svm.cost)
    rf.fit <- randomForest(HeartDisease ~ ., data = trainData, 
                           mtry = sqrt(11), ntree = 1000)
    glm.fit <- glm(glm.formula, family = "binomial", data = trainData)
    # save the prediction values
    overall.prediction.glm[,j] <- predict(glm.fit, X.test, type="response")
    overall.prediction.rf[,j] <- predict(rf.fit, X.test)
    overall.prediction.svm[,j] <- predict(svm.fit, X.test)
  }
  
  prediction.glm <- rowMeans(overall.prediction.glm)
  prediction.rf <- rowMeans(overall.prediction.rf)
  prediction.svm <- rowMeans(overall.prediction.svm)
  prediction.aggregate <- rep(0, length(prediction.glm))
  
  for(j in 1:length(prediction.glm)) {
    # realize glm probability prediction as 0 and 1 using threshold
    if(prediction.glm[j] < threshold) {
      prediction.glm[j] = 0
    }
    else {
      prediction.glm[j] = 1
    }
    
    if(prediction.rf[j] < 1.6) {
      prediction.rf[j] = 0
    }
    else {
      prediction.rf[j] = 1
    }
    
    if(prediction.svm[j] < 1.6) {
      prediction.svm[j] = 0
    }
    else {
      prediction.svm[j] = 1
    }
    
    prediction.aggregate[j] <- mean(prediction.glm[j] + prediction.rf[j] +
                                      prediction.svm[j])
    
    # realize aggregate probability prediction as 0 and 1 using threshold
    if(prediction.aggregate[j] < 2/3){
      prediction.aggregate[j] = 0
    }
    else{
      prediction.aggregate[j] = 1
    }
  }
  
  misclassification.glm[i] = helper.f(prediction.glm, y.test)
  misclassification.rf[i] = helper.f(prediction.rf, y.test)
  misclassification.svm[i] = helper.f(prediction.svm, y.test)
  misclassification.aggregate[i] = helper.f(prediction.aggregate, y.test)
}

temp <- tibble(glm = misclassification.glm,
               rf = misclassification.rf,
               svm = misclassification.svm,
               aggregate = misclassification.aggregate)

write.csv(temp, "Final-models-and-CV-errors/CV-errors.csv")