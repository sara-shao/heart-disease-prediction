library(tidyverse)
library(glmnet)
library(stringr)
library(bnstruct)
library(e1071)
library(mice)

heart <- read.csv(file = 'heart.csv')
heart <- heart %>% 
  mutate(Cholesterol = case_when(
    Cholesterol == 0 ~ NA_integer_,
    TRUE ~ Cholesterol
  ),
  RestingBP = case_when(
    RestingBP == 0 ~ NA_integer_,
    TRUE ~ RestingBP
  )) 

# exclude one observation with 
heart.full <- heart %>% 
  filter(!(is.na(Cholesterol)&is.na(RestingBP)))
#impute 5 chains of full data-set
heart.mice.ls <- mice(heart.full, seed = 1)
# separate the five chains
heart.imp.1 <- complete(heart.mice.ls, 1)
heart.imp.2 <- complete(heart.mice.ls, 2)
heart.imp.3 <- complete(heart.mice.ls, 3)
heart.imp.4 <- complete(heart.mice.ls, 4)
heart.imp.5 <- complete(heart.mice.ls, 5)

# combine imputed data into a list
heart.imp.ls <- list(heart.imp.1, heart.imp.2, heart.imp.3, heart.imp.4,
                     heart.imp.5)

# selecting and saving the optimal logistic model for each chain
glm.aic.select <- function(i){
  heart <- heart.imp.ls[[i]]
  heart <- heart %>% 
    mutate(Sex = as.factor(Sex),
           ChestPainType= as.factor(ChestPainType),
           RestingECG= as.factor(RestingECG),
           ExerciseAngina = as.factor(ExerciseAngina),
           ST_Slope = as.factor(ST_Slope),
           HeartDisease = as.factor(HeartDisease))
  # full model for the logistic model of each chain
  fullmod <- glm(HeartDisease ~.^2,family="binomial",data = heart)
  # selected model for the logistic model of each chain
  selectedmod <- step(fullmod,direction="backward", trace = FALSE)
  # generate file name
  file.name <- str_c("Final-models-and-CV-errors/Chain ", i, "/glm.RDS")
  saveRDS(selectedmod, file.name)
}

get_misclassification <- function(bestmod, X_test, y_test) {
  prediction <- predict(bestmod, X_test)
  tab <- table(y_test, prediction)
  misclassification_rate <- 1 - sum(diag(tab))/sum(tab)
  return(misclassification_rate)
}

svm.select <- function(i){
  heart <- heart.imp.ls[[i]]
  
  heart <- heart %>% 
    filter(Cholesterol != 0) %>% 
    filter(RestingBP!=0) %>% 
    mutate(Sex = as.factor(Sex)) %>% 
    mutate(ChestPainType= as.factor(ChestPainType)) %>% 
    mutate(RestingECG= as.factor(RestingECG)) %>%
    mutate(ExerciseAngina = as.factor(ExerciseAngina)) %>%
    mutate(ST_Slope = as.factor(ST_Slope)) %>%
    mutate(HeartDisease = as.factor(HeartDisease))
  
  set.seed(1)
  shuffled_heart <- heart[sample(nrow(heart)),]
  folds <- cut(seq(1,nrow(shuffled_heart)),breaks=5,labels=FALSE)
  
  # error
  misclassfication.linear <- rep(0, 5)
  misclassfication.poly <- rep(0, 5)
  misclassfication.radial <- rep(0, 5)
  
  for(i in 1:5){
    
    #Segment your data by fold using the which() function 
    testIndexes <- which(folds==i,arr.ind=TRUE)
    testData <- shuffled_heart[testIndexes, ]
    y.test <- testData$HeartDisease
    X.test <- testData[, -12]
    trainData <- shuffled_heart[-testIndexes, ]
    
    # Need to choose parameters
    set.seed(1)
    tune.out.linear <- tune(svm, HeartDisease ~ ., data = trainData, 
                            kernel = "linear",
                            ranges = list(cost = 10^seq(-2, 1, by = 0.25)))
    tune.out.poly <- tune(svm, HeartDisease ~ ., data = trainData, 
                          kernel = "poly", degree=2,
                          ranges = list(cost = 10^seq(-2, 1, by = 0.25)))
    tune.out.radial <- tune(svm, HeartDisease ~ ., data = trainData, 
                            kernel = "radial",
                            ranges = list(cost = 10^seq(-2, 1, by = 0.25)))
    
    bestmod.linear <- tune.out.linear$best.model
    bestmod.poly <- tune.out.poly$best.model
    bestmod.radial <- tune.out.radial$best.model
    
    misclassfication.linear[i] = get_misclassification(bestmod.linear, 
                                                       X.test, y.test)
    misclassfication.poly[i] = get_misclassification(bestmod.poly, 
                                                     X.test, y.test)
    misclassfication.radial[i] = get_misclassification(bestmod.radial, 
                                                       X.test, y.test)
  }
  
  misclassfication.linear
  misclassfication.poly
  misclassfication.radial
  temp <- tibble(linear = misclassfication.linear, poly = misclassfication.poly,
                 radial = misclassfication.radial)
  write.csv(temp, file = str_c("Final-models-and-CV-errors/Chain ", i, 
                               "/SVM-CVs.csv"))
  
  # save model name
  mod.name <- c("linear", "poly", "radial")[which.min(colMeans(temp))]
  
  # Tuning best SVM
  tune.out.mod <- tune(svm, HeartDisease ~ ., data = heart, kernel = mod.name,
                       ranges = list(cost = 10^seq(-2, 1, by = 0.25)))
  bestmod.mod <- tune.out.mod$best.model
  bestmod.param <- tune.out.mod$best.parameters
  bestmod.mod
  bestmod.param
  temp <- list(model = mod.name, cost = bestmod.param, best.model = bestmod.mod)
  saveRDS(temp, file = str_c("Final-models-and-CV-errors/Chain ", i, 
                             "/SVM-specs.RDS"))
}

for(i in 1:5){
  # save glm models for each chain
  glm.aic.select(i)
  # save SVM specification for each chain
  svm.select(i)
  file.name <- str_c("Final-models-and-CV-errors/Chain ", i, "/heart.imp.", i,
                     ".csv")
  write.csv(heart.imp.ls[[i]], file.name)
}
