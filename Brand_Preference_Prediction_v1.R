#### Project name:Brand Preference ####

#### Load packages ####
install.packages("readr")
install.packages("caret")
install.packages("ggplot2", dependencies=TRUE)
install.packages("caret")

library(readr)
library(lattice)
library(ggplot2)
library(caret)
library(party)
library(plyr)
library(C50) 

#### Import Data ####
#read train file
complete <- read.csv("CompleteResponses.csv", header = TRUE, check.names=FALSE,sep = ",", quote = "\'", as.is = TRUE)
#read predict file
incomplete <- read.csv("SurveyIncomplete.csv", header = TRUE, check.names=FALSE,sep =",", quote = "\'", as.is = TRUE)

#### Data structure####
str(complete) # 10.000 obs of 7 vars 
str(incomplete) # 5.000 obs of 7 vars

#### Data Analysis####
names(complete)
names(incomplete)

head(complete, 5)
head(incomplete, 5)

tail(complete, 5)
tail(incomplete, 5)

summary(complete)
summary(incomplete)

#Rename some variables
complete$brand[complete$brand=="0"] <-"Acer"
complete$brand[complete$brand=="1"] <-"Sony"
#Convert brand to factor
complete$brand <- factor(complete$brand)

summary(complete)

any(is.na(complete))  # confirm if any "NA" values in ds
any(is.na(incomplete))   # confirm if any "NA" values in ds

#Explorative Plots

ggplot(complete, aes(x=brand, fill=brand)) + geom_bar() + ggtitle("Brand") +
      geom_text(stat="count",aes(label=..count..,y=..count..), vjust=10) 

ggplot(complete, aes(x=salary, fill=brand)) + geom_histogram(color="grey", bins=20) + ggtitle("Brand - Salary")

ggplot(complete, aes(x=age, fill=brand)) + geom_histogram(color="grey", bins=20) + ggtitle("Brand - Age")

ggplot(complete, aes(x=elevel, fill=brand)) + geom_histogram(color="grey", bins=20) + ggtitle("Brand - Education Level")

ggplot(complete, aes(x=zipcode, fill=brand)) + geom_histogram(color="grey", bins=20) + ggtitle("Brand - Zip Code")


p <- ggplot(complete, aes(x=salary, y=age, colour = brand))
p <- p + geom_point() + ggtitle("Brand - Salary + Age")
p
# This plot showed us how that  "age and salary"  had some correlation with the brand variable. 
# We can split the age into 3 bins and the salary into 5 bins to see it with more clarity.


# change data types to factor
complete$elevel <- factor(complete$elevel)
complete$car <- factor(complete$car)
complete$zipcode <- factor(complete$zipcode)
# change data types so train and predict data sets match 
incomplete$elevel <- factor(incomplete$elevel)   ###  education level code doesn't have numeric meaning, is not intended for calculations
incomplete$car <- factor(incomplete$car)         ###  car brand code doesn't have numeric meaning, is not intended for calculations
incomplete$zipcode <- factor(incomplete$zipcode) ###  zipcode doesn't have numeric meaning, is not intended for calculations
incomplete$brand <- factor(incomplete$brand)     ###  brand code doesn't have numeric meaning, is not intended for calculations

# correlations <- cor(incomplete[,c(1,2,3,4,5,6)])  ### see correlations between  1 and 6 variables, #7 variable is to be predicted
# print(correlations)   

#### Determining the variables ####
#decision Tree Plot
ct1<-ctree(brand~., data=complete, controls = ctree_control(maxdepth=3))
plot(ct, main="Plot Decision Tree") 


#-------------------#### Create Train/Test sets ####--------------------------

set.seed(123) # set random seed (random selection can be reproduced)

# create the training partition that is 75% of total obs
inTraining <- createDataPartition(complete$brand, p = .75, list = FALSE)
trainSet <- complete[inTraining,]
testSet <- complete[-inTraining,]

str(trainSet) # 7501 obs of 7 var 
str(testSet) # 2499 obs of 7 var

#Train Control | Using "Cross validation" method
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 5)


#### Train and test model on train ds#####
#First I trained the model including all variables -car #


#KNN1 with all variables  -car | because I consider that it variable no tiene sentido inclurila, no guarda relacion
KNNfit1<-train(brand~.-car, data= trainSet, method="knn", trControl=fitControl, 
         preProcess=c("center", "scale"), tuneLength=5)
KNNfit1 
#  k   Accuracy   Kappa  
# 13  0.8208536  0.6198988

#KNN2 with Salary and Age variables
KNNfit2<-train(brand~salary+age, data= trainSet, method="knn", trControl=fitControl, 
               preProcess=c("center", "scale"), tuneLength=5)
KNNfit2 
#  k   Accuracy   Kappa  
#  13  0.9167299  0.8233277

#KNN3 with Salary variable
KNNfit3<-train(brand~salary, data= trainSet, method="knn", trControl=fitControl,
               preProcess=c("center", "scale"), tuneLength=5)
KNNfit3 
#  k   Accuracy   Kappa  
#  13  0.7081108  0.3737227


#RF1 with all variables -car
RFfit1 <-train(brand~.-car, data= trainSet, method="parRF",trControl=fitControl, ntree=50, do.trace=10)
RFfit1
#  mtry  Accuracy   Kappa     
#  19    0.9179673  0.8259138

#RF2 with Salary and Age variables
RFfit2 <-train(brand~salary+age, data= trainSet, method="parRF",trControl=fitControl, ntree=50, do.trace=10)
RFfit2
#  Accuracy   Kappa    
#  0.9034484  0.7947727

#RF3 with Salary variable
RFfit3 <-train(brand~salary, data= trainSet, method="parRF",trControl=fitControl, ntree=50, do.trace=10)
RFfit3
#  Accuracy   Kappa    
#  0.6419175  0.2385625



#CTree1 with all variables -car
ctreefit1 <- train(brand~.-car, data = trainSet, method = "ctree", trControl = fitControl) 
ctreefit1
#  mincriterion  Accuracy   Kappa    
#  0.01          0.9142500  0.8179274

#CTree2 with Salary and Age variables
ctreefit2 <- train(brand~salary+age, data = trainSet, method = "ctree", trControl = fitControl) 
ctreefit2
#  mincriterion  Accuracy   Kappa    
#  0.01          0.9147621  0.8188386

#CTree3 with Salary variable
ctreefit3 <- train(brand~salary, data = trainSet, method = "ctree", trControl = fitControl) 
ctreefit3
#  mincriterion  Accuracy   Kappa    
#  0.01          0.7263475  0.4182787


#C5.0 with all variables -car
C50fit1 <- train(brand~.-car, data= trainSet, method="C5.0", trControl=fitControl)
C50fit1 
# model  winnow  trials  Accuracy   Kappa    
# rules   TRUE   20      0.9186984  0.8267278

#### C5.0 Methods
C50fit2 <- train(brand~salary+age, data= trainSet, method="C5.0", trControl=fitControl)
C50fit2
# model  winnow  trials  Accuracy   Kappa 
# tree    TRUE   20      0.9186152  0.8275404

C50fit3 <- train(brand~salary, data= trainSet, method="C5.0", trControl=fitControl)
C50fit3
# model  winnow  trials  Accuracy   Kappa 
# rules   TRUE    1      0.7243280  0.4219681

#### summarize the distributions ####
results <- resamples(list(KNN1=KNNfit1, KNN2=KNNfit2, KNN3=KNNfit3,RF1=RFfit1,RF2=RFfit2,RF3=RFfit3,
                    CTree1=ctreefit1,CTree2=ctreefit2,CTree3=ctreefit3, C50fit1=C50fit1, C50fit2=C50fit2, C50fit3=C50fit3))
summary(results)

#### Predictions of the models #### antes de elegir el modelo a aplicar en el data set incomplete, testeamos los mejores modelos

## Make predictions ##
testPredRF1<-predict(RFfit1,testSet)
testPredKNN2<-predict(KNNfit2,testSet)
testPredCTree2<-predict(ctreefit2,testSet)
testPredC50fit1<-predict(C50fit1,testSet)
testPredC50fit2<-predict(C50fit2,testSet)

#### Performance measurment####

predRF1<-postResample(testPredRF1, testSet$brand)
predRF1
# Accuracy     Kappa 
# 0.9187551 0.8281817 

predKNN2<-postResample(testPredKNN2, testSet$brand)
predKNN2
# Accuracy     Kappa 
# 0.9244139 0.8398827 

predCTree2<-postResample(testPredCTree2, testSet$brand)
predCTree2
# Accuracy     Kappa 
# 0.9232013 0.8362564

PredC50fit1<-postResample(testPredC50fit1, testSet$brand)
PredC50fit1
# Accuracy     Kappa 
# 0.9308812 0.8533993 

PredC50fit2<-postResample(testPredC50fit2, testSet$brand)
PredC50fit2
# Accuracy     Kappa 
# 0.9232013 0.8375509


#### GETTING RESULTS FROM INCOMPLETE DATA ####
# Apply the model to the incomplete data set 
Pred_Incomp <-predict(RFfit1, newdata = incomplete)

summary(Pred_Incomp)
# acer sony 
# 1901 3099 

Pred_Incomp_V1 <-predict(C50fit1, newdata = incomplete)

summary(Pred_Incomp)
# acer sony 
# 1901 3099 







