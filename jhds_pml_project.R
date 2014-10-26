# Background
# Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is 
# now possible to collect a large amount of data about personal 
# activity relatively inexpensively. These type of devices are part 
# of the quantified self movement - a group of enthusiasts who take 
# measurements about themselves regularly to improve their health, 
# to find patterns in their behavior, or because they are tech geeks. 
# One thing that people regularly do is quantify how much of a particular 
# activity they do, but they rarely quantify how well they do it. In this 
# project, your goal will be to use data from accelerometers on the belt, 
# forearm, arm, and dumbell of 6 participants. They were asked to perform 
# barbell lifts correctly and incorrectly in 5 different ways. More 
# information is available from the website here: 
# http://groupware.les.inf.puc-rio.br/har 
# (see the section on the Weight Lifting Exercise Dataset). 
# 
# Data 
# The training data for this project are available here: 
# https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv
# The test data are available here: 
# https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv
# 
# The data for this project come from this source: 
# http://groupware.les.inf.puc-rio.br/har. 
# If you use the document you create for this class for any purpose 
# please cite them as they have been very generous in allowing their 
# data to be used for this kind of assignment. 
# 
# What you should submit
# The goal of your project is to predict the manner in which they 
# did the exercise. This is the "classe" variable in the training set. 
# You may use any of the other variables to predict with. You should 
# create a report describing how you built your model, how you used 
# cross validation, what you think the expected out of sample error 
# is, and why you made the choices you did. You will also use your 
# prediction model to predict 20 different test cases. 
# 
# 1. Your submission should consist of a link to a Github repo with 
# your R markdown and compiled HTML file describing your analysis. 
# Please constrain the text of the writeup to < 2000 words and the 
# number of figures to be less than 5. It will make it easier for 
# the graders if you submit a repo with a gh-pages branch so the 
# HTML page can be viewed online (and you always want to make it 
# easy on graders :-).
# 2. You should also apply your machine learning algorithm to the 
# 20 test cases available in the test data above. Please submit 
# your predictions in appropriate format to the programming assignment 
# for automated grading. See the programming assignment for additional 
# details. 
# 
# Reproducibility 
# Due to security concerns with the exchange of R code, your code will 
# not be run during the evaluation by your classmates. Please be sure 
# that if they download the repo, they will be able to view the compiled 
# HTML version of your analysis. 

# plan
# 1. get data
# 2. clean data
# 3. model
# 4. validate
# 5. score/predict


# clear screen, source: http://stackoverflow.com/questions/2824965/clear-r-console-programmatically
cls<-function() cat(rep("\n",100))
cls()


## set the working directory
setwd("c:/_coursera")  # use / instead of \ in windows 

## download training csv data file and save to variable, make NA and #DIV/0! null
trn_Url<-"http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
download.file(trn_Url,destfile=".\\pml_trn.csv")
trn_data<-read.csv("pml_trn.csv",na.strings=c("NA","#DIV/0!"))
#how many rows and columns are in the file
dim(trn_data)
head(trn_data)
summary(trn_data)
str(trn_data)

## download test csv data file as predicting data and save to variable, make NA and #DIV/0! null
prd_Url<-"http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(prd_Url,destfile=".\\pml_prd.csv")
prd_data<-read.csv("pml_prd.csv",na.strings=c("NA","#DIV/0!"))
#inspect data
dim(prd_data)
head(prd_data)
summary(prd_data)
str(prd_data)

# remove any column that has an NA and get rid of the first 7 columns which are not 
#features to model: X, user_name, raw_timestamp_part_1, raw_timestamp_part_2, 
#cvtd_timestamp, new_window, num_window
trn_data2<-trn_data[ , colSums(is.na(trn_data)) == 0]
trn_data2<-trn_data2[,-c(1,2,3,4,5,6,7)]
str(prd_data2)
head(trn_data2)
summary(trn_data2)

prd_data2<-prd_data[ , colSums(is.na(prd_data)) == 0]
prd_data2<-prd_data2[,-c(1,2,3,4,5,6,7)]
str(prd_data2)
head(prd_data2)
summary(prd_data2)


library(caret)
library(randomForest)
library(doParallel)
library(foreach)
set.seed(78237611)
partx<-createDataPartition(y=trn_data2$classe, p=0.70, list=FALSE )
trn<-trn_data2[partx,]
tst<-trn_data2[-partx,]

#create 5 random forests with 100 trees each. Parallel processing helps alot
# with speed
registerDoParallel()
a<-trn[-ncol(trn)]
b<-trn$classe
rf<-foreach(ntree=rep(100, 5), .combine=randomForest::combine, .packages='randomForest', na.action=na.omit) %dopar% {
        randomForest(a,b,ntree=ntree) 
}

#look at confusion matrix for accuracy
predictions_trn<-predict(rf, newdata=trn)
confusionMatrix(predictions_trn,trn$classe)

predictions_tst<-predict(rf, newdata=tst)
confusionMatrix(predictions_tst,tst$classe)

# accuracy is very high. code below (from COURSERA) is used
# for submissions.  It is predicting for 20 obs and writing
# the prediction 1 per file for a total of 20 files.
pml_write_files = function(x){
        n = length(x)
        for(i in 1:n){
                filename = paste0("problem_id_",i,".txt")
                write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
        }
}

predictions<-predict(rf, newdata=prd_data2)
predictions

pml_write_files(predictions)