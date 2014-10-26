Practical Machine Learning Course Project Writeup
========================================================
        
Background
-----------------------
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 



Data 
---------
The training data for this project are available here: 
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here: 
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har. If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment. 

What you should submit
----------
The goal of your project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases. 
1. Your submission should consist of a link to a Github repo with your R markdown and compiled HTML file describing your analysis. Please constrain the text of the writeup to < 2000 words and the number of figures to be less than 5. It will make it easier for the graders if you submit a repo with a gh-pages branch so the HTML page can be viewed online (and you always want to make it easy on graders :-).
2. You should also apply your machine learning algorithm to the 20 test cases available in the test data above. Please submit your predictions in appropriate format to the programming assignment for automated grading. See the programming assignment for additional details. 
        

Plan and Process
-------
1. access/clean data
2. model
3. validate
4. score/predict

download dataset and deal with NA and #DIV/0! values.  Save testing as a prediction dataset.  Remove any remove any column that has an NA and get rid of the first 7 columns which are not features to model: X, user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp, new_window, num_window



```r
trn_Url <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
download.file(trn_Url, destfile = ".\\pml_trn.csv")
trn_data <- read.csv("pml_trn.csv", na.strings = c("NA", "#DIV/0!"))
trn_data2 <- trn_data[, colSums(is.na(trn_data)) == 0]
trn_data2 <- trn_data2[, -c(1, 2, 3, 4, 5, 6, 7)]

prd_Url <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(prd_Url, destfile = ".\\pml_prd.csv")
prd_data <- read.csv("pml_prd.csv", na.strings = c("NA", "#DIV/0!"))
prd_data2 <- prd_data[, colSums(is.na(prd_data)) == 0]
prd_data2 <- prd_data2[, -c(1, 2, 3, 4, 5, 6, 7)]
```


breakup training data into train and test.  Then create 5 random forests with 100 trees each and used parallel processing to help with speed

```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(randomForest)
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
library(doParallel)
```

```
## Loading required package: foreach
## Loading required package: iterators
## Loading required package: parallel
```

```r
library(foreach)
set.seed(78237611)
partx <- createDataPartition(y = trn_data2$classe, p = 0.7, list = FALSE)
trn <- trn_data2[partx, ]
tst <- trn_data2[-partx, ]

registerDoParallel()
a <- trn[-ncol(trn)]
b <- trn$classe
rf <- foreach(ntree = rep(100, 5), .combine = randomForest::combine, .packages = "randomForest", 
    na.action = na.omit) %dopar% {
    randomForest(a, b, ntree = ntree)
}
```


Review confusion matrices for accuracy.  Training is 100% and Testing is very high.

```r
predictions_trn <- predict(rf, newdata = trn)
confusionMatrix(predictions_trn, trn$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3906    0    0    0    0
##          B    0 2658    0    0    0
##          C    0    0 2396    0    0
##          D    0    0    0 2252    0
##          E    0    0    0    0 2525
## 
## Overall Statistics
##                                 
##                Accuracy : 1     
##                  95% CI : (1, 1)
##     No Information Rate : 0.284 
##     P-Value [Acc > NIR] : <2e-16
##                                 
##                   Kappa : 1     
##  Mcnemar's Test P-Value : NA    
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    1.000    1.000    1.000    1.000
## Specificity             1.000    1.000    1.000    1.000    1.000
## Pos Pred Value          1.000    1.000    1.000    1.000    1.000
## Neg Pred Value          1.000    1.000    1.000    1.000    1.000
## Prevalence              0.284    0.193    0.174    0.164    0.184
## Detection Rate          0.284    0.193    0.174    0.164    0.184
## Detection Prevalence    0.284    0.193    0.174    0.164    0.184
## Balanced Accuracy       1.000    1.000    1.000    1.000    1.000
```

```r

predictions_tst <- predict(rf, newdata = tst)
confusionMatrix(predictions_tst, tst$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1673    2    0    0    0
##          B    1 1137    8    0    0
##          C    0    0 1018   14    5
##          D    0    0    0  949    0
##          E    0    0    0    1 1077
## 
## Overall Statistics
##                                         
##                Accuracy : 0.995         
##                  95% CI : (0.993, 0.996)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.993         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.999    0.998    0.992    0.984    0.995
## Specificity             1.000    0.998    0.996    1.000    1.000
## Pos Pred Value          0.999    0.992    0.982    1.000    0.999
## Neg Pred Value          1.000    1.000    0.998    0.997    0.999
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.193    0.173    0.161    0.183
## Detection Prevalence    0.285    0.195    0.176    0.161    0.183
## Balanced Accuracy       0.999    0.998    0.994    0.992    0.998
```


Used the model to predict 20 records.  Submitting using the code below from Coursera which writes 1 prediction per file for a total of 20 files.

```r
pml_write_files = function(x) {
    n = length(x)
    for (i in 1:n) {
        filename = paste0("problem_id_", i, ".txt")
        write.table(x[i], file = filename, quote = FALSE, row.names = FALSE, 
            col.names = FALSE)
    }
}

predictions <- predict(rf, newdata = prd_data2)
predictions
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

```r

pml_write_files(predictions)
```


Summary
----------------------
The random forest model predicts well for this scenario.
