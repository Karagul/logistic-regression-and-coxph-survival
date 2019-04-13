## basis cox survival model introduction

####clear the work enviroment
rm(list=ls())
setwd('C:/Users/PINGAN/Desktop/test_code/logist_regression/')


###### 1、reading the data
data_raw<-read.csv('stroke_survival.csv',header=T,fileEncoding="UTF-8")
feature_name<-c('sex','age','waist','BMI','sbp','dbp','smoking','time','label')
data<-data_raw[,feature_name]


###### 2、define the time window
window<-5
data$label[data$time>=window & data$label==1 ]<-0
data$time[data$time>=window]<-window



#### 3、impute the missing data

## method1:mice
# library(mice)
# init<-mice(data,maxit=0)
# meth<-init$method
# data<-mice(data,m=1,method=meth,seed=2099)

## method2:random
# library(Hmisc)
# var<-c('sex','age','waist','BMI','sbp','dbp','smoking')
# continus<-c('age','waist','BMI','sbp','dbp')
# category<-c('sex','smoking')
# data_cont<-subset(data,select=continus)
# data_cate<-subset(data,select=category)
# data_label<-subset(data,select=label)
# for (i in 1:ncol(data_cont)){
#   data_cont[,i]<-impute(data_cont[,i],mean)
# }
# 
# getmode <- function(v) {
#   uniqv <- unique(v)
#   uniqv[which.max(tabulate(match(v, uniqv)))]
# }
# 
# for (j in 1:ncol(data_cate)){
#   data_cate[,j][is.na(data_cate[,j])]<-getmode(data_cate[,j])
# }
# data<-cbind(data_cate,data_cont,data_label)


## method3: manual imputation
data$age[is.na(data$age)]<-mean(data$age,na.rm = T)
data$waist[is.na(data$waist)]<-mean(data$waist,na.rm = T)
data$BMI[is.na(data$BMI)]<-mean(data$BMI,na.rm = T)
data$sbp[is.na(data$sbp)]<-mean(data$sbp,na.rm = T)
data$dbp[is.na(data$dbp)]<-mean(data$dbp,na.rm = T)
data$sex[is.na(data$sex)]<-0
data$smoking[is.na(data$smoking)]<-0


###### 4、split training and testing set
library(caret)
set.seed(666)
trainind<-createDataPartition(y=data$label,p=0.75,list=FALSE)
data_train<-data[trainind,]
data_test<-data[-trainind,]



###### 5、basic survival model
library(survival)
library(survivalROC)

cox_model<-coxph(Surv(time,label)~sex+age+waist+BMI+sbp+dbp+smoking,data=data_train)
summary(cox_model)


###### 6、model evaluation
##1)traditional auc method
library(pROC)
pred_train<-1-summary(survfit(cox_model,data_train),time=window)$surv
pred_test<-1-summary(survfit(cox_model,data_test),time=window)$surv

auc_train<-roc(data_train$label~as.numeric(pred_train))
auc_train<-auc_train$auc[1]

auc_test<-roc(data_test$label~as.numeric(pred_test))
auc_test<-auc_test$auc[1]
cat('训练集auc为',auc_train,'\n')
cat('测试集auc为',auc_test)

##2)c_index method 
library(survcomp)
pred_train_cindex<-predict(cox_model,data_train)
pred_test_cindex<-predict(cox_model,data_test)
cind_train<-concordance.index(pred_train_cindex,data_train$time,data_train$label,method='noether')
cind_train<-cind_train$c.index

cind_test<-concordance.index(pred_test_cindex,data_test$time,data_test$label,method='noether')
cind_test<-cind_test$c.index

cat('训练集c_index为',cind_train,'\n')
cat('测试集c_index为',cind_test)












