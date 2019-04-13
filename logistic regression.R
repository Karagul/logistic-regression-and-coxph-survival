#####1、descriptive statistics
#####2、LR model
#####3、model evaluation
#####4、score system

####rename the variable
# colnames(data)[colnames(data)=='stroke']<-'label'

####clear the work enviroment
rm(list=ls())
setwd('C:/Users/PINGAN/Desktop/test_code/logist_regression/')


###### reading the data
data_raw<-read.csv('stroke.csv',header=T,fileEncoding="UTF-8")
feature_name<-c('sex','age','waist','BMI','sbp','dbp','smoking','label')
data<-data_raw[,feature_name]


## caterria 1:missing<50%
missing_rate<-sapply(data,function(x)sum(is.na(x)))/nrow(data)
colnames(data)[missing_rate<0.5]
data<-data[missing_rate<0.5]


#### impute the missing data

## method1:mice
# library(mice)
# tempData <- mice(data,m=5,maxit=0,meth='mean',seed=500)
# data<-complete(tempData)

## method2:random
library(Hmisc)
var<-c('sex','age','waist','BMI','sbp','dbp','smoking')
continus<-c('age','waist','BMI','sbp','dbp')
category<-c('sex','smoking')
data_cont<-subset(data,select=continus)
data_cate<-subset(data,select=category)
data_label<-subset(data,select=label)
for (i in 1:ncol(data_cont)){
  data_cont[,i]<-impute(data_cont[,i],mean)
}

getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

for (j in 1:ncol(data_cate)){
  data_cate[,j][is.na(data_cate[,j])]<-getmode(data_cate[,j])
}
data<-cbind(data_cate,data_cont,data_label)


## method3: manual imputation
# data$age[is.na(data$age)]<-mean(data$age,na.rm = T)
# data$waist[is.na(data$wasit)]<-mean(data$wasit,na.rm = T)
# data$BMI[is.na(data$BMI)]<-mean(data$BMI,na.rm = T)
# data$sbp[is.na(data$sbp)]<-mean(data$sbp,na.rm = T)
# data$dbp[is.na(data$dbp)]<-mean(data$dbp,na.rm = T)
# data$sex[is.na(data$sex)]<-0
# data$smoking[is.na(data$smoking)]<-0



## caterria 2: label can not be missing
data<-subset(data,!is.na(data$label))

#### 1、descriptive statistics.In case of nonnorm or non chi-square, 
#### you can add the condition and transfer to rank test and fisher test.  
library(tableone)

var<-c('sex','age','waist','BMI','sbp','dbp','smoking')
continus<-c('age','waist','BMI','sbp','dbp')
category<-c('sex','smoking')

table_o<-CreateTableOne(vars=var,strata=c('label'),factorVars=category,data=data)
nonnorm<-c('age','BMI','sbp','dbp')
exact_var<-c('smoking','sex')
ttest_result<-data.frame(print(table_o))
rank_result<-data.frame(print(table_o,nonnormal=nonnorm,exact=exact_var))



##  caterria 3: significant in statistical test
var_sig<-colnames(data)[as.numeric(as.character(ttest_result$p))[-1]<0.5]
data<-data[colnames(data) %in% c(var_sig,'label')]


#### spliting the data
library(caret)
set.seed(666)
trainind<-createDataPartition(data$label,p=0.75,list=FALSE)
data_train<-data[trainind,]
data_test<-data[-trainind,]


###2、build LR model
model<-glm(label~.,data=data_train,family=binomial)
summ<-summary(model)
model_result<-cbind(beta=summ$coefficients[,1],
                OR=summ$coefficients[,1],
                exp(confint.lm(model)),
                'p_value'=summ$coefficients[,4])
results<-round(model_result[order(model_result[,5]),],7)
     

### 3、model evaluation incluing auc and calibration
p_train<-predict(model,data_train,type='response')
p_test<-predict(model,data_test,type='response')
library(pROC)
auc_train<-roc(data_train$label~p_train)
auc_test<-roc(data_test$label~p_test)
cat('训练集的AUC为',auc_train$auc,'\n')
cat('测试集的AUC为',auc_test$auc)


## calibration curve plot and hoslem test
library(ResourceSelection)
hl<-hoslem.test(data_train$label,p_train,g=10)
cal_ob<-hl$observed
cal_pred<-hl$expected
result<-data.frame(cal_ob[,1])
result[,1]<-NULL
result['observation']=data.frame(cal_ob[,2]/(cal_ob[,1]+cal_ob[,2]))
result['expection']=data.frame(cal_pred[,2]/(cal_pred[,1]+cal_pred[,2]))

x<-seq(1:10)
y<-matrix(result['observation'])[[1]]
z<-matrix(result['expection'])[[1]]
plot(x,y,type='b',pch=21,col='red',lty=3,xlab='declies of risk',
     ylab = 'proportion od event(%)')
lines(x,z,type='b',pch=22,col='blue',lty=3)
legend('topleft',inset=0.05,c('observation','expection'),lty=c(1,2),
       pch=c(21,22),col=c('red','blue'))
title('calibration curve of the stroke prediction model')



model_eval<-function(prop,y_true,y_pred){
  cutoff<-sort(y_pred,decreasing = T)[floor(length(y_pred)*prop)]
  y_pred1<-ifelse(y_pred>cutoff,1,0)
  table(y_pred1)
  cf<-as.matrix.data.frame(table(y_true,y_pred1))
  tp<-cf[2,2]
  fp<-cf[1,2]
  tn<-cf[1,1]
  fn<-cf[2,1]
  accuracy<-(tp+tn)/(tp+tn+fp+fn)
  sensitivity<-tp/(tp+fn)
  specificity<-tn/(tn+fp)
  precision<-tp/(tp+fp)
  return(c(prop,cutoff,accuracy,sensitivity,specificity,precision))
}

prop<-seq(0.01,0.5,by=0.01)
model_metrics<-matrix(nrow=1,ncol=8)
for(i in prop){
  model_result<-model_eval(i,data_test$label,p_test)
  model_metrics<-rbind(model_metrics,model_result)
}
model_results<-data.frame(model_metrics[-1,1:6])
colnames(model_results)<-c('prop','cutoff','accuracy','sensitivity','specificity','precision')


###4、 score sysytem
feature<-c('sbp','waist','smoking','label')
data_new<-data[feature]
sbp_breaks<-c(60,120,160,200)
waist_breaks<-c(50,80,100,130)
data_new$sbp1<-cut(data_new$sbp,breaks = sbp_breaks)
data_new$waist1<-cut(data_new$waist,breaks = waist_breaks)
data_new$smoking1<-factor(data_new$smoking)

data_final<-subset(data_new,select=c('sbp1','waist1','smoking1','label'))
lr_model<-glm(label~sbp1+waist1+smoking1,data=data_final,family = binomial)
summ<-summary(lr_model)
coeff<-summ$coefficients[,1]
score_result<-cbind(beta=coeff,OR=exp(coeff),score=round(coeff*10))
