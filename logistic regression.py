# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 10:39:31 2019

@author: PINGAN
"""
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from tableone import TableOne
import os
os.chdir('C:/Users/PINGAN/Desktop/test_code/logist_regression/')


data_raw=pd.read_csv('stroke.csv',na_values='\\N',encoding='gbk')
feature_name=['age','sex','BMI','smoking','label']
data=data_raw[feature_name]

##impute the missing data including the continus data and category data

continus=['age','BMI']
category=['sex','smoking']
imputer1=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer1=imputer1.fit(data.loc[:,continus])
data.loc[:,continus]=imputer1.transform(data.loc[:,continus])


imputer2=Imputer(missing_values='NaN',strategy='most_frequent',axis=0)
imputer2=imputer2.fit(data.loc[:,category])
data.loc[:,category]=imputer2.transform(data.loc[:,category])


##### descriptive statistics
groupby=['label']
nonnormal=['age']
labels={'label':'morbidity'}
mytable=TableOne(data,feature_name,category,groupby,nonnormal,labels=labels,pval=True)
mytables=mytable.tableone.reset_index()

#### build the logistic regression model
data_num=data.drop(['label'],axis=1)
data_label=data['label']
data_new=sm.add_constant(data_num,has_constant='add')
x_train,x_test,y_train,y_test=train_test_split(data_num,data_label,random_state=33,test_size=0.25)
logit=sm.GLM(y_train,x_train,family=sm.families.Binomial())
result=logit.fit()
pred_train=result.predict(x_train)
pred_test=result.predict(x_test)


####model result statistics
p_value=result.pvalues
params=result.params
OddsRatio=np.exp(params)
lr_result=pd.concat([params,OddsRatio,p_value],axis=1)
lr_result.columns=['beta','OddsRatio','p_value']

auc_train=metrics.roc_auc_score(y_train,pred_train)
auc_test=metrics.roc_auc_score(y_test,pred_test)
print('训练集的auc为',auc_train)
print('测试集的auc为',auc_test)



####model evaluation
def model_evaluation(y_true,y_pred):
    y_pred=np.array(y_pred)
    if(len(y_pred.shape)==1):
        y_pred=np.stack([1-y_pred,y_pred]).transpose((1,0))
    cf=metrics.confusion_matrix(y_true,y_pred.argmax(axis=1)).astype(np.float32)
    accuracy=(cf[0][0]+cf[1][1])/np.sum(cf)
    sensitivity=cf[1][1]/(cf[1][1]+cf[1][0])
    specificity=cf[0][0]/(cf[0][0]+cf[0][1])
    precision=cf[1][1]/(cf[1][1]+cf[0][1])
    return [accuracy,sensitivity,specificity,precision]

#### top * selection
top=0.2
locs=int(len(pred_test)*top)
pred_test1=pd.DataFrame(np.array(pred_test)).reset_index()
pred_test1.columns=['index','risk']
pred_test1=pred_test1.sort_values(by='risk',ascending=False)

pred_test1.risk=0
pred_test1['risk'][:locs]=1
pred_test1=pred_test1.sort_values(by='index',ascending='True')
pred_test2=np.array(pred_test1['risk'])

model_eval=model_evaluation(y_test,pred_test2) 

model_eval.append(auc_test)   
model_eval=np.reshape(np.array(model_eval),[1,5]) 
model_evals=pd.DataFrame(model_eval,columns=['auucracy','sensitivity','specificity','precision','auroc'])

#### fixed threshold selection
#threshold=0.005
#pred_test3=np.array(pred_test)
#pred_test4=np.where(pred_test3>threshold,1,0)
#model_eval=model_evaluation(y_test,pred_test4)  
#model_eval.append(auc_test) 
#model_eval=np.reshape(np.array(model_eval),[1,5]) 
#model_evals=pd.DataFrame(model_eval,columns=['auucracy','sensitivity','specificity','precision','auroc'])



