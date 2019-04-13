# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 08:54:16 2018
@author: JIAWENXIAO502
"""
import numpy as np
import pandas as pd
from sklearn import metrics
from xgboost import XGBClassifier 
from sklearn.externals import joblib
from sklearn.cross_validation import train_test_split
import warnings
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm
from scipy.stats import ttest_ind
from scipy.stats import chi2_contingency
warnings.filterwarnings("ignore")
data_raw=pd.read_csv('D:/Users/JIAWENXIAO502/Desktop/data/atrial_fibrillation/af_data.csv',
                     na_values='\\N',encoding='gbk')

data_raw=data_raw.loc[~data_raw['fld22_desc'].str.contains("房颤|心房颤动", na=False),:]

data_raw=data_raw[~(data_raw['sex'].isnull())]
weights = np.array(data_raw.weights)
heights = np.array(data_raw.heights)
BMI = weights / ((heights/100)**2)
data_raw['BMI']= pd.DataFrame(BMI, index = data_raw.index, columns=['BMI'])
data_raw["sex"]=data_raw["sex"].map({"M":1,"F":0})
features=['age','sex', 'heart_rate','sbp','dbp','fasting','ggt','cr',
          'glu_pyr','glu_oxa','whitebloodcell','redbloodcell','triglyceride',
          'cholesterol','bun','plt','hba1c','BMI','waist','smoking','education']
data=data_raw[features]
data_label=data_raw['y']



#remove outlier based on rules from dr.yang
data.loc[data.heart_rate>120, 'heart_rate']=120
data.loc[data.heart_rate<40, 'heart_rate']=40
data.loc[data.sbp>170, 'sbp']=170
data.loc[data.sbp<75, 'sbp']=75
data.loc[data.dbp>120, 'dbp']=120
data.loc[data.dbp<50, 'dbp']=50
data.loc[data.fasting>12, 'fasting']=12
data.loc[data.fasting<3, 'fasting']=3
data.loc[data.ggt>100, 'ggt']=100
data.loc[data.ggt<0, 'ggt']=0

data.loc[data.cr>150, 'cr']=150
data.loc[data.cr<0, 'cr']=0
data.loc[data.glu_pyr>100, 'glu_pyr']=100
data.loc[data.glu_pyr<0, 'glu_pyr']=0
data.loc[data.glu_oxa>100, 'glu_oxa']=100
data.loc[data.glu_oxa<0, 'glu_oxa']=0
data.loc[data.whitebloodcell>20, 'whitebloodcell']=20
data.loc[data.whitebloodcell<0, 'whitebloodcell']=0
data.loc[data.redbloodcell>10, 'redbloodcell']=10
data.loc[data.redbloodcell<0, 'redbloodcell']=0

data.loc[data.triglyceride>10, 'triglyceride']=10
data.loc[data.triglyceride<0, 'triglyceride']=0
data.loc[data.cholesterol>10, 'cholesterol']=10
data.loc[data.cholesterol<0, 'cholesterol']=0
data.loc[data.bun>10, 'bun']=10
data.loc[data.bun<0, 'bun']=0
data.loc[data.plt>500, 'plt']=500
data.loc[data.plt<0, 'plt']=0
data.loc[data.hba1c>12, 'hba1c']=12
data.loc[data.hba1c<0, 'hba1c']=0
data.loc[data.BMI>40, 'BMI']= 40 
data.loc[data.BMI<10, 'BMI']=10
data.loc[data.waist>140, 'waist']= 140 
data.loc[data.waist<50, 'waist']=50

data.heart_rate=data.heart_rate.fillna(data.heart_rate.mean())
data.sbp=data.sbp.fillna(data.sbp.mean())
data.dbp=data.dbp.fillna(data.dbp.mean())
data.fasting=data.fasting.fillna(data.fasting.mean())
data.ggt=data.ggt.fillna(data.ggt.mean())
data.cr=data.cr.fillna(data.cr.mean())
data.glu_pyr=data.glu_pyr.fillna(data.glu_pyr.mean())
data.glu_oxa=data.glu_oxa.fillna(data.glu_oxa.mean())
data.whitebloodcell=data.whitebloodcell.fillna(data.whitebloodcell.mean())
data.redbloodcell=data.redbloodcell.fillna(data.redbloodcell.mean())
data.triglyceride=data.triglyceride.fillna(data.triglyceride.mean())
data.cholesterol=data.cholesterol.fillna(data.cholesterol.mean())
data.bun=data.bun.fillna(data.bun.mean())
data.plt=data.plt.fillna(data.plt.mean())
data.hba1c=data.hba1c.fillna(data.hba1c.mean())
data.BMI=data.BMI.fillna(data.BMI.mean())
data.waist=data.waist.fillna(data.waist.mean())
data.smoking=data.smoking.fillna(0)


############3、统计检验 
features=data.columns.tolist()
category=['sex','smoking','education']
continus=[i  for i in features if i  not in category]
result_tmp=np.zeros((len(continus),3))
i=0
for col in  continus:
    pos_index=data_label[data_label==1]
    neg_index=data_label[data_label==0]
    pos_data=data[col].ix[pos_index.index]
    neg_data=data[col].ix[neg_index.index]
    t,p=ttest_ind(pos_data,neg_data,equal_var=False)  #要加上异方差性，这样才会和R语言的P值结果一致
    result_tmp[i,:]=[pos_data.mean(),neg_data.mean(),p]
    i=i+1


result_tmp1=np.zeros((len(category),3))
i=0
for col in  category:
    pos_index=data_label[data_label==1]
    neg_index=data_label[data_label==0]
    pos_data=data[col].ix[pos_index.index]
    neg_data=data[col].ix[neg_index.index]
    pos_sum=pos_data.value_counts().reset_index()   
    neg_sum=neg_data.value_counts().reset_index()
    data_sum=pd.merge(pos_sum,neg_sum,how='right',on='index')
    data_sum=data_sum.drop(['index'],axis=1)
    data_sum=data_sum.fillna(0)
    data_summary=np.array(data_sum).T  
    g,p,dof,expectd=chi2_contingency(data_summary) #对于类别型变量，需要每一行代表一类，这样才能在两类中进行统计分析   
    result_tmp1[i,:]=[pos_data.mean(),neg_data.mean(),p]
    i=i+1   
        
result_tmp2=np.concatenate([result_tmp,result_tmp1],axis=0)    
features=continus+ category
ttest_result=pd.DataFrame(result_tmp2,index=features,columns=['pos_mean','neg_mean','p_value'])
ttest_result=ttest_result.sort_values(by='p_value',ascending=True)

data['age2']=data['age']**2
data['sex*age2']=data['age']*data['sex']
feature_select=['age','sex','BMI','sbp','age2','sex*age2','cholesterol','heart_rate','cr','dbp']


data_new=data[feature_select]
data_new = sm.add_constant(data_new,has_constant='add')
x_train,x_test,y_train,y_test=train_test_split(data_new,data_label,random_state=33,test_size=0.25)
logit=sm.GLM(y_train,x_train,family=sm.families.Binomial())  
result=logit.fit()  
train_pred=result.predict(x_train)
test_pred=result.predict(x_test)
auc_train=roc_auc_score(y_train,train_pred)
auc_test=roc_auc_score(y_test,test_pred) 


p_value2=result.pvalues
params=result.params
OddsRatio=np.exp(params)
lr_result=pd.concat([params,OddsRatio,p_value2],axis=1)
lr_result.columns=['beta','OR','p_value']
#lr_result.to_csv('D:/Users/JIAWENXIAO502/Desktop/data/stroke_life/lr_result.csv')


def print_metrics_binary(y_true, predictions, verbose=0):
    predictions = np.array(predictions)
    if (len(predictions.shape) == 1):
        predictions = np.stack([1 - predictions, predictions]).transpose((1, 0))
    cf = metrics.confusion_matrix(y_true, predictions.argmax(axis=1))
    cf = cf.astype(np.float32)
    accuracy = (cf[0][0] + cf[1][1]) / np.sum(cf)
    sensitivity = cf[1][1] / (cf[1][0] + cf[1][1]) 
    specificity = cf[0][0] / (cf[0][0] + cf[0][1])
    precision = cf[1][1] / (cf[1][1] + cf[0][1])
    auroc = metrics.roc_auc_score(y_true, predictions[:, 1])
    return [accuracy,sensitivity,specificity, precision,auroc]


top=0.2
locs=int(len(test_pred)*top)
test_pred=test_pred.reset_index()
test_pred=test_pred.drop(['index'],axis=1)
test_pred=test_pred.reset_index()
test_pred.columns=['index','risk']
test_pred1=test_pred.sort_values(by='risk',ascending=False)

test_pred1.risk=0
test_pred1['risk'][:locs]=1
test_pred1=test_pred1.sort_values(by='index',ascending=True)
test_pred2=np.array(test_pred1['risk'])

#
#test_eval=print_metrics_binary(y_test,test_pred)
#test_eval=np.array(test_eval).reshape(1,5)
#test_eval[0,4]=auc_test
#test_evals=pd.DataFrame(test_eval,columns=['accuracy','sensitivity','specificity','precision','auroc'])
