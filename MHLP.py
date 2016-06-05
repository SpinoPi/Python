
# coding: utf-8

import numpy as np
import pandas as pd
from sklearn import *
np.random.seed(123)

dA = pd.read_csv('E01LP.csv')
lA=list(dA.columns)
lA[0]='DATE'
dA.columns=lA
dB = dA[(dA['ASU RUNNING TIME']>23.99)==True]
dB = dB.ix[:,['MPAIR FLOW','MPAIR TEMPERATURE','MPGAN FLOW','MPGAN TEMPERATURE','LPGAN FLOW','LPGAN TEMPERATURE']]
dB.describe().T

target=dB['MPGAN TEMPERATURE']
data=dB.ix[:,['MPAIR FLOW','MPAIR TEMPERATURE','MPGAN FLOW','LPGAN FLOW','LPGAN TEMPERATURE']]
train_data,test_data,train_target,test_target=cross_validation.train_test_split(data, target,test_size=0.24,random_state=0)
clf=tree.DecisionTreeClassifier(criterion='gini',max_depth=11,min_samples_split=5)
clf_fit=clf.fit(train_data,train_target)
train_est=clf.predict(train_data)
test_est=clf.predict(test_data)

lA = list(dB.ix[list(test_data.index)]['MPGAN TEMPERATURE'])
Sum=0
l=len(test_data)
for i in range(l):
    if abs(lA[i]-test_est[i])<=1:
        Sum=Sum+1
print(Sum/len(test_data))

Sum=0
lB=[]
lC=list(test_data.index)
for i in range(l):
    if abs(lA[i]-test_est[i])<=2:
        Sum=Sum+1
    else:
        lB.append(lC[i])
print(Sum/len(test_data))
dC=dA.ix[lB]
list(dC['DATE'])

test_est=clf.predict(data)
lA = list(dB.ix[list(data.index)]['MPGAN TEMPERATURE'])
Sum=0
lB=[]
lC=list(data.index)
for i in range(l):
    if abs(lA[i]-test_est[i])<=1:
        Sum=Sum+1
    else:
        lB.append(lC[i])
print(Sum/len(test_data))
dC=dA.ix[lB]
list(dC['DATE'])

