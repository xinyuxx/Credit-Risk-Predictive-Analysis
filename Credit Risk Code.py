#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 14:42:51 2019

@author: Team 5
"""

from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import pickle
import streamlit as st
import pickle
import numpy as np
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
import warnings
from sklearn.svm import SVC
warnings.filterwarnings('ignore')
import pandas as pd




#1.Load the data from the computer

risk_data=pd.read_csv('/Users/xinyuhuang/Desktop/CIS432/Team Project 2/heloc_dataset_v1.csv')

#2. Cleaning Data:
#replace the numerical data with the np.nan and then replace them into mean of that column
#replace the categorical data with the np.nan and then replace them into the mode of that column


risk_data = risk_data.replace(-9,np.nan)
risk_data = risk_data.replace(-8,np.nan)
risk_data = risk_data.replace(-7,np.nan)

cols = ["MaxDelq2PublicRecLast12M", "MaxDelqEver"]
risk_data[cols]=risk_data[cols].replace(np.nan,risk_data[cols].mode())



risk_data = risk_data.replace(np.nan,risk_data.mean())
risk_data = risk_data.replace(np.nan,risk_data.mean())
risk_data = risk_data.replace(np.nan,risk_data.mean())
risk_data = risk_data.replace("Bad",1)
risk_data = risk_data.replace("Good",0)
risk_data = risk_data.round()

describe = risk_data.describe() 

#3.Create the Train Set and Test Set


#For the decision tree model ,we split the data into train set and test set
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(risk_data, test_size=0.2, random_state=40)

train_set.head()
test_set.head()

Y_test = test_set.RiskPerformance
Y_train = train_set.RiskPerformance

test_set = test_set.drop(columns = 'RiskPerformance')
train_set = train_set.drop(columns = 'RiskPerformance')
#For other models, we split data into X_test, y_test, X_train and y_train

Y=np.array(risk_data['RiskPerformance'])
del risk_data['RiskPerformance']
X=np.array(risk_data)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=40)


#4. Tune different models:

#A.logistics regression:
pipe_logistic = Pipeline([('minmax', MinMaxScaler()), ('lr', 
                          LogisticRegression(penalty='l1',tol=0.00001,C=100))])
pipe_logistic.fit(X_train, y_train)

print('Accuracy_logisctic: ', pipe_logistic.score(X_test, y_test))

#B.KNN:
 #standard scaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn import neighbors
knn = neighbors.KNeighborsClassifier(1)
knn.fit(X_train_scaled, y_train)

print('Accuracy_knn: ',knn.score(X_test_scaled, y_test))

#C.AdaBoost:

from sklearn.ensemble import AdaBoostClassifier
abc = AdaBoostClassifier(n_estimators=100,learning_rate=0.7)
abc = abc.fit(X_train_scaled,y_train)

print('Accuracy_adaboost: ',abc.score(X_test_scaled, y_test))

#D. Random Forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10, criterion='gini', 
                             max_depth=9, min_samples_split=8, max_features=5)
rfc.fit(X_train_scaled,y_train)

print('Accuracy_randomforest: ',rfc.score(X_test_scaled, y_test))

#E. Decision Tree:
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(max_depth =  4)
dtc.fit(train_set, Y_train)

from sklearn.metrics import accuracy_score
prediction = dtc.predict(test_set)
accuracy_score(Y_test, prediction)

bestDepth = pd.DataFrame({'Depth':np.repeat(0.111,31),'Accuracy':np.repeat(0.111,31)})   

for d in range(1,31):
    dtc = DecisionTreeClassifier(max_depth=d)
    dtc = dtc.fit(train_set, Y_train)
    prediction = dtc.predict(test_set)
    bestDepth['Depth'][d] = d
    bestDepth['Accuracy'][d] = accuracy_score(Y_test,prediction)
     
    
#5.constructing the interface :
#After comparing different models, we found that the logistics regression did best
#with an accuracy of 0.721. Therefore, we decide to build the interface based on
#the logistics regression model.
    
pipe_logistic = Pipeline([('minmax', MinMaxScaler()), ('lr', 
                          LogisticRegression(penalty='l1',tol=0.00001,C=100))])
pipe_logistic.fit(X_train, y_train)

print('Accuracy: ', pipe_logistic.score(X_test, y_test))

pickle.dump(X_train, open('X_train.sav', 'wb'))
pickle.dump(pipe_logistic, open('pipe_logistic.sav', 'wb'))
pickle.dump(X_test, open('X_test.sav', 'wb'))
pickle.dump(y_test, open('y_test.sav', 'wb'))

X_test = pickle.load(open('X_test.sav', 'rb'))
y_test = pickle.load(open('y_test.sav', 'rb'))
dic = {0: 'Good', 1: 'Bad'}

def test_demo(index):
    values = X_test[index]  # Input the value from dataset

    # Create four sliders in the sidebar
    a = st.sidebar.slider('ExternalRiskEstimate', 0.0,94.0, values[0], 1.0)
    b = st.sidebar.slider('MSinceOldestTradeOpen', 0.0,803.0, values[1], 1.0)
    c = st.sidebar.slider('MSinceMostRecentTradeOpen', 0.0,383.0, values[2], 1.0)
    d= st.sidebar.slider('AverageMInFile', 0.0,383.0, values[3], 1.0)  
    e= st.sidebar.slider('NumSatisfactoryTrades', 0.0,79.0, values[4], 1.0)
    f= st.sidebar.slider('NumTrades60Ever2DerogPubRec', 0.0,19.0, values[5], 1.0)
    g= st.sidebar.slider('NumTrades90Ever2DerogPubRec', 0.0,19.0, values[6], 1.0)
    h= st.sidebar.slider('PercentTradesNeverDelq', 0.0,100.0, values[7], 1.0)
    i= st.sidebar.slider('MSinceMostRecentDel', 0.0,83.0, values[8], 1.0)
    j= st.sidebar.slider('MaxDelq2PublicRecLast12M', 0.0,9.0, values[9], 1.0)
    k= st.sidebar.slider('MaxDelqEver', 0.0,8.0, values[10], 1.0)
    l= st.sidebar.slider('NumTotalTrades', 0.0,104.0, values[11], 1.0)
    m= st.sidebar.slider('NumTradesOpeninLast12M', 0.0,19.0, values[12], 1.0)
    n= st.sidebar.slider('PercentInstallTrades', 0.0,100.0, values[13], 1.0)
    o= st.sidebar.slider('MSinceMostRecentInqexcl7days', 0.0,24.0, values[14], 1.0)
    p= st.sidebar.slider('NumInqLast6M', 0.0,66.0, values[15], 1.0)
    q= st.sidebar.slider('NumInqLast6Mexcl7days', 0.0,66.0, values[16], 1.0)
    r= st.sidebar.slider('NetFractionRevolvingBurden', 0.0,232.0, values[17], 1.0)
    s= st.sidebar.slider('NetFractionInstallBurden', 0.0,471.0, values[18], 1.0)
    t= st.sidebar.slider('NumRevolvingTradesWBalance', 0.0,32.0, values[19], 1.0)
    u= st.sidebar.slider('NumInstallTradesWBalance', 0.0,23.0, values[20], 1.0)
    v= st.sidebar.slider('NumBank2NatlTradesWHighUtilization', 0.0,18.0, values[21], 1.0)
    w= st.sidebar.slider('PercentTradesWBalance', 0.0,100.0, values[22], 1.0)



    pipe = pickle.load(open('pipe_logistic.sav', 'rb'))
    res = pipe.predict(np.array([a, b, c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w]).reshape(1, -1))[0]
    st.write('Prediction:  ', dic[res])
    pred = pipe.predict(X_test)
    score = pipe.score(X_test, y_test)
    cm = metrics.confusion_matrix(y_test, pred)
    st.write('Accuracy: ', score)
    st.write('Confusion Matrix: ', cm)


# title
st.title('Credit Risk Assessment')
# show data
if st.checkbox('Show dataframe'):
    st.write(X_test)


number = st.text_input('Choose a row of information in the dataset:', 5)  # Input the index number

test_demo(int(number))  # Run the test function
    