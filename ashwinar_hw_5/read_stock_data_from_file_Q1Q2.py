# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 14:37:29 2018

@author: ashwinar
this scripts reads your ticker file (e.g. MSFT.csv) and
constructs a list of lines
"""

"""

Ashwin Arunkumar
Class: CS 677
Date: 11/24/2023
Homework Problem # 1
Description of Problem (just a 1-2 line summary!): 
Question 1 :
1. load the Excel (”raw data” worksheet) data into Pandas dataframe 
2. combine NSP labels into two groups: N (normal - these labels are assigned) and Abnormal (everything else).
We will use existing class 1 for normal and define class 0 for abnormal.

Question 2: Use Naive Bayesian NB classifier to answer these questions: 
1. split your set 50/50, train NB on Xtrain and predict class labels in Xtest.Use for remainder of assignment/models. 
2. what is the accuracy? 
3. compute the confusion matrix

"""

import os
import csv
import statistics
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score

import matplotlib.pyplot as plt
import xlrd


## This is code for Q1

input_dir = os.getcwd()
clinicalRecords = os.path.join(input_dir,'CTG.xls')

# df_clinicalRecords = pd.read_csv(clinicalRecords)
df_clinicalRecords = pd.read_excel(clinicalRecords,sheet_name=2)
df_clinicalRecords = df_clinicalRecords.drop(0, axis=0)
print(df_clinicalRecords)
df_clinicalRecords['CLASS'] = df_clinicalRecords['NSP'].apply(lambda x: 1 if x == 1 else 0)

required_features = ['LB','ALTV','Min','Mean','CLASS']
df_clinicalRecords_filtered = df_clinicalRecords.drop(df_clinicalRecords.columns.difference(required_features), axis=1)

# print(df_clinicalRecords_filtered.dropna().astype(int))

## This is code for Q2
X = df_clinicalRecords_filtered.dropna().astype(int)[['LB', 'ALTV', 'Min','Mean']].values # Features
y = df_clinicalRecords_filtered.dropna().astype(int)[['CLASS']].values # Class labels

#Data set 50/50 split between train and test.
X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size=0.5,random_state=120)

NB_Classifier = MultinomialNB()
NB_Classifier.fit(X_train,Y_train.ravel())
prediction = NB_Classifier.predict(X_test)
print(prediction)
#
accuracy = accuracy_score(Y_test, prediction)
print(f"Accuracy of the NB classifier prediction is: {accuracy * 100 :.2f}%")

conf_matrix = confusion_matrix(Y_test, prediction)
conf_matrix_df = pd.DataFrame(conf_matrix, index=['Actual Positive', 'Actual Negative'], columns=['Predicted Positive', 'Predicted Negative'])
print("\nConfusion Matrix:")
TP = conf_matrix[[0],[0]][0]
FP = conf_matrix[[0],[1]][0]
TN = conf_matrix[[1],[1]][0]
FN = conf_matrix[[1],[0]][0]
print("True Positive is ", TP)
print("False Positive is ", FP)
print("False Negative is ", FN)
print("True Negative is ", TN)
print("True Positive Rate is : ", round(TP/(TP+FN),2)*100)
print("True Negative Rate is : ", round(TN/(TN+FP),2)*100)

