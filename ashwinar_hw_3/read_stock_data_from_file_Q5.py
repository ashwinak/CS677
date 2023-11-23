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
Date: 11/21/2023
Homework Problem # 5
Description of Problem (just a 1-2 line summary!): 
Question 5 :
1. Use the same Xtrain and Xtest as before. Train your logistic regression classifier on Xtrain and compute its accuracy for Xtest 
2. summarize your performance measures in the table
3. is your logistic regression better than your simple classifier for any of the measures from the previous table? 
4. is your logistic regression better than your k-NN classifier (using the best k ∗) for any of the measures from the previous table? 
5. consider a bill x that contains the last 4 digits of your BUID as feature values. 
What is the class label predicted for this bill x by logistic regression? Is it the same label as predicted by k-NN?
"""

import os
import csv
import statistics
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt



## This is code for Q5

input_dir = os.getcwd()
bankNotes = os.path.join(input_dir,'data_banknote_authentication.csv')

df_bankNotes = pd.read_csv(bankNotes)

column_names = ['F1-Variance', 'F2-Skewness', 'F3-curtosis', 'F4-Entropy', 'F5-Class']

if len(column_names) == len(df_bankNotes.columns):
    df_bankNotes.columns = column_names

df_bankNotes['Color'] = df_bankNotes['F5-Class'].apply(lambda x: 'Green' if x ==0 else 'Red')

X = df_bankNotes[['F1-Variance', 'F2-Skewness', 'F3-curtosis']].values
y = df_bankNotes[['Color']].values

#Data set 50/50 split between train and test.
X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size=0.5,random_state=120)

# Data preprocessing using standard scalar before fitting into KNN model.
# print(X_test)
scalar = StandardScaler()
X_train = scalar.fit_transform(X_train) # For training data to both learn parameters and apply transformation.
X_test = scalar.transform(X_test) # For applying learned transformation to the new data set.

# Fit into model and calculate the accuracy

logRegression = LogisticRegression()
logRegression.fit(X_train,Y_train.ravel())
predictions = logRegression.predict(X_test)
print("Accuracy from Logistic regression model is: " + str(round(np.mean(predictions == Y_test.ravel())*100,2))) #compare predictions from the KNN model against Y_test output.

if Y_test.ravel().shape == predictions.shape:
    TP = FP = TN = FN = TPR = TNR = accuracy = totalCount = 0
    for (i), value in np.ndenumerate(Y_test.ravel()):
        if (Y_test.ravel()[i] == 'Green') and (predictions[i]== 'Green'):
            TP+=1
        elif (Y_test.ravel()[i] == 'Red') and (predictions[i] == 'Red'):
            TN+=1
        elif (Y_test.ravel()[i] == 'Red') and (predictions[i] == 'Green'):
            FP+=1
        elif (Y_test.ravel()[i] == 'Green') and (predictions[i] == 'Red'):
            FN+=1
    print("True Positive is " + str(TP))
    print("True Negative is " + str(TN))
    print("False Positive is " + str(FP))
    print("False Negative is " + str(FN))
    print("True Positive Rate is : " + str(round(TP/(TP+FN),2)*100))
    print("True Negative Rate is : " + str(round(TN/(TN+FP),2)*100))


#BU ID : U08370453. Last 4 digits F1-Variance = 0, F2-Skewness = 4, F3-curtosis = 5, F4-Entropy = 3

# # print(X_test)
logRegression_BUID = LogisticRegression()
logRegression_BUID.fit(X_train,Y_train.ravel())
logRegression_BUID.fit(X_train,Y_train.ravel()) # Used to train the KNN model based on different neighbor values.
predictions_BUID = logRegression_BUID.predict(np.array([[0, 4, 5]])) # used to make predictions on new data.

print("Prediction using logistic regression for last 4 of BUID is " +str(predictions_BUID))



