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
Date: 11/20/2023
Homework Problem # 4
Description of Problem (just a 1-2 line summary!): 
Question 4 :
1. take your best value k ∗. For each of the four features f1, . . . , f4, drop that feature from both Xtrain and Xtest. ...
2. did accuracy increase in any of the 4 cases compared with accuracy when all 4 features are used? 
3. which feature, when removed, contributed the most to loss of accuracy? 
4. which feature, when removed, contributed the least to loss of accuracy?

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
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix



## This is code for Q4

#load csv to dataframe
input_dir = os.getcwd()
bankNotes = os.path.join(input_dir,'data_banknote_authentication.csv')

df_bankNotes = pd.read_csv(bankNotes)
#add column names
column_names = ['F1-Variance', 'F2-Skewness', 'F3-curtosis', 'F4-Entropy', 'F5-Class']

if len(column_names) == len(df_bankNotes.columns):
    df_bankNotes.columns = column_names

#add color column with values populated.
df_bankNotes['Color'] = df_bankNotes['F5-Class'].apply(lambda x: 'Green' if x ==0 else 'Red')

#function for calculating accuracy based on a specific feature drop.
def featureDrop(featureName=None):
    if featureName == 'F1-Variance':
        X = df_bankNotes[['F2-Skewness', 'F3-curtosis','F4-Entropy']].values
        y = df_bankNotes[['Color']].values
    elif featureName == 'F2-Skewness':
        X = df_bankNotes[['F1-Variance','F3-curtosis','F4-Entropy']].values
        y = df_bankNotes[['Color']].values
    elif featureName == 'F3-curtosis':
        X = df_bankNotes[['F1-Variance', 'F2-Skewness','F4-Entropy']].values
        y = df_bankNotes[['Color']].values
    elif featureName == 'F4-Entropy':
        X = df_bankNotes[['F1-Variance', 'F2-Skewness', 'F3-curtosis']].values
        y = df_bankNotes[['Color']].values
    elif featureName == None:
        X = df_bankNotes[['F1-Variance', 'F2-Skewness', 'F3-curtosis','F4-Entropy']].values
        y = df_bankNotes[['Color']].values

    #Data set 50/50 split between train and test.
    X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size=0.5,random_state=120)

    # Data preprocessing using standard scalar before fitting into KNN model.
    scalar = StandardScaler()
    X_train = scalar.fit_transform(X_train) # For training data to both learn parameters and apply transformation.
    X_test = scalar.transform(X_test) # For applying learned transformation to the new data set.

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train,Y_train.ravel()) # Used to train the KNN model based on different neighbor values.
    predictions = knn.predict(X_test) # used to make predictions on new data.
    print("Accuracy from KNN model for k=5 after dropping " +str(featureName) + ": " + str(round(np.mean(predictions == Y_test.ravel())*100,2))) #compare predictions from the KNN model against Y_test output.


featureDrop('F1-Variance')
featureDrop('F2-Skewness')
featureDrop('F3-curtosis')
featureDrop('F4-Entropy')
featureDrop()



