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
Homework Problem # 3
Description of Problem (just a 1-2 line summary!): 
Question 3: 

Use Decision Tree to answer these questions: 

1. using split your set 50/50, train NB DT on Xtrain and predict class labels in Xtest 

2. what is the accuracy? 3. compute the confusion matrix


"""

import os
# import csv
# import statistics
import pandas as pd
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
# import xlrd


## This is code for Q4
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

X = df_clinicalRecords_filtered.dropna().astype(int)[['LB', 'ALTV', 'Min','Mean']].values # Features
y = df_clinicalRecords_filtered.dropna().astype(int)[['CLASS']].values # Class labels

#Data set 50/50 split between train and test.
X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size=0.5,random_state=120)

err_rate = []
estimator = []
depth = []
for n in range (1,11):
    for d in range (1,6):
        model = RandomForestClassifier(n_estimators =n,max_depth =d,criterion ='entropy')
        model.fit (X_train, Y_train.ravel())
        prediction = model.predict(X_test)
        accuracy = accuracy_score(Y_test, prediction)
        print("\nFor n=" + str(n) + " and d= " + str(d))
        print(f"    Accuracy of Randmom Forest Classifier with : {accuracy * 100 :.2f}%")
        conf_matrix = confusion_matrix(Y_test, prediction)
        conf_matrix_df = pd.DataFrame(conf_matrix, index=['Actual Positive', 'Actual Negative'], columns=['Predicted Positive', 'Predicted Negative'])
        print("Confusion Matrix:")
        TP = conf_matrix[[0],[0]][0]
        FP = conf_matrix[[0],[1]][0]
        TN = conf_matrix[[1],[1]][0]
        FN = conf_matrix[[1],[0]][0]
        print("    True Positive is ", TP)
        print("    False Positive is ", FP)
        print("    False Negative is ", FN)
        print("    True Negative is ", TN)
        print("    True Positive Rate is : ", round(TP/(TP+FN),2)*100)
        print("    True Negative Rate is : ", round(TN/(TN+FP),2)*100)
        # Compute and print the error rate
        error_rate = (conf_matrix[0, 1] + conf_matrix[1, 0]) / len(Y_test)
        print(f"Error Rate: {error_rate:.2%}")
        err_rate.append([error_rate*100])
        estimator.append(n)
        depth.append(d)

print("Minimum error rate is ", min(err_rate))

# Convert data to NumPy arrays
N_values = np.array(estimator)
d_values = np.array(depth)
err_rate_reshaped = np.tile(err_rate, (1, 50))
error_rates = np.array(err_rate_reshaped)

# print(N_values.shape)
# print(d_values.shape)
# print(error_rates)

# Create a meshgrid for N and d values
N, d = np.meshgrid(N_values, d_values)

# Create a pcolormesh plot
plt.pcolormesh(N, d, error_rates, shading='auto', cmap='viridis', edgecolors='k', linewidths=0.5)
plt.colorbar(label='Error Rate')
plt.xlabel('N values')
plt.ylabel('d values')
plt.title('Error Rates for Different estimator and depth Combinations')

# Show the plot
plt.show()