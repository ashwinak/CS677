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
Date: 11/12/2023
Homework Problem # 2
Description of Problem (just a 1-2 line summary!): 
Question 2 :
1. split your dataset X into training Xtrain and Xtesting parts (50/50 split). Using ”pairplot” from seaborn package, 
plot pairwise relationships in Xtrain separately for class 0 and class 1. Save your results into 2 pdf files ”good bills.pdf” and ”fake bills.pdf” 
2. visually examine your results. Come up with three simple comparisons that you think may be sufficient to detect a fake bill. 
3. apply your simple classifier to Xtest and compute predicted class labels 
4. comparing your predicted class labels with true labels, com- pute the following:
5. summarize your findings in the table as shown below: 
6. does you simple classifier gives you higher accuracy on iden- tifying ”fake” bills or ”real” bills” Is your accuracy better than 50% (”coin” flipping)?
"""

import os
import csv
import statistics
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

## This is code for Q2

input_dir = os.getcwd()
bankNotes = os.path.join(input_dir,'data_banknote_authentication.csv')

#read csv to a data frame
df_bankNotes = pd.read_csv(bankNotes)

column_names = ['F1-Variance', 'F2-Skewness', 'F3-curtosis', 'F4-Entropy', 'F5-Class']

#add columns to the dataframe
if len(column_names) == len(df_bankNotes.columns):
    df_bankNotes.columns = column_names

#add color column and populate values based on the class.
df_bankNotes['Color'] = df_bankNotes['F5-Class'].apply(lambda x: 'Green' if x ==0 else 'Red')

# split features and target. Features to X data frame and target to y dataframe.
X = df_bankNotes[['F1-Variance', 'F2-Skewness', 'F3-curtosis', 'F4-Entropy','F5-Class']]
y = df_bankNotes[['Color']]

#split train,test and target data 50/50.
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.5,random_state=120)

# scatter plots for X_train
sns.pairplot(X_train, hue='F5-Class')
plt.show()

# scatter plots for X_train with class =0
class0 = X_train[X_train['F5-Class'] == 0]
sns.pairplot(class0, hue='F5-Class').savefig("class0.pdf")
plt.show()

# scatter plots for X_train with class =1
class1 = X_train[X_train['F5-Class'] == 1]
sns.pairplot(class1, hue='F5-Class').savefig("class1.pdf")
plt.show()

#Simple classifier with 3 features.

X_test['Predicted Class'] = ''
for index, row in X_test.iterrows():
    if (row['F1-Variance'] and row['F2-Skewness'] >= 0) and (row['F3-curtosis'] < 0):
        X_test.at[index, 'Predicted Class'] = 0
    else:
        X_test.at[index, 'Predicted Class'] = 1

print("### Test data with predictions")
print(X_test.head())

X_test.to_csv('xtest.csv', index=False)

# Compare the outcome of manual classifier with class labels and compute confusion matrix for the same.
def ComparePredictions(data,classLabel,predictionLabel):
    TP = FP = TN = FN = TPR = TNR = accuracy = totalCount = 0
    for index, row in data.iterrows():
        totalCount+=1
        if data.loc[index, classLabel] == data.loc[index, predictionLabel]:
            accuracy+=1
    print("Predicted accuracy for manual classifier is : " + str(round(accuracy/(totalCount),2)*100))

    for index, row in data.iterrows():
        if (row[classLabel] == 0) and (row[predictionLabel]) == 0:
            TP+=1
        elif (row[classLabel] == 1) and (row[predictionLabel]) == 1:
            TN+=1
        elif (row[classLabel] == 1) and (row[predictionLabel]) == 0:
            FP+=1
        elif (row[classLabel] == 0) and (row[predictionLabel]) == 1:
            FN+=1
    print("True Positive is " + str(TP))
    print("True Negative is " + str(TN))
    print("False Positive is " + str(FP))
    print("False Negative is " + str(FN))
    print("True Positive Rate is : " + str(round(TP/(TP+FN),2)*100))
    print("True Negative Rate is : " + str(round(TN/(TN+FP),2)*100))

print("Confusion matrix for X_test dataset")

ComparePredictions(X_test,'F5-Class','Predicted Class')

print("Confusion matrix for full data set")

df_bankNotes['Predicted Class'] = ''
for index, row in df_bankNotes.iterrows():
    if (row['F1-Variance'] and row['F2-Skewness'] >= 0) and (row['F3-curtosis'] < 0):
        df_bankNotes.at[index, 'Predicted Class'] = 0
    else:
        df_bankNotes.at[index, 'Predicted Class'] = 1

ComparePredictions(df_bankNotes,'F5-Class','Predicted Class')

#Simple classifier with 3 features predicting lable based on last 4 of BUID as feature values.

X_test_BUID = pd.DataFrame(data=[[0, 4, 5]], columns=['F1-Variance', 'F2-Skewness', 'F3-curtosis'])

X_test_BUID['Predicted Class'] = ''
for index, row in X_test_BUID.iterrows():
    if (row['F1-Variance'] and row['F2-Skewness'] >= 0) and (row['F3-curtosis'] < 0):
        X_test_BUID.at[index, 'Predicted Class'] = 0
        print("Prediction using simple classifier  for last 4 of BUID is Green")
    else:
        X_test_BUID.at[index, 'Predicted Class'] = 1
        print("Prediction using simple classifier for last 4 of BUID is Red")
