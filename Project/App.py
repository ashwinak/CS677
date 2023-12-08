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
Date: 12/12/2023
Term project 
Description of Problem (just a 1-2 line summary!): 
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import FunctionTransformer
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import seaborn as sns


# A total of 225K rows are part of this data set. For this project, only 200 rows will be used per flow type to train the model.
# flow type 1 : BENIGN : 1000
# flow type 2 : Ddos : 1000

input_dir = os.getcwd()
DDoS_Data = os.path.join(input_dir, 'DDOS_Capture.csv')
df_DDoS = pd.read_csv(DDoS_Data)

# Removing leading or trailing spaces.
df_DDoS = df_DDoS.rename(columns=lambda x: x.strip())

condition_Benign = df_DDoS['Label'] == 'BENIGN'
condition_DDoS  = df_DDoS['Label'] == 'DDoS'

random_rows_Benign = df_DDoS[condition_Benign].sample(n=1000, random_state=42)
random_rows_DDoS = df_DDoS[condition_DDoS].sample(n=1000, random_state=42)
df_DDoS_filtered = pd.concat([random_rows_Benign, random_rows_DDoS], ignore_index=True)
df_DDoS_filtered_Drop_Correlated = pd.concat([random_rows_Benign, random_rows_DDoS], ignore_index=True)

df_DDoS_filtered.drop(['Fwd Header Length.1'], axis=1, inplace=True)
df_DDoS_filtered_Drop_Correlated.drop(['Fwd Header Length.1'], axis=1, inplace=True)

df_DDoS_filtered.to_csv('DDOS_Capture_Filtered.csv',index=False)

# Removing leading or trailing spaces.
# df_DDoS_filtered = df_DDoS_filtered.rename(columns=lambda x: x.strip())

# df_DDoS_filtered_Drop_Correlated = df_DDoS_filtered_Drop_Correlated.rename(columns=lambda x: x.strip())
# df_DDoS_filtered_Drop_Correlated.columns = df_DDoS_filtered_Drop_Correlated.columns.str.strip('.123').str.rstrip('.123').str.lstrip('.123')

# create a new class label for DDOS and Benign flows. Convert text labels to integer labels.
df_DDoS_filtered['Class_Label'] = df_DDoS_filtered['Label'].apply(lambda x: 0 if x =='BENIGN' else 1)

# print(df_DDoS_filtered)

# Selecting columns for training. There are a total of 86 features available. All 86 except dependent variable will be used in the training set.

y = df_DDoS_filtered[['Class_Label']].values
df_DDoS_filtered.drop(['Class_Label', 'Label','Flow ID','Source IP','Destination IP','Timestamp'], axis=1, inplace=True)
X = df_DDoS_filtered.values

#Data set 50/50 split between train and test.l
X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size=0.5,random_state=120)

logRegression = LogisticRegression()
logRegression.fit(X_train,Y_train.ravel())
predictions = logRegression.predict(X_test)
print("Accuracy from Logistic regression model is: " + str(round(np.mean(predictions == Y_test.ravel())*100,2)))

# code for correlation matrices with file save
sns.heatmap(df_DDoS_filtered.corr(),annot=True, vmin=-2,vmax=3,cmap="coolwarm", fmt=".2f",linewidths=1, linecolor='black',annot_kws={"size": 10})
plt.title("DF")
plt.figure()
# sns.heatmap(correlation_matrix_df1, annot=True, vmin=-2,vmax=3,cmap="coolwarm", fmt=".2f",linewidths=1, linecolor='black',annot_kws={"size": 10})
# plt.title("DF1")
# plt.show()


df_DDoS_filtered_Drop_Correlated.drop(['Label','Flow ID','Source IP','Destination IP','Timestamp'], axis=1, inplace=True)

# Based on the correlation matrices, the following features are deleted.

threshold = 0.8

# Identify columns to drop based on the correlation matrix
columns_to_drop = np.where(np.abs((df_DDoS_filtered_Drop_Correlated.corr()) > threshold))

columns_to_drop = [(df_DDoS_filtered_Drop_Correlated.columns[x], df_DDoS_filtered_Drop_Correlated.columns[y])
                   for x, y in zip(*columns_to_drop)
                   if x != y and x < y]
# Drop the identified columns

for col1, col2 in columns_to_drop:
    if col2 in df_DDoS_filtered_Drop_Correlated.columns:
        df_DDoS_filtered_Drop_Correlated.drop(col2, axis=1, inplace=True)

# print(df_DDoS_filtered_Drop_Correlated.columns)

# After dropping highly correlated features, the total number of features is reduced to 41.


# Selecting columns for training. There are a total of 86 features available. All 86 except dependent variable will be used in the training set.

# y = df_DDoS_filtered[['Class_Label']].values
# df_DDoS_filtered.drop(['Class_Label', 'Label','Flow ID','Source IP','Destination IP','Timestamp'], axis=1, inplace=True)
X = df_DDoS_filtered_Drop_Correlated.values

#Data set 50/50 split between train and test.l
X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size=0.5,random_state=120)

logRegression = LogisticRegression()
logRegression.fit(X_train,Y_train.ravel())
predictions = logRegression.predict(X_test)
print("Accuracy from Logistic regression model after removing highly correlated features : " + str(round(np.mean(predictions == Y_test.ravel())*100,2)))



