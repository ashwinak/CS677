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
1. load the data into Pandas dataframe. Extract two dataframes with the above 4 features: df 0 for surviving patients (DEATH EVENT = 0) 
and df 1 for deceased patients (DEATH EVENT = 1) 
2. for each dataset, construct the visual representations of correponding correlation matrices M0 (from df 0) and M1 (from df 1) 
and save the plots into two separate files 
3. examine your correlation matrix plots visually and answer the following: 
(a) which features have the highest correlation for surviving patients? 
(b) which features have the lowest correlation for surviving patients? 
(c) which features have the highest correlation for deceased patients? 
(d) which features have the lowest correlation for deceased patients?
(e) are results the same for both cases?

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



## This is code for Q1

input_dir = os.getcwd()
clinicalRecords = os.path.join(input_dir,'heart_failure_clinical_records_dataset.csv')

df_clinicalRecords = pd.read_csv(clinicalRecords)

required_features = ['creatinine_phosphokinase','serum_creatinine','serum_sodium','platelets','DEATH_EVENT']
df_clinicalRecords_filtered = df_clinicalRecords.drop(df_clinicalRecords.columns.difference(required_features), axis=1)

df_0 = df_clinicalRecords_filtered[df_clinicalRecords_filtered['DEATH_EVENT'] == 0]
df_1 = df_clinicalRecords_filtered[df_clinicalRecords_filtered['DEATH_EVENT'] == 1]


print("Death_Event = 0")
print(df_0)

print("Death_Event = 1")
print(df_1)

# code for two different correlation matrices
correlation_matrix_df0 = df_0.corr().round(2)#.fillna(0)
correlation_matrix_df1 = df_1.corr().round(2)#.fillna(1)

# code for correlation matrices with file save
sns.heatmap(correlation_matrix_df0, annot=True, vmin=-1,vmax=1,cmap="coolwarm", fmt=".2f",linewidths=1, linecolor='black',annot_kws={"size": 8})
plt.savefig("M0_surviving_patients.pdf",bbox_inches='tight',dpi=300)
plt.figure()
sns.heatmap(correlation_matrix_df1, annot=True, vmin=-1,vmax=1,cmap="coolwarm", fmt=".2f",linewidths=1, linecolor='black',annot_kws={"size": 8})
plt.savefig("M1_surviving_patients.pdf",bbox_inches='tight', dpi=300)


## Code to find various high and low correlation matrix feature pairs.

# code for highest negative correlation
max_negative_correlation_pair = correlation_matrix_df1[correlation_matrix_df1 < 0].unstack().idxmax()

# code for lowest negative correlation
min_negative_correlation_pair = correlation_matrix_df1[correlation_matrix_df1 < 0].unstack().idxmin()

print("Pair with the highest negative correlation:")
print(max_negative_correlation_pair, correlation_matrix_df1.loc[max_negative_correlation_pair])

print("\nPair with the lowest negative correlation:")
print(min_negative_correlation_pair, correlation_matrix_df1.loc[min_negative_correlation_pair])

# code for lowest positive correlation
min_positive_correlation_pair = correlation_matrix_df1[(correlation_matrix_df1 >= 0) & (correlation_matrix_df1 < 1)].unstack().idxmin()

print("Pair with the lowest positive correlation:")
print(min_positive_correlation_pair, correlation_matrix_df1.loc[min_positive_correlation_pair])

## Correlation pairs for df0_surviving_patients.
highest_correlation_feature_pair = correlation_matrix_df0.unstack().sort_values(ascending=False)
# print(highest_correlation_feature_pair)

# Select feature pairs with highest correlation. Closer to 1 is highest.
highest_correlation_feature_pair = highest_correlation_feature_pair[highest_correlation_feature_pair.index.get_level_values(0) != highest_correlation_feature_pair.index.get_level_values(1)]

print("%%%%%%%%%%%%%%% Feature pairs with top highest correlation for M0_surviving_patients:")
print(highest_correlation_feature_pair.head())

lowest_correlation_feature_pair = correlation_matrix_df0.unstack().sort_values(ascending=True)
# print(lowest_correlation_feature_pair)

# Select feature pairs with highest correlation. Closer to 1 is highest.
lowest_correlation_feature_pair = lowest_correlation_feature_pair[lowest_correlation_feature_pair.index.get_level_values(0) != lowest_correlation_feature_pair.index.get_level_values(1)]

print("%%%%%%%%%%%%%%% Feature pairs with top lowest correlation for M0_surviving_patients:")
print(lowest_correlation_feature_pair.head())

## Correlation pairs for df1_deceased_patients.
highest_correlation_feature_pair = correlation_matrix_df1.unstack().sort_values(ascending=False)
# print(highest_correlation_feature_pair)

# Select feature pairs with highest correlation. Closer to 1 is highest.
highest_correlation_feature_pair = highest_correlation_feature_pair[highest_correlation_feature_pair.index.get_level_values(0) != highest_correlation_feature_pair.index.get_level_values(1)]

print("%%%%%%%%%%%%%%% Feature pairs with top highest correlation M1_deceased_patients:")
print(highest_correlation_feature_pair.head())

lowest_correlation_feature_pair = correlation_matrix_df1.unstack().sort_values(ascending=True)
# print(lowest_correlation_feature_pair)

# Select feature pairs with highest correlation. Closer to 1 is highest.
lowest_correlation_feature_pair = lowest_correlation_feature_pair[lowest_correlation_feature_pair.index.get_level_values(0) != lowest_correlation_feature_pair.index.get_level_values(1)]

print("%%%%%%%%%%%%%%% Feature pairs with top lowest correlation M1_deceased_patients:")
print(lowest_correlation_feature_pair.head())



# X = df_clinicalRecords[['F1-Variance', 'F2-Skewness', 'F3-curtosis']].values
# y = df_clinicalRecords[['Color']].values
#
# #Data set 50/50 split between train and test.
# X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size=0.5,random_state=120)
#
# # Data preprocessing using standard scalar before fitting into KNN model.
# # print(X_test)
# scalar = StandardScaler()
# X_train = scalar.fit_transform(X_train) # For training data to both learn parameters and apply transformation.
# X_test = scalar.transform(X_test) # For applying learned transformation to the new data set.
#
# # Fit into model and calculate the accuracy
#
# logRegression = LogisticRegression()
# logRegression.fit(X_train,Y_train.ravel())
# predictions = logRegression.predict(X_test)
# print("Accuracy from Logistic regression model is: " + str(round(np.mean(predictions == Y_test.ravel())*100,2))) #compare predictions from the KNN model against Y_test output.
#
# if Y_test.ravel().shape == predictions.shape:
#     TP = FP = TN = FN = TPR = TNR = accuracy = totalCount = 0
#     for (i), value in np.ndenumerate(Y_test.ravel()):
#         if (Y_test.ravel()[i] == 'Green') and (predictions[i]== 'Green'):
#             TP+=1
#         elif (Y_test.ravel()[i] == 'Red') and (predictions[i] == 'Red'):
#             TN+=1
#         elif (Y_test.ravel()[i] == 'Red') and (predictions[i] == 'Green'):
#             FP+=1
#         elif (Y_test.ravel()[i] == 'Green') and (predictions[i] == 'Red'):
#             FN+=1
#     print("True Positive is " + str(TP))
#     print("True Negative is " + str(TN))
#     print("False Positive is " + str(FP))
#     print("False Negative is " + str(FN))
#     print("True Positive Rate is : " + str(round(TP/(TP+FN),2)*100))
#     print("True Negative Rate is : " + str(round(TN/(TN+FP),2)*100))
#
#
# #BU ID : U08370453. Last 4 digits F1-Variance = 0, F2-Skewness = 4, F3-curtosis = 5, F4-Entropy = 3
#
# # # print(X_test)
# logRegression_BUID = LogisticRegression()
# logRegression_BUID.fit(X_train,Y_train.ravel())
# logRegression_BUID.fit(X_train,Y_train.ravel()) # Used to train the KNN model based on different neighbor values.
# predictions_BUID = logRegression_BUID.predict(np.array([[0, 4, 5]])) # used to make predictions on new data.
#
# print("Prediction using logistic regression for last 4 of BUID is " +str(predictions_BUID))
#
#
#
