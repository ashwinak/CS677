# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 14:37:29 2018
@author: ashwinar@bu.edu

This is a machine learning algorithm implemenation to detect DDoS attacks based on the sflow data received from the network equipement.
Various ML algorithms are compared against its accuracy for accurate prediction of DDoS attacks.

"""
"""
Ashwin Arunkumar
Class: CS 677
Date: 12/12/2023
Term project 
Description of Project (just a 1-2 line summary!):

This is a machine learning algorithm implemenation to detect DDoS attacks based on the sflow data received from the network equipement.
Various ML algorithms are compared against its accuracy. 
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import copy

# import matplotlib.pyplot as plt
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.preprocessing import FunctionTransformer
# import warnings
# warnings.filterwarnings("ignore", category=UserWarning)
# import seaborn as sns


# A total of 225K rows are part of this data set. For this project, only 200 rows will be used per flow type to train the model.
# flow type 1 : BENIGN : 1000
# flow type 2 : Ddos : 1000

fileName = "DDOS_Capture.csv"
new_ClassLabel = 'Class_Label'

def dataPreProcessing(fileName,class_feature, class0_Label,class1_Label,new_ClassLabel,sample_count=len(pd.read_csv(fileName))):
    """
    This function takes input file name, class_feature column, existing class labels, new computed class label  and length of sample_count.

    Parameters:
    - fileName (str): Name of the input file in current working directory.
    - class_feature : This is called the dependent variable or y column in data pre processing.
    - class0_Label : The values in class features is expected to be in binary. This label maps to one of the binary value.
    - class1_Label : The values in class features is expected to be in binary. This label maps to one of the binary value.
    - new_ClassLabel : This is the new class label computed based on values from class_features column.
    - sample_count : This is the number of samples that will be filtered and stored in a dataframe. IOW, the number of rows from the input file.

    Returns:
    dataframe: This function returns the the data frame post processing.
    """

    input_dir = os.getcwd()
    DDoS_Data = os.path.join(input_dir, fileName)
    df_DDoS = pd.read_csv(DDoS_Data)

    # Removing leading or trailing spaces.
    df_DDoS = df_DDoS.rename(columns=lambda x: x.strip())
    condition_Benign = df_DDoS[class_feature] == class0_Label
    condition_DDoS  = df_DDoS[class_feature] == class1_Label
    random_rows_class0_Label = df_DDoS[condition_Benign].sample(n=sample_count, random_state=42)
    random_rows_class1_Label = df_DDoS[condition_DDoS].sample(n=sample_count, random_state=42)
    df_DDoS_filtered = pd.concat([random_rows_class0_Label, random_rows_class1_Label], ignore_index=True)
    df_DDoS_filtered.to_csv('DDOS_Capture_Filtered.csv',index=False)

    # create a new class label for DDOS and Benign flows. Convert text labels to integer labels.
    df_DDoS_filtered[new_ClassLabel] = df_DDoS_filtered[class_feature].apply(lambda x: 0 if x =='BENIGN' else 1)
    return df_DDoS_filtered

# Selecting columns for training. There are a total of 86 features available. All 86 except dependent variable will be used in the training set.
def trainTestLogisticRegression(dataFrame,new_ClassLabel,feature_list_toDrop):
    """
    This function takes dataframe obtained from dataPreProcessing function, new_classLabel value and feature list to drop if they cannot be
    passed to ML for training. Reasons for dropping features could be if the values are strings or float values.

    Parameters:
    - dataFrame : pre processed dataframe returned from previous function.
    - new_ClassLabel : This is the new class label computed based on values from class_features column.
    - feature_list_toDrop : List of features to drop if they cannot be passed to ML training model.

    Returns:
    Accuracy of the training model.
    """
    y = dataFrame[[new_ClassLabel]].values
    dataFrame.drop([new_ClassLabel,*feature_list_toDrop], axis=1, inplace=True)
    X = dataFrame.values

    #Data set 50/50 split between train and test
    X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size=0.5,random_state=120)

    logRegression = LogisticRegression()
    logRegression.fit(X_train,Y_train.ravel())
    predictions = logRegression.predict(X_test)
    print("Total columns used for training is: ", len(dataFrame.columns))
    print("     Accuracy from Logistic regression model is: " + str(round(np.mean(predictions == Y_test.ravel())*100,2)))

def applyCorrelationMatrix(dataFrame,new_ClassLabel,feature_list_toDrop,threshold):
    """
    This function takes dataframe obtained from dataPreProcessing function, new_classLabel value and feature list to drop if they cannot be
    passed to ML for training. Reasons for dropping features could be if the values are strings or float values.
    This function also takes the threshold value i.e the value that can be used to drop highly correlated features in a given data frame.

    Parameters:
    - dataFrame : pre processed dataframe returned from previous function.
    - new_ClassLabel : This is the new class label computed based on values from class_features column.
    - feature_list_toDrop : List of features to drop if they cannot be passed to ML training model.

    Returns:
    Accuracy of the training model after dropping highly correlated features.
    """
    df_full = copy.copy(dataFrame)
    dataFrame.drop([new_ClassLabel,*feature_list_toDrop], axis=1, inplace=True)
    # Based on the correlation matrices, the following features are deleted.
    columns_to_drop = np.where(np.abs((dataFrame.corr()) > threshold))
    columns_to_drop = [(dataFrame.columns[x], dataFrame.columns[y])
                       for x, y in zip(*columns_to_drop)
                       if x != y and x < y]
    # Drop the identified columns
    for col1, col2 in columns_to_drop:
        if col2 in df_full.columns:
            df_full.drop(col2, axis=1, inplace=True)
    print("\nDropping high positive Correlated features...")
    trainTestLogisticRegression(df_full,new_ClassLabel,feature_list_toDrop)

if __name__ == "__main__":
    df_DDoS_filtered = dataPreProcessing(fileName,'Label','BENIGN','DDoS',new_ClassLabel,1000)
    df_full = copy.copy(df_DDoS_filtered)
    feature_list_toDrop = ['Label','Flow ID','Source IP','Destination IP','Timestamp']
    trainTestLogisticRegression(df_DDoS_filtered,'Class_Label',feature_list_toDrop)
    applyCorrelationMatrix(df_full,new_ClassLabel,feature_list_toDrop,0.8)


