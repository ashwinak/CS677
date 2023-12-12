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
Homework Problem # 2
Description of Problem (just a 1-2 line summary!): 
Question 2: In this question you will compare a number of different models using linear systems (including linear regres- sion). 
You choose one feature X as independent variable X and another feature Y as dependent. 
Your choice of X and Y will depend on your facilitator group as follows: 
1. Group 1: X: creatinine phosphokinase (CPK), Y : platelets

"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import FunctionTransformer,StandardScaler
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from sklearn import svm
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.linear_model import LogisticRegression
import seaborn as sns



## This is code for Q1.

### Function to calculate linear kernel SVM
def linearKernelSVM():
    input_dir = os.getcwd()
    seedRecords = os.path.join(input_dir, 'seeds_dataset.csv')
    df_seedRecords = pd.read_csv(seedRecords,sep='\t', header=None)
    feature_names = ['area', 'perimeter', 'compactness','length of kernel','width of kernel','asymmetry coefficient','length of kernel groove.','class']
    df_seedRecords.columns = feature_names
    # Drop rows where 'class' has the value 1
    df_seedRecords_filtered = df_seedRecords.drop(df_seedRecords[df_seedRecords['class'] == 1].index)
    # print(df_seedRecords_filtered)
    df_seedRecords_filtered.to_csv('seeds_dataset_filtered.csv')
    df_seedRecords_filtered['Label'] = df_seedRecords['class'].apply(lambda x: 0 if x == 2 else 1)
    X = df_seedRecords_filtered[['area', 'perimeter', 'compactness','length of kernel','width of kernel','asymmetry coefficient','length of kernel groove.']].values
    Y = df_seedRecords_filtered['Label'].values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=120)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    # Y = df_seedRecords['class'].values
    svm_classifier = svm.SVC(kernel ='linear')
    svm_classifier.fit(X_train,Y_train)
    y_pred = svm_classifier.predict(X_test)
    # new_x = scaler.transform(np.asmatrix([6 , 160]))
    # predicted = svm_classifier.predict(new_x)
    # accuracy1 = svm_classifier.score(X_test, Y_test)
    # print(accuracy1*100)
    accuracy = accuracy_score(Y_test, y_pred)
    print("Accuracy from Linear Kernel SVM is ", round(accuracy*100,2))
    #                       Predicted Negative     Predicted Positive
    # Actual Negative        TN                    FP
    # Actual Positive        FN                    TP

    conf_matrix = confusion_matrix(Y_test, y_pred)
    print("Confusion Matrix:Linear Kernel SVM")
    TN = conf_matrix[[0],[0]][0]
    FP = conf_matrix[[0],[1]][0]
    FN = conf_matrix[[1],[0]][0]
    TP = conf_matrix[[1],[1]][0]
    print("    True Positive is ", TP)
    print("    False Positive is ", FP)
    print("    False Negative is ", FN)
    print("    True Negative is ", TN)
    print("    True Negative Rate is : ", round(TN/(TN+FP),2)*100)
    print("    True Positive Rate is : ", round(TP/(TP+FN),2)*100)
    sns.pairplot(df_seedRecords_filtered, x_vars=['area', 'perimeter', 'compactness','length of kernel','width of kernel','asymmetry coefficient','length of kernel groove.'], y_vars='Label', kind='scatter')
    plt.show()

### Function to calculate Gaussian kernel SVM
def GaussianKernelSVM():
    input_dir = os.getcwd()
    seedRecords = os.path.join(input_dir, 'seeds_dataset.csv')
    df_seedRecords = pd.read_csv(seedRecords,sep='\t', header=None)
    feature_names = ['area', 'perimeter', 'compactness','length of kernel','width of kernel','asymmetry coefficient','length of kernel groove.','class']
    df_seedRecords.columns = feature_names
    # Drop rows where 'class' has the value 1
    df_seedRecords_filtered = df_seedRecords.drop(df_seedRecords[df_seedRecords['class'] == 1].index)
    # print(df_seedRecords_filtered)
    df_seedRecords_filtered.to_csv('seeds_dataset_filtered.csv')
    df_seedRecords_filtered['Label'] = df_seedRecords['class'].apply(lambda x: 0 if x == 2 else 1)
    X = df_seedRecords_filtered[['area', 'perimeter', 'compactness','length of kernel','width of kernel','asymmetry coefficient','length of kernel groove.']].values
    Y = df_seedRecords_filtered['Label'].values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=120)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    # Y = df_seedRecords['class'].values
    svm_classifier = svm.SVC(kernel ='rbf')
    svm_classifier.fit(X_train,Y_train)
    y_pred = svm_classifier.predict(X_test)
    # new_x = scaler.transform(np.asmatrix([6 , 160]))
    # predicted = svm_classifier.predict(new_x)
    # accuracy1 = svm_classifier.score(X_test, Y_test)
    # print(accuracy1*100)
    accuracy = accuracy_score(Y_test, y_pred)
    print("\nAccuracy from Gaussian KernelSVM is ", round(accuracy*100,2))
    #                       Predicted Negative     Predicted Positive
    # Actual Negative        TN                    FP
    # Actual Positive        FN                    TP

    conf_matrix = confusion_matrix(Y_test, y_pred)
    print("Confusion Matrix:GaussianKernelSVM")
    TN = conf_matrix[[0],[0]][0]
    FP = conf_matrix[[0],[1]][0]
    FN = conf_matrix[[1],[0]][0]
    TP = conf_matrix[[1],[1]][0]
    print("    True Positive is ", TP)
    print("    False Positive is ", FP)
    print("    False Negative is ", FN)
    print("    True Negative is ", TN)
    print("    True Negative Rate is : ", round(TN/(TN+FP),2)*100)
    print("    True Positive Rate is : ", round(TP/(TP+FN),2)*100)

### Function to calculate polynomial kernel SVM
def polynomialKernelSVM():
    input_dir = os.getcwd()
    seedRecords = os.path.join(input_dir, 'seeds_dataset.csv')
    df_seedRecords = pd.read_csv(seedRecords,sep='\t', header=None)
    feature_names = ['area', 'perimeter', 'compactness','length of kernel','width of kernel','asymmetry coefficient','length of kernel groove.','class']
    df_seedRecords.columns = feature_names
    # Drop rows where 'class' has the value 1
    df_seedRecords_filtered = df_seedRecords.drop(df_seedRecords[df_seedRecords['class'] == 1].index)
    # print(df_seedRecords_filtered)
    df_seedRecords_filtered.to_csv('seeds_dataset_filtered.csv')
    df_seedRecords_filtered['Label'] = df_seedRecords['class'].apply(lambda x: 0 if x == 2 else 1)
    X = df_seedRecords_filtered[['area', 'perimeter', 'compactness','length of kernel','width of kernel','asymmetry coefficient','length of kernel groove.']].values
    Y = df_seedRecords_filtered['Label'].values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=120)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    # Y = df_seedRecords['class'].values
    svm_classifier = svm.SVC(kernel ='poly',degree =3)
    svm_classifier.fit(X_train,Y_train)
    y_pred = svm_classifier.predict(X_test)
    # new_x = scaler.transform(np.asmatrix([6 , 160]))
    # predicted = svm_classifier.predict(new_x)
    # accuracy1 = svm_classifier.score(X_test, Y_test)
    # print(accuracy1*100)
    accuracy = accuracy_score(Y_test, y_pred)
    print("\nAccuracy from polynomial KernelSVM with degree = 3 is ", round(accuracy*100,2))
    #                       Predicted Negative     Predicted Positive
    # Actual Negative        TN                    FP
    # Actual Positive        FN                    TP

    conf_matrix = confusion_matrix(Y_test, y_pred)
    print("Confusion Matrix:polynomialKernelSVM")
    TN = conf_matrix[[0],[0]][0]
    FP = conf_matrix[[0],[1]][0]
    FN = conf_matrix[[1],[0]][0]
    TP = conf_matrix[[1],[1]][0]
    print("    True Positive is ", TP)
    print("    False Positive is ", FP)
    print("    False Negative is ", FN)
    print("    True Negative is ", TN)
    print("    True Negative Rate is : ", round(TN/(TN+FP),2)*100)
    print("    True Positive Rate is : ", round(TP/(TP+FN),2)*100)

linearKernelSVM()
GaussianKernelSVM()
polynomialKernelSVM()

# This is the code for Q2.

def logisticRegression():
    input_dir = os.getcwd()
    seedRecords = os.path.join(input_dir, 'seeds_dataset.csv')
    df_seedRecords = pd.read_csv(seedRecords,sep='\t', header=None)
    feature_names = ['area', 'perimeter', 'compactness','length of kernel','width of kernel','asymmetry coefficient','length of kernel groove.','class']
    df_seedRecords.columns = feature_names
    # Drop rows where 'class' has the value 1
    df_seedRecords_filtered = df_seedRecords.drop(df_seedRecords[df_seedRecords['class'] == 1].index)
    # print(df_seedRecords_filtered)
    df_seedRecords_filtered.to_csv('seeds_dataset_filtered.csv')
    df_seedRecords_filtered['Label'] = df_seedRecords['class'].apply(lambda x: 0 if x == 2 else 1)
    X = df_seedRecords_filtered[['area', 'perimeter', 'compactness','length of kernel','width of kernel','asymmetry coefficient','length of kernel groove.']].values
    Y = df_seedRecords_filtered['Label'].values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=120)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    logRegression = LogisticRegression()
    logRegression.fit(X_train,Y_train.ravel())
    predictions = logRegression.predict(X_test)
    accuracy = accuracy_score(Y_test, predictions)
    print("\nAccuracy from Logistic Regression is ", round(accuracy*100,2))
    #                       Predicted Negative     Predicted Positive
    # Actual Negative        TN                    FP
    # Actual Positive        FN                    TP

    conf_matrix = confusion_matrix(Y_test, predictions)
    print("Confusion Matrix:logisticRegression")
    TN = conf_matrix[[0],[0]][0]
    FP = conf_matrix[[0],[1]][0]
    FN = conf_matrix[[1],[0]][0]
    TP = conf_matrix[[1],[1]][0]
    print("    True Positive is ", TP)
    print("    False Positive is ", FP)
    print("    False Negative is ", FN)
    print("    True Negative is ", TN)
    print("    True Negative Rate is : ", round(TN/(TN+FP),2)*100)
    print("    True Positive Rate is : ", round(TP/(TP+FN),2)*100)
logisticRegression()