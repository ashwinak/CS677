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

"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import FunctionTransformer
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


## This is code for Q2. Select features for Group 1 and store surviving and deceased patients in its own data frame.

### Function to calculate simple linear regression.
def SimpleLinearRegression(DEATH_EVENT):
    input_dir = os.getcwd()
    clinicalRecords = os.path.join(input_dir, 'heart_failure_clinical_records_dataset.csv')
    df_clinicalRecords = pd.read_csv(clinicalRecords)
    required_features = ['creatinine_phosphokinase', 'platelets', 'DEATH_EVENT']
    df_clinicalRecords_filtered = df_clinicalRecords.drop(df_clinicalRecords.columns.difference(required_features),
                                                          axis=1)
    df_0 = df_clinicalRecords_filtered[df_clinicalRecords_filtered['DEATH_EVENT'] == 0]  # .values
    df_1 = df_clinicalRecords_filtered[df_clinicalRecords_filtered['DEATH_EVENT'] == 1]  # .values
    if DEATH_EVENT == 0:
        X = df_0[['creatinine_phosphokinase']]  # X is a matrix of 2d array.
        y = df_0[['platelets']]  # Y dependent variable can be 1d array.
        title = 'Simple Linear Regression for surviving patients'
    elif DEATH_EVENT == 1:
        X = df_1[['creatinine_phosphokinase']]  # X is a matrix of 2d array.
        y = df_1[['platelets']]  # Y dependent variable can be 1d array.
        title = 'Simple Linear Regression for deceased patients'

    # (a) fit the model on Xtrain. Data set 50/50 split between train and test.
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.5, random_state=120)
    LinearRegressModel = LinearRegression(fit_intercept=True)
    LinearRegressModel.fit(X_train, Y_train)

    # (b) print the weights (a, b, . . .)
    # Simple linear regression eq is (Y = slope * X_surviving + intercept)
    slope = LinearRegressModel.coef_[0]  # slope value after model training
    intercept = LinearRegressModel.intercept_  # intercept value after model training.
    print(title + " score is ", LinearRegressModel.score(X_train, Y_train))
    print(" Slope is :", slope[0])
    print(" Intercept is :", intercept[0])

    # (c) compute predicted values using Xtest
    y_predict = LinearRegressModel.predict(X_test)

    # (d) plot (if possible) predicted and actual values in Xtest
    plt.scatter(X_test, Y_test, color="blue")
    plt.plot(X_test, y_predict, color="blue", lw=3)
    plt.xlabel('X_test')
    plt.ylabel('Y_test')
    plt.title(title)
    plt.savefig(title+ ".pdf",bbox_inches='tight',dpi=300)
    plt.show()

    # (e) compute (and print) the corresponding loss function
    sse = np.sum((Y_test - y_predict) ** 2, axis=0)
    print(" SSE (Sum of Squared Errors) is :", sse)

SimpleLinearRegression(0)
SimpleLinearRegression(1)


### Function to calculate quadratic linear regression.

def QuadraticLinearRegression(DEATH_EVENT):
    input_dir = os.getcwd()
    clinicalRecords = os.path.join(input_dir, 'heart_failure_clinical_records_dataset.csv')
    df_clinicalRecords = pd.read_csv(clinicalRecords)
    required_features = ['creatinine_phosphokinase', 'platelets', 'DEATH_EVENT']
    df_clinicalRecords_filtered = df_clinicalRecords.drop(df_clinicalRecords.columns.difference(required_features),
                                                          axis=1)
    df_0 = df_clinicalRecords_filtered[df_clinicalRecords_filtered['DEATH_EVENT'] == 0]  # .values
    df_1 = df_clinicalRecords_filtered[df_clinicalRecords_filtered['DEATH_EVENT'] == 1]  # .values
    if DEATH_EVENT == 0:
        X = df_0[['creatinine_phosphokinase']]  # X is a matrix of 2d array.
        y = df_0[['platelets']]  # Y dependent variable can be 1d array.
        title = 'Quadratic Polynomial Regression for surviving patients'
    elif DEATH_EVENT == 1:
        X = df_1[['creatinine_phosphokinase']]  # X is a matrix of 2d array.
        y = df_1[['platelets']]  # Y dependent variable can be 1d array.
        title = 'Quadratic Polynomial Regression for deceased patients'

    # (a) fit the model on Xtrain. Data set 50/50 split between train and test.
    # Transform the input features to include polynomial terms (degree=2)

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.5, random_state=120)

    poly = PolynomialFeatures(degree=2)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    # Y_train = poly.fit_transform(Y_train)

    LinearRegressModel = LinearRegression()
    LinearRegressModel.fit(X_train_poly, Y_train)

    # (b) print the weights (a, b, . . .)
    # Simple linear regression eq is (Y = slope * X_surviving + intercept)

    slope = LinearRegressModel.coef_  # slope value after model training
    intercept = LinearRegressModel.intercept_  # intercept value after model training.
    # weights = np.polyfit(X_train_poly[:, 1], Y_train, 2)
    print(title + " score is ", LinearRegressModel.score(X_train_poly, Y_train))
    print(" Slope is :", slope)
    print(" Intercept is :", intercept)
    # print(" Weights are ", weights.ravel())

    # (c) compute predicted values using Xtest
    y_predict = LinearRegressModel.predict(X_test_poly)

    # (d) plot (if possible) predicted and actual values in Xtest
    x_smooth = np.linspace(X.min(), X.max(), 100)
    x_smooth_poly = poly.transform(x_smooth.reshape(-1, 1))
    y_smooth = LinearRegressModel.predict(x_smooth_poly)

    # Plot the original data points and the cubic polynomial curve
    plt.scatter(X, y, color='blue')
    plt.plot(x_smooth, y_smooth, color='blue', lw=3)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.savefig(title+ ".pdf",bbox_inches='tight',dpi=300)
    plt.show()

    # (e) compute (and print) the corresponding loss function
    sse = np.sum((Y_test - y_predict) ** 2, axis=0)
    print(" SSE (Sum of Squared Errors) is :", sse)


QuadraticLinearRegression(0)
QuadraticLinearRegression(1)

def cubicPolynomialModel(DEATH_EVENT):
    input_dir = os.getcwd()
    clinicalRecords = os.path.join(input_dir, 'heart_failure_clinical_records_dataset.csv')
    df_clinicalRecords = pd.read_csv(clinicalRecords)
    required_features = ['creatinine_phosphokinase', 'platelets', 'DEATH_EVENT']
    df_clinicalRecords_filtered = df_clinicalRecords.drop(df_clinicalRecords.columns.difference(required_features),axis=1)
    df_0 = df_clinicalRecords_filtered[df_clinicalRecords_filtered['DEATH_EVENT'] == 0]  # .values
    df_1 = df_clinicalRecords_filtered[df_clinicalRecords_filtered['DEATH_EVENT'] == 1]  # .values
    if DEATH_EVENT == 0:
        X = df_0[['creatinine_phosphokinase']]  # X is a matrix of 2d array.
        y = df_0[['platelets']]  # Y dependent variable can be 1d array.
        title = 'Cubic Polynomial Regression for surviving patients'
    elif DEATH_EVENT == 1:
        X = df_1[['creatinine_phosphokinase']]  # X is a matrix of 2d array.
        y = df_1[['platelets']]  # Y dependent variable can be 1d array.
        title = 'Cubic Polynomial Regression for deceased patients'

    # (a) fit the model on Xtrain. Data set 50/50 split between train and test.
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.5, random_state=120)

    poly = PolynomialFeatures(degree=3, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    LinearRegressModel = LinearRegression()
    LinearRegressModel.fit(X_train_poly, Y_train)

    # (b) print the weights (a, b, . . .)
    slope = LinearRegressModel.coef_  # slope value after model training
    intercept = LinearRegressModel.intercept_  # intercept value after model training.
    # weights = np.polyfit(X_train_poly[:, 1], Y_train, 3)

    print(title + " score is ", LinearRegressModel.score(X_train_poly, Y_train))
    print(" Slope is :", slope)
    print(" Intercept is :", intercept)
    # print(" Weights are ", weights.ravel())

    # (c) compute predicted values using Xtest
    y_predict = LinearRegressModel.predict(X_test_poly)

    # (d) plot (if possible) predicted and actual values in Xtest
    x_smooth = np.linspace(X.min(), X.max(), 100)
    x_smooth_poly = poly.transform(x_smooth.reshape(-1, 1))
    y_smooth = LinearRegressModel.predict(x_smooth_poly)
    plt.scatter(X, y, color='blue')
    plt.plot(x_smooth, y_smooth, color='blue', lw=3)
    plt.xlabel('x_smooth')
    plt.ylabel('y_smooth')
    plt.title(title)
    plt.savefig(title+ ".pdf",bbox_inches='tight',dpi=300)
    plt.show()

    # (e) compute (and print) the corresponding loss function
    sse = np.sum((Y_test - y_predict) ** 2, axis=0)
    print(" SSE (Sum of Squared Errors) is :", sse)

cubicPolynomialModel(0)
cubicPolynomialModel(1)


def GLM_logX(DEATH_EVENT):
    input_dir = os.getcwd()
    clinicalRecords = os.path.join(input_dir, 'heart_failure_clinical_records_dataset.csv')
    df_clinicalRecords = pd.read_csv(clinicalRecords)
    required_features = ['creatinine_phosphokinase', 'platelets', 'DEATH_EVENT']
    df_clinicalRecords_filtered = df_clinicalRecords.drop(df_clinicalRecords.columns.difference(required_features),
                                                          axis=1)
    df_0 = df_clinicalRecords_filtered[df_clinicalRecords_filtered['DEATH_EVENT'] == 0]  # .values
    df_1 = df_clinicalRecords_filtered[df_clinicalRecords_filtered['DEATH_EVENT'] == 1]  # .values
    if DEATH_EVENT == 0:
        X = df_0[['creatinine_phosphokinase']]  # X is a matrix of 2d array.
        y = df_0[['platelets']]  # Y dependent variable can be 1d array.
        title = 'Generalized linear model logX for surviving patients'
    elif DEATH_EVENT == 1:
        X = df_1[['creatinine_phosphokinase']]  # X is a matrix of 2d array.
        y = df_1[['platelets']]  # Y dependent variable can be 1d array.
        title = 'Generalized linear model logX for deceased patients'

    log_transformer = FunctionTransformer(func=np.log1p, inverse_func=np.expm1)
    X_log = log_transformer.fit_transform(X)

    # (a) fit the model on Xtrain. Data set 50/50 split between train and test.
    X_train, X_test, Y_train, Y_test = train_test_split(X_log, y, test_size=0.5, random_state=120)

    LinearRegressModel = LinearRegression()
    LinearRegressModel.fit(X_train, Y_train)

    # (b) print the weights (a, b, . . .)
    slope = LinearRegressModel.coef_  # slope value after model training
    intercept = LinearRegressModel.intercept_  # intercept value after model training.

    print(title + " score is ", LinearRegressModel.score(X_train, Y_train))
    print(" Slope is :", slope)
    print(" Intercept is :", intercept)

    # (c) compute predicted values using Xtest
    y_predict = LinearRegressModel.predict(X_test)

    # (d) plot (if possible) predicted and actual values in Xtest
    x_smooth = np.linspace(X.min(), X.max(), 100)
    x_smooth_log = log_transformer.transform(x_smooth.reshape(-1, 1))
    y_smooth = LinearRegressModel.predict(x_smooth_log)
    plt.scatter(X, y, color='blue')
    plt.plot(x_smooth, y_smooth, color='blue', lw=3)
    plt.xlabel('x_smooth')
    plt.ylabel('y_smooth')
    plt.title(title)
    plt.savefig(title+ ".pdf",bbox_inches='tight',dpi=300)
    plt.show()

    # (e) compute (and print) the corresponding loss function
    sse = np.sum((Y_test - y_predict) ** 2, axis=0)
    print(" SSE (Sum of Squared Errors) is :", sse)

GLM_logX(0)
GLM_logX(1)

def GLM_logXY(DEATH_EVENT):
    input_dir = os.getcwd()
    clinicalRecords = os.path.join(input_dir, 'heart_failure_clinical_records_dataset.csv')
    df_clinicalRecords = pd.read_csv(clinicalRecords)
    required_features = ['creatinine_phosphokinase', 'platelets', 'DEATH_EVENT']
    df_clinicalRecords_filtered = df_clinicalRecords.drop(df_clinicalRecords.columns.difference(required_features),
                                                          axis=1)
    df_0 = df_clinicalRecords_filtered[df_clinicalRecords_filtered['DEATH_EVENT'] == 0]  # .values
    df_1 = df_clinicalRecords_filtered[df_clinicalRecords_filtered['DEATH_EVENT'] == 1]  # .values
    if DEATH_EVENT == 0:
        X = df_0[['creatinine_phosphokinase']]  # X is a matrix of 2d array.
        y = df_0[['platelets']]  # Y dependent variable can be 1d array.
        title = 'Generalized linear model logXY for surviving patients'
    elif DEATH_EVENT == 1:
        X = df_1[['creatinine_phosphokinase']]  # X is a matrix of 2d array.
        y = df_1[['platelets']]  # Y dependent variable can be 1d array.
        title = 'Generalized linear model logXY for deceased patients'

    log_transformer = FunctionTransformer(func=np.log1p, inverse_func=np.expm1)
    X_log = log_transformer.fit_transform(X)
    y_log = log_transformer.fit_transform(y.values.reshape(-1, 1)).flatten()

    # (a) fit the model on Xtrain. Data set 50/50 split between train and test.
    X_train, X_test, Y_train, Y_test = train_test_split(X_log, y_log, test_size=0.5, random_state=120)

    LinearRegressModel = LinearRegression()
    LinearRegressModel.fit(X_train, Y_train)

    # (b) print the weights (a, b, . . .)
    slope = LinearRegressModel.coef_  # slope value after model training
    intercept = LinearRegressModel.intercept_  # intercept value after model training.

    print(title + " score is ", LinearRegressModel.score(X_train, Y_train))
    print(" Slope is :", slope)
    print(" Intercept is :", intercept)

    # (c) compute predicted values using Xtest
    y_predict = LinearRegressModel.predict(X_test)

    # (d) plot (if possible) predicted and actual values in Xtest
    x_smooth = np.linspace(X.min(), X.max(), 100)
    x_smooth_log = log_transformer.transform(x_smooth.reshape(-1, 1))
    y_smooth = LinearRegressModel.predict(x_smooth_log)
    y_smooth = log_transformer.inverse_transform(y_smooth.reshape(-1, 1)).flatten()

    plt.scatter(X, y, color='blue')
    plt.plot(x_smooth, y_smooth, color='blue', lw=3)
    plt.xlabel('x_smooth')
    plt.ylabel('y_smooth')
    plt.title(title)
    plt.savefig(title+ ".pdf",bbox_inches='tight',dpi=300)
    plt.show()

    # (e) compute (and print) the corresponding loss function
    sse = np.sum((Y_test - y_predict) ** 2, axis=0)
    print(" SSE (Sum of Squared Errors) is :", sse)


GLM_logXY(0)
GLM_logXY(1)

# Code for Q3

SSE_List_0 = [1.323136e+12, 1.322724e+12, 1.338190e+12, 1.315832e+12, 17.841136537858077]
SSE_List_1 = [6.529056e+11, 6.529151e+11, 9.440193e+11, 6.158507e+11, 12.406651661766794]

print("Min SSE for surviving patients is ", min(SSE_List_0))
print("Max SSE for surviving patients is ", max(SSE_List_0))
print("Min SSE for deceased patients is ", min(SSE_List_1))
print("Max SSE for deceased patients is ", max(SSE_List_1))
