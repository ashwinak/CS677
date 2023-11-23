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
Date: 11/19/2023
Homework Problem # 1
Description of Problem (just a 1-2 line summary!): 
1. load the data into dataframe and add column ”color”. For each class 0, this should contain ”green” and for each class 1 it should contain ”red” 
2. for each class and for each feature f1, f2, f3, f4, compute its mean µ() and standard deviation σ(). 
3. examine your table. Are there any obvious patterns in the distribution of banknotes in each class

"""

import os
import csv
import statistics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## This is code for Q1,

#function to compute mean
def computeMean(returns):
    mean = statistics.mean(returns)
    return round(mean, 6)

#function to compute standard deviation
def computeSTD(returns):
    std = statistics.stdev(returns)
    return round(std, 6)

def computeAll(df,col):
    computeList = []
    for i, row in df.iterrows():
        computeList.append(row[col])
    print(col + " all Mean is " + str(round(computeMean(computeList),2)))
    print(col + " all STD is " + str(round(computeSTD(computeList),2)))

def computeClass(df,col,F5Class):
    computeList = []
    for i, row in df.iterrows():
        # print(i,row['F5-Class'])
        if (row['F5-Class'] == F5Class):
            computeList.append(row[col])
    print(col + " for class " + str(F5Class) + " Mean is " + str(round(computeMean(computeList),2)))
    print(col + " for class " + str(F5Class) + " STD is " + str(round(computeSTD(computeList),2)))

input_dir = os.getcwd()
bankNotes = os.path.join(input_dir,'data_banknote_authentication.csv')

df_bankNotes = pd.read_csv(bankNotes)

column_names = ['F1-Variance', 'F2-Skewness', 'F3-curtosis', 'F4-Entropy', 'F5-Class']

if len(column_names) == len(df_bankNotes.columns):
    df_bankNotes.columns = column_names

df_bankNotes['Color'] = df_bankNotes['F5-Class'].apply(lambda x: 'Green' if x ==0 else 'Red')
print(df_bankNotes)

computeAll(df_bankNotes,"F1-Variance")
computeClass(df_bankNotes,"F1-Variance",0)
computeClass(df_bankNotes,"F1-Variance",1)
print()
computeAll(df_bankNotes,"F2-Skewness")
computeClass(df_bankNotes,"F2-Skewness",0)
computeClass(df_bankNotes,"F2-Skewness",1)
print()
computeAll(df_bankNotes,"F3-curtosis")
computeClass(df_bankNotes,"F3-curtosis",0)
computeClass(df_bankNotes,"F3-curtosis",1)
print()
computeAll(df_bankNotes,"F4-Entropy")
computeClass(df_bankNotes,"F4-Entropy",0)
computeClass(df_bankNotes,"F4-Entropy",1)






