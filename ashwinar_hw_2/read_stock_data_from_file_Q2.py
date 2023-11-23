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
Predicting labels: We will now describe a procedure to predict labels for each day in years 4 and 5 from ”true” labels in training years 1,2 and 3.

1. For W = 2, 3, 4, compute predicted labels for each day in year 4 and 5 based on true labels in years 1,2 and 3 only. 
Perform this for your ticker and for ”spy”.

2. for each W = 2, 3, 4, compute the accuracy - what percent- age of true labels (both positive and negative) have you predicted correctly for the last two years. 

3. which W ∗ value gave you the highest accuracy for your stock and and which W ∗ valuegave you the highest accuracy for S&P-500?

"""

import os
import csv
import statistics
import pandas as pd
import numpy as np

## This is code for Q2.1, JNPR ticker
ticker1 = 'JNPR'
input_dir = os.getcwd()
ticker_file1 = os.path.join(input_dir, ticker1 + '.csv')
df_jnpr = pd.read_csv(ticker_file1)

df_jnpr['True Label'] = df_jnpr['Return'].apply(lambda x: '+' if x >=0 else '-')

## This is code for Q2.1,2,3 for ticker JNPR

### W = 4
past_labels = {}
i = 1
for index, row in df_jnpr.iterrows(): # W = 1
    if int(row['Year']) ==2016:
        current_value = row['True Label'] # curr
        past_labels[i] = [current_value]
    i+=1

i = 1
for index, row in df_jnpr.iterrows(): # W = 2
    if (i>252):
        i = 1
    if int(row['Year']) == 2017:
        current_value = row['True Label']
        past_labels[i].append(current_value)
    i+=1

i = 1
for index, row in df_jnpr.iterrows(): # W = 3
    if (i>252):
        i = 1
    if int(row['Year']) == 2018:
        current_value = row['True Label']
        past_labels[i].append(current_value)
    i+=1

i = 1
for index, row in df_jnpr.iterrows(): # W = 4
    if (i>252):
        i = 1
    if int(row['Year']) == 2019:
        current_value = row['True Label']
        past_labels[i].append(current_value)
    i+=1

df_jnpr['Predicted True Label'] = ''
condition = df_jnpr['Year'] >= 2019
upday = {}
downday = {}
for i in past_labels:
    upday[i] = past_labels[i].count('+')
    downday[i] = past_labels[i].count('-')

i=1
for index, row in df_jnpr.iterrows():
    if int(row['Year']) >=2019:
        if i == 253:
            break
        if(upday[i] >= downday[i]):
            df_jnpr.loc[index+1, 'Predicted True Label'] = '+'
            i+=1
        elif (upday[i] < downday[i]):
            df_jnpr.loc[index+1, 'Predicted True Label'] = '-'
            i+=1
        else:
            df_jnpr.loc[index+1, 'Predicted True Label'] = '+'
            i+=1

print()
print("                         ###### JNPR Ticker ######")
print()
condition = df_jnpr['Year'] >= 2019

print("                         ###### W = 4 ######")
print(df_jnpr[condition].head())
df_jnpr.to_csv('w4prediction_jnpr.csv',index=False)

accuracy = 0
totalCount = 0
for index, row in df_jnpr.iterrows():
    if int(row['Year']) >=2019:
        totalCount+=1
        if df_jnpr.loc[index, 'True Label'] == df_jnpr.loc[index, 'Predicted True Label']:
            accuracy+=1
print("Predicted accuracy for W = 4 is: " + str(round(accuracy/(totalCount/2),4)*100))

# W=3
past_labels = {}
i = 1
for index, row in df_jnpr.iterrows(): # W = 1
    if int(row['Year']) ==2016:
        current_value = row['True Label'] # curr
        past_labels[i] = [current_value]
    i+=1

i = 1
for index, row in df_jnpr.iterrows(): # W = 2
    if (i>252):
        i = 1
    if int(row['Year']) == 2017:
        current_value = row['True Label']
        past_labels[i].append(current_value)
    i+=1

i = 1
for index, row in df_jnpr.iterrows(): # W = 3
    if (i>252):
        i = 1
    if int(row['Year']) == 2018:
        current_value = row['True Label']
        past_labels[i].append(current_value)
    i+=1

df_jnpr['Predicted True Label'] = ''
condition = df_jnpr['Year'] >= 2019
upday = {}
downday = {}
for i in past_labels:
    upday[i] = past_labels[i].count('+')
    downday[i] = past_labels[i].count('-')

i=1
for index, row in df_jnpr.iterrows():
    if int(row['Year']) >=2019:
        if i == 253:
            break
        if(upday[i] >= downday[i]):
            df_jnpr.loc[index+1, 'Predicted True Label'] = '+'
            i+=1
        elif (upday[i] < downday[i]):
            df_jnpr.loc[index+1, 'Predicted True Label'] = '-'
            i+=1
        else:
            df_jnpr.loc[index+1, 'Predicted True Label'] = '+'
            i+=1
print("                         ###### W = 3 ######")
print(df_jnpr[condition].head())
df_jnpr.to_csv('w3prediction_jnpr.csv',index=False)

accuracy = 0
totalCount = 0
for index, row in df_jnpr.iterrows():
    if int(row['Year']) >=2019:
        totalCount+=1
        if df_jnpr.loc[index, 'True Label'] == df_jnpr.loc[index, 'Predicted True Label']:
            accuracy+=1
print("Predicted accuracy for W = 3 is: " + str(round(accuracy/(totalCount/2),4)*100))

# W=2
past_labels = {}
i = 1
for index, row in df_jnpr.iterrows(): # W = 1
    if int(row['Year']) ==2016:
        current_value = row['True Label'] # curr
        past_labels[i] = [current_value]
    i+=1

i = 1
for index, row in df_jnpr.iterrows(): # W = 2
    if (i>252):
        i = 1
    if int(row['Year']) == 2017:
        current_value = row['True Label']
        past_labels[i].append(current_value)
    i+=1

df_jnpr['Predicted True Label'] = ''
condition = df_jnpr['Year'] >= 2019
upday = {}
downday = {}
for i in past_labels:
    upday[i] = past_labels[i].count('+')
    downday[i] = past_labels[i].count('-')

i=1
for index, row in df_jnpr.iterrows():
    if int(row['Year']) >=2019:
        if i == 253:
            break
        if(upday[i] >= downday[i]):
            df_jnpr.loc[index+1, 'Predicted True Label'] = '+'
            i+=1
        elif (upday[i] < downday[i]):
            df_jnpr.loc[index+1, 'Predicted True Label'] = '-'
            i+=1
        else:
            df_jnpr.loc[index+1, 'Predicted True Label'] = '+'
            i+=1
print("                         ###### W = 2 ######")
print(df_jnpr[condition].head())
df_jnpr.to_csv('w2prediction_jnpr.csv',index=False)

accuracy = 0
totalCount = 0
for index, row in df_jnpr.iterrows():
    if int(row['Year']) >=2019:
        totalCount+=1
        if df_jnpr.loc[index, 'True Label'] == df_jnpr.loc[index, 'Predicted True Label']:
            accuracy+=1
print("Predicted accuracy for W = 2 is: " + str(round(accuracy/(totalCount/2),4)*100))

#################### SPY Ticker #####################################################
## This is code for Q2.1, SPY ticker

ticker1 = 'SPY'
input_dir = os.getcwd()
ticker_file1 = os.path.join(input_dir, ticker1 + '.csv')
df_spy = pd.read_csv(ticker_file1)

df_spy['True Label'] = df_spy['Return'].apply(lambda x: '+' if x >=0 else '-')

## This is code for Q2.1,2,3 for ticker JNPR

### W = 4
past_labels = {}
i = 1
for index, row in df_spy.iterrows(): # W = 1
    if int(row['Year']) ==2016:
        current_value = row['True Label'] # curr
        past_labels[i] = [current_value]
    i+=1

i = 1
for index, row in df_spy.iterrows(): # W = 2
    if (i>252):
        i = 1
    if int(row['Year']) == 2017:
        current_value = row['True Label']
        past_labels[i].append(current_value)
    i+=1

i = 1
for index, row in df_spy.iterrows(): # W = 3
    if (i>252):
        i = 1
    if int(row['Year']) == 2018:
        current_value = row['True Label']
        past_labels[i].append(current_value)
    i+=1

i = 1
for index, row in df_spy.iterrows(): # W = 4
    if (i>252):
        i = 1
    if int(row['Year']) == 2019:
        current_value = row['True Label']
        past_labels[i].append(current_value)
    i+=1

df_spy['Predicted True Label'] = ''
condition = df_spy['Year'] >= 2019
upday = {}
downday = {}
for i in past_labels:
    upday[i] = past_labels[i].count('+')
    downday[i] = past_labels[i].count('-')

i=1
for index, row in df_spy.iterrows():
    if int(row['Year']) >=2019:
        if i == 253:
            break
        if(upday[i] >= downday[i]):
            df_spy.loc[index+1, 'Predicted True Label'] = '+'
            i+=1
        elif (upday[i] < downday[i]):
            df_spy.loc[index+1, 'Predicted True Label'] = '-'
            i+=1
        else:
            df_spy.loc[index+1, 'Predicted True Label'] = '+'
            i+=1

print()
print("                         ###### SPY Ticker ######")
print()
condition = df_spy['Year'] >= 2019

print("                         ###### W = 4 ######")
print(df_spy[condition].head())
df_spy.to_csv('w4prediction_spy.csv',index=False)

accuracy = 0
totalCount = 0
for index, row in df_spy.iterrows():
    if int(row['Year']) >=2019:
        totalCount+=1
        if df_spy.loc[index, 'True Label'] == df_spy.loc[index, 'Predicted True Label']:
            accuracy+=1
print("Predicted accuracy for W = 4 is: " + str(round(accuracy/(totalCount/2),4)*100))

# W=3
past_labels = {}
i = 1
for index, row in df_spy.iterrows(): # W = 1
    if int(row['Year']) ==2016:
        current_value = row['True Label'] # curr
        past_labels[i] = [current_value]
    i+=1

i = 1
for index, row in df_spy.iterrows(): # W = 2
    if (i>252):
        i = 1
    if int(row['Year']) == 2017:
        current_value = row['True Label']
        past_labels[i].append(current_value)
    i+=1

i = 1
for index, row in df_spy.iterrows(): # W = 3
    if (i>252):
        i = 1
    if int(row['Year']) == 2018:
        current_value = row['True Label']
        past_labels[i].append(current_value)
    i+=1

df_spy['Predicted True Label'] = ''
condition = df_spy['Year'] >= 2019
upday = {}
downday = {}
for i in past_labels:
    upday[i] = past_labels[i].count('+')
    downday[i] = past_labels[i].count('-')

i=1
for index, row in df_spy.iterrows():
    if int(row['Year']) >=2019:
        if i == 253:
            break
        if(upday[i] >= downday[i]):
            df_spy.loc[index+1, 'Predicted True Label'] = '+'
            i+=1
        elif (upday[i] < downday[i]):
            df_spy.loc[index+1, 'Predicted True Label'] = '-'
            i+=1
        else:
            df_spy.loc[index+1, 'Predicted True Label'] = '+'
            i+=1
print("                         ###### W = 3 ######")
print(df_spy[condition].head())
df_spy.to_csv('w3prediction_spy.csv',index=False)

accuracy = 0
totalCount = 0
for index, row in df_spy.iterrows():
    if int(row['Year']) >=2019:
        totalCount+=1
        if df_spy.loc[index, 'True Label'] == df_spy.loc[index, 'Predicted True Label']:
            accuracy+=1
print("Predicted accuracy for W = 3 is: " + str(round(accuracy/(totalCount/2),4)*100))

# W=2
past_labels = {}
i = 1
for index, row in df_spy.iterrows(): # W = 1
    if int(row['Year']) ==2016:
        current_value = row['True Label'] # curr
        past_labels[i] = [current_value]
    i+=1

i = 1
for index, row in df_spy.iterrows(): # W = 2
    if (i>252):
        i = 1
    if int(row['Year']) == 2017:
        current_value = row['True Label']
        past_labels[i].append(current_value)
    i+=1

df_spy['Predicted True Label'] = ''
condition = df_spy['Year'] >= 2019
upday = {}
downday = {}
for i in past_labels:
    upday[i] = past_labels[i].count('+')
    downday[i] = past_labels[i].count('-')

i=1
for index, row in df_spy.iterrows():
    if int(row['Year']) >=2019:
        if i == 253:
            break
        if(upday[i] >= downday[i]):
            df_spy.loc[index+1, 'Predicted True Label'] = '+'
            i+=1
        elif (upday[i] < downday[i]):
            df_spy.loc[index+1, 'Predicted True Label'] = '-'
            i+=1
        else:
            df_spy.loc[index+1, 'Predicted True Label'] = '+'
            i+=1
print("                         ###### W = 2 ######")
print(df_spy[condition].head())
df_spy.to_csv('w2prediction_spy.csv',index=False)

accuracy = 0
totalCount = 0
for index, row in df_spy.iterrows():
    if int(row['Year']) >=2019:
        totalCount+=1
        if df_spy.loc[index, 'True Label'] == df_spy.loc[index, 'Predicted True Label']:
            accuracy+=1
print("Predicted accuracy for W = 2 is: " + str(round(accuracy/(totalCount/2),7)*100))
