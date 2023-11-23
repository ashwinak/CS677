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
Homework Problem # 3
Description of Problem (just a 1-2 line summary!): 
Question 3. One of the most powerful methods to (potentially) improve predictions is to combine predictions by some ”averaging”. 
This is called ensemble learning.

1. compute ensemble labels for year 4 and 5 for both your stock and S&P-500. 
2. for both S&P-500 and your ticker, what percentage of labels in year 4 and 5 do you compute correctly by using ensemble?
3. did you improve your accuracy on predicting ”−” labels by using ensemble compared to W = 2, 3, 4? 
4. did you improve your accuracy on predicting ”+” labels by using ensemble compared to W = 2, 3, 4?

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

## This is code for Q3.1,2,3 for ticker JNPR

### W = 4, store same day labels across 4 years in a dict with key betweeen day 1 to day 252
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
w432_ensemble = {}
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
            w432_ensemble[i] = [df_jnpr.loc[index+1, 'Predicted True Label']]
            i+=1
        elif (upday[i] < downday[i]):
            df_jnpr.loc[index+1, 'Predicted True Label'] = '-'
            w432_ensemble[i] = [df_jnpr.loc[index+1, 'Predicted True Label']]
            i+=1
        else:
            df_jnpr.loc[index+1, 'Predicted True Label'] = '+'
            w432_ensemble[i] = [df_jnpr.loc[index+1, 'Predicted True Label']]
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
            w432_ensemble[i].append(df_jnpr.loc[index+1, 'Predicted True Label'])
            i+=1
        elif (upday[i] < downday[i]):
            df_jnpr.loc[index+1, 'Predicted True Label'] = '-'
            w432_ensemble[i].append(df_jnpr.loc[index+1, 'Predicted True Label'])
            i+=1
        else:
            df_jnpr.loc[index+1, 'Predicted True Label'] = '+'
            w432_ensemble[i].append(df_jnpr.loc[index+1, 'Predicted True Label'])
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
            w432_ensemble[i].append(df_jnpr.loc[index+1, 'Predicted True Label'])
            i+=1
        elif (upday[i] < downday[i]):
            df_jnpr.loc[index+1, 'Predicted True Label'] = '-'
            w432_ensemble[i].append(df_jnpr.loc[index+1, 'Predicted True Label'])
            i+=1
        else:
            df_jnpr.loc[index+1, 'Predicted True Label'] = '+'
            w432_ensemble[i].append(df_jnpr.loc[index+1, 'Predicted True Label'])
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

### Ensemble Lable based on W=4,3,2
# print(w432_ensemble)
upday_en = {}
downday_en = {}
for i in w432_ensemble:
    upday_en[i] = w432_ensemble[i].count('+')
    downday_en[i] = w432_ensemble[i].count('-')

i=1
for index, row in df_jnpr.iterrows():
    if int(row['Year']) >=2019:
        if i == 253:
            break
        if(upday_en[i] > downday_en[i]):
            df_jnpr.loc[index+1, 'Ensemble Label'] = '+'
            i+=1
        elif (upday[i] < downday[i]):
            df_jnpr.loc[index+1, 'Ensemble Label'] = '-'
            i+=1
        else:
            df_jnpr.loc[index+1, 'Ensemble Label'] = '+'
            i+=1

print("                         ###### Ensemble Label ######")
print(df_jnpr[condition].head())
df_jnpr.to_csv('w432prediction_jnpr_ens.csv',index=False)

accuracy = 0
totalCount = 0
for index, row in df_jnpr.iterrows():
    if int(row['Year']) >=2019:
        totalCount+=1
        if df_jnpr.loc[index, 'True Label'] == df_jnpr.loc[index, 'Ensemble Label']:
            accuracy+=1
print("Predicted accuracy for Ensemble Label for JNPR is: " + str(round(accuracy/(totalCount/2),4)*100))


#################### SPY Ticker #####################################################
## This is code for Q2.1, SPY ticker
## This is code for Q2.1, SPY ticker
ticker1 = 'SPY'
input_dir = os.getcwd()
ticker_file1 = os.path.join(input_dir, ticker1 + '.csv')
df_spy = pd.read_csv(ticker_file1)

df_spy['True Label'] = df_spy['Return'].apply(lambda x: '+' if x >=0 else '-')

## This is code for Q3.1,2,3 for ticker JNPR

### W = 4, store same day labels across 4 years in a dict with key betweeen day 1 to day 252
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
w432_ensemble = {}
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
            w432_ensemble[i] = [df_spy.loc[index+1, 'Predicted True Label']]
            i+=1
        elif (upday[i] < downday[i]):
            df_spy.loc[index+1, 'Predicted True Label'] = '-'
            w432_ensemble[i] = [df_spy.loc[index+1, 'Predicted True Label']]
            i+=1
        else:
            df_spy.loc[index+1, 'Predicted True Label'] = '+'
            w432_ensemble[i] = [df_spy.loc[index+1, 'Predicted True Label']]
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
            w432_ensemble[i].append(df_spy.loc[index+1, 'Predicted True Label'])
            i+=1
        elif (upday[i] < downday[i]):
            df_spy.loc[index+1, 'Predicted True Label'] = '-'
            w432_ensemble[i].append(df_spy.loc[index+1, 'Predicted True Label'])
            i+=1
        else:
            df_spy.loc[index+1, 'Predicted True Label'] = '+'
            w432_ensemble[i].append(df_spy.loc[index+1, 'Predicted True Label'])
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
            w432_ensemble[i].append(df_spy.loc[index+1, 'Predicted True Label'])
            i+=1
        elif (upday[i] < downday[i]):
            df_spy.loc[index+1, 'Predicted True Label'] = '-'
            w432_ensemble[i].append(df_spy.loc[index+1, 'Predicted True Label'])
            i+=1
        else:
            df_spy.loc[index+1, 'Predicted True Label'] = '+'
            w432_ensemble[i].append(df_spy.loc[index+1, 'Predicted True Label'])
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
print("Predicted accuracy for W = 2 is: " + str(round(accuracy/(totalCount/2),4)*100))

### Ensemble Lable based on W=4,3,2
# print(w432_ensemble)
upday_en = {}
downday_en = {}
for i in w432_ensemble:
    upday_en[i] = w432_ensemble[i].count('+')
    downday_en[i] = w432_ensemble[i].count('-')

i=1
for index, row in df_spy.iterrows():
    if int(row['Year']) >=2019:
        if i == 253:
            break
        if(upday_en[i] > downday_en[i]):
            df_spy.loc[index+1, 'Ensemble Label'] = '+'
            i+=1
        elif (upday[i] < downday[i]):
            df_spy.loc[index+1, 'Ensemble Label'] = '-'
            i+=1
        else:
            df_spy.loc[index+1, 'Ensemble Label'] = '+'
            i+=1

print("                         ###### Ensemble Label ######")
print(df_spy[condition].head())
df_spy.to_csv('w432prediction_spy_ens.csv',index=False)

accuracy = 0
totalCount = 0
for index, row in df_spy.iterrows():
    if int(row['Year']) >=2019:
        totalCount+=1
        if df_spy.loc[index, 'True Label'] == df_spy.loc[index, 'Ensemble Label']:
            accuracy+=1
print("Predicted accuracy for Ensemble Label for SPY is: " + str(round(accuracy/(totalCount/2),4)*100))
