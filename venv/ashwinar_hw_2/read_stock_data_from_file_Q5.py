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
Homework Problem # 5
Description of Problem (just a 1-2 line summary!): 
Question 5 :At the beginning of year 4 you start with $100 dollars and trade for 2 years based on predicted labels. 
1. take your stock. Plot the growth of your amount for 2 years if you trade based on best W ∗ and on ensemble.
On the same graph, plot the growth of your portfolio for ”buy-and- hold” strategy 


"""

import os
import csv
import statistics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import scipy.interpolate as interp


## This is code for Q2.1, JNPR ticker
ticker1 = 'JNPR'
input_dir = os.getcwd()
ticker_file1 = os.path.join(input_dir, ticker1 + '.csv')
df_jnpr = pd.read_csv(ticker_file1)

df_jnpr['True Label'] = df_jnpr['Return'].apply(lambda x: '+' if x >=0 else '-')

## This is code for Q4.1,2,3 for ticker JNPR

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


## compute TP,FP, TN, FN, TPR, TNR for W = 4

TP=FP=TN=FN=TPR=TNR=0

for index, row in df_jnpr.iterrows():
    if int(row['Year']) >=2019:
        totalCount+=1
        if df_jnpr.loc[index, 'True Label'] == '+' and df_jnpr.loc[index, 'Predicted True Label'] == '+':
            TP+=1
        elif df_jnpr.loc[index, 'True Label'] == '-' and df_jnpr.loc[index, 'Predicted True Label'] == '-':
            TN+=1
        elif df_jnpr.loc[index, 'True Label'] == '-' and df_jnpr.loc[index, 'Predicted True Label'] == '+':
            FP+=1
        elif df_jnpr.loc[index, 'True Label'] == '+' and df_jnpr.loc[index, 'Predicted True Label'] == '-':
            FN+=1

TPR = (TP/(TP+FN))*100
TNR = (TN/(TN+FP))*100
print("True Positive for W=4 is: " + str(TP))
print("False Positive for W=4 is: " + str(FP))
print("True Negative for W=4 is: " + str(TN))
print("False Negative for W=4 is: " + str(FN))
print("True Positive Rate for W=4 is: " + str(TPR))
print("True Negative Rate for W=4 is: " + str(TNR))

# Plot graph with 100$ investment for W=4 and Ensemble. Also plot graph for buy and hold strategy.

upday = {}
downday = {}
for i in past_labels:
    upday[i] = past_labels[i].count('+')
    downday[i] = past_labels[i].count('-')

i=1
cash = 100
xaxis = []
yaxis = []
for index, row in df_jnpr.iterrows():
    if int(row['Year']) >=2019:
        if i == 253:
            i=1
        if(upday[i] >= downday[i]):
            df_jnpr.loc[index+1, 'Predicted True Label'] = '+'
            cash = round(cash * (1 + float(df_jnpr.loc[index + 1, 'Return'])),6)
            xaxis.append(df_jnpr.loc[index+1, 'Date'])
            yaxis.append(cash)
            # cash = round(cash * (1+float(row[13])),2)
        i+=1

print("Total cash based on W=4 is " + "$"  + str(cash))

## plotting graph for Q5.

plt.plot (xaxis,yaxis,label='W=4')
plt.xticks(xaxis[::100])





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

## compute TP,FP, TN, FN, TPR, TNR for W = 3

TP=FP=TN=FN=TPR=TNR=0

for index, row in df_jnpr.iterrows():
    if int(row['Year']) >=2019:
        totalCount+=1
        if df_jnpr.loc[index, 'True Label'] == '+' and df_jnpr.loc[index, 'Predicted True Label'] == '+':
            TP+=1
        elif df_jnpr.loc[index, 'True Label'] == '-' and df_jnpr.loc[index, 'Predicted True Label'] == '-':
            TN+=1
        elif df_jnpr.loc[index, 'True Label'] == '-' and df_jnpr.loc[index, 'Predicted True Label'] == '+':
            FP+=1
        elif df_jnpr.loc[index, 'True Label'] == '+' and df_jnpr.loc[index, 'Predicted True Label'] == '-':
            FN+=1

TPR = (TP/(TP+FN))*100
TNR = (TN/(TN+FP))*100
print("True Positive for W=3 is: " + str(TP))
print("False Positive for W=3 is: " + str(FP))
print("True Negative for W=3 is: " + str(TN))
print("False Negative for W=3 is: " + str(FN))
print("True Positive Rate for W=3 is: " + str(TPR))
print("True Negative Rate for W=3 is: " + str(TNR))


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

## compute TP,FP, TN, FN, TPR, TNR for W = 2

TP=FP=TN=FN=TPR=TNR=0

for index, row in df_jnpr.iterrows():
    if int(row['Year']) >=2019:
        totalCount+=1
        if df_jnpr.loc[index, 'True Label'] == '+' and df_jnpr.loc[index, 'Predicted True Label'] == '+':
            TP+=1
        elif df_jnpr.loc[index, 'True Label'] == '-' and df_jnpr.loc[index, 'Predicted True Label'] == '-':
            TN+=1
        elif df_jnpr.loc[index, 'True Label'] == '-' and df_jnpr.loc[index, 'Predicted True Label'] == '+':
            FP+=1
        elif df_jnpr.loc[index, 'True Label'] == '+' and df_jnpr.loc[index, 'Predicted True Label'] == '-':
            FN+=1

TPR = (TP/(TP+FN))*100
TNR = (TN/(TN+FP))*100
print("True Positive for W=2 is: " + str(TP))
print("False Positive for W=2 is: " + str(FP))
print("True Negative for W=2 is: " + str(TN))
print("False Negative for W=2 is: " + str(FN))
print("True Positive Rate for W=2 is: " + str(TPR))
print("True Negative Rate for W=2 is: " + str(TNR))

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

## compute TP,FP, TN, FN, TPR, TNR for Ensemble Label

TP=FP=TN=FN=TPR=TNR=0

for index, row in df_jnpr.iterrows():
    if int(row['Year']) >=2019:
        totalCount+=1
        if df_jnpr.loc[index, 'True Label'] == '+' and df_jnpr.loc[index, 'Ensemble Label'] == '+':
            TP+=1
        elif df_jnpr.loc[index, 'True Label'] == '-' and df_jnpr.loc[index, 'Ensemble Label'] == '-':
            TN+=1
        elif df_jnpr.loc[index, 'True Label'] == '-' and df_jnpr.loc[index, 'Ensemble Label'] == '+':
            FP+=1
        elif df_jnpr.loc[index, 'True Label'] == '+' and df_jnpr.loc[index, 'Ensemble Label'] == '-':
            FN+=1

TPR = (TP/(TP+FN))*100
TNR = (TN/(TN+FP))*100
print("True Positive for Ensemble Label is: " + str(TP))
print("False Positive for Ensemble Label is: " + str(FP))
print("True Negative for Ensemble Label is: " + str(TN))
print("False Negative for Ensemble Label is: " + str(FN))
print("True Positive Rate for Ensemble Label is: " + str(TPR))
print("True Negative Rate for Ensemble Label is: " + str(TNR))

yaxis_en = []
i=1
cash_en = 100
for index, row in df_jnpr.iterrows():
    if int(row['Year']) >=2019:
        if i == 253:
            i=1
        if(upday_en[i] >= downday_en[i]):
            df_jnpr.loc[index+1, 'Ensemble Label'] = '+'
            cash_en = round(cash_en * (1 + float(df_jnpr.loc[index + 1, 'Return'])),6)
            yaxis_en.append(cash_en)
            # cash = round(cash * (1+float(row[13])),2)
        i+=1
x=0
yaxis_en.extend([x for i in range(26)])


print("Total cash based on Ensemble Label is " + "$"  + str(cash_en))

# x_inter = interp.interp1d(np.arange(xaxis.si), xaxis)
# x_ = x_inter(np.linspace(0,xaxis.size-1,yaxis_en.size))

plt.plot (xaxis,yaxis_en,label='Ensemble Label')


yaxis_bhs = []
i=1
cash_bhs = 100
yaxis_bhs.append(cash_bhs)
for index, row in df_jnpr.iterrows():
    if int(row['Year']) >=2019:
        if(df_jnpr.loc[index, 'Date'] == '2019-01-02'):
            NoOfShares = cash_bhs/df_jnpr.loc[index, 'Close']
        if(df_jnpr.loc[index, 'Date'] == '2020-12-30'):
            cash_bhs = NoOfShares * df_jnpr.loc[index, 'Close']
            yaxis_bhs.append(cash_bhs)
        i+=1

print("Total cash based on BHS strategy is " + "$"  + str(cash_bhs))

## plotting graph for Q5 Ensemble label

x=0
yaxis_bhs.extend([x for i in range(376)])
plt.plot (xaxis,yaxis_bhs,label='Buy Hold Strategy')
plt.xlabel('Years')
plt.ylabel('Cash')
plt.show()

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

## compute TP,FP, TN, FN, TPR, TNR for  W=4

TP=FP=TN=FN=TPR=TNR=0

for index, row in df_spy.iterrows():
    if int(row['Year']) >=2019:
        totalCount+=1
        if df_spy.loc[index, 'True Label'] == '+' and df_spy.loc[index, 'Predicted True Label'] == '+':
            TP+=1
        elif df_spy.loc[index, 'True Label'] == '-' and df_spy.loc[index, 'Predicted True Label'] == '-':
            TN+=1
        elif df_spy.loc[index, 'True Label'] == '-' and df_spy.loc[index, 'Predicted True Label'] == '+':
            FP+=1
        elif df_spy.loc[index, 'True Label'] == '+' and df_spy.loc[index, 'Predicted True Label'] == '-':
            FN+=1

TPR = (TP/(TP+FN))*100
TNR = (TN/(TN+FP))*100
print("True Positive for W=4 is: " + str(TP))
print("False Positive for W=4 is: " + str(FP))
print("True Negative for W=4 is: " + str(TN))
print("False Negative for W=4 is: " + str(FN))
print("True Positive Rate for W=4 is: " + str(TPR))
print("True Negative Rate for W=4 is: " + str(TNR))

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

## compute TP,FP, TN, FN, TPR, TNR for  W=3

TP=FP=TN=FN=TPR=TNR=0

for index, row in df_spy.iterrows():
    if int(row['Year']) >=2019:
        totalCount+=1
        if df_spy.loc[index, 'True Label'] == '+' and df_spy.loc[index, 'Predicted True Label'] == '+':
            TP+=1
        elif df_spy.loc[index, 'True Label'] == '-' and df_spy.loc[index, 'Predicted True Label'] == '-':
            TN+=1
        elif df_spy.loc[index, 'True Label'] == '-' and df_spy.loc[index, 'Predicted True Label'] == '+':
            FP+=1
        elif df_spy.loc[index, 'True Label'] == '+' and df_spy.loc[index, 'Predicted True Label'] == '-':
            FN+=1

TPR = (TP/(TP+FN))*100
TNR = (TN/(TN+FP))*100
print("True Positive for W=3 is: " + str(TP))
print("False Positive for W=3 is: " + str(FP))
print("True Negative for W=3 is: " + str(TN))
print("False Negative for W=3 is: " + str(FN))
print("True Positive Rate for W=3 is: " + str(TPR))
print("True Negative Rate for W=3 is: " + str(TNR))



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

## compute TP,FP, TN, FN, TPR, TNR for  W=4

TP=FP=TN=FN=TPR=TNR=0

for index, row in df_spy.iterrows():
    if int(row['Year']) >=2019:
        totalCount+=1
        if df_spy.loc[index, 'True Label'] == '+' and df_spy.loc[index, 'Predicted True Label'] == '+':
            TP+=1
        elif df_spy.loc[index, 'True Label'] == '-' and df_spy.loc[index, 'Predicted True Label'] == '-':
            TN+=1
        elif df_spy.loc[index, 'True Label'] == '-' and df_spy.loc[index, 'Predicted True Label'] == '+':
            FP+=1
        elif df_spy.loc[index, 'True Label'] == '+' and df_spy.loc[index, 'Predicted True Label'] == '-':
            FN+=1

TPR = (TP/(TP+FN))*100
TNR = (TN/(TN+FP))*100
print("True Positive for W=2 is: " + str(TP))
print("False Positive for W=2 is: " + str(FP))
print("True Negative for W=2 is: " + str(TN))
print("False Negative for W=2 is: " + str(FN))
print("True Positive Rate for W=2 is: " + str(TPR))
print("True Negative Rate for W=2 is: " + str(TNR))


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

## compute TP,FP, TN, FN, TPR, TNR for  Ensemble Label

TP=FP=TN=FN=TPR=TNR=0

for index, row in df_spy.iterrows():
    if int(row['Year']) >=2019:
        totalCount+=1
        if df_spy.loc[index, 'True Label'] == '+' and df_spy.loc[index, 'Ensemble Label'] == '+':
            TP+=1
        elif df_spy.loc[index, 'True Label'] == '-' and df_spy.loc[index, 'Ensemble Label'] == '-':
            TN+=1
        elif df_spy.loc[index, 'True Label'] == '-' and df_spy.loc[index, 'Ensemble Label'] == '+':
            FP+=1
        elif df_spy.loc[index, 'True Label'] == '+' and df_spy.loc[index, 'Ensemble Label'] == '-':
            FN+=1

TPR = (TP/(TP+FN))*100
TNR = (TN/(TN+FP))*100
print("True Positive for Ensemble Label is: " + str(TP))
print("False Positive for Ensemble Label is: " + str(FP))
print("True Negative for Ensemble Label is: " + str(TN))
print("False Negative for Ensemble Label is: " + str(FN))
print("True Positive Rate for Ensemble Label is: " + str(TPR))
print("True Negative Rate for Ensemble Label is: " + str(TNR))
