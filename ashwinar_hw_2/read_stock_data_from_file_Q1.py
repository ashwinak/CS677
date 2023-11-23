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
Homework Problem # 1
Description of Problem (just a 1-2 line summary!): 
Question 1: You have a csv table of daily returns for your stosk and for S&P-500 (”spy” ticker). 
1. If your initial dataframe were  you will add an additional column ”True Label” and have data as shown in Table 2. 
Your daily ”true labels” sequence is +, −, +, · · · +, −.

2. compute the default probability p∗ that the next day is a ”up” day.

3. take years 1, 2 and 3 What is the probability that after seeing k consecutive ”down days”, the next day is an ”up day”? 

4. take years 1, 2 and 3. What is the probability that after seeing k consecutive ”up days”, the next day is still an ”up day”? 

"""

import os
import csv
import statistics
import pandas as pd
import numpy as np

## This is code for Q1.1, JNPR ticker
ticker1 = 'JNPR'
input_dir = os.getcwd()
ticker_file1 = os.path.join(input_dir, ticker1 + '.csv')
df_jnpr = pd.read_csv(ticker_file1)

df_jnpr['True Label'] = df_jnpr['Return'].apply(lambda x: '+' if x >=0 else '-')
print()
print("                         ###### JNPR Ticker ######")
print()
print(df_jnpr)

## This is code for Q1.2. Computing default probability for next day to be up.

Lminus_jnpr = 0
Lplus_jnpr = 0
for i, row in df_jnpr.iterrows():
    if int(row['Year']) <=2018 and row['True Label'] == '-':
        Lminus_jnpr+=1
    elif int(row['Year']) <=2018 and row['True Label'] == '+':
        Lplus_jnpr+=1
# print("Total Down days betwen 2016 to 2018 is " + str(Lminus_jnpr))
# print("Total up days betwen 2016 to 2018 is " + str(Lplus_jnpr))

print("Default probability of next day up for JNPR is (in percentage): " + str(round(Lplus_jnpr/(Lplus_jnpr+Lminus_jnpr)*100,2)))

## This is code for Q1.3, K=1 starting -ve days

k1Pattern = 0
totalCount = 0
for index, row in df_jnpr.iterrows():
    if int(row['Year']) <=2018 and row['True Label'] == '-':
        current_value = row['True Label']
        next_row = df_jnpr.loc[index + 1] if index + 1 < len(df_jnpr) else None
        next_value = next_row['True Label'] if next_row is not None else None
        totalCount +=1
        if(current_value != next_value): # pattern check for "-+"
            k1Pattern+=1

# print("Total k=1 pattern (-+) found between 2016 to 2018 is " + str(k1Pattern) + "  count: "+str(totalCount))
print("Probability of K=1 pattern (-+) between 2016 to 2018 is (in percentage) " + str(round((k1Pattern/totalCount)*100,2)))

## This is code for Q1.3, K=2 starting -ve days

k2Pattern = 0
totalCount = 0
for index, row in df_jnpr.iterrows():
    if int(row['Year']) <=2018 and row['True Label'] == '-':
        current_value = row['True Label']
        next_row = df_jnpr.loc[index + 1] if index + 1 < len(df_jnpr) else None
        next_value = next_row['True Label'] if next_row is not None else None
        totalCount +=1
        if(current_value == next_value):
            nn_row = df_jnpr.loc[index + 2] if index + 2 < len(df_jnpr) else None
            nn_value = nn_row['True Label'] if next_row is not None else None
            if (next_value !=nn_value): # pattern check for "--+"
                k2Pattern+=1

# print("Total k=2 pattern  found between 2016 to 2018 is " + str(k2Pattern) + "  count: "+str(totalCount))
print("Probability of K=2 pattern (--+) between 2016 to 2018 is (in percentage) " + str(round((k2Pattern/(totalCount))*100,2)))

## This is code for Q1.3, K=3 starting -ve days

k3Pattern = 0
totalCount = 0
for index, row in df_jnpr.iterrows():
    if int(row['Year']) <=2018 and row['True Label'] == '-':
        current_value = row['True Label']
        next_row = df_jnpr.loc[index + 1] if index + 1 < len(df_jnpr) else None
        next_value = next_row['True Label'] if next_row is not None else None
        totalCount +=1
        if(current_value == next_value):
            nn_row = df_jnpr.loc[index + 2] if index + 2 < len(df_jnpr) else None
            nn_value = nn_row['True Label'] if next_row is not None else None
            if (next_value == nn_value):
                nnn_row = df_jnpr.loc[index + 3] if index + 3 < len(df_jnpr) else None
                nnn_value = nnn_row['True Label'] if next_row is not None else None
                if (nn_value != nnn_value): # pattern check for "---+"
                    k3Pattern+=1

# print("Total k=3 pattern  found between 2016 to 2018 is " + str(k3Pattern)+ "  count: "+str(totalCount))
print("Probability of K=3 pattern (---+) between 2016 to 2018 is (in percentage)" + str(round((k3Pattern/totalCount)*100,2)))

######################

## This is code for Q1.3, K=1 starting +ve days

k1Pattern = 0
totalCount = 0
for index, row in df_jnpr.iterrows():
    if int(row['Year']) <=2018 and row['True Label'] == '+':
        current_value = row['True Label']
        next_row = df_jnpr.loc[index + 1] if index + 1 < len(df_jnpr) else None
        next_value = next_row['True Label'] if next_row is not None else None
        totalCount +=1
        if(current_value == next_value): # pattern check for "++"
            k1Pattern+=1

# print("Total k=1 pattern found between 2016 to 2018 is " + str(k1Pattern))
print("Probability of K=1 pattern (++) between 2016 to 2018 is (in percentage) " + str(round((k1Pattern/totalCount)*100,2)))

## This is code for Q1.3, K=2 starting +ve days

k2Pattern = 0
totalCount = 0
for index, row in df_jnpr.iterrows():
    if int(row['Year']) <=2018 and row['True Label'] == '+':
        current_value = row['True Label']
        next_row = df_jnpr.loc[index + 1] if index + 1 < len(df_jnpr) else None
        next_value = next_row['True Label'] if next_row is not None else None
        totalCount +=1
        if(current_value == next_value):
            nn_row = df_jnpr.loc[index + 2] if index + 2 < len(df_jnpr) else None
            nn_value = nn_row['True Label'] if next_row is not None else None
            if (next_value == nn_value): # pattern check for "+++"
                k2Pattern+=1

# print("Total k=1 pattern found between 2016 to 2018 is " + str(k2Pattern))
print("Probability of K=2 pattern (+++) between 2016 to 2018 is (in percentage) " + str(round((k2Pattern/totalCount)*100,2)))

## This is code for Q1.3, K=3 starting +ve days

k3Pattern = 0
totalCount = 0
for index, row in df_jnpr.iterrows():
    if int(row['Year']) <=2018 and row['True Label'] == '+':
        current_value = row['True Label']
        next_row = df_jnpr.loc[index + 1] if index + 1 < len(df_jnpr) else None
        next_value = next_row['True Label'] if next_row is not None else None
        totalCount +=1
        if(current_value == next_value):
            nn_row = df_jnpr.loc[index + 2] if index + 2 < len(df_jnpr) else None
            nn_value = nn_row['True Label'] if next_row is not None else None
            if (next_value == nn_value):
                nnn_row = df_jnpr.loc[index + 3] if index + 3 < len(df_jnpr) else None
                nnn_value = nnn_row['True Label'] if next_row is not None else None
                if (nn_value == nnn_value): # pattern check for "++++"
                    k3Pattern+=1

# print("Total k=3 pattern found between 2016 to 2018 is " + str(k3Pattern))
print("Probability of K=3 pattern (++++) between 2016 to 2018 is (in percentage) " + str(round((k3Pattern/totalCount)*100,2)))


#################################################################################################################################################
## This is code for Q1.1, SPY ticker

ticker2 = 'SPY'
input_dir = os.getcwd()
ticker_file2 = os.path.join(input_dir, ticker2 + '.csv')
df_spy = pd.read_csv(ticker_file2)

df_spy['True Label'] = df_spy['Return'].apply(lambda x: '+' if x >=0 else '-')
print()
print("                         ###### SPY Ticker ######")
print()

print(df_spy)

## This is code for Q1.2. Computing default probability for next day to be up.
Lminus_spy = 0
Lplus_spy = 0
for i, row in df_spy.iterrows():
    if int(row['Year'] <=2018 and row['True Label'] == '-'):
        Lminus_spy+=1
    elif int(row['Year'] <=2018 and row['True Label'] == '+'):
        Lplus_spy+=1


# print("Total Down days betwen 2016 to 2018 is " + str(Lminus_spy))
# print("Total up days betwen 2016 to 2018 is " + str(Lplus_spy))

print("Default probability of next day up for SPY is (in percentage): " + str(round(Lplus_spy/(Lplus_spy+Lminus_spy)*100,2)))

## This is code for Q1.3, K=1 starting -ve days

k1Pattern = 0
totalCount = 0
for index, row in df_spy.iterrows():
    if int(row['Year']) <=2018 and row['True Label'] == '-':
        current_value = row['True Label']
        next_row = df_spy.loc[index + 1] if index + 1 < len(df_spy) else None
        next_value = next_row['True Label'] if next_row is not None else None
        totalCount +=1
        if(current_value != next_value): # pattern check for "-+"
            k1Pattern+=1

# print("Total k=1 pattern (-+) found between 2016 to 2018 is " + str(k1Pattern))
print("Probability of K=1 pattern (-+) between 2016 to 2018 is (in percentage) " + str(round((k1Pattern/totalCount)*100,2)))

## This is code for Q1.3, K=2 starting -ve days

k2Pattern = 0
totalCount = 0
for index, row in df_spy.iterrows():
    if int(row['Year']) <=2018 and row['True Label'] == '-':
        current_value = row['True Label']
        next_row = df_spy.loc[index + 1] if index + 1 < len(df_spy) else None
        next_value = next_row['True Label'] if next_row is not None else None
        totalCount +=1
        if(current_value == next_value):
            nn_row = df_spy.loc[index + 2] if index + 2 < len(df_spy) else None
            nn_value = nn_row['True Label'] if next_row is not None else None
            if (next_value !=nn_value): # pattern check for "--+"
                k2Pattern+=1

# print("Total k=1 pattern  found between 2016 to 2018 is " + str(k2Pattern))
print("Probability of K=2 pattern (--+) between 2016 to 2018 is (in percentage) " + str(round((k2Pattern/totalCount)*100,2)))

## This is code for Q1.3, K=3 starting -ve days

k3Pattern = 0
totalCount = 0
for index, row in df_spy.iterrows():
    if int(row['Year']) <=2018 and row['True Label'] == '-':
        current_value = row['True Label']
        next_row = df_spy.loc[index + 1] if index + 1 < len(df_spy) else None
        next_value = next_row['True Label'] if next_row is not None else None
        totalCount +=1
        if(current_value == next_value):
            nn_row = df_spy.loc[index + 2] if index + 2 < len(df_spy) else None
            nn_value = nn_row['True Label'] if next_row is not None else None
            if (next_value == nn_value):
                nnn_row = df_spy.loc[index + 3] if index + 3 < len(df_spy) else None
                nnn_value = nnn_row['True Label'] if next_row is not None else None
                if (nn_value != nnn_value): # pattern check for "---+"
                    k3Pattern+=1

# print("Total k=3 pattern  found between 2016 to 2018 is " + str(k3Pattern))
print("Probability of K=3 pattern (---+) between 2016 to 2018 is (in percentage) " + str(round((k3Pattern/totalCount)*100,2)))

######################

## This is code for Q1.3, K=1 starting +ve days

k1Pattern = 0
totalCount = 0
for index, row in df_spy.iterrows():
    if int(row['Year']) <=2018 and row['True Label'] == '+':
        current_value = row['True Label']
        next_row = df_spy.loc[index + 1] if index + 1 < len(df_spy) else None
        next_value = next_row['True Label'] if next_row is not None else None
        totalCount +=1
        if(current_value == next_value): # pattern check for "++"
            k1Pattern+=1

# print("Total k=1 pattern found between 2016 to 2018 is " + str(k1Pattern))
print("Probability of K=1 pattern (++) between 2016 to 2018 is (in percentage) " + str(round((k1Pattern/totalCount)*100,2)))

## This is code for Q1.3, K=2 starting +ve days

k2Pattern = 0
totalCount = 0
for index, row in df_spy.iterrows():
    if int(row['Year']) <=2018 and row['True Label'] == '+':
        current_value = row['True Label']
        next_row = df_spy.loc[index + 1] if index + 1 < len(df_spy) else None
        next_value = next_row['True Label'] if next_row is not None else None
        totalCount +=1
        if(current_value == next_value):
            nn_row = df_spy.loc[index + 2] if index + 2 < len(df_spy) else None
            nn_value = nn_row['True Label'] if next_row is not None else None
            if (next_value == nn_value): # pattern check for "+++"
                k2Pattern+=1

# print("Total k=1 pattern found between 2016 to 2018 is " + str(k2Pattern))
print("Probability of K=2 pattern (+++) between 2016 to 2018 is (in percentage) " + str(round((k2Pattern/totalCount)*100,2)))

## This is code for Q1.3, K=3 starting +ve days

k3Pattern = 0
totalCount = 0
for index, row in df_spy.iterrows():
    if int(row['Year']) <=2018 and row['True Label'] == '+':
        current_value = row['True Label']
        next_row = df_spy.loc[index + 1] if index + 1 < len(df_spy) else None
        next_value = next_row['True Label'] if next_row is not None else None
        totalCount +=1
        if(current_value == next_value):
            nn_row = df_spy.loc[index + 2] if index + 2 < len(df_spy) else None
            nn_value = nn_row['True Label'] if next_row is not None else None
            if (next_value == nn_value):
                nnn_row = df_spy.loc[index + 3] if index + 3 < len(df_spy) else None
                nnn_value = nnn_row['True Label'] if next_row is not None else None
                if (nn_value == nnn_value): # pattern check for "++++"
                    k3Pattern+=1

# print("Total k=3 pattern found between 2016 to 2018 is " + str(k3Pattern))
print("Probability of K=3 pattern (++++) between 2016 to 2018 is (in percentage) " + str(round((k3Pattern/totalCount)*100,2)))

