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
Date: 11/07/2023
Homework Problem # 4
Description of Problem (just a 1-2 line summary!): 
Question 4: You listen to the oracle and follow its advice. How much much money will you have on the last trading day of 2020 2019:

"""

import os
import csv
import statistics

# The code below stores the absolute path of the csv file for both tickers. Later it is added to a list data structure.

ticker1 = 'JNPR'
input_dir = os.getcwd()
ticker_file1 = os.path.join(input_dir, ticker1 + '.csv')

ticker2 = 'SPY'
input_dir = os.getcwd()
ticker_file2 = os.path.join(input_dir, ticker2 + '.csv')

#The ticket_list is iterated to read both the csv file and read its contents.

ticker_List = [ticker_file1, ticker_file2]
for ticker in ticker_List:
    try:
        with open(ticker) as f:
            lines = f.read().splitlines()
        if (ticker == ticker_file1):
            ticker_name = ticker1
            print('opened file for ticker: ', ticker_name)
        else:
            ticker_name = ticker2
            print('opened file for ticker: ', ticker_name)
        """    your code for assignment 1 goes here
        """
# function to compute total retuns on the last day of 2019 and 2020 when invested only on positive return days
        def readCSVPositiveReturn(file,year):
            cash = 100
            with open(file, 'r',newline='') as returnPositive:
                reader = csv.reader(returnPositive)
                return_listPositive = []
                next(reader)
                for row in reader:
                    if (int(row[1]) <= int(year)) and (float(row[13]) >= 0):
                        cash = round(cash * (1+float(row[13])),2)
            print("Total cash on last day of the FY#" + str(year) + " from ticker " + ticker_name)
            print("$" +str(cash))

        readCSVPositiveReturn(ticker,2019)
        readCSVPositiveReturn(ticker,2020)

    except Exception as e:
            print(e)
            print('failed to read stock data for ticker: ', ticker)
