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
Homework Problem # 3
Description of Problem (just a 1-2 line summary!): 
Question 3: Compute the aggregate table across all 5 years, one table for both your stock and one table for S&P-500 (using data for ”spy”).

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
        if(ticker == ticker_file1):
            ticker_name = ticker1
            print('opened file for ticker: ', ticker_name)
        else:
            ticker_name = ticker2
            print('opened file for ticker: ', ticker_name)
        """    your code for assignment 1 goes here
        """

        Total_Negative_Return_Days = []
        Total_Positive_Return_Days = []
        Total_Up_Days = []
        Total_Down_Days = []

# Function to compute mean
        def computeMean(returns):
            mean = statistics.mean(returns)
            return round(mean, 6)

#function to compute standard deviation

        def computeSTD(returns):
            std = statistics.stdev(returns)
            return round(std, 6)

# Function to read full CSV file.

        def readCSVReturn(file, day):
            with open(file, 'r') as returnAll:
                readerReturn = csv.reader(returnAll)
                return_listAll = []
                for row in readerReturn:
                    if (row[4] == day):
                        return_listAll.append(float(row[13]))
            print("Mean of returns for " + str(day))
            print(computeMean(return_listAll))

            print("Standard Deviation of returns for " + str(day))
            print(computeSTD(return_listAll))

# Function to read CSV file with rows where return value is negative.

        def readCSVNegativeReturn(file, day):
            with open(file, 'r') as returnNegative:
                readerNegative = csv.reader(returnNegative)
                return_listNegative = []
                for row in readerNegative:
                    if (row[4] == day) and (float(row[13]) < 0):
                        return_listNegative.append(float(row[13]))
            print("Count of Negative returns for " + str(day))
            print(len(return_listNegative))
            Total_Negative_Return_Days.append(len(return_listNegative))
            Total_Down_Days.append(sum(return_listNegative))

            print("Mean of Negative returns for " + str(day))
            print(computeMean(return_listNegative))

            print("Standard Deviation of Negative returns for ")
            print(computeSTD(return_listNegative))

# Function to read CSV file with rows where return value is positive.

        def readCSVPositiveReturn(file, day):
            with open(file, 'r') as returnPositive:
                reader = csv.reader(returnPositive)
                return_listPositive = []
                for row in reader:
                    if (row[4] == day) and (float(row[13]) >= 0):
                        return_listPositive.append(float(row[13]))
            print("Count of Positive returns for " + str(day))
            print(len(return_listPositive))
            Total_Positive_Return_Days.append(len(return_listPositive))
            Total_Up_Days.append(sum(return_listPositive))


            print("Mean of Positive returns for " + str(day))
            print(computeMean(return_listPositive))

            print("Standard Deviation of Positive returns for ")
            print(computeSTD(return_listPositive))

# Call functions to find aggregate for each day of every year between 2016 - 2020

        readCSVReturn(ticker, "Monday")
        readCSVReturn(ticker, "Tuesday")
        readCSVReturn(ticker, "Wednesday")
        readCSVReturn(ticker, "Thursday")
        readCSVReturn(ticker, "Friday")

        readCSVNegativeReturn(ticker, "Monday")
        readCSVNegativeReturn(ticker, "Tuesday")
        readCSVNegativeReturn(ticker, "Wednesday")
        readCSVNegativeReturn(ticker, "Thursday")
        readCSVNegativeReturn(ticker, "Friday")

        readCSVPositiveReturn(ticker, "Monday")
        readCSVPositiveReturn(ticker, "Tuesday")
        readCSVPositiveReturn(ticker, "Wednesday")
        readCSVPositiveReturn(ticker, "Thursday")
        readCSVPositiveReturn(ticker, "Friday")

        print("Total Positive return Days " + str(sum(Total_Positive_Return_Days)))
        print("Total Negative return Days " + str(sum(Total_Negative_Return_Days)))

        print("Total loss on Down days " + str(sum(Total_Down_Days)))
        print("Total Gain on Up Days " + str(sum(Total_Up_Days)))

    except Exception as e:
        print(e)
        print('failed to read stock data for ticker: ', ticker)
