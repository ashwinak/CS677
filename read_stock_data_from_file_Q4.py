# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 14:37:29 2018

@author: ashwinar
this scripts reads your ticker file (e.g. MSFT.csv) and
constructs a list of lines
"""
import os
import csv
import statistics

ticker1 = 'JNPR'
input_dir = r'/home/ashwinak/Documents/Projects/Python/CS677/'
ticker_file1 = os.path.join(input_dir, ticker1 + '.csv')

ticker2 = 'SPY'
input_dir = r'/home/ashwinak/Documents/Projects/Python/CS677/'
ticker_file2 = os.path.join(input_dir, ticker2 + '.csv')

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
