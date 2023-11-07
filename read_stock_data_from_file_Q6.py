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
            Return_List = dict()

            with open(file, 'r',newline='') as returnPositive:
                reader = csv.reader(returnPositive)
                return_listPositive = []
                next(reader)
                for row in reader:
                    Return_List.update({float(row[13]):float(row[12])})

            #Top 10 worst return days
            SortedTop10Negative = []
            SortedNegativeList = dict(sorted(Return_List.items()))
            for (i,(k,v)) in enumerate(SortedNegativeList.items()):
                if (i <=9):
                    SortedTop10Negative.append(v)

            #Top 10 best return days
            SortedTop10Positive = []
            SortedPositveList = dict(sorted(Return_List.items(),reverse=True))
            for (i,(k,v)) in enumerate(SortedPositveList.items()):
                if (i <=9):
                    SortedTop10Positive.append(v)

            #Top 5 worst return days
            SortedTop5Negative = []
            # SortedNegativeList = dict(sorted(Return_List.items()))
            for (i,(k,v)) in enumerate(SortedNegativeList.items()):
                if (i <=4):
                    SortedTop5Negative.append(v)

            #Top 5 best return days
            SortedTop5Positive = []
            # SortedPositveList = dict(sorted(Return_List.items(),reverse=True))
            for (i,(k,v)) in enumerate(SortedPositveList.items()):
                if (i <=4):
                    SortedTop5Positive.append(v)


            with open(file, 'r',newline='') as returnPositive:
                cash = 100
                reader = csv.reader(returnPositive)
                return_listPositive = []
                next(reader)
                for row in reader:
                    if (int(row[1]) <= int(year)) and ((float(row[13]) >= 0) and float(row[12]) in SortedTop10Positive):
                        cash = round(cash * (1 - float(row[13])),2)
                    elif (int(row[1]) <= int(year)) and (float(row[13]) >= 0):
                        cash = round(cash * (1 + float(row[13])),2)

                print("Total cash (excluding best 10 day trading) on last day of the FY#" + str(year) + " from ticker " + ticker_name)
                print("$" +str(cash))

            with open(file, 'r',newline='') as returnPositive:
                cash = 100
                reader = csv.reader(returnPositive)
                return_listPositive = []
                next(reader)
                for row in reader:
                    # print(cash)
                    if (int(row[1]) <= int(year)) and ((float(row[13]) >= 0) or float(row[12]) in SortedTop10Negative):
                        cash = round(cash * (1+float(row[13])),2)
                print("Total cash (including worst 10 day trading) on last day of the FY#" + str(year) + " from ticker " + ticker_name)
                print("$" +str(cash))


            with open(file, 'r',newline='') as returnPositive:
                cash = 100
                reader = csv.reader(returnPositive)
                return_listPositive = []
                next(reader)
                for row in reader:
                    if (int(row[1]) <= int(year)) and ((float(row[13]) >= 0) and float(row[12]) in SortedTop5Positive):
                        cash = round(cash * (1 - float(row[13])),2)
                    elif (int(row[1]) <= int(year)) and (float(row[13]) >= 0):
                        cash = round(cash * (1 + float(row[13])),2)

                print("Total cash (excluding best 5 day trading) on last day of the FY#" + str(year) + " from ticker " + ticker_name)
                print("$" +str(cash))

            with open(file, 'r',newline='') as returnPositive:
                cash = 100
                reader = csv.reader(returnPositive)
                return_listPositive = []
                next(reader)
                for row in reader:
                    # print(cash)
                    if (int(row[1]) <= int(year)) and ((float(row[13]) >= 0) or float(row[12]) in SortedTop5Negative):
                        cash = round(cash * (1+float(row[13])),2)
                print("Total cash (including worst 5 day trading) on last day of the FY#" + str(year) + " from ticker " + ticker_name)
                print("$" +str(cash))

        readCSVPositiveReturn(ticker,2020)

    except Exception as e:
        print(e)
        print('failed to read stock data for ticker: ', ticker)
