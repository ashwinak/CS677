# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5, 2023.
@author: ashwinar@bu.edu

This is a machine learning algorithm implementation to detect DDoS attacks based on the sflow data received from the network element.
Various ML algorithms are compared against its accuracy for accurate prediction of DDoS attacks. The DDoS flow is labeled as 1 from the class_label.
From the confusion matrix, the true positive rate determines the rate of accurate predictions on the DDoS traffic.
A false positive indiates model predicting as DDoS, however it is a BENIGN traffic.

"""
"""
Ashwin Arunkumar
Class: CS 677
Date: 12/12/2023
Term project 
Description of Project (just a 1-2 line summary!):

This is a machine learning algorithm implemenation to detect DDoS attacks based on the sflow data received from the network equipement.
Various ML algorithms are compared against its accuracy. 
"""

import os
import copy
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier



# A total of 225K rows are part of this data set. For this project, only 1000 rows will be used per flow type to train the model.
# flow type 1 : BENIGN : 1000
# flow type 2 : Ddos : 1000

fileName = "DDOS_Capture.csv"
new_ClassLabel = 'Class_Label'

def dataPreProcessing(fileName,class_feature, class0_Label,class1_Label,new_ClassLabel,sample_count=len(pd.read_csv(fileName))):
    """
    This function takes input file name, class_feature column, existing class labels, new computed class label  and length of sample_count.

    Parameters:
    - fileName (str): Name of the input file in current working directory.
    - class_feature : This is called the dependent variable or y column in data pre processing.
    - class0_Label : The values in class features is expected to be in binary. This label maps to one of the binary value.
    - class1_Label : The values in class features is expected to be in binary. This label maps to one of the binary value.
    - new_ClassLabel : This is the new class label computed based on values from class_features column.
    - sample_count : This is the number of samples that will be filtered and stored in a dataframe. IOW, the number of rows from the input file used for training.

    Sample function call: df_DDoS_filtered = dataPreProcessing(fileName,'Label','BENIGN','DDoS',new_ClassLabel,2000)

    Returns:
    dataframe: This function returns the the data frame post processing.
    """
    err_rate = []
    input_dir = os.getcwd()
    DDoS_Data = os.path.join(input_dir, fileName)
    df_DDoS = pd.read_csv(DDoS_Data)

    # Removing leading or trailing spaces.
    df_DDoS = df_DDoS.rename(columns=lambda x: x.strip())
    df_DDoS = df_DDoS.sample(frac=1.0, random_state=42).reset_index(drop=True)
    condition_Benign = df_DDoS[class_feature] == class0_Label
    condition_DDoS  = df_DDoS[class_feature] == class1_Label
    random_rows_class0_Label = df_DDoS[condition_Benign].sample(n=sample_count, random_state=142)
    random_rows_class1_Label = df_DDoS[condition_DDoS].sample(n=sample_count, random_state=142)
    df_DDoS_filtered = pd.concat([random_rows_class0_Label, random_rows_class1_Label], ignore_index=True)

    # create a new class label for DDOS and Benign flows. Convert text labels to integer labels.
    df_DDoS_filtered[new_ClassLabel] = df_DDoS_filtered[class_feature].apply(lambda x: 0 if x =='BENIGN' else 1)
    df_DDoS_filtered = df_DDoS_filtered.sample(frac=1.0, random_state=42).reset_index(drop=True) #shuffle rows so that during 50/50 split, the training happens with both type of flows
    df_DDoS_filtered.to_csv('DDOS_Capture_Filtered.csv',index=False)
    # Class_Label = 1 determines if the flow is of type DDoS. This  will translate to True positive value in confusion Matrix.
    return df_DDoS_filtered

# Selecting features for training. There are a total of 86 features available. All 86 except dependent variable will be used in the training set.
def applyLogisticRegression(dataFrame,new_ClassLabel,feature_list_toDrop):
    """
    This function takes dataframe obtained from dataPreProcessing function, new_classLabel value and feature list to drop if they cannot be
    passed to ML for training. Reasons for dropping features could be if the values are strings or float values.

    Parameters:
    - dataFrame : pre processed dataframe returned from previous function.
    - new_ClassLabel : This is the new class label computed based on values from class_features column.
    - feature_list_toDrop : List of features to drop if they cannot be passed to ML training model.

    Returns:
    Accuracy of the training model.
    """
    err_rate = []
    y = dataFrame[[new_ClassLabel]].values
    dataFrame.drop([new_ClassLabel,*feature_list_toDrop], axis=1, inplace=True)
    X = dataFrame.values

    #Data set 50/50 split between train and test
    X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size=0.2,random_state=120)

    # Data preprocessing using standard scalar before fitting into KNN model.
    # print(X_test)
    scalar = StandardScaler()
    X_train = scalar.fit_transform(X_train) # For training data to both learn parameters and apply transformation.
    X_test = scalar.transform(X_test) # For applying learned transformation to the new data set.

    logRegression = LogisticRegression()
    logRegression.fit(X_train,Y_train.ravel())
    predictions = logRegression.predict(X_test)
    # print("predictions ", predictions)
    # print("Y_test ", Y_test)
    # print("Total features used for training is: ", len(dataFrame.columns))
    accuracy = accuracy_score(Y_test, predictions)
    # conf_matrix_df = pd.DataFrame(conf_matrix, index=['Actual Positive', 'Actual Negative'], columns=['Predicted Positive', 'Predicted Negative'])
    conf_matrix = confusion_matrix(Y_test, predictions)
    # print("Confusion Matrix:")
    TN = conf_matrix[[0],[0]][0]
    FP = conf_matrix[[0],[1]][0]
    FN = conf_matrix[[1],[0]][0]
    TP = conf_matrix[[1],[1]][0]
    # print("    True Positive is ", TP)
    # print("    False Positive is ", FP)
    # print("    False Negative is ", FN)
    # print("    True Negative is ", TN)
    # print("    True Negative Rate is : ", round(TN/(TN+FP),2)*100)
    # Compute and print the error rate
    # error_rate = (conf_matrix[0, 1] + conf_matrix[1, 0]) / len(Y_test)
    # print(f"Error Rate: {error_rate:.2%}")
    # err_rate.append([error_rate*100])
    print("\nTotal features used for training is: ", len(dataFrame.columns))
    print(f"    Accuracy from Logistic regression model is: {accuracy * 100 :.2f}%")
    print("    True Positive Rate is (Rate of DDoS flows predicted as DDoS by the model) : ", round(TP/(TP+FN),2)*100)
    print("    False Positive Rate is (Rate of Benign flows predicted as DDoS by the model) : ", round(FP/(TP+FN),2)*100)
    return (accuracy*100)


def applyCorrelationMatrix(dataFrame,new_ClassLabel,feature_list_toDrop,threshold):
    """
    This function takes dataframe obtained from dataPreProcessing function, new_classLabel value and feature list to drop if they cannot be
    passed to ML for training. Reasons for dropping features could be if the values are strings or float values.
    This function also takes the threshold value i.e the value that can be used to drop highly correlated features in a given data frame.

    Parameters:
    - dataFrame : pre processed dataframe returned from previous function.
    - new_ClassLabel : This is the new class label computed based on values from class_features column.
    - feature_list_toDrop : List of features to drop if they cannot be passed to ML training model.

    Returns:
    Accuracy of the training model after dropping highly correlated features.
    """
    err_rate = []
    df_full = copy.copy(dataFrame)
    dataFrame.drop([new_ClassLabel,*feature_list_toDrop], axis=1, inplace=True)
    # Based on the correlation matrices, the following features are deleted.
    columns_to_drop = np.where(np.abs((dataFrame.corr()) > threshold))
    columns_to_drop = [(dataFrame.columns[x], dataFrame.columns[y])
                       for x, y in zip(*columns_to_drop)
                       if x != y and x < y]
    # Drop the identified columns
    for col1, col2 in columns_to_drop:
        if col2 in df_full.columns:
            df_full.drop(col2, axis=1, inplace=True)
    print("\n%%% Dropping high positive Correlated features")
    accuracy = applyLogisticRegression(df_full,new_ClassLabel,feature_list_toDrop)
    return (accuracy)

def applyNaiveBayes(dataFrame,new_ClassLabel,feature_list_toDrop):
    """
    This function takes dataframe obtained from dataPreProcessing function, new_classLabel value and feature list to drop if they cannot be
    passed to ML for training. Reasons for dropping features could be if the values are strings or float values.

    Parameters:
    - dataFrame : pre processed dataframe returned from previous function.
    - new_ClassLabel : This is the new class label computed based on values from class_features column.
    - feature_list_toDrop : List of features to drop if they cannot be passed to ML training model.

    Returns:
    Accuracy of the training model.
    """
    err_rate = []
    y = dataFrame[[new_ClassLabel]].values
    dataFrame.drop([new_ClassLabel,*feature_list_toDrop], axis=1, inplace=True)
    X = dataFrame.values

    #Data set 50/50 split between train and test
    X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size=0.5,random_state=3620)

    # Data preprocessing using standard scalar before fitting into KNN model.
    # print(X_test)
    scalar = StandardScaler()
    X_train = scalar.fit_transform(X_train) # For training data to both learn parameters and apply transformation.
    X_test = scalar.transform(X_test) # For applying learned transformation to the new data set.

    clf = tree.DecisionTreeClassifier(criterion ='entropy')
    clf = clf.fit(X_train,Y_train.ravel())
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(Y_test, predictions)
    # conf_matrix_df = pd.DataFrame(conf_matrix, index=['Actual Positive', 'Actual Negative'], columns=['Predicted Positive', 'Predicted Negative'])
    conf_matrix = confusion_matrix(Y_test, predictions)
    # print("Confusion Matrix:")
    TN = conf_matrix[[0],[0]][0]
    FP = conf_matrix[[0],[1]][0]
    FN = conf_matrix[[1],[0]][0]
    TP = conf_matrix[[1],[1]][0]
    # print("    True Positive is ", TP)
    # print("    False Positive is ", FP)
    # print("    False Negative is ", FN)
    # print("    True Negative is ", TN)
    # print("    True Negative Rate is : ", round(TN/(TN+FP),2)*100)
    # Compute and print the error rate
    # error_rate = (conf_matrix[0, 1] + conf_matrix[1, 0]) / len(Y_test)
    # print(f"Error Rate: {error_rate:.2%}")
    # err_rate.append([error_rate*100])
    print("\nTotal features used for training is: ", len(dataFrame.columns))
    print(f"   Total Accuracy of the NB Decision Tree Classifier is: {accuracy * 100 :.2f}%")
    print("    True Positive Rate is (Rate of DDoS flows predicted as DDoS by the model) : ", round(TP/(TP+FN),2)*100)
    print("    False Positive Rate is (Rate of Benign flows predicted as DDoS by the model) : ", round(FP/(TP+FN),2)*100)
    return (accuracy*100)


def applyKNN(dataFrame,new_ClassLabel,feature_list_toDrop,neighbor):
    """
    This function takes dataframe obtained from dataPreProcessing function, new_classLabel value and feature list to drop if they cannot be
    passed to ML for training. Reasons for dropping features could be if the values are strings or float values.

    Parameters:
    - dataFrame : pre processed dataframe returned from previous function.
    - new_ClassLabel : This is the new class label computed based on values from class_features column.
    - feature_list_toDrop : List of features to drop if they cannot be passed to ML training model.
    - neighbor : hyperparameter: range of K neighbors to train the model.

    Returns:
    Accuracy of the training model for various n values.
    """

    y = dataFrame[[new_ClassLabel]].values
    dataFrame.drop([new_ClassLabel,*feature_list_toDrop], axis=1, inplace=True)
    X = dataFrame.values

    #Data set 50/50 split between train and test
    X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size=0.5,random_state=120)

    # Data preprocessing using standard scalar before fitting into KNN model.
    scalar = StandardScaler()
    X_train = scalar.fit_transform(X_train) # For training data to both learn parameters and apply transformation.
    X_test = scalar.transform(X_test) # For applying learned transformation to the new data set.

    xaxis = []
    yaxis = []
    err_rate = []
    k = 0
    print("\nTotal features used for training is: ", len(dataFrame.columns))
    for i in range(1,neighbor,1):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train,Y_train.ravel()) # Used to train the KNN model based on different neighbor values.
        xaxis.append(i)
        predictions = knn.predict(X_test) # used to make predictions on new data.
        yaxis.append(round(np.mean(predictions == Y_test.ravel())*100,2))
        accuracy = accuracy_score(Y_test, predictions)
        conf_matrix = confusion_matrix(Y_test, predictions)
        # conf_matrix_df = pd.DataFrame(conf_matrix, index=['Actual Positive', 'Actual Negative'], columns=['Predicted Positive', 'Predicted Negative'])
        # print("Confusion Matrix:")
        TN = conf_matrix[[0],[0]][0]
        FP = conf_matrix[[0],[1]][0]
        FN = conf_matrix[[1],[0]][0]
        TP = conf_matrix[[1],[1]][0]

        # print("    True Positive is ", TP)
        # print("    False Positive is ", FP)
        # print("    False Negative is ", FN)
        # print("    True Negative is ", TN)
        # print("    True Negative Rate is : ", round(TN/(TN+FP),2)*100)
        # Compute and print the error rate
        # error_rate = (conf_matrix[0, 1] + conf_matrix[1, 0]) / len(Y_test)
        # print(f"Error Rate: {error_rate:.2%}")
        # err_rate.append([error_rate*100])
        # print("max " , max(yaxis))
        # print("current " ,round(np.mean(predictions == Y_test.ravel())*100,2))
        # print("i ", i)
        if (round(np.mean(predictions == Y_test.ravel())*100,2)) >= max(yaxis):
            k=i
        # compare predictions from the KNN model against Y_test output.
        # print("     Accuracy from KNN model for k=" + str(i) + ": "+ f"{accuracy * 100 :.2f}%")

    print("    The best accuracy seen with KNN=" + str(k) +" is ", f"{(round(max(yaxis),2))}%")
    print("    True Positive Rate is (Rate of DDoS flows predicted as DDoS by the model) : ", round(TP/(TP+FN),2)*100)
    print("    False Positive Rate is (Rate of Benign flows predicted as DDoS by the model) : ", round(FP/(TP+FN),2)*100)
    return max(yaxis)

    # # Plot graph for accuracy
    # plt.plot(xaxis,yaxis,label='KNN Accuracy')
    # plt.xlabel("K-Value")
    # plt.ylabel("Accuracy")
    # plt.title("KNN Model classifier")
    # plt.show()

def applyRandomForest(dataFrame,new_ClassLabel,feature_list_toDrop,subtree,max_depth):
    """
    This function takes dataframe obtained from dataPreProcessing function, new_classLabel value and feature list to drop if they cannot be
    passed to ML for training. Reasons for dropping features could be if the values are strings or float values.

    Parameters:
    - dataFrame : pre processed dataframe returned from previous function.
    - new_ClassLabel : This is the new class label computed based on values from class_features column.
    - feature_list_toDrop : List of features to drop if they cannot be passed to ML training model.
    - neighbor : range of K neighbors to train the model.
    - subtree : hyperparameter : N - number of (sub)trees to use and
    - max_depth: hyperparameter : d - max depth of each subtree

    Returns:
    Accuracy of the training model for various n and d values along with confusion matrices for each combination of n and d
    """
    accuracy_list = []
    err_rate = []
    k = 0
    y = dataFrame[[new_ClassLabel]].values
    dataFrame.drop([new_ClassLabel,*feature_list_toDrop], axis=1, inplace=True)
    X = dataFrame.values

    #Data set 50/50 split between train and test
    X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size=0.5,random_state=120)

    # Data preprocessing using standard scalar before fitting into KNN model.
    # print(X_test)
    scalar = StandardScaler()
    X_train = scalar.fit_transform(X_train) # For training data to both learn parameters and apply transformation.
    X_test = scalar.transform(X_test) # For applying learned transformation to the new data set.

    err_rate = []
    estimator = []
    depth = []
    n1=0
    d1=0
    print("\nTotal features used for training is: ", len(dataFrame.columns))
    for n in range (1,subtree,1):
        for d in range (1,max_depth):
            model = RandomForestClassifier(n_estimators =n,max_depth =d,criterion ='entropy')
            model.fit (X_train, Y_train.ravel())
            predictions = model.predict(X_test)
            accuracy = accuracy_score(Y_test, predictions)
            accuracy_list.append(accuracy)
            # print("\nFor n=" + str(n) + " and d=" + str(d))
            # print(f"    Accuracy of Randmom Forest Classifier with : {accuracy * 100 :.2f}%")
            conf_matrix = confusion_matrix(Y_test, predictions)
            # conf_matrix_df = pd.DataFrame(conf_matrix, index=['Actual Positive', 'Actual Negative'], columns=['Predicted Positive', 'Predicted Negative'])
            # print("Confusion Matrix:")
            TN = conf_matrix[[0],[0]][0]
            FP = conf_matrix[[0],[1]][0]
            FN = conf_matrix[[1],[0]][0]
            TP = conf_matrix[[1],[1]][0]
            # print("    True Positive Rate is (DDoS prediction accuracy) : ", round(TP/(TP+FN),2)*100)
            # print("    True Positive is ", TP)
            # print("    False Positive is ", FP)
            # print("    False Negative is ", FN)
            # print("    True Negative is ", TN)
            # print("    True Negative Rate is : ", round(TN/(TN+FP),2)*100)
            # Compute and print the error rate
            # error_rate = (conf_matrix[0, 1] + conf_matrix[1, 0]) / len(Y_test)
            # print(f"Error Rate: {error_rate:.2%}")
            # err_rate.append([error_rate*100])
            estimator.append(n)
            depth.append(d)
            if (accuracy) >= max(accuracy_list):
                n1=n
                d1=d
    # Run the training again with only the best estimators and max_depth.This time the test and predictions will be saved to CSV for plotting graph to show top n DDOS flows.
    model = RandomForestClassifier(n_estimators =n1,max_depth =d1,criterion ='entropy')
    model.fit (X_train, Y_train.ravel())
    predictions = model.predict(X_test)
    selected_feature_names = ['Source Port','Destination Port','Protocol','Flow Duration']
    feature_names = dataFrame.columns
    # print(feature_names)
    df_DDoS_prediction = pd.DataFrame(scalar.inverse_transform(X_test), columns=feature_names).filter(items=selected_feature_names)
    df_DDoS_prediction["predictions"] = predictions
    condition = df_DDoS_prediction['predictions'] == 1
    df_DDoS_prediction['srcPort_dstPort_Protocol'] = df_DDoS_prediction.iloc[:, :3].apply(lambda row: '_'.join(map(str, row)), axis=1)
    df_DDoS_prediction[condition].to_csv('X_test_prediction.csv', index=False)
    df_DDoS_prediction = df_DDoS_prediction[condition].sort_values(by='Flow Duration', ascending=False)
    columns_for_graph = ['srcPort_dstPort_Protocol','Flow Duration']
    df_DDoS_prediction = df_DDoS_prediction[columns_for_graph]
    df_DDoS_prediction['Flow Duration'] = df_DDoS_prediction['Flow Duration'] / (1000 * 60 * 60)
    # print(df_DDoS_prediction)

    # Create a scatter plot
    plt.figure(figsize=(34, 6))
    plt.scatter(df_DDoS_prediction.head(75)['srcPort_dstPort_Protocol'], df_DDoS_prediction.head(75)['Flow Duration'], color='blue', marker='o', label='Data Points')
    # ax.text(2, 20, 'Text Here', fontsize=12, color='red')
    plt.xlabel('srcPort_dstPort_Protocol', fontsize=12)
    plt.ylabel('Flow Duration(in hrs)', fontsize=12)
    plt.title('Top 75 DDoS flows', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.legend(['DDoS flow info vs flow duration'])
    plt.savefig('DDoS_Flows.pdf')
    plt.show()

    #compare predictions from the Randmo forest model against Y_test output.
    print("    The best accuracy seen with Random Forest using estimator " + str(n1) +" and max depth "+ str(d1) +":" , f"{(round(max(accuracy_list)*100,2))}%")
    print("    True Positive Rate is (Rate of DDoS flows predicted as DDoS by the model) : ", round(TP/(TP+FN),2)*100)
    print("    False Positive Rate is (Rate of Benign flows predicted as DDoS by the model) : ", round(FP/(TP+FN),2)*100)
    return (max(accuracy_list)*100)


if __name__ == "__main__":
    df_DDoS_filtered = dataPreProcessing(fileName,'Label','BENIGN','DDoS',new_ClassLabel,2000)
    # print("df_DDoS_filtered", df_DDoS_filtered.columns)
    df_full_Corr = copy.copy(df_DDoS_filtered)
    df_full_NB = copy.copy(df_DDoS_filtered)
    df_full_KNN = copy.copy(df_DDoS_filtered)
    df_full_RandomForest = copy.copy(df_DDoS_filtered)
    feature_list_toDrop = ['Label','Flow ID','Source IP','Destination IP','Timestamp']

    # Applying various training models and comparing its accuracy.
    print("\n#1) Logistic Regression")
    LogRegress_Accuracy = applyLogisticRegression(df_DDoS_filtered,'Class_Label',feature_list_toDrop) #Model logistic regression
    LogRegress_Accuracy_CorrMatrix = applyCorrelationMatrix(df_full_Corr,new_ClassLabel,feature_list_toDrop,0.8) # Apply correlation matrix, remove closest features and compute accuracy of logistic regression.
    print("\n#2) Naive Bayes")
    NB_Accuracy = applyNaiveBayes(df_full_NB,new_ClassLabel,feature_list_toDrop) # Model Naive bayes
    print("\n#3) KNN Classifier")
    KNN_Accuracy = applyKNN(df_full_KNN,new_ClassLabel,feature_list_toDrop,10) # Model KNN
    print("\n#4) Random_Forest")
    RandomForest_Accuracy = applyRandomForest(df_full_RandomForest,new_ClassLabel,feature_list_toDrop,10,6) # Model RandomForest

    # plot graph to compare accuracy of various algorithms.
    xaxis = ["LogisticRegression","LogisticRegression_CorrMatrix","Naive Bayes","KNN","Random_Forest"]
    yaxis = [LogRegress_Accuracy,LogRegress_Accuracy_CorrMatrix,NB_Accuracy,KNN_Accuracy,RandomForest_Accuracy]
    fig, p0 = plt.subplots(figsize=(12, 6))
    p0.plot(xaxis, yaxis, label='DDoS Algorithm Accuracy')
    # ax.text(2, 20, 'Text Here', fontsize=12, color='red')
    p0.set_xlabel('Machine Learning Algorithm', fontsize=12)
    p0.set_ylabel('Accuracy', fontsize=12)
    p0.set_title('Machine Learning Algorithm accuracy to predict DDoS Traffic', fontsize=14)
    p0.legend()
    plt.savefig('ML_Algo_Accuracy.pdf')
    plt.show()

