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
Date: 11/24/2023
Homework Problem # 2
Description of Problem (just a 1-2 line summary!): 
Question 2: In this question you will compare a number of different models using linear systems (including linear regres- sion). 
You choose one feature X as independent variable X and another feature Y as dependent. 
Your choice of X and Y will depend on your facilitator group as follows: 
1. Group 1: X: creatinine phosphokinase (CPK), Y : platelets

"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import FunctionTransformer,StandardScaler
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
from sklearn import svm
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn.cluster import KMeans
from scipy.stats import mode
from sklearn.metrics import pairwise_distances_argmin_min


## This is code for Q3.1

### Function to calculate linear kernel SVM
def KMeansClustering():
    input_dir = os.getcwd()
    seedRecords = os.path.join(input_dir, 'seeds_dataset.csv')
    df_seedRecords = pd.read_csv(seedRecords,sep='\t', header=None)
    feature_names = ['area', 'perimeter', 'compactness','length of kernel','width of kernel','asymmetry coefficient','length of kernel groove.','class']
    df_seedRecords.columns = feature_names
    # df_seedRecords_filtered = df_seedRecords.drop(columns=['class'])
    df_seedRecords.to_csv('seeds_dataset_filtered_Q3.csv')
    X = df_seedRecords[['area', 'perimeter', 'compactness','length of kernel','width of kernel','asymmetry coefficient','length of kernel groove.']].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Calculate distortions for different values of k
    distortions = []
    k_values = range(1, 9)
    for k in k_values:
        kmeans = KMeans(n_clusters=k, init='random', random_state=143)
        kmeans.fit(X_scaled)
        distortions.append(kmeans.inertia_)

    # Find the "knee" or elbow point
    # knee_point = np.argmin(distortions) + 1  # Add 1 because of zero-based indexing
    knee_point = 3 # The elblow/knee point is selected based on the graph.

    print(f"Best k using the 'knee' method: {knee_point}")

    # Plot the distortion vs. k
    plt.plot(k_values, distortions, marker='o')
    plt.title('Distortion vs. Number of Clusters (k)')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Distortion')
    plt.ylim(0, max(distortions) + 1000)
    plt.scatter(knee_point, distortions[knee_point - 1], color='red', label='Knee Point')
    plt.legend()
    plt.show()


## This is code for Q3.2
def bestKMeansClustering():
    input_dir = os.getcwd()
    seedRecords = os.path.join(input_dir, 'seeds_dataset.csv')
    df_seedRecords = pd.read_csv(seedRecords,sep='\t', header=None)
    feature_names = ['area', 'perimeter', 'compactness','length of kernel','width of kernel','asymmetry coefficient','length of kernel groove.','class']
    df_seedRecords.columns = feature_names
    # df_seedRecords_filtered = df_seedRecords.drop(columns=['class'])
    df_seedRecords.to_csv('seeds_dataset_filtered_Q3.csv')
    X = df_seedRecords[['area', 'perimeter', 'compactness','length of kernel','width of kernel','asymmetry coefficient','length of kernel groove.']].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Calculate distortions for different values of k
    distortions = []
    k_values = range(1, 9)
    for k in k_values:
        kmeans = KMeans(n_clusters=k, init='random', random_state=143)
        kmeans.fit(X_scaled)
        distortions.append(kmeans.inertia_)
    # Find the "knee" or elbow point
    # knee_point = np.argmin(distortions) + 1  # Add 1 because of zero-based indexing
    knee_point = 3 # The elblow/knee point is selected based on the graph.


# Re-run KMeans with the optimal number of clusters
    kmeans = KMeans(n_clusters=knee_point, random_state=143)
    kmeans.fit(X_scaled)

    # Randomly pick two features (fi and fj) at random
    np.random.seed(42)
    fi, fj = np.random.choice(X_scaled.shape[1], size=2, replace=False)
    # Plot the datapoints using fi and fj as axes
    plt.scatter(X_scaled[:, fi], X_scaled[:, fj], c=kmeans.labels_, cmap='viridis', edgecolors='k', marker='o')
    plt.scatter(kmeans.cluster_centers_[:, fi], kmeans.cluster_centers_[:, fj], s=200, color='red', marker='X', label='Centroids')
    plt.title(f'K-Means Clustering (k={knee_point}) - Features {fi} and {fj}')
    plt.xlabel(f'Feature {fi}')
    plt.ylabel(f'Feature {fj}')
    plt.legend()
    plt.show()



## This is code for Q3.3
def clusterLabels():
    input_dir = os.getcwd()
    seedRecords = os.path.join(input_dir, 'seeds_dataset.csv')
    df_seedRecords = pd.read_csv(seedRecords,sep='\t', header=None)
    feature_names = ['area', 'perimeter', 'compactness','length of kernel','width of kernel','asymmetry coefficient','length of kernel groove.','class']
    df_seedRecords.columns = feature_names
    # df_seedRecords_filtered = df_seedRecords.drop(columns=['class'])
    df_seedRecords.to_csv('seeds_dataset_filtered_Q3.csv')
    X = df_seedRecords[['area', 'perimeter', 'compactness','length of kernel','width of kernel','asymmetry coefficient','length of kernel groove.']].values
    y = df_seedRecords[['class']].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    knee_point = 3 # The elblow/knee point is selected based on the graph.
    kmeans = KMeans(n_clusters=knee_point, random_state=42)
    kmeans.fit(X_scaled)

    # Find the majority class for each cluster
    cluster_labels = np.zeros_like(kmeans.labels_)
    cluster_sizes = np.zeros(knee_point)
    cluster_class_label = []
    for cluster in range(knee_point):
        mask = (kmeans.labels_ == cluster)
        if np.any(mask):
            majority_class = mode(y[mask]).mode[0]
            cluster_labels[mask] = majority_class
            centroid = kmeans.cluster_centers_[cluster]
            cluster_sizes[cluster] = np.sum(mask)
            cluster_class_label.append(majority_class)
            print(f"\nCluster {cluster + 1} - Centroid: {centroid}, Assigned Label: {majority_class}")
        else:
            cluster_labels[mask] = -1
            cluster_sizes[cluster] = 0
    print("\n")
    for cluster in range(knee_point):
        print("cluster " + str(cluster) + ": the cluster label " +  str(cluster_class_label[cluster]) + " its cluster size is " + str(cluster_sizes[cluster]))


## This is code for Q3.4

def KMeansClassifier():
    input_dir = os.getcwd()
    seedRecords = os.path.join(input_dir, 'seeds_dataset.csv')
    df_seedRecords = pd.read_csv(seedRecords,sep='\t', header=None)
    feature_names = ['area', 'perimeter', 'compactness','length of kernel','width of kernel','asymmetry coefficient','length of kernel groove.','class']
    df_seedRecords.columns = feature_names
    # df_seedRecords_filtered = df_seedRecords.drop(columns=['class'])
    df_seedRecords.to_csv('seeds_dataset_filtered_Q3.csv')
    X = df_seedRecords[['area', 'perimeter', 'compactness','length of kernel','width of kernel','asymmetry coefficient','length of kernel groove.']].values
    y = df_seedRecords[['class']].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # knee_point = 8
    knee_point = 3 # The elblow/knee point is selected based on the graph.
    kmeans = KMeans(n_clusters=knee_point, random_state=42)
    kmeans.fit(X_scaled)

    # Find the majority class for each cluster
    cluster_labels = np.zeros_like(kmeans.labels_)
    cluster_sizes = np.zeros(knee_point)
    for cluster in range(knee_point):
        mask = (kmeans.labels_ == cluster)
        if np.any(mask):
            majority_class = mode(y[mask]).mode[0]
            cluster_labels[mask] = majority_class
            cluster_sizes[cluster] = np.sum(mask)
        else:
            cluster_labels[mask] = -1
            cluster_sizes[cluster] = 0
    largest_cluster_index = np.argmax(cluster_sizes)
    print("\n")

    # Print information about the largest cluster
    print(f"Largest Cluster - Cluster {largest_cluster_index + 1}")
    print(f"Size: {cluster_sizes[largest_cluster_index]} data points")
    print(f"Centroid: {kmeans.cluster_centers_[largest_cluster_index]}")

    # Find the largest 3 clusters with labels 1, 2, and 3
    # largest_clusters_indices = np.argsort(cluster_sizes)[-3:]
    # print("largest_clusters_indices ", largest_clusters_indices)
    largest_clusters_indices = [2,1,0] # Selected highest size of a cluster per class label i.e. label 1,label 2 and label 3
    # largest_clusters_indices = [6,0,3] # Selected highest size of a cluster per class label i.e. label 1,label 2 and label 3

    clusters_A_B_C = kmeans.cluster_centers_[largest_clusters_indices]

    # Assign labels based on the nearest centroid for each data point
    closest_cluster_labels = pairwise_distances_argmin_min(X_scaled, clusters_A_B_C)[0] + 1

    # print("closest_cluster_labels " , closest_cluster_labels )
    # # Print the assigned labels for each data point
    # for i, assigned_label in enumerate(closest_cluster_labels):
    #     print(f"Data Point {i + 1} - Assigned Label: {assigned_label}")

    #  accuracy
    accuracy = accuracy_score(y, closest_cluster_labels)
    print(f"\nAccuracy of Kmeans classifier for label 1,2 and 3 data points is : {accuracy:.2%}")


## This is code for Q3.5
def KMeansClassifierVsSVM():
    input_dir = os.getcwd()
    seedRecords = os.path.join(input_dir, 'seeds_dataset.csv')
    df_seedRecords = pd.read_csv(seedRecords,sep='\t', header=None)
    feature_names = ['area', 'perimeter', 'compactness','length of kernel','width of kernel','asymmetry coefficient','length of kernel groove.','class']
    df_seedRecords.columns = feature_names
    # Drop rows where 'class' has the value 1
    df_seedRecords_filtered = df_seedRecords.drop(df_seedRecords[df_seedRecords['class'] == 1].index)
    # print(df_seedRecords_filtered)
    df_seedRecords_filtered.to_csv('seeds_dataset_filtered_Q3_5.csv')
    df_seedRecords_filtered['Label'] = df_seedRecords['class'].apply(lambda x: 0 if x == 2 else 1)
    X = df_seedRecords_filtered[['area', 'perimeter', 'compactness','length of kernel','width of kernel','asymmetry coefficient','length of kernel groove.']].values
    df_seedRecords_filtered['class'] = df_seedRecords_filtered['class'] - 1 # This is to match the label values with closest_cluster_labels format
    y = df_seedRecords_filtered[['class']].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # knee_point = 8
    knee_point = 3 # The elblow/knee point is selected based on the graph.

    kmeans = KMeans(n_clusters=knee_point, random_state=42)
    kmeans.fit(X_scaled)

    # Find the majority class for each cluster
    cluster_labels = np.zeros_like(kmeans.labels_)
    cluster_sizes = np.zeros(knee_point)
    cluster_class_label = []
    for cluster in range(knee_point):
        mask = (kmeans.labels_ == cluster)
        if np.any(mask):
            majority_class = mode(y[mask]).mode[0]
            cluster_labels[mask] = majority_class
            cluster_sizes[cluster] = np.sum(mask)
            cluster_class_label.append(majority_class)
        else:
            cluster_labels[mask] = -1
            cluster_sizes[cluster] = 0
    largest_cluster_index = np.argmax(cluster_sizes)
    print("\n")
    for cluster in range(knee_point):
        print("cluster " + str(cluster) + ": the cluster label " +  str(cluster_class_label[cluster]) + " its cluster size is " + str(cluster_sizes[cluster]))


    # Find the largest 3 clusters with labels 1, 2, and 3
    largest_clusters_indices = np.argsort(cluster_sizes)[-2:]
    print(largest_clusters_indices)
    # largest_clusters_indices = [2,1,0] # Selected highest size of a cluster per class label i.e. label 1,label 2 and label 3

    # largest_clusters_indices = [3,7] # Selected highest size of a cluster per class label i.e. label 2 and label 3. Here Label 1 is removed from the original input dataframe.
    clusters_A_B_C = kmeans.cluster_centers_[largest_clusters_indices]
    # Assign labels based on the nearest centroid for each data point

    closest_cluster_labels = pairwise_distances_argmin_min(X_scaled, clusters_A_B_C)[0] + 1
    #  accuracy
    accuracy = accuracy_score(y, closest_cluster_labels)
    print(f"\nAccuracy of Kmeans classifier for label 2 and 3 data points is  : {accuracy:.2%}")
    #                       Predicted Negative     Predicted Positive
    # Actual Negative        TN                    FP
    # Actual Positive        FN                    TP
    conf_matrix = confusion_matrix(y, closest_cluster_labels)
    print("Confusion Matrix:Kmeans Classifier")
    TN = conf_matrix[[0],[0]][0]
    FP = conf_matrix[[0],[1]][0]
    FN = conf_matrix[[1],[0]][0]
    TP = conf_matrix[[1],[1]][0]
    print("    True Positive is ", TP)
    print("    False Positive is ", FP)
    print("    False Negative is ", FN)
    print("    True Negative is ", TN)
    print("    True Negative Rate is : ", round(TN/(TN+FP),2)*100)
    print("    True Positive Rate is : ", round(TP/(TP+FN),2)*100)


KMeansClustering()
bestKMeansClustering()
clusterLabels()
KMeansClassifier()
KMeansClassifierVsSVM()






