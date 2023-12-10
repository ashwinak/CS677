#
# x = "Wednesday"
# y = x [1:5][100:500] + x [2::4]
# print(y)
#
# z = "university" #ytisrevinu
# print (z [::-1][1:4])


# x = {(10,), (20,)}
# print(x)
# # y = {[10], [20]}
# # print(y)
# z = {10, 20}
# print(z)
#
# x = [[10], 20]
# y = x.copy()
# y[0][0] = 100
# print(x, y)
#
# m = {10: 20, 30: 40}
# for x, y in m.items():
#     print(x, y, end=" ")

# def f(n):
#     if n == 1:
#         return 1
#     else:
#         return 2 ** n * f(n - 1)
#
# print(f(3))
#
# # f(3) = 2^3 *f(2) 8 * 4
# # f(2) = 2^2 * f(1) = 4*1
# # f(1) = 1

#
# x = [10 , 20, 30]
# y = x
# print(id(x) == id(y))

# import numpy as np
#
# x = np.array([1, 2, 3])
#
# print(type(x))
#
# x.arr
# print(type(x.tolist()))

# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
#
# # Assuming you have a dataset, X and y are your features and target variable
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Create some example data
# import numpy as np
# np.random.seed(42)
# X = np.random.randn(100, 2)
# y = (X[:, 0] + 2 * X[:, 1] > 0).astype(int)
#
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# sns.scatterplot(x = X_train, y =y_train, color='red')
# plt.title('Training Set Visualization')
# plt.xlabel('X-axis')
# plt.ylabel('Y=axis')
# plt.legend()
# plt.show()
#
# # # Visualize the test set
# # sns.scatterplot(X_test, y_test, hue=y_test, marker='x', label='Test Set')
# # plt.title('Test Set Visualization')
# # plt.legend()
# # plt.show()

# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd
# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split
# # # Example data (replace this with your actual dataset)
# # X, y = make_classification(n_samples=100, n_features=4, n_informative=2, n_redundant=0, random_state=42)
# #
# # # Split the data into training and testing sets
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# #
# # # Combine X_train and y_train into a DataFrame for easier plotting
# # train_data = pd.DataFrame(data=X_train, columns=[f'feature_{i}' for i in range(X_train.shape[1])])
# # train_data['class'] = y_train
# #
# # # Use Seaborn pairplot with hue to plot pairwise relationships for class 0 and class 1
# # sns.pairplot(train_data, hue='class', markers=['o', 's'])
# # plt.show()
#
#
# import pandas as pd
# import numpy as np
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.preprocessing import LabelEncoder
#
# # data = pd. DataFrame (
# #     {'Day': [1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 ,9 ,10] ,
# #      'Weather':['sunny','rainy','sunny','rainy','sunny','overcast',
# #                 'sunny','overcast','rainy','rainy'],
# #      'Temperature': ['hot','mild','cold','cold','cold','mild',
# #                      'hot','hot','hot','mild'],
# #      'Wind': ['low','high','low','high','high','low','low',
# #               'high','high','low'],
# #      'Play': ['no','yes','yes','no','yes','yes','yes',
# #               'yes','no','yes']},
# #     columns = ['Day','Weather','Temperature','Wind','Play']
# # )
# # input_data = data [['Weather', 'Temperature', 'Wind']]
# # dummies = [pd.get_dummies(data [c]) for c in input_data.columns ]
# #
# #
# # binary_data = pd.concat(dummies , axis =1)
# # X = binary_data[0:10].values
# # le = LabelEncoder ()
# # Y = le.fit_transform(data['Play'].values)
# # print(X)
# # print("blah")
# # print(Y)
# # knn_classifier = KNeighborsClassifier (n_neighbors = 3)  # 3 neighbor
# # knn_classifier.fit (X,Y)
# # arr = np.asarray([[0 ,0 ,1 ,1 ,0 ,0 ,0 ,1]])
# # # new_instance = np.asmatrix(arr)
# # prediction = knn_classifier.predict(arr)
# # print(prediction)
#
# import numpy as np
# from sklearn . linear_model import LinearRegression
# X = np.array([[1,1],[1,2],[2,2],[2,3]])
# y = np.dot(X,np.array([1,2]))+3  #[3 5 6 8]
# print(y)


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import csv
# input_dir = os.getcwd()
# clinicalRecords = os.path.join(input_dir,'heart_failure_clinical_records_dataset.csv')
# df_clinicalRecords = pd.read_csv(clinicalRecords)
# required_features = ['creatinine_phosphokinase','platelets','DEATH_EVENT']
# df_clinicalRecords_filtered = df_clinicalRecords.drop(df_clinicalRecords.columns.difference(required_features), axis=1)
# df_0 = df_clinicalRecords_filtered[df_clinicalRecords_filtered['DEATH_EVENT'] == 0]#.values
# df_1 = df_clinicalRecords_filtered[df_clinicalRecords_filtered['DEATH_EVENT'] == 1]#.values
# X = df_0[['creatinine_phosphokinase']] # X is a matrix of 2d array.
# y = df_0[['platelets']] #Y dependent variable can be 1d array.
# title = 'Quadratic Linear Regression for surviving patients'


# Create a sample DataFrame
# data = {'X': [1, 2, 3, 4, 5],
#         'Y': [2, 4, 5, 4, 5]}

# X = df_0['creatinine_phosphokinase'].values
# Y = df_0['platelets'].values
# data = {'X': X,
#         'Y': Y}
#
#
# df = pd.DataFrame(data)
#
# # Extract X and Y values
# x = df['X'].values
# y = df['Y'].values
#
# # Fit a quadratic model (degree=2)
# coefficients = np.polyfit(x, y, deg=2)
#
# # Create a polynomial function using the coefficients
# poly_func = np.poly1d(coefficients)
#
# # Generate points along the curve for smooth visualization
# x_smooth = np.linspace(x.min(), x.max(), 100)
# y_smooth = poly_func(x_smooth)
#
# # Plot the original data points and the quadratic polynomial curve
# plt.scatter(x, y, color='black', label='Data Points')
# plt.plot(x_smooth, y_smooth, color='blue', label='Quadratic Polynomial Curve', linewidth=3)
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Quadratic Polynomial Regression')
# plt.legend()
# plt.show()



# from sklearn.metrics import confusion_matrix
#
# # Actual values
# y_true = [0, 1, 0, 1]
#
# # Predicted values
# # 1 is positive
# # 0 is negative
# y_pred = [1, 0, 1, 0]
#
#
# # Calculate confusion matrix
# conf_matrix = confusion_matrix(y_true, y_pred)
#
# print("Confusion Matrix:")
# TN = conf_matrix[[0],[0]][0]
# FP = conf_matrix[[0],[1]][0]
# FN = conf_matrix[[1],[0]][0]
# TP = conf_matrix[[1],[1]][0]
# print("    True Positive is ", TP)
# print("    False Positive is ", FP)
# print("    False Negative is ", FN)
# print("    True Negative is ", TN)
# # print("    True Positive Rate is : ", round(TP/(TP+FN),2)*100)
# # print("    True Negative Rate is : ", round(TN/(TN+FP),2)*100)
# print(conf_matrix)
import matplotlib.pyplot as plt

# Your data
x_values = [1, 2, 3, 4, 5]
y1_values = [10, 15, 25, 30, 35]
y2_values = [5, 10, 20, 25, 30]

# Create a figure and subplots
fig, axs = plt.subplots(2, 1, figsize=(8, 6))

# Plot the first set of data
axs[0].plot(x_values, y1_values, label='Line 1')
axs[0].set_xlabel('X-axis Label 1')
axs[0].set_ylabel('Y-axis Label 1')
axs[0].set_title('Plot 1')
axs[0].legend()

# Plot the second set of data
axs[1].plot(x_values, y2_values, label='Line 2')
axs[1].set_xlabel('X-axis Label 2')
axs[1].set_ylabel('Y-axis Label 2')
axs[1].set_title('Plot 2')
axs[1].legend()

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plots
plt.show()
