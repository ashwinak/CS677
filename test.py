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

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
# # Example data (replace this with your actual dataset)
# X, y = make_classification(n_samples=100, n_features=4, n_informative=2, n_redundant=0, random_state=42)
#
# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Combine X_train and y_train into a DataFrame for easier plotting
# train_data = pd.DataFrame(data=X_train, columns=[f'feature_{i}' for i in range(X_train.shape[1])])
# train_data['class'] = y_train
#
# # Use Seaborn pairplot with hue to plot pairwise relationships for class 0 and class 1
# sns.pairplot(train_data, hue='class', markers=['o', 's'])
# plt.show()


import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

data = pd. DataFrame (
    {'Day': [1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 ,9 ,10] ,
     'Weather':['sunny','rainy','sunny','rainy','sunny','overcast',
                'sunny','overcast','rainy','rainy'],
     'Temperature': ['hot','mild','cold','cold','cold','mild',
                     'hot','hot','hot','mild'],
     'Wind': ['low','high','low','high','high','low','low',
              'high','high','low'],
     'Play': ['no','yes','yes','no','yes','yes','yes',
              'yes','no','yes']},
    columns = ['Day','Weather','Temperature','Wind','Play']
)
input_data = data [['Weather', 'Temperature', 'Wind']]
dummies = [pd.get_dummies(data [c]) for c in input_data.columns ]


binary_data = pd.concat(dummies , axis =1)
X = binary_data[0:10].values
le = LabelEncoder ()
Y = le.fit_transform(data['Play'].values)
print(X)
print("blah")
print(Y)
knn_classifier = KNeighborsClassifier (n_neighbors = 3)  # 3 neighbor
knn_classifier.fit (X,Y)
arr = np.asarray([[0 ,0 ,1 ,1 ,0 ,0 ,0 ,1]])
# new_instance = np.asmatrix(arr)
prediction = knn_classifier.predict(arr)
print(prediction)