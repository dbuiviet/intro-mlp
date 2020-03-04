import sys

from sklearn.datasets import load_iris
iris_dataset = load_iris()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

import pandas as pd

from pandas.plotting import scatter_matrix

sys.path.append('d:\Workspaces\Python\intro-mlp\src')
# print(sys.path)
import mglearn

print('Key of iris dataset:\n{}'.format(iris_dataset.keys()))
# print(iris_dataset['DESCR'][:200]+'\n...')
# print('Target name: {}'.format(iris_dataset['target_names']))
# print('Feature name: {}'.format(iris_dataset['feature_names']))
# print('Type of data: {}'.format(type(iris_dataset['data'])))
# print('Shape of data: {}'.format(iris_dataset['data'].shape))
# print('First five column of data: \n{}'.format(iris_dataset['data'][:5]))
# print("Target:\n{}".format(iris_dataset['target']))
print('X_train Shape: {}'.format(X_train.shape)) # 75%
print('X_test Shape: {}'.format(X_test.shape))  # 25%

# create dataframe from data in X_train
# label the columns using the strings in iris_dataset.feature_names
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# create a scatter matrix from the dataframe, color by y_train
grr = scatter_matrix(iris_dataframe, c=y_train, figsize=(15,15), marker='o', hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)