# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 18:48:44 2021
Iris data
@author: user
"""
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data[:, 2:4]  # we only take the first two features.
y = iris.target


x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

plt.figure(2, figsize=(8, 6))
plt.clf()

# Plot the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set3,
            edgecolor='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
