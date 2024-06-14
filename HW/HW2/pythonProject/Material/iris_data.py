# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 18:48:44 2021
Iris data
@author: user
"""
import matplotlib.pyplot as plt
from sklearn import datasets


def load():
    iris = datasets.load_iris()
    X = iris.data[:, 2:4]  # we only take the petal length and petal width.
    y = iris.target

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

    plt.figure(2, figsize=(8, 6))
    plt.clf()

    # Plot the training points
    colors = ['red', 'green', 'blue']
    labels = ['setosa', 'versicolor', 'virginica']
    for i, color in enumerate(colors):
        idx = y == i
        plt.scatter(X[idx, 0], X[idx, 1], c=color, label=labels[i], edgecolor='k')

    plt.xlabel('Speal length')
    plt.ylabel('Speal width')
    plt.legend(loc='best')
    plt.show()
    return X, y, x_min, x_max, y_min, y_max
