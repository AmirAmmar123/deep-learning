# -*- coding: utf-8 -*-
"""
Created on Wed May 22 16:58:25 2024

@author: Machine learning class exercise
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio


# All the functions will be here

def sigmoid(z):
    """
    y = sigmoid(z) implements a sigmoid function for an input z
    """
    y = 1 / (1 + np.exp(-z))
    return y

z = np.linspace(-5, 5, 1000)
y = sigmoid(z)
plt.plot(z, y), plt.grid(axis = 'both')

def compute_cost(X, y, theta):
    """
    compute_cost computes cost for logistic regression.
    Computes the cost of using theta parameters for logistic regression
    Input arguments: X -  input matrix (np array), observations (features) in
    rows. y -  (np array) output vector (labels) for each input sample,
    theta - (np array) parameters vector, weights of the measured features
    return: J - the cost function for theta,
    grad_J - the gradient vector of J_theta
    Usage: J, grad = computeCost(X, y, theta)

    """
    m = y.size
    J = []
    grad_J = np.zeros(theta.shape)
    z = np.dot(X, theta)
    h_theta = sigmoid(z)
    J = - (1 / m) * (np.dot(y.T, np.log(h_theta)) + np.dot((1 - y).T, np.log(1 - h_theta)))
    grad_J = (1/m) * np.dot(X.T, (h_theta - y))
    return J, grad_J

def grad_descent_logreg(X, y, theta, alpha, num_iter):
    """
    grad_descent_logreg implements gradient descent algorithm for logistic
    regression.
    Input arguments: X -  input matrix (np array), observations (features) in
    rows. y -  (np array) output vector (labels) for each input sample,
    theta - (np array) parameters vector, weights of the measured features
    alpha - learning step, num_iter - number of iterations
    return: theta, J_iter - the cost function for each iteration.
    Usage: theta, J_iter = grad_descent_logreg(X, y, theta, alpha, num_iter)
    """

    J_iter = np.zeros(num_iter)
    for k in range(num_iter):
        J_iter[k], grad = compute_cost(X, y, theta)
        theta = theta - alpha * grad
    return theta, J_iter

def plot_log_reg_line(X, y, theta):
    """
    plot_reg_line plots the data points and regression line for logistic regrssion
    Input arguments: X - np array (m, n) - independent variable.
    y - np array (m,1) - target variable
    theta - parameters
    The function is for 2-d input- x2 = -(theta[0] + theta[1]*x1)/theta[2]
    """
    ind = 1
    x1_min = 0.9 * X[:, ind].min()
    x1_max = 1.1 * X[:, ind].max()
    x2_min = - (theta[0] + theta[1]*x1_min) / theta[2]
    x2_max = - (theta[0] + theta[1]*x1_max) / theta[2]
    x1lh = np.array([x1_min, x1_max])
    x2lh = np.array([x2_min, x2_max])
    x1 = X[:, 1]
    x2 = X[:, 2]
    plt.plot(x1[y[:,0] == 0], x2[y[:,0] == 0], 'ro',
             x1[y[:,0] == 1], x2[y[:,0] == 1], 'go',
             x1lh, x2lh, 'b-')
    plt.xlabel( 'x1' ), plt.ylabel( 'x2' ), plt.title('data'),
    plt.grid(axis = 'both'), plt.show()



if __name__ == "__main__":
    Xdata = np.genfromtxt("admittance_data.csv", delimiter = ',', skip_header = 1)
    Xdata.shape
    X_orig = Xdata[:,0:2]
    ## another option for reading the csv file with pandas.
    ## You should import pandas before using it: import pandas as pd
    # df = pd.read_csv("admittance_data.csv")
    # data = np.array(df)
    # X_orig = data[:, 0:2]
    y = Xdata[:, 2]
    plt.figure(1)
    plt.plot(X_orig[y == 0, 0], X_orig[y == 0, 1], 'ro',
             X_orig[y == 1, 0], X_orig[y == 1, 1], 'go'),
    plt.grid(axis = 'both'),
    plt.show()

    m = y.size
    onesvec  = np.ones((m, 1))
    X = np.concatenate((onesvec, X_orig), axis = 1)
    n = X.shape[1]
    theta = np.zeros((n, 1))
    alpha = 0.001225
    num_iter = 100000
    y = y.reshape([y.shape[0], 1])
    theta, J_iter = grad_descent_logreg(X, y, theta, alpha, num_iter)
    plt.figure(1), plt.plot(J_iter), plt.grid(axis = 'both'), plt.show()

    plt.figure(2), plot_log_reg_line(X, y, theta)


    # too many iterations for a simple problem.
    # Normalization of the data would help

    Xn = (X_orig - np.mean(X_orig, axis = 0)) / np.std(X_orig, axis = 0)
    onesvec = np.ones((m ,1))
    X = np.concatenate((onesvec, Xn), axis = 1)
    n = X.shape[1]
    theta = np.zeros((n,1))
    y = y.reshape([y.shape[0], 1])
    J, grad_J = compute_cost(X, y, theta)
    alpha = 0.1
    num_iters = 500
    theta, J_iter = grad_descent_logreg(X, y, theta, alpha, num_iters)
    plt.figure(1), plt.plot(J_iter), plt.grid(axis = 'both')
    plt.show()

    plt.figure(2), plot_log_reg_line(X, y, theta),  plt.show()



if __name__ == "__main__":
    pass







