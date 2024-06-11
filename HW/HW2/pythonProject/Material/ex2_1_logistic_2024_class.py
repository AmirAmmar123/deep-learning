# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 23:57:11 2021
Machine Learning
Class exercise 2
exercise 2.1
@author: class exercise
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd


def sigmoid(Z):
    """
    Compute the sigmoid of Z
    Arguments:  - a scalar or numpy array of any size.
    Return: A - sigmoid(Z)
    """
    ### your code here
    
    return A
    

def computeCost(X, y, theta):
    """
    COMPUTECOST Compute cost for logistic regression.
    Computes the cost of using theta parameters for logistic regression
    Input arguments: X -  input matrix (np array), observations (features) in rows.
    y -  (np array) output vector (labels) for each input sample, 
    theta - (np array) parameters vector, weights of the measured features
    Output arguments:
    return - J - the cost function for theta
    Usage: J = computeCost(X, y, theta)     
    """
    
    m = y.size
    J = 0
    grad_J = np.zeros(theta.shape)
    Z = np.dot(X,theta)
    h_theta = 
    J = 
    grad_J = 
    return J, grad_J




def gradDescent_log(X, y, theta, alpha, num_iters):
    """
    gradDescent - Batch implementation of GD algorithm
    for logistic regression using matrix-vector operations
    Input arguments: 
    X - (m, n) numpy matrix - each row is a feature vector, 
    y - (m,1) np array of target values,
    theta - (n,1) initial parameter vector, 
    alpha - learning rate
    num_iters - number of iterations.
    returns theta - parameters vector and J_iter - (num_iter,1) 
    cost function vector.
    
    """
    # m = y.size
    # n = theta.size
    # y = y.reshape((1, m))
    # X = X.T
    J_iter = np.zeros((num_iters, 1))
    for iter in range(num_iters):
        J_iter[iter], grad = 
        theta = 
        plt.plot(J_iter)
    return theta, J_iter

def plot_logreg_line(X, y, theta):
    """
    plot_reg_line plots the data points and regression line for logistic regrssion
    Input arguments: X - np array (m, n) - independent variable.
    y - np array (m,1) - target variable
    theta - parameters
    The function is for 2-d input - x2 = -(theta[0] + theta[1]*x1)/theta[2]
    """
    ind = 1          
    x1_min = 1.1*X[:,ind].min()
    x1_max = 1.1*X[:,ind].max()
    x2_min = 
    x2_max = 
    x1lh = np.array([x1_min, x1_max])
    x2lh = np.array([x2_min, x2_max])
    x1 = X[: , 1]
    x2 = X[: , 2]
   
    
    plt.xlabel('x1'), plt.ylabel('x2') 
    plt.title('data')
    plt.grid()
    plt.show()
    


    
Xdata = pd.read_csv("admittance_data.csv")
data = Xdata.to_numpy()
X_orig = data[:,0:2]
y = data[:,2]
m = y.size # the training set size

x1 = X_orig[: , 0]
x2 = X_orig[: , 1]
plt.plot(x1[y==0], x2[y==0], 'ro',x1[y==1], x2[y==1], 'go')
plt.grid()
plt.show()





# too many iterations for a simple problem.
# Normalization of the data would help

x1n = 
x2n = 

Xn = (X_orig - np.mean(X_orig, axis = 0)) / np.std(X_orig, axis = 0)

onesvec = np.ones((m ,1))
X = np.concatenate((onesvec, Xn), axis = 1)
n = X.shape[1]
theta = np.zeros((n,1))
y = y.reshape([y.shape[0], 1])
J, grad_J = 
alpha = 0.1
num_iters = 500
theta, J_iter = 
plt.plot(J_iter)
plt.show()

plot_logreg_line(X, y, theta)






Xtest = np.array([60, 45])

X_test = Xtest.reshape(1, Xtest.shape[0])
X_test_n = (X_test - np.mean(X_orig, axis = 0)) / np.std(X_orig, axis = 0)
X_testn= np.concatenate((np.ones((1,1)), X_test_n), axis = 1)

Z = 
y_pred = 
print('the probability of success = ', y_pred)













