import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.io as sio
import pandas as pd
from sklearn import datasets
# separate train and test randomly
from sklearn.model_selection import train_test_split


def sigmoid(Z):
    """
    Compute the sigmoid activation function.

    Args:
        Z: Input value to apply the sigmoid function to.

    Returns:
        np.array: Result of applying the sigmoid function to the input value.
    """
    return 1.0 / (1 + np.exp(-Z))


def ReLU(Z):
    """
    Compute the ReLU activation function.

    Args:
        Z: Input value to apply the ReLU function to.

    Returns:
        np.array: Result of applying the ReLU function to the input value.
    """
    return np.maximum(0, Z)


def DReLU(Z):
    """
    Compute the derivative of the Rectified Linear Unit (ReLU) activation function.

    Args:
        Z: Input value to compute the derivative for.

    Returns:
        np.array: Result of the derivative calculation based on the input value, array if 0's and 1's.
    """
    return (Z > 0) * 1


def init_parameters(Lin, Lout):
    """
    Initialize the parameters for a neural network layer using the Xavier initialization method.

    Args:
        Lin: Number of input units.
        Lout: Number of output units.

    Returns:
        numpy.ndarray: Initialized weight parameters for the layer.
    """
    factor = np.sqrt(6 / (Lin + Lout))
    return 2 * factor * (np.random.rand(Lout, Lin + 1) - 0.5)


def ff_predict(Theta1, Theta2, X, y):
    """
    Perform forward propagation to predict the output of a neural network.

    Args:
        Theta1: Parameters of the first layer.
        theta2: Parameters of the second layer.
        X: Input data.
        y: True labels.

    Returns:
        numpy.ndarray: Predicted output of the neural network.
    """
    m = X.shape[0]
    num_outputs = Theta2.shape[0]
    p = np.zeros((m, 1))
    X_0 = np.ones((m, 1))
    X1 = np.concatenate((X_0, X), axis=1)
    z2 = np.dot(X1, Theta1.T)
    a2 = ReLU(z2)
    a2_0 = np.ones((a2.shape[0], 1))
    a2 = np.concatenate((a2_0, a2), axis=1)
    z3 = np.dot(a2, Theta2.T)
    a3 = sigmoid(z3)
    p = np.argmax(a3.T, axis=0)
    p = p.reshape(p.shape[0], 1)
    detectp = np.sum(p == y) / m * 100
    return (p, detectp)


#
# Theta1 = np.load('theta_1.npy')
# Theta2 = np.load('theta_2.npy')


Theta1 = init_parameters(64, 16)
Theta2 = init_parameters(16, 10)

digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images.reshape(n_samples, -1)

X_train, X_test, y_train, y_test = train_test_split(
    data,
    digits.target,
    test_size=0.3,
    shuffle=False
)

y_train = y_train.reshape((y_train.shape[0], 1))
y_test = y_test.reshape((y_test.shape[0], 1))

num_outputs = np.unique(y_train).size


p, acc, = ff_predict(Theta1, Theta2, X_test, y_test)

print('accuracy for X_test= ', acc)

