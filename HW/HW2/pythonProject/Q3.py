import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple, List
from Q1 import plot_data, not_normolized
from Material.map_feature import map_feature
from Material.plotDecisionBoundaryfunctions import plotDecisionBoundary1
from Material.Admittance_class_2024 import sigmoid


def compute_cost(theta: np.ndarray, X: np.ndarray, y: np.ndarray, lambda_: float) -> float:
    """
    Compute the regularized cost for logistic regression.

    Args:
        theta (np.ndarray): Parameters for logistic regression.
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Labels.
        lambda_ (float): Regularization parameter.

    Returns:
        float: The cost value.
    """
    m = len(y)
    h_theta = sigmoid(X.dot(theta))
    cost = (1 / m) * np.sum(-y * np.log(h_theta) - (1 - y) * np.log(1 - h_theta))
    reg_term = (lambda_ / (2 * m)) * np.sum(theta[1:] ** 2)
    J = cost + reg_term
    return J


def compute_gradient(theta: np.ndarray, X: np.ndarray, y: np.ndarray, lambda_: float) -> np.ndarray:
    """
    Compute the regularized gradient for logistic regression.

    Args:
        theta (np.ndarray): Parameters for logistic regression.
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Labels.
        lambda_ (float): Regularization parameter.

    Returns:
        np.ndarray: The gradient vector.
    """
    m = len(y)
    h_theta = sigmoid(X.dot(theta))
    error = h_theta - y
    grad = (1 / m) * (X.T.dot(error)) + (lambda_ / m) * np.r_[[[0]], theta[1:].reshape(-1, 1)]
    return grad.flatten()


def gd_reg(X: np.ndarray, y: np.ndarray, theta: np.ndarray, alpha: float, num_iters: int, lambda_: float = 0) -> Tuple[
    np.ndarray, List[float]]:
    """
    Perform gradient descent with regularization to learn theta.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Labels.
        theta (np.ndarray): Initial parameters.
        alpha (float): Learning rate.
        num_iters (int): Number of iterations for gradient descent.
        lambda_ (float): Regularization parameter.

    Returns:
        Tuple[np.ndarray, List[float]]: Final parameters and list of cost values per iteration.
    """
    J_iter = []
    for i in range(num_iters):
        grad = compute_gradient(theta, X, y, lambda_)
        theta -= alpha * grad.reshape(-1, 1)
        J_iter.append(compute_cost(theta, X, y, lambda_))
    return theta, J_iter


def read_data(path: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, int]:
    """
    Read data from a CSV file.

    Args:
        path (str): Path to the CSV file.

    Returns:
        Tuple[pd.DataFrame, np.ndarray, np.ndarray, int]: DataFrame, features, labels, and number of examples.
    """
    Xdata = pd.read_csv(path)
    data = Xdata.to_numpy()
    X_orig = data[:, 0:2]
    y = data[:, 2]
    m = y.size
    return Xdata, X_orig, y, m


def predict(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Predict labels using learned parameters theta.

    Args:
        X (np.ndarray): Feature matrix.
        theta (np.ndarray): Learned parameters.

    Returns:
        np.ndarray: Predicted labels.
    """
    probability = sigmoid(X.dot(theta))
    return (probability >= 0.5).astype(int)


def main() -> None:
    """
    Main function to execute logistic regression with regularization,
    plot decision boundaries, and calculate prediction error on a test set.
    """
    Xdata, X_orig, y, m = read_data('Material/email_data_2.csv')
    plot_data(X_orig, y, 'rd', 'go')
    not_normolized(Xdata, X_orig, y, m, 'rd', 'go', 0.01, 100000)

    x1 = X_orig[:, 0]
    x2 = X_orig[:, 1]
    degree = 6
    X = map_feature(x1, x2, degree)
    y = y.reshape(y.shape[0], 1)
    alpha = 0.01
    num_iters = 30000
    lambda_values = [0, 0.01, 0.05, 0.5, 1, 5, 10]

    for lambda_ in lambda_values:
        theta = np.zeros((X.shape[1], 1))
        theta, J_iter = gd_reg(X, y, theta, alpha, num_iters, lambda_)
        plt.figure()
        plt.title(f'Decision Boundary with λ = {lambda_}')
        plotDecisionBoundary1(theta, X, y, degree)
        plt.show()

        test_file_path = 'Material/email_data_3_2024.csv'
        X_test_data, X_test_orig, y_test, m_test = read_data(test_file_path)

        x1_test = X_test_orig[:, 0]
        x2_test = X_test_orig[:, 1]
        X_test = map_feature(x1_test, x2_test, degree)

        y_test = y_test.reshape(y_test.shape[0], 1)

        # Predictions
        predictions = predict(X_test, theta)

        # Calculate prediction error
        prediction_error = np.mean(predictions != y_test) * 100
        print(f'Prediction error on the test set: {prediction_error:.2f}%')

        # The regularization parameter 𝜆 (lambda) in logistic regression affects the decision boundary as follows:
        #
        # - Higher 𝜆 leads to a simpler decision boundary by penalizing large coefficients, reducing overfitting but
        # potentially increasing bias. - Lower 𝜆 allows for a more complex decision boundary that closely fits the
        # training data, potentially leading to overfitting but lower bias.
        #
        # Choosing the right 𝜆 balances model complexity and generalization ability, crucial for optimal performance
        # on unseen data.
        #


if __name__ == '__main__':
    main()
