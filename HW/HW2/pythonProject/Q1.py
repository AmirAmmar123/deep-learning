import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Material.Admittance_class_2024 import grad_descent_logreg, plot_log_reg_line, compute_cost

def load_data():
    """
    Load data from CSV file.

    Returns:
        tuple: DataFrame, feature matrix, labels, number of examples.
    """
    Xdata = np.genfromtxt("Material/admittance_data.csv", delimiter=',', skip_header=1)
    X_orig = Xdata[:, 0:2]
    Xdata = pd.read_csv("Material/admittance_data.csv")
    data = Xdata.to_numpy()
    X_orig = data[:, 0:2]
    y = data[:, 2]
    m = y.size
    return Xdata, X_orig, y, m

def draw_succession(X_orig, y):
    """
    Plot exam scores and admission status.
    """
    plt.figure()
    plt.plot(X_orig[y == 0, 0], X_orig[y == 0, 1], 'ro',
             X_orig[y == 1, 0], X_orig[y == 1, 1], 'go')
    plt.grid(axis='both')
    plt.show()

def calc_GD(X, y, theta, alpha, num_iter):
    """
    Compute gradient descent for logistic regression.

    Returns:
        tuple: Optimized parameters, cost per iteration.
    """
    return grad_descent_logreg(X, y, theta, alpha, num_iter)

def data_normalization(X, y):
    """
    Normalize features in the dataset.

    Returns:
        tuple: Normalized feature matrix, labels, means, standard deviations.
    """
    meanX = np.mean(X, axis=0).reshape(-1, 1)
    std_deviationX = np.std(X, axis=0).reshape(-1, 1)
    X[:, 1] = (X[:, 1] - meanX[1, 0]) / std_deviationX[1, 0]
    X[:, 2] = (X[:, 2] - meanX[2, 0]) / std_deviationX[2, 0]

    meany = np.mean(y, axis=0).reshape(-1, 1)
    std_deviationy = np.std(y, axis=0).reshape(-1, 1)

    return X, y, meanX, std_deviationX, meany, std_deviationy

def not_normolized(Xdata, X_orig, y, m):
    """
    Perform logistic regression without normalizing data.
    """
    onesvec = np.ones((m, 1))
    X = np.concatenate((onesvec, X_orig), axis=1)
    n = X.shape[1]
    theta = np.zeros((n, 1))
    y = y.reshape([y.shape[0], 1])
    J, grad_J = compute_cost(X, y, theta)
    alpha = 0.001
    num_iters = 90000
    theta, J_iter = calc_GD(X, y, theta, alpha, num_iters)

    plt.figure()
    plt.plot(J_iter)
    plt.title("Cost per Iteration (Not Normalized)")
    plt.show()

    plt.figure()
    plot_log_reg_line(X, y, theta)
    plt.title("Decision Boundary (Not Normalized)")
    plt.close()

def normlized(Xdata, X_orig, y, m):
    """
    Perform logistic regression with normalized data.
    """
    onesvec = np.ones((m, 1))
    X_with_ones = np.concatenate((onesvec, X_orig), axis=1)
    X_normalized, y_normalized, meanX, std_deviationX, meany, std_deviationy = data_normalization(X_with_ones, y)
    n = X_normalized.shape[1]
    theta = np.zeros((n, 1))
    y = y.reshape([y.shape[0], 1])
    J, grad_J = compute_cost(X_normalized, y, theta)
    alpha = 0.001
    num_iters = 90000
    theta, J_iter = calc_GD(X_normalized, y, theta, alpha, num_iters)

    plt.figure()
    plt.plot(J_iter)
    plt.title("Cost per Iteration (Normalized)")
    plt.show()

    plt.figure()
    plot_log_reg_line(X_normalized, y, theta)
    plt.title("Decision Boundary (Normalized)")

def main():
    """
    Run logistic regression with and without data normalization.
    """
    not_normolized(*load_data())
    normlized(*load_data())

    # Conclusion #1: Better separation with normalization.
    # Conclusion #2: Faster convergence with normalization.

if __name__ == "__main__":
    main()
