import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from typing import List, Tuple


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def compute_cost(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    m = len(y)
    h = sigmoid(X @ theta)
    cost = (-1 / m) * (y.T @ np.log(h) + (1 - y).T @ np.log(1 - h))
    return cost


def gradient_descent(X: np.ndarray, y: np.ndarray, theta: np.ndarray, alpha: float, num_iters: int) -> np.ndarray:
    m = len(y)
    J_history = []

    for _ in range(num_iters):
        h = sigmoid(X @ theta)
        theta -= (alpha / m) * (X.T @ (h - y))
        J_history.append(compute_cost(X, y, theta))

    return theta, J_history


def one_vs_all(X: np.ndarray, y: np.ndarray, num_labels: int, alpha: float, num_iters: int) -> np.ndarray:
    m, n = X.shape
    all_theta = np.zeros((num_labels, n))

    for i in range(num_labels):
        theta = np.zeros(n)
        y_i = np.where(y == i, 1, 0)
        theta, _ = gradient_descent(X, y_i, theta, alpha, num_iters)
        all_theta[i] = theta

    return all_theta


def predict_one_vs_all(all_theta: np.ndarray, X: np.ndarray) -> np.ndarray:
    predictions = sigmoid(X @ all_theta.T)
    return np.argmax(predictions, axis=1)


def plot_decision_boundaries(X: np.ndarray, y: np.ndarray, all_theta: np.ndarray) -> None:
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

    plt.figure(figsize=(8, 6))
    plt.clf()

    colors = ['red', 'green', 'blue']
    labels = ['setosa', 'versicolor', 'virginica']
    for i, color, label in zip(range(3), colors, labels):
        Z = sigmoid(np.c_[np.ones((xx.ravel().shape[0], 1)), xx.ravel(), yy.ravel()] @ all_theta[i])
        Z = Z.reshape(xx.shape)
        plt.contour(xx, yy, Z, levels=[0.5], colors=color, linestyles=['-'])
        plt.scatter(X[y == i, 0], X[y == i, 1], c=color, label=label, edgecolor='k')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(loc='best')
    plt.title('One-vs-All Logistic Regression on Iris Data')
    plt.show()


def main() -> None:
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # Only using the first two features for visualization purposes
    y = iris.target

    # Add intercept term to X
    X = np.c_[np.ones((X.shape[0], 1)), X]

    # Define the test indices
    test_indices = np.concatenate([np.arange(35, 50), np.arange(85, 100), np.arange(135, 150)])

    # Extract the training set
    train_indices = np.setdiff1d(np.arange(len(X)), test_indices)
    X_train, y_train = X[train_indices], y[train_indices]

    # Extract the test set and labels
    X_test, y_test = X[test_indices], y[test_indices]

    # Train the classifiers
    alpha = 0.09
    num_iters = 2500
    num_labels = 3
    all_theta = one_vs_all(X_train, y_train, num_labels, alpha, num_iters)

    # Predict the test set
    predicted_labels = predict_one_vs_all(all_theta, X_test)

    # Calculate error rate
    error_rate = np.mean(predicted_labels != y_test)
    print("Predicted Labels:", predicted_labels)
    print("Error Rate:", error_rate)

    # Plot decision boundaries for the first two features as an example
    plot_decision_boundaries(X[:, 1:3], y, all_theta[:, [0, 1, 2]])  # Only for visualization purposes


if __name__ == '__main__':
    main()
