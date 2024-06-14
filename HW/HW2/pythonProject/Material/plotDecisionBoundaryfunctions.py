import numpy as np
import matplotlib.pyplot as plt


def polyFeatureVector(x1, x2, degree):
    """
    Generate polynomial features from two input features x1 and x2 up to a specified degree.

    Parameters:
    x1 (np.ndarray): Input feature vector 1.
    x2 (np.ndarray): Input feature vector 2.
    degree (int): Maximum degree of polynomial features.

    Returns:
    np.ndarray: Matrix of polynomial features with shape (m, n) where m is the number of examples and n is the number of features.
    """
    x1.shape = (x1.size, 1)
    x2.shape = (x2.size, 1)
    Xp = np.ones(shape=(x1[:, 0].size, 1))

    for j in range(1, degree + 1):
        for k in range(j + 1):
            p = (x1 ** (j - k)) * (x2 ** k)
            Xp = np.append(Xp, p, axis=1)
    return Xp


def plotDecisionBoundary1(theta, X, y, d):
    """
    Plot the decision boundary for logistic regression.

    Parameters:
    theta (np.ndarray): Parameter vector for the logistic regression model.
    X (np.ndarray): Input feature matrix.
    y (np.ndarray): Output vector.
    d (int): Degree of polynomial features used.

    Returns:
    None
    """
    x1 = X[:, 1]
    x2 = X[:, 2]

    plt.plot(x1[y[:, 0] == 0], x2[y[:, 0] == 0], 'ro', label='Negative Class')
    plt.plot(x1[y[:, 0] == 1], x2[y[:, 0] == 1], 'go', label='Positive Class')

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Decision Boundary')
    plt.legend()
    plt.grid(True)

    u = np.linspace(np.min(x1), np.max(x1), 50)
    v = np.linspace(np.min(x2), np.max(x2), 50)
    z = np.zeros((len(u), len(v)))

    for i in range(len(u)):
        for j in range(len(v)):
            z[i, j] = np.dot(polyFeatureVector(np.array([u[i]]), np.array([v[j]]), d), theta)

    z = z.T  # Transpose z to match contour plot expectations
    plt.contour(u, v, z, levels=[0], colors='blue', linewidths=2)  # Specify colors and linewidths for contour plot
    plt.show()
