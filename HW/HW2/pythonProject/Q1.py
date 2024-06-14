import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Material.Admittance_class_2024 import grad_descent_logreg, plot_log_reg_line, compute_cost


def calculate_probabilities(english_test_grade: float, math_test_grade: float, not_norm_theta: np.array,
                            norm_theta: np.array, meanX: np.array, std_deviationX: np.array) -> None:
    """
    Calculate and print the acceptance probabilities for given test grades.
    """
    # Unnormalized
    z_un = not_norm_theta[0] + not_norm_theta[1] * english_test_grade + not_norm_theta[2] * math_test_grade
    sigmoid_un = 1 / (1 + np.exp(-z_un))
    print("UNNORMALIZED PROBABILITY: The acceptance probability of this student is: ", sigmoid_un)

    # Normalized
    normalized_english_test_grade = (english_test_grade - meanX[1]) / std_deviationX[1]
    normalized_math_test_grade = (math_test_grade - meanX[2]) / std_deviationX[2]

    z_n = norm_theta[0] + norm_theta[1] * normalized_english_test_grade + norm_theta[2] * normalized_math_test_grade
    sigmoid_n = 1 / (1 + np.exp(-z_n))
    print("NORMALIZED PROBABILITY: The acceptance probability of this student is: ", sigmoid_n)


def load_data(path: str) -> tuple[pd.DataFrame, np.array, np.array, int]:
    """
    Load data from CSV file.

    Returns:
        tuple: DataFrame, feature matrix, labels, number of examples.
    """
    Xdata = np.genfromtxt(path, delimiter=',', skip_header=1)
    X_orig = Xdata[:, 0:2]
    Xdata = pd.read_csv(path)
    data = Xdata.to_numpy()
    X_orig = data[:, 0:2]
    y = data[:, 2]
    m = y.size
    return Xdata, X_orig, y, m


def plot_data(X_orig: np.array, y: np.array, shape1: str = 'ro', shape2: str = 'go') -> None:
    """
    Plot exam scores and admission status.
    """
    plt.figure()
    plt.plot(X_orig[y == 0, 0], X_orig[y == 0, 1], shape1,
             X_orig[y == 1, 0], X_orig[y == 1, 1], shape2)
    plt.grid(axis='both')
    plt.show()


def data_normalization(X: np.array, y: np.array) -> tuple[np.array, np.array, np.array, np.array, np.array, np.array]:
    """
    Normalize features in the dataset.

    Returns:
        tuple: Normalized feature matrix, labels, means, standard deviations.
    """
    meanX = np.mean(X, axis=0).reshape(-1, 1)
    std_deviationX = np.std(X, axis=0).reshape(-1, 1)
    X[:, 1] = (X[:, 1] - meanX[1, 0]) / std_deviationX[1, 0]
    X[:, 2] = (X[:, 2] - meanX[2, 0]) / std_deviationX[2, 0]

    mean = np.mean(y, axis=0).reshape(-1, 1)
    std_deviationy = np.std(y, axis=0).reshape(-1, 1)

    return X, y, meanX, std_deviationX, mean, std_deviationy


def add_onec_vec(X_orig: np.array, m: int) -> np.array:
    """
    Add vector of ones that have n*1 shape.
    """
    return np.concatenate((np.ones((m, 1)), X_orig), axis=1)


def initialize_log_reg_params(X_orig: np.array, y: np.array, m: int) -> tuple:
    """
    Initialize logistic regression parameters.

    Args:
        X_orig (np.array): Original feature matrix.
        y (np.array): Target vector.
        m (int): Number of examples.

    Returns:
        tuple: Initialized feature matrix with ones, reshaped target vector, and initial theta.
    """
    X = add_onec_vec(X_orig, m)
    n = X.shape[1]
    theta = np.zeros((n, 1))
    y = y.reshape([y.shape[0], 1])
    return X, y, theta


def not_normolized(Xdata: pd.DataFrame, X_orig: np.array, y: np.array, m: int, shape1: str = 'ro'
                   , shape2: str = 'go',
                   alpha=0.001, num_iters=90000) -> np.array:
    """
    Perform logistic regression without normalizing data.

    Returns:
        np.array: Optimized theta.
    """
    X, y, theta = initialize_log_reg_params(X_orig, y, m)
    J, grad_J = compute_cost(X, y, theta)

    theta, J_iter = grad_descent_logreg(X, y, theta, alpha, num_iters)

    plt.figure()
    plt.plot(J_iter)
    plt.title("Cost per Iteration (Not Normalized)")
    plt.show()

    plt.figure()
    plot_log_reg_line(X, y, theta, shape1, shape2)
    plt.title("Decision Boundary (Not Normalized)")
    plt.close()

    return theta


def normlized(Xdata: pd.DataFrame, X_orig: np.array, y: np.array, m: int) -> tuple[np.array, np.array, np.array]:
    """
    Perform logistic regression with normalized data.

    Returns:
        tuple: Optimized theta, means, standard deviations.
    """
    onesvec = np.ones((m, 1))

    X_with_ones = np.concatenate((onesvec, X_orig), axis=1)

    X_normalized, y_normalized, meanX, std_deviationX, mean, std = data_normalization(X_with_ones, y)

    n = X_normalized.shape[1]

    theta = np.zeros((n, 1))

    y = y.reshape([y.shape[0], 1])

    alpha = 0.01
    num_iters = 2500
    theta, J_iter = grad_descent_logreg(X_normalized, y, theta, alpha, num_iters)

    plt.figure()
    plt.plot(J_iter)
    plt.title("Cost per Iteration (Normalized)")
    plt.show()

    plt.figure()
    plot_log_reg_line(X_normalized, y, theta)
    plt.title("Decision Boundary (Normalized)")

    return theta, meanX, std_deviationX


def main() -> None:
    """
    Run logistic regression with and without data normalization.
    """
    Xdata, X_orig, y, m = load_data("Material/admittance_data.csv")

    plot_data(X_orig, y)

    not_norm_theta = not_normolized(Xdata, X_orig, y, m)

    norm_theta, mean, std = normlized(Xdata, X_orig, y, m)

    ENGLISH_TEST_GRADE = 65

    MATH_TEST_GRADE = 41

    calculate_probabilities(ENGLISH_TEST_GRADE, MATH_TEST_GRADE, not_norm_theta, norm_theta, mean, std)


if __name__ == "__main__":
    main()
