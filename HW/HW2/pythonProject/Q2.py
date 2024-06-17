import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Q1 import (
    plot_data, load_data, add_onec_vec,
    not_normolized, plot_log_reg_line
)
from Material.Admittance_class_2024 import grad_descent_logreg, sigmoid


def plot_log_quad_line(X, y, theta):
    """Plot decision boundary for quadratic logistic regression."""
    if theta.shape[0] == X.shape[1]:
        ind = 1
    else:
        ind = 0

    x1_min = X[:, ind].min()
    x1_max = X[:, ind].max()
    x1 = np.linspace(x1_min, x1_max, 10000)
    x2 = -((theta[3] * (x1 ** 2) + theta[1] * x1 + theta[0]) * (1 / theta[2]))

    plt.plot(X[y[:, 0] == 0, ind], X[y[:, 0] == 0, ind + 1], 'go',
             X[y[:, 0] == 1, ind], X[y[:, 0] == 1, ind + 1], 'ro',
             x1, x2, 'b-')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Quadratic Logistic Regression')
    plt.grid(axis='both')
    plt.show()


def transform_features_quadratic(X):
    """Transform input features to include a quadratic term."""
    X = add_onec_vec(X, X.shape[0])
    x1_squared = X[:, 1] ** 2
    X_poly = np.hstack([X, x1_squared.reshape(-1, 1)])
    return X_poly


def quad_logistic_regression(X, y, alpha=0.0001, num_iters=90000):
    """Perform quadratic logistic regression and plot results."""
    X_poly = transform_features_quadratic(X)
    theta = np.zeros((X_poly.shape[1], 1))
    y = y.reshape([y.shape[0], 1])
    theta, J_iter = grad_descent_logreg(X_poly, y, theta, alpha, num_iters)

    plot_log_quad_line(X, y, theta)

    plt.figure()
    plt.plot(J_iter)
    plt.title("Cost per Iteration (Quadratic Logistic Regression)")
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.grid(True)
    plt.show()

    return theta


def predict_log_reg(X, theta):
    """Predict binary labels using logistic regression."""
    return (sigmoid(np.dot(X, theta)) >= 0.5).astype(int)



def test_results(correct_linear: float,  y_test: np.array, accuracy_linear:float, correct_quad:float, accuracy_quad:float) -> None:
    """
        Print test results and accuracy.
    :param correct_linear:
    :param y_test:
    :param accuracy_linear:
    :param correct_quad:
    :param accuracy_quad:
    :return:
    """
    print(f"Linear Model: {correct_linear} out of {y_test.size} correctly classified.")
    print(f"Linear Model Accuracy: {accuracy_linear * 100:.2f}%")
    print(f"Quadratic Model: {correct_quad} out of {y_test.size} correctly classified.")
    print(f"Quadratic Model Accuracy: {accuracy_quad * 100:.2f}%")

def test(theta_linear, theta_quad):
    """Test linear and quadratic models on test data, print accuracy, and plot results."""
    Xdata = pd.read_csv('Material/email_data_test_2024.csv')
    X_test = Xdata.iloc[:, :-1].values
    y_test = Xdata.iloc[:, -1].values
    m = X_test.shape[0]


    X_test_linear = add_onec_vec(X_test, m)
    y_pred_linear = predict_log_reg(X_test_linear, theta_linear)
    correct_linear = np.sum(y_pred_linear.flatten() == y_test)
    accuracy_linear = correct_linear / y_test.size


    X_test_quad = transform_features_quadratic(X_test)
    y_pred_quad = predict_log_reg(X_test_quad, theta_quad)
    correct_quad = np.sum(y_pred_quad.flatten() == y_test)
    accuracy_quad = correct_quad / y_test.size

    test_results(correct_linear, y_test, accuracy_linear, correct_quad, accuracy_quad)


    plot_log_reg_line(X_test_linear, y_test.reshape((y_test.shape[0], 1)), theta_linear, 'go', 'rd')


    plt.figure()
    plot_log_quad_line(X_test, y_test.reshape((y_test.shape[0], 1)), theta_quad)


def main():
    """Main function to run logistic regression and test models."""
    Xdata, X_orig, y, m = load_data('Material/email_data_1.csv')

    plot_data(X_orig, y, 'go', 'dr')

    # Linear model
    linear_theta = not_normolized(Xdata, X_orig, y, m, 'go', 'dr')

    # Quadratic model
    theta_quad = quad_logistic_regression(X_orig, y, alpha=1, num_iters=20000)

    # Test models
    test(linear_theta, theta_quad)


if __name__ == '__main__':
    main()
