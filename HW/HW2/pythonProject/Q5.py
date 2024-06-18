import matplotlib.pyplot as plt

import numpy as np


def plot_points_and_boundary(X: np.array, y: np.array, theta: np.array, b=0):
    """
    Plots the data points and the decision boundary.
    """
    ind = 1
    x1_min = 1.1 * X[:, ind].min()
    x1_max = 1.1 * X[:, ind].max()
    x2_min = - (b + theta[0] * x1_min) / theta[1]
    x2_max = - (b + theta[0] * x1_max) / theta[1]  # This is the missing line
    x1lh = np.array([x1_min, x1_max])
    x2lh = np.array([x2_min, x2_max])
    x1 = X[:, 0]
    x2 = X[:, 1]
    plt.plot(x1[y[:, 0] == 1], x2[y[:, 0] == 1], 'go',
             x1[y[:, 0] == -1], x2[y[:, 0] == -1], 'rx',
             x1lh, x2lh, 'b-')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(f'Data')
    plt.grid(axis='both')
    plt.show()


def perceptron_train(X: np.array, y: np.array, plotflag: bool, max_iter: int, init_bias: int):
    """
    Implements the perceptron learning algorithm.

    Input arguments:
    X : data matrix, where each row is one observation
    y : labels (1 or -1)
    plotflag : 1 if to plot
    max_iter : maximum number of iterations

    Returns:
    theta, k - number of iterations (until a decision boundary classifies all the samples correctly)
    """

    num_correct = 0
    mat_shape = X.shape
    nrow = ncol = 0
    if len(mat_shape) > 1:
        nrow = mat_shape[0]
        ncol = mat_shape[1]
    else:
        X = X.reshape(X.shape[0], 1)

    current_index = 0
    theta = np.zeros((ncol, 1))
    b = init_bias
    j = 0
    k = 0
    is_first_iter = 1
    while num_correct < nrow and k < max_iter:
        j = j + 1
        xt = X[current_index, :]
        xt = xt.reshape(xt.shape[0], 1)
        yt = y[current_index]
        # ---------------------------------------------------------------------
        a = yt * (np.dot(theta.T, xt) + b)  # This is the missing line
        # ---------------------------------------------------------------------
        if is_first_iter == 1 or a <= 0:  # Modified comparison to <=
            # -----------------------------------------------------------------
            theta = theta + yt * xt  # Update theta
            b = b + yt  # Update bias
            num_correct = 0  # Reset num_correct
            k += 1  # Increment error count
            # -----------------------------------------------------------------
            is_first_iter = 0
            if plotflag :
                plot_points_and_boundary(X, y, theta, b)
                plt.pause(0.01)
        else:
            num_correct += 1
        current_index += 1
        if current_index >= nrow:  # Changed condition to >=
            current_index = 0
    return theta, b, k


def load_dat() -> tuple[np.array, np.array]:
    """
    Loads and returns the data from a .npz file.

    Returns:
        tuple: A tuple containing two numpy arrays X and y.
    """
    
    npzfile = np.load("./Material/Perceptron_exercise_2.npz")
    sorted(npzfile.files)
    X = npzfile['arr_0']
    y = npzfile['arr_1']
    return X, y


def main() -> None:
    """
    Executes the main logic of the program, including loading data, training the perceptron model, and printing the results.

    Returns:
        None
    """
    X, y = load_dat()
    int_biases = [0,1,2,3, -10]
    plotflag = True
    max_iter = 100
    for int_bias in int_biases:
        thata, b, k = perceptron_train(X, y, plotflag, max_iter, int_bias)
        print("Theta:", thata)
        print("Bias (b):", b)
        print("Number of iterations (k):", k)


if __name__ == '__main__':
    main()
