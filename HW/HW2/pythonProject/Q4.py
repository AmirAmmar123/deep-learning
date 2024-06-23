import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from typing import List, Tuple


def three_binary_classification(X: np.ndarray, y: np.ndarray) -> List[make_pipeline]:
    """
    Train three binary classifiers using one-versus-all classification.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Labels.

    Returns:
        List[make_pipeline]: A list containing three trained classifiers.
    """
    classifiers = []
    for i in range(3):
        y_binary = (y == i).astype(int)
        clf = make_pipeline(StandardScaler(), LogisticRegression())
        clf.fit(X, y_binary)
        classifiers.append(clf)

    return classifiers


def plot_decision_boundaries(X: np.ndarray, y: np.ndarray) -> None:
    """
    Plot the decision boundaries for the first two features using one-versus-all classification.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Labels.
    """
    # Train three binary classifiers using one-versus-all classification with the first two features
    classifiers = []
    for i in range(3):
        y_binary = (y == i).astype(int)
        clf = make_pipeline(StandardScaler(), LogisticRegression())
        clf.fit(X[:, :2], y_binary)  # Train using only the first two features
        classifiers.append(clf)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

    plt.figure(figsize=(8, 6))
    plt.clf()

    colors = ['red', 'green', 'blue']
    labels = ['setosa', 'versicolor', 'virginica']
    for i, (clf, color, label) in enumerate(zip(classifiers, colors, labels)):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contour(xx, yy, Z, levels=[0], colors=color, linestyles=['-'])
        plt.scatter(X[y == i, 0], X[y == i, 1], c=color, label=label, edgecolor='k')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(loc='best')
    plt.title('One-vs-All Logistic Regression on Iris Data')
    plt.show()


def evaluate_classifiers(classifiers: List[make_pipeline], X_test: np.ndarray, y_test: np.ndarray) -> Tuple[
    np.ndarray, float]:
    """
    Predict and calculate the error rate using the trained classifiers.

    Args:
        classifiers (List[make_pipeline]): List of trained classifiers.
        X_test (np.ndarray): Test feature matrix.
        y_test (np.ndarray): Test labels.

    Returns:
        Tuple[np.ndarray, float]: Predicted labels and error rate.
    """
    predictions = np.zeros((X_test.shape[0], len(classifiers)))

    for i, clf in enumerate(classifiers):
        predictions[:, i] = clf.predict(X_test)

    # Get the predicted class with the highest confidence
    predicted_labels = np.argmax(predictions, axis=1)

    # Calculate error rate
    correct_predictions = np.sum(predicted_labels == y_test)
    error_rate = 1 - correct_predictions / len(y_test)

    return predicted_labels, error_rate


def main() -> None:
    """
    Main function to load the Iris dataset, train classifiers, evaluate them, and plot decision boundaries.
    """
    # Load the Iris dataset
    iris = datasets.load_iris()
    X = iris.data  # We take all the features: sepal length, sepal width, petal length, petal width
    y = iris.target

    # Define the test indices
    test_indices = np.concatenate([np.arange(35, 50), np.arange(85, 100), np.arange(135, 150)])

    # Extract the training set
    train_indices = np.setdiff1d(np.arange(len(X)), test_indices)
    X_train, y_train = X[train_indices], y[train_indices]

    # Extract the test set and labels
    X_test, y_test = X[test_indices], y[test_indices]

    # Train the classifiers
    classifiers = three_binary_classification(X_train, y_train)

    # Evaluate the classifiers on the test set
    predicted_labels, error_rate = evaluate_classifiers(classifiers, X_test, y_test)

    print("Predicted Labels:", predicted_labels)
    print("Error Rate:", error_rate)

    # Plot decision boundaries for the first two features as an example
    plot_decision_boundaries(X, y)  # Only for visualization purposes


if __name__ == '__main__':
    main()
