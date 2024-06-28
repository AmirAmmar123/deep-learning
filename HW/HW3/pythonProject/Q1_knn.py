import numpy as np
import pandas as pd
import seaborn as sns
from typing import Any, Tuple, List


def knn(x_test: np.ndarray, x_train: np.ndarray, y_train: np.ndarray, k: int, features: list) -> float:
    correct_predictions = 0
    """
    Performs k-Nearest Neighbors classification on test data using training data and labels.
    
    Args:
        x_test (np.ndarray): Test data points.
        x_train (np.ndarray): Training data points.
        y_train (np.ndarray): Labels for the training data.
        k (int): Number of nearest neighbors to consider.
        features (list): List of feature indices to use for distance calculation.
    
    Returns:
        float: Accuracy percentage of the k-NN classification on the test data.
    """

    for test_sample in x_test:
        distances = np.linalg.norm(x_train[:, features] - test_sample[features], axis=1)
        k_nearest_indices = np.argsort(distances)[:k]
        k_nearest_labels = y_train[k_nearest_indices]

        # Find the most common label (mode) in the k nearest neighbors
        unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
        most_common_label = unique_labels[np.argmax(counts)]

        correct_predictions += most_common_label == test_sample[-1]

    return correct_predictions / len(x_test) * 100


def load_penguins_data() -> pd.DataFrame:
    sns.set(style='white', context='notebook', rc={'figure.figsize': (14, 10)})
    return pd.read_csv(
        "https://github.com/allisonhorst/palmerpenguins/raw/5b5891f01b52ae26ad8cb9755ec93672f49328a8/data"
        "/penguins_size.csv"
    )


def add_label(arr: np.ndarray, label: int) -> np.ndarray:
    labels = np.full((arr.shape[0], 1), label)
    return np.hstack((arr, labels))


def pick_up(data: pd.DataFrame, pick_Adelie: int, pick_Chinstrap: int, pick_Gentoo: int, headers: List) -> Tuple[
    np.ndarray, np.ndarray]:
    """
    Constructs test and training matrices with labels for penguin species based on the provided data and selection criteria.

    Args:
        data (pd.DataFrame): Input DataFrame containing penguin data.
        pick_Adelie (int): Number of Adelie penguin samples to pick.
        pick_Chinstrap (int): Number of Chinstrap penguin samples to pick.
        pick_Gentoo (int): Number of Gentoo penguin samples to pick.
        headers (List): List of headers to include in the matrices.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Test matrix and training matrix with labels for the selected penguin species.
    """

    penguin_data = data.dropna()

    Adelie_data = penguin_data[penguin_data['species_short'] == 'Adelie']
    Gentoo_data = penguin_data[penguin_data['species_short'] == 'Gentoo']
    Chinstrap_data = penguin_data[penguin_data['species_short'] == 'Chinstrap']

    Adelie_train = Adelie_data.head(pick_Adelie)
    Gentoo_train = Gentoo_data.head(pick_Gentoo)
    Chinstrap_train = Chinstrap_data.head(pick_Chinstrap)

    Adelie_test = Adelie_data.iloc[pick_Adelie:]
    Gentoo_test = Gentoo_data.iloc[pick_Gentoo:]
    Chinstrap_test = Chinstrap_data.iloc[pick_Chinstrap:]

    # Constructing the training matrix with labels
    train_matrix = np.vstack((
        add_label(Adelie_train[headers].values, 0),
        add_label(Gentoo_train[headers].values, 1),
        add_label(Chinstrap_train[headers].values, 2),
    ))

    # Constructing the test matrix with labels
    test_matrix = np.vstack((
        add_label(Adelie_test[headers].values, 0),
        add_label(Gentoo_test[headers].values, 1),
        add_label(Chinstrap_test[headers].values, 2),
    ))

    return test_matrix, train_matrix


def main() -> None:
    """
    Executes the main workflow of loading penguins data, selecting features, running k-NN algorithm with different k
    values, and printing the accuracy results.

    No arguments are passed to the main function.

    Returns:
        None
    """

    HEADERS1 = ["culmen_length_mm", "culmen_depth_mm", "flipper_length_mm", "body_mass_g"]
    data = load_penguins_data()
    test_mat, train_mat = pick_up(data, pick_Adelie=100, pick_Chinstrap=50, pick_Gentoo=34, headers=HEADERS1)

    X_train = train_mat[:, :-1]
    y_train = train_mat[:, -1]
    X_test = test_mat
    FEATURES1 = [0, 1, 2, 3]
    FEATURES2 = [0, 2]

    for k in [1, 3, 5]:
        accuracy = knn(X_test, X_train, y_train, k,
                       FEATURES2)
        print(f'Accuracy for k={k}: {accuracy:.2f}%')
    print(f'Headers chosen {[HEADERS1[i] for i in FEATURES2]}\n')

    for k in [1, 3, 5]:
        accuracy = knn(X_test, X_train, y_train, k,
                       FEATURES1)
        print(f'Accuracy for k={k}: {accuracy:.2f}%')
    print(f'Headers chosen {[HEADERS1[i] for i in FEATURES1]}')


if __name__ == "__main__":
    main()
