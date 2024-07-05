from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from numpy import ndarray, dtype
from time import sleep
from Q1_knn import load_penguins_data, pick_up




class Kmeans:
    """
    Represents a KMeans clustering algorithm with methods for initialization, centroid calculation, assigning samples
    to clusters, fitting the model to data, predicting cluster labels, calculating clustering error, and plotting
    clusters.

    Args:
        k (int): The number of clusters.
        max_iter (int): The maximum number of iterations for convergence.
        local_min (int, optional): The local minimum clustering error threshold. Defaults to 100000000000000.

    Returns:
        None
    """

    EPS = 0.05

    def __init__(self, k: int, max_iter: int, local_min: int = 10000000000000000000000000000000000):
        self.total_clusters = k
        self.max_iterations = max_iter
        self.local_min = local_min
        self.restart = 1
        self.inertias = []
        self.centroids = None
        self.Xdata = None
        self.error = None
        self.labels = None

    def init_centroid(self) -> ndarray[Any, dtype[Any]]:
        return self.Xdata[np.random.choice(self.Xdata.shape[0], self.total_clusters, replace=False)]

    def assign_samples(self) -> ndarray[Any, dtype[Any]]:
        """
        Assigns each data point to the nearest centroid based on the Euclidean distance.

        Returns:
            numpy.ndarray: An array containing the cluster labels for each data point.
        """

        for i, point in enumerate(self.Xdata):
            self.labels[i] = self.find_nearest_centroid(point)
        return self.labels

    def centroid_calc(self):
        """
        Calculates new centroids based on the current cluster assignments by updating the centroids to the mean of
        the data points in each cluster.

        Returns:
            None
        """

        self.labels = self.assign_samples()
        new_centroids = np.zeros_like(self.centroids)
        for i in range(self.total_clusters):

            points = self.Xdata[self.labels == i]

            if len(points) > 0:
                new_centroids[i] = points.mean(axis=0)
        self.centroids = new_centroids

    def fit_predict(self, Xdata) -> ndarray[Any, dtype[Any]]:
        """
        Fits the KMeans model to the input data and predicts the cluster labels for each data point iteratively until
        convergence and with local minima clustering error.

        Args:
            Xdata (numpy.ndarray): The input data to fit the KMeans model.

        Returns:
            numpy.ndarray: An array of predicted cluster labels for each data point.
    """

        self.Xdata = Xdata
        while True:
            self.init_data()
            for _ in range(self.max_iterations):
                old_centroids = np.copy(self.centroids)
                self.centroid_calc()
                if np.allclose(old_centroids, self.centroids):
                    break
            if self.local_min is not None and np.abs(self.local_min - self.calc_error()) <= self.EPS:
                return self.labels
            self.local_min = self.error



    def calc_error(self) -> float | int:
        """
    Calculates the error of the clustering based on the distance between data points and their assigned centroids.
    Returns:
        float: The calculated error of the clustering.
    """

        self.error = (1 / self.Xdata.shape[0]) * sum(
            np.linalg.norm(point - self.centroids[self.labels[i]])
            for i, point in enumerate(self.Xdata)
        )
        return self.error

    def plot_clusters(self, title: str, labels) -> None:

        plt.scatter(self.Xdata[:, 1], self.Xdata[:, 0], c=labels, cmap='viridis', alpha=0.5)

        self._extracted_from_plot_init_(title)

    def plot_init(self, title: str) -> None:
        """
    Plots the initial state of the clusters with centroids based on the input data.

    Args:
        title (str): The title of the plot.

    Returns:
        None
    """

        plt.scatter(self.Xdata[:, 1], self.Xdata[:, 0], alpha=0.5)
        self._extracted_from_plot_init_(title)

    def _extracted_from_plot_init_(self, title):
        plt.scatter(
            self.centroids[:, 1],
            self.centroids[:, 0],
            c='red',
            marker='x',
            s=100,
            label='Centroids',
        )
        plt.legend()
        plt.title(title)
        plt.show()
        sleep(1)

    def find_nearest_centroid(self, point):
        distances = np.linalg.norm(self.centroids - point, axis=1)
        return np.argmin(distances)

    def init_data(self):
        self.centroids = self.init_centroid()
        self.labels = np.zeros(self.Xdata.shape[0], dtype=int)
        # self.plot_init(f'Cluster plot init #{self.restart}')

    def predict(self, Xtest: np.array ) -> np.array:
        pass


def test(train_mat,  FEATURES, title):
    _process_kmeans(
        train_mat, FEATURES, title
    )
def main() -> None:
    """
    Executes the main process of the KMeans clustering algorithm on penguins data with different sets of features.

    Returns:
        None
    """

    HEADERS = ["culmen_length_mm", "culmen_depth_mm", "flipper_length_mm", "body_mass_g"]
    penguins_data = load_penguins_data()
    test_mat, train_mat = pick_up(penguins_data, pick_Adelie=146, pick_Chinstrap=68, pick_Gentoo=120, headers=HEADERS)

    FEATURES1 = [0, 2]
    title1 = 'Cluster Plot final with 2 features'
    FEATURES2 = [0, 1, 2, 3]
    title2 = 'Cluster Plot final with 4 features'
    test(train_mat, FEATURES1, title1)
    test(train_mat, FEATURES2, title2)


def _process_kmeans(train_mat, features, title):

    kmeans = Kmeans(k=3, max_iter=100)
    y_label = kmeans.fit_predict(train_mat[:, features])
    kmeans.plot_clusters(title, y_label)


if __name__ == '__main__':
    main()
