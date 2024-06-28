import numpy as np
import matplotlib.pyplot as plt
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

    def __init__(self, k: int, max_iter: int, local_min: int = 100000000000000):
        self.total_clusters = k
        self.max_iterations = max_iter
        self.local_min = local_min
        self.restart = 1
        self.centroids = None
        self.Xdata = None
        self.error = None
        self.labels = None

    def init_centroid(self):
        return self.Xdata[np.random.choice(self.Xdata.shape[0], self.total_clusters, replace=False)]

    def assign_samples(self):
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
            # Select points that belong to the current centroid
            points = self.Xdata[self.labels == i]
            # Calculate the mean of these points to update the centroid
            if len(points) > 0:
                new_centroids[i] = points.mean(axis=0)
        self.centroids = new_centroids

    def fit_predict(self, Xdata):
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
            self.centroids = self.init_centroid()
            self.labels = np.zeros(self.Xdata.shape[0], dtype=int)
            self.plot_init(f'Cluster plot init #{self.restart}')
            for _ in range(self.max_iterations):
                old_centroids = np.copy(self.centroids)
                self.centroid_calc()
                # Check for convergence (if centroids do not change)
                if np.allclose(old_centroids, self.centroids):
                    break
            if self.local_min is not None and self.local_min <= self.calc_error():
                return self.labels
            if self.error is not None and self.local_min <= self.error:
                break
            self.local_min = self.error
            self.restart += 1

    def calc_error(self):
        self.error = (1 / self.Xdata.shape[0]) * sum(
            np.linalg.norm(point - self.centroids[self.labels[i]])
            for i, point in enumerate(self.Xdata)
        )
        return self.error

    def plot_clusters(self, title: str = 'Cluster Plot with Centroids', labels: np.array = np.array([])):
        plt.scatter(self.Xdata[:, 1], self.Xdata[:, 0], c=labels, cmap='viridis', alpha=0.5)
        plt.scatter(self.centroids[:, 1], self.centroids[:, 0], c='red', marker='x', s=100, label='Centroids')
        plt.legend()
        plt.title(title)
        plt.show()

    def plot_init(self, title: str):
        plt.scatter(self.Xdata[:, 1], self.Xdata[:, 0], alpha=0.5)
        plt.scatter(self.centroids[:, 1], self.centroids[:, 0], c='red', marker='x', s=100, label='Centroids')
        plt.legend()
        plt.title(title)
        plt.show()

    def find_nearest_centroid(self, point):
        distances = np.linalg.norm(self.centroids - point, axis=1)
        return np.argmin(distances)


def main() -> None:
    HEADERS = ["culmen_length_mm", "culmen_depth_mm", "flipper_length_mm", "body_mass_g"]
    penguins_data = load_penguins_data()
    test_mat, train_mat = pick_up(penguins_data, pick_Adelie=146, pick_Chinstrap=68, pick_Gentoo=120, headers=HEADERS)

    FEATURES1 = [0, 2]
    _process_kmeans(
        train_mat, FEATURES1, 'Cluster Plot final with 2 features'
    )
    FEATURES2 = [0, 1, 2, 3]
    _process_kmeans(
        train_mat, FEATURES2, 'Cluster Plot final with 4 features'
    )


# TODO Rename this here and in `main`
def _process_kmeans(train_mat, features, title):
    kmeans = Kmeans(k=3, max_iter=100)
    y_label = kmeans.fit_predict(train_mat[:, features])
    kmeans.plot_clusters(title, y_label)


if __name__ == '__main__':
    main()
