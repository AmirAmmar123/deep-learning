# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 00:57:56 2022
Hiararchical clustering and plotting using a dendrogram
@author: Scipy example
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


# You should do the exercise manually on paper, and plot the dendrogram also
# manually, then please compare your results to those achieved using
# AgglomerativeClustering and plot_dendrogram.


# Simple 2-d example of using AgglomerativeClustering and plot_dendrogram.
# We use here "single linkage",  which means that the distance between two clusters
# is defined by the distance between the closest pair of elements.
# Other choices are also common - average linkage or complete linkage.
# complete linkage is defined by the distance between the furthest pair of elements.

###--------- replace the example here with the exercise data for comparison
X = np.array([[1, 1], [2, 1], [5, 4], [6, 5], [6.5, 6]
              ])
####### -------------------------------------------------------

clustering = AgglomerativeClustering().fit(X)
clustering.labels_
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
model = model.fit(X)
plt.title("Hierarchical Clustering Dendrogram")
# plot the top three levels of the dendrogram
plot_dendrogram(model, truncate_mode="level", p=2)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()
