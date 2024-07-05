from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt

"""
@author: Amir Ammar
"""


 # Generate random blobs, (Gaussian 2d meo)
blob_centers = np.array([
                             [0.2, 1.2], [-1.5, 2.1],
                             [-2.7, 1.8], [-2.7, 2.7],
                             [-2.7, 1.7]
    ]
                            )

blob_std = np.array([0.355, 0.29, 0.11, 0.11, 0.11])

X, y = make_blobs(n_samples=2700, centers=blob_centers, cluster_std=blob_std, random_state=42)

plt.scatter(X[:, 0], X[:, 1])
plt.show()

inertia = np.zeros((11,))
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    y_pred = kmeans.fit_predict(X)
    # saving the best inertia so far
    inertia[k] = kmeans.inertia_
    print(kmeans.cluster_centers_)
    print('-------------Centroids with 2d features-------------------')

plt.plot(inertia[1:])
plt.show()
#
#k = 5
#kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)

# y_perd = kmeans.fit_predict(X)

# centroids after clustering
centroids = kmeans.cluster_centers_

X_new = np.array([[-1.5, 2]])


plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()
plt.figure(1)
plt.subplot(111)
# get the points belonging to each cluster center and plot them
# plt.plot(X[y_perd == 0, 0], X[y_perd == 0, 1], 'go', label='cluster 0')
# plt.plot(X[y_perd == 1, 0], X[y_perd == 1, 1], 'bo', label='cluster 1')
# plt.plot(X[y_perd == 2, 0], X[y_perd == 2, 1], 'ro', label='cluster 2')
# plt.plot(X[y_perd == 3, 0], X[y_perd == 3, 1], 'yo', label='cluster 3')
# plt.plot(X[y_perd == 4, 0], X[y_perd == 4, 1], 'bo', label='cluster 4')
# plt.show()
