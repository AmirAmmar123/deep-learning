from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from Q1_knn import load_penguins_data, pick_up
"""
@author: Amir Ammar
"""
HEADERS = ["culmen_length_mm", "culmen_depth_mm", "flipper_length_mm", "body_mass_g"]
penguins_data = load_penguins_data()
test_mat, train_mat = pick_up(penguins_data, pick_Adelie=146, pick_Chinstrap=68, pick_Gentoo=120, headers=HEADERS)

k=3
FEATURES = [0, 1,2,3]
kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
y_pred = kmeans.fit_predict(train_mat[:, FEATURES])


#
#k = 5
#kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)

# y_perd = kmeans.fit_predict(X)

# centroids after clustering
centroids = kmeans.cluster_centers_

X_new = np.array([[-1.5, 2]])


plt.scatter(train_mat[:, FEATURES][:, 0], train_mat[:, FEATURES][:, 1], c=y_pred)
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
