import numpy as np
import matplotlib.pyplot as plt 




def compute_cost(X, y, theta):
    m = y.shape[0]
    z = np.dot(X, theta) - y
    J = 1/(2*m) * np.dot(z.T, z)
    # J = 1 / (2 * m) * np.sum(z)

    return J



def gd_ol(X, y, theta, alpha, num_iter):
    m = y.shape[0]
    J_iter = np.zeros((m * num_iter))
    k = 0
    for j in range(num_iter):
       randindex = np.random.permutation(m)
       for i in range(m):
          xi = X[randindex[i],: ]
          xi = xi.reshape(1, xi.shape[0])
          yi = y[randindex[i],: ]
          delta = np.dot(xi, theta) - yi
          theta = theta - alpha * delta * xi.T
          J_iter[k] = compute_cost(X, y, theta)
          k += 1 
    return theta, J_iter 


if __name__ == '__main__':

    data = np.load('./Cricket.npz')
    y_x = data['arr_0']
    x =  y_x[:,1]
    y = y_x[:, 0]

    x = x.reshape(x.shape[0], 1)
    y = y.reshape(y.shape[0], 1)

    plt.figure(1)
    plt.plot(x,y, 'ro')
    plt.grid(axis = 'both')
    plt.show()


    m = y.size

    onesVec = np.ones((m,1))

    X = np.concatenate((onesVec, x), axis=1) ## concate verticals
    
    alpha = 0.0001
    num_iter = 100
    n = X.shape[1]
    theta = np.zeros((n,1))
    theta, J_iter = gd_ol(X, y, theta, alpha, num_iter)









