import numpy as np
from Q4 import gd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import copy
class ReadData:
    def __init__(self, path, delimiter):
        self.path = path
        self.delimiter = delimiter
        self.data = None

    def load(self):
        self.data = np.loadtxt(self.path, delimiter=self.delimiter)
        return self.data


def UpgradeData(data):
     x0 = np.ones((data.shape[0], 1),dtype=float)
     x1 = data[:,0].reshape(-1,1)
     x2 = data[:,1].reshape(-1,1)
     X = np.concatenate( (x0,x1,x2), axis=1)
     y = data[:,2].reshape(-1,1)

     X_org = copy.deepcopy(X)
     y_org = copy.deepcopy(y)

     return *data_normalization(X,y),X_org, y_org

def data_normalization(X,y):
    meanX = np.mean(X, axis=0).reshape(-1,1)
    std_deviationX = np.std(X, axis=0).reshape(-1,1)
    X[:,1] = (X[:,1] - meanX[1,0])/std_deviationX[1,0]
    X[:, 2] = (X[:, 2] - meanX[2, 0]) / std_deviationX[2, 0]

    meany = np.mean(y, axis=0).reshape(-1,1)
    std_deviationy = np.std(y, axis=0).reshape(-1,1)

    # y[:,0] = (y[:,0] - meany[0,0]) /std_deviationy[0,0] after a discussion with Yizhar, there is no need to normalize the Y

    return X,y,meanX,std_deviationX,meany,std_deviationy


class LR:
   def __init__(self, *args):
      self.X, self.y = args
      self.X = self.X[:, 1:3]  # Only use the features, excluding the bias term
      self.x1_front_home_size = self.X[:, 0].reshape(-1, 1)
      self.x2_rooms = self.X[:, 1].reshape(-1, 1)
      self.X = np.hstack((self.x1_front_home_size, self.x2_rooms))

   def plotLR(self):
      model = LinearRegression()
      model.fit(self.X, self.y)

      fig = plt.figure()
      ax = fig.add_subplot(111, projection='3d')

      ax.scatter(self.X[:, 0], self.X[:, 1], self.y, color='blue', marker='.', label='Data points')

      # Create a grid for X1 and X2 to plot the regression plane
      X1_range = np.linspace(self.X[:, 0].min(), self.X[:, 0].max(), 10)
      X2_range = np.linspace(self.X[:, 1].min(), self.X[:, 1].max(), 10)
      X1_grid, X2_grid = np.meshgrid(X1_range, X2_range)
      X_grid = np.c_[X1_grid.ravel(), X2_grid.ravel()]
      y_pred_grid = model.predict(X_grid).reshape(X1_grid.shape)

      # Plot the regression plane
      ax.plot_surface(X1_grid, X2_grid, y_pred_grid, color='red', alpha=0.5, rstride=100, cstride=100)

      ax.set_xlabel('Area')
      ax.set_ylabel('Number of rooms')
      ax.set_zlabel('Price in thousands$')
      ax.set_title('Linear Regression')
      ax.legend()
      plt.show()

      front_length_home = np.array([[100, 5]])
      expected_price = model.predict(front_length_home)

      for i, length in enumerate(front_length_home):
         print(f"Expected price for front length home {length[0]} price: {expected_price[i]} NIS")

      print("Model coefficients =", model.coef_)
      print("Model intercept =", model.intercept_)


if __name__ == "__main__":
    # Initialize the ReadData class with the path to the file and the delimiter
    reader = ReadData('./DATA/houses.txt', ',')
    # Load the data into a numpy array
    data = reader.load()
    X,y, Xmean,Xsd,ymean,ysd,X_org,y_org = UpgradeData(data)
    theta = gd(X,y,np.random.uniform(1, 2, size=(3, 1)), 0.003  ,1000, 'online')
    lr = LR(X_org,y_org)
    lr.plotLR()

    inv_X_tr_X= np.linalg.inv(np.dot(X.T,X))
    X_T_y = np.dot(X.T,y)
    theta_expected = np.dot(inv_X_tr_X,X_T_y)
    print("Normalized Theta gd on-line:\n", theta)

    print("\n(X^T * X)^(-1) * X^T * y={Normal Equations}\n",theta_expected)


