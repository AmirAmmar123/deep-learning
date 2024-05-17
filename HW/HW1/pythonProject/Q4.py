import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from Q3 import MinibatchGradientDescent
from Q2 import VectorizationOfTheCostFunc

sns.set()  # Setting seaborn style for plots


class Loader:
   def __init__(self, *paths):
      self.paths = paths
      self.data = None

   def load_data(self):
      self.data = {path: np.load(path) for path in self.paths}
      return self.data

   def plotter(self, **kwargs):
      X, Y = self.data.values()
      plt.xlabel(kwargs['X'])
      plt.ylabel(kwargs['Y'])
      plt.scatter(X, Y, color='blue', marker='.')
      plt.title(kwargs['Title'])
      plt.show()


class LR:
   def __init__(self, *args):
      self.X, self.Y = args

   def plotLR(self):
      X = self.X.reshape(-1, 1)
      model = LinearRegression()
      model.fit(X, self.Y)
      Y_pred = model.predict(X)

      plt.scatter(X, self.Y, color='blue', marker='.')
      plt.plot(X, Y_pred, color='red', label='Linear Regression Line')

      front_length_home = np.array([15, 27])
      expected_price = model.predict(front_length_home[:, np.newaxis])

      for i, length in enumerate(front_length_home):
         print(f"Expected price for front length home {length} price: {expected_price[i]} NIS")

      plt.xlabel('X')
      plt.ylabel('Y')
      plt.title('Linear Regression')
      plt.legend()
      plt.show()

      print("Model slope a1 =", model.coef_[0])
      print("Model intercept a0 =", model.intercept_)

      mbgd = MinibatchGradientDescent(self, 0.001, np.array([[0.5], [3]]), self.X.shape[0], 2000)
      theta = mbgd.MiniBatch()
      print(f'Mini batch Gradient Descent a0 = {theta[0, 0]}')
      print(f'Mini batch Gradient Descent a1 = {theta[1, 0]}')


def h(X, theta):
   return np.dot(theta.T, X.reshape(-1, 1))


def d_j(theta, X, y, j):
   s = 0
   m = X.shape[0]
   for i in range(m):
      s += h(X[i, :], theta) - y[i, 0] * X[i, j]
   return s


def gd(X, y, theta, alpha, num_iter, method):
   m = y.shape[0]
   J_iter = np.zeros((m * num_iter))
   votcf = VectorizationOfTheCostFunc(theta, X, y)
   k = 0
   for j in range(num_iter):
      randindex = np.random.permutation(m)
      for i in range(m):
         if method == 'online':
            xi = X[randindex[i], :].reshape(1, -1)
            yi = y[randindex[i], :]
            delta = np.dot(xi, theta) - yi
            theta -= alpha * delta * xi.T
            votcf.setTheta(theta)
            J_iter[k] = votcf.J()
         elif method == 'batch':
            xi = X
            yi = y
            delta = np.dot(xi, theta) - yi
            theta -= alpha * delta * xi
            votcf.setTheta(theta)
            J_iter[k] = votcf.J()
            votcf.setTheta(theta)
            J_iter[k] = votcf.J()
         k += 1
   plotJAsFofIter(J_iter, alpha, 500)
   return theta


def plotJAsFofIter(j_iter, alpha, section):
   plt.plot(np.arange(0, j_iter[:section].size), j_iter[:section])
   plt.xlabel('Iteration')
   plt.ylabel('J(Theta)')
   plt.title(f'J(theta) as a function of #iteration for alpha {alpha}')
   plt.show()


class MLExpr:
   def __init__(self, X_m_2, y, theta_init):
      self.X = np.concatenate((np.ones((X_m_2.size, 1)), X_m_2, np.power(X_m_2, 2)), axis=1)
      self.y = y
      self.theta = theta_init

   def predictTheta(self, how, alpha, iteration):
      return gd(self.X, self.y, self.theta, alpha, iteration, how)


def PolynomialRegressionofSecondOrder(X, y, theta_predicted_gd):
   poly_features = PolynomialFeatures(degree=2, include_bias=False)
   X_poly = poly_features.fit_transform(X)

   lin_reg = LinearRegression()
   lin_reg.fit(X_poly, y)

   plt.scatter(X, y, color='blue', label='Original data', marker='.')

   X_range = np.linspace(np.min(X), np.max(X), 100).reshape(-1, 1)
   X_range_poly = poly_features.transform(X_range)
   plt.plot(X_range, lin_reg.predict(X_range_poly), color='red', label='Fitted polynomial curve')

   plt.xlabel('X')
   plt.ylabel('y')
   plt.title('Polynomial Regression of Second Order')
   plt.legend()
   plt.show()

   coefficients = lin_reg.coef_
   intercept = lin_reg.intercept_

   print("Coefficients:", coefficients)
   print("Intercept:", intercept)
   print("\nPredicted values:")
   print(f"Coefficients: {theta_predicted_gd[1:, 0]}")
   print(f"Intercept: {theta_predicted_gd[0, 0]}")

   # Predict prices for front lengths of 15 and 27 meters
   front_lengths = np.array([15, 27]).reshape(-1, 1)
   front_lengths_poly = poly_features.transform(front_lengths)
   predicted_prices = lin_reg.predict(front_lengths_poly)

   for length, price in zip(front_lengths, predicted_prices):
      print(f'Predicted price for a house with a front length of {length[0]} meters is {float(price):.2f} NIS')



if __name__ == '__main__':
   loader = Loader('./DATA/TA_Xhouses.npy', './DATA/TA_yprice.npy')
   data = loader.load_data()
   loader.plotter(X='Length of the house front in meters', Y='House price in hundreds of thousands of NIS',
                  Title="House price (NIS'000s) based on front length (m).")
   print('Linear Regression:')
   lr = LR(data['./DATA/TA_Xhouses.npy'], data['./DATA/TA_yprice.npy'])
   lr.plotLR()
   print('\nPolynomial Regression of Second Order:')
   mlexper = MLExpr(data['./DATA/TA_Xhouses.npy'], data['./DATA/TA_yprice.npy'], np.random.uniform(1, 2, size=(3, 1)))
   theta = mlexper.predictTheta('online', 0.00001, 1000)
   PolynomialRegressionofSecondOrder(lr.X, mlexper.y, theta)


   #from the results we can see that there is differencies between the Linear Regression prediction and Polynomial Regression of Second Order prediction