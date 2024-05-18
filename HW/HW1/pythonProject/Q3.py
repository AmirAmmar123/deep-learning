import numpy as np
from typing import Callable,Iterable
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()# data visualization library
from sklearn.linear_model import LinearRegression


class ReadData:
   def __init__(self, path = './DATA/Cricket.npz'):
      self.path = path
      self.data = self.data = np.loadtxt(self.path)
   def load(self):
      return self.data



class OldFaithful:
   def __init__(self, rd:ReadData):
      self.X = rd.data[:,0]
      self.Y = rd.data[:,1]
   def plot_data(self):
      plt.scatter(self.X, self.Y, color='red', marker='x')
      plt.xlabel('Duration of eruption (minutes)')
      plt.ylabel('Time to next eruption (minutes)')
      plt.title('Old Faithful Eruptions')
      plt.show()


class MinibatchGradientDescent:
   def __init__(self,Data:OldFaithful, alpha:float, ThetaInit:Iterable, minibatchSize:int, epochs:int ):
      self.theta = ThetaInit
      self.epochs = min(epochs,2000) # iteration
      self.minibatchSize = min(minibatchSize, Data.Y.shape[0]) # size of the Data
      self.alpha = alpha
      self.Y = Data.Y
      self.X = np.ones((Data.X.shape[0], 2), dtype='float64')
      self.upperBound = self.Y.shape[0] - self.minibatchSize
      self.lowerBound = 0
      self.X[:,1] = Data.X.flatten()

   def MiniBatch(self):
      for x in range(0, self.epochs):
         subX, subY = self.randomSubVector()
         x_flatten = subX[:,1]
         res  = self.deJ_deThetaj(subX, subY.reshape(-1,1)).flatten()
         result0 = np.sum(res * 1)
         result1 = np.sum(res * x_flatten)
         self.theta[0, 0] = self.theta[0, 0] - self.alpha * 1 / (self.minibatchSize) * result0
         self.theta[1,0] = self.theta[1,0] - self.alpha *  1/(self.minibatchSize ) * result1
      return self.theta

   def deJ_deThetaj(self, subX, subY):
      return  np.dot(subX,self.theta) - subY


   def randomSubVector(self):
      if self.upperBound <= 0: return self.X, self.Y
      start = np.random.randint(self.lowerBound,self.upperBound)
      return self.X[start:start + self.minibatchSize, :], self.Y[start : start + self.minibatchSize]


class LinearRegressionPlotter():
   def __init__(self, of: OldFaithful, mbgd: MinibatchGradientDescent):
      self.of = of
      self.mbgd = mbgd
   def plotLinearRegression(self):

      # Fit linear regression model
      plt.scatter(self.of.X, self.of.Y, color='red', marker='x')
      plt.xlabel('Duration of eruption (minutes)')
      plt.ylabel('Time to next eruption (minutes)')
      plt.title('Old Faithful Eruptions')

      model = LinearRegression(fit_intercept=True)
      model.fit(self.of.X[:, np.newaxis], self.of.Y)

      # Predict time to next eruption using linear regression
      current_durations = np.array([2.1, 3.5, 5.2])
      expected_times = model.predict(current_durations[:, np.newaxis])

      for i, duration in enumerate(current_durations):
         print(f"Expected time until next eruption for duration {duration} minutes: {expected_times[i]} minutes")

      # Plot linear regression line
      xfit = np.linspace(1.5, 5, 10000)
      yfit = model.predict(xfit[:, np.newaxis])
      plt.plot(xfit, yfit)

      theta = self.mbgd.MiniBatch()
      print("Model intercept a0 = ", model.intercept_)
      print("Model slope a1 = ", model.coef_[0])

      print(f'Mini batch Gradient Descent a0 = {theta[0,0]}')
      print(f'Mini batch Gradient Descent a1 = {theta[1,0]}')

      plt.show()


class VectorizationOfTheCostFunc:
   def __init__(self, Theta_vec , X_mat, Y_vec ):
      self.Theta_vec = Theta_vec
      self.X_mat = X_mat
      self.Y_vec = Y_vec
      self.m = Y_vec.shape[0]
      self.z = np.dot(self.X_mat, self.Theta_vec) - self.Y_vec

   def J(self):
      return (1/(2*self.m)) *  np.dot(self.z.T, self.z)

   def setTheta(self, otherTheta):
      self.Theta_vec = otherTheta
      self.z = np.dot(self.X_mat, self.Theta_vec)


if __name__ == '__main__':
   RD = ReadData( path= './DATA/faithful.txt')
   of = OldFaithful(RD)
   # of.plot_data()
   for alpha in [ 0.01, 0.001, 0.02, 0.002, ] :
      mbgd = MinibatchGradientDescent(of,alpha, np.array( [ [0.5], [10] ] ), 30, 2000)
      LRP = LinearRegressionPlotter(of, mbgd)
      LRP.plotLinearRegression()
      V = VectorizationOfTheCostFunc(mbgd.theta, mbgd.X, mbgd.Y.reshape(mbgd.Y.shape[0],1)).J()
      print(f'cost function value for theta = {mbgd.theta[0,0], mbgd.theta[1,0]}, J(theta) = {V}')
      print(f'result for alpha = {alpha}\n--------------------------------------------------------------\n')


