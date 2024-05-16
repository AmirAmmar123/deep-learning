import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()# data visualization library
from sklearn.linear_model import LinearRegression



class LR:
   """
    Class for generating linear regression models and visualizing them.

    Parameters:
        a1 : float
            Slope of the linear regression model.
        a0 : float
            Intercept of the linear regression model.
        sample : int
            Number of samples to generate for the model.
        num : int
            Number of points to use for plotting the model.
        stop : float
            End value for the x-axis range for plotting.
        start : float
            Start value for the x-axis range for plotting.
        funci : {0, 1}
            Indicator for the type of linear model:
            - 0: y = a0 + a1 * x + ε, where ε is random noise following a normal distribution.
            - 1: y = a0 + a1 + ε, where ε is random noise following a normal distribution.

    Attributes:
        x : numpy.ndarray
            Generated random x values for the model.
        y : numpy.ndarray
            Generated random y values for the model.
        model : sklearn.linear_model.LinearRegression
            Linear regression model fitted to the generated data.
        xfit : numpy.ndarray
            Points for the x-axis used for plotting the model.
        yfit : numpy.ndarray
            Points for the y-axis used for plotting the model.
    """
   def __init__(self, **kwargs):
      # preparing the data
      self.a1 = kwargs['a1']
      self.a0 = kwargs['a0']
      self.sample = kwargs['sample']
      self.num = kwargs['num']
      self.stop = kwargs['stop']
      self.start = kwargs['start']
      self.funci = kwargs['funci']

      self.x = 10 * np.random.rand( self.sample )
      self.y = self.func()

      plt.scatter(self.x, self.y)
      self.model =  LinearRegression(fit_intercept=True)
      self.model.fit(self.x[:, np.newaxis], self.y)
      self.xfit = np.linspace(self.start, self.stop, self.num)
      self.yfit = self.model.predict(self.xfit[:, np.newaxis])
      plt.scatter(self.x, self.y)
      plt.plot(self.xfit, self.yfit)
      plt.show()

      print("Model slope a1 = ", self.model.coef_[0])
      print("Model intercept a0 = ", self.model.intercept_)

   def func(self):
      if self.funci == 0 : return self.a0 + self.a1 * self.x + np.random.randn(self.sample )
      else : return  self.a1 + self.a0 * self.x + np.random.normal(0, 25, self.sample)

if __name__ == '__main__':

   LR(a1=1.8, a0=0.2, sample=100, start=0, stop=10, num=10000, funci = 0)
   LR(a1=5, a0=2.7, sample=500, start=0, stop=35, num=10000, funci = 1)
