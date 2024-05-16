import numpy as np
import matplotlib.pyplot as plt
import copy

class ReadData:
   def __init__(self, path = './DATA/Cricket.npz', name = 'arr_0'):
      self.path = path
      self.data = self.data = np.load(self.path)
      self.name = name
      self.x = self.data[self.name][:,1]
      self.y = self.data[self.name][:,0]
   def load(self):
      return self.data

   def plotdata(self):
      plt.scatter(self.x, self.y)
      plt.ylabel('Chirps/Second')
      plt.xlabel('Temperature (º F)')
      plt.show()

   def getForInfForGD(self):
      return self.x.reshape(-1,1), self.y.reshape(-1,1)

# part 2
class VectorizationOfTheCostFunc:
   def __init__(self, Theta_vec , X_mat, Y_vec ):
      self.Theta_vec = Theta_vec
      self.X_mat = X_mat
      self.Y_vec = Y_vec
      self.m = Y_vec.shape[0]
      self.z = np.dot(self.X_mat, self.Theta_vec) - self.Y_vec

   def J(self):
      return (1/2*self.m) *  np.dot(self.z.T, self.z)

   def setTheta(self, otherTheta):
      self.Theta_vec = otherTheta
      self.z = np.dot(self.X_mat, self.Theta_vec)

class RegLinePlotter:
   def __init__(self, X, y, theta):
      self.theta = theta
      self.X = X
      self.y = y
      self.ind = 1 if self.X.shape[1] == 2 else 0
      self.x_min = self.X[:, self.ind].min()
      self.x_max = self.X[:, self.ind].max()
      self.Xlh = np.array([[self.x_min], [self.x_max]])  # Define Xlh based on x_min and x_max
      self.yprd_lh = np.dot(np.hstack((np.ones((2, 1)), self.Xlh)), self.theta)  # Calculate yprd_lh

   def plotdata(self):
      plt.plot(self.X[:, self.ind], self.y, 'go')
      plt.plot(self.Xlh, self.yprd_lh, 'r-')  # Plot the line separately
      plt.axis((self.x_min-5, self.x_max+5, min(self.y)-5, max(self.y)+5))  # Adjust axis limits
      plt.xlabel('Temperature (º F)')
      plt.ylabel('Chirps/Second')
      plt.title('Regression data')
      plt.grid()
      plt.show()

   def find_y_value(self, x_value):
      y_value = self.theta[0] + self.theta[1] * x_value
      return y_value

def h(theta_0, theta_i, x_i):
   return theta_0 + theta_i * x_i




# part3
class GradientDescent:
   def __init__(self, rd: ReadData, **kwargs):
      self.x, self.y = rd.getForInfForGD()
      self.theta = np.array([ [ kwargs['theta0'] ], [ kwargs['theta1'] ] ], dtype='float64')
      self.iteration = kwargs['iteration']
      self.alpha = kwargs['alpha']
      self.m = self.y.shape[0]
      self.X = np.ones((self.x.shape[0], 2), dtype='float64')
      self.X[:, 1] = copy.deepcopy(self.x.flatten())
      self.VOTCF = VectorizationOfTheCostFunc(self.theta,self.X, self.y)
      self.J_theta = np.zeros((self.iteration))


   def BATCH(self):
      result = {}
      for alpha_ in self.alpha:
         th = copy.deepcopy(self.theta)  # Make a deep copy of self.theta
         for j in range(0, self.iteration):
            th[0, 0] = th[0, 0] - alpha_ * (1 / self.m) * self.de_j(th,0)
            th[1, 0] = th[1, 0] - alpha_ * (1 / self.m) * self.de_j(th,1)
            self.VOTCF.setTheta(th)
            self.J_theta[j] =  self.VOTCF.J()
         self.plotThetaAsFuncOfIter(alpha_)
         yield  {'alpha': alpha_, 'theta': [th[0, 0], th[1, 0]],'text': f'with alpha:{alpha_} results: Theta0={th[0, 0]}\tTheta1={th[1, 0]}\n'}

   def plotThetaAsFuncOfIter(self,alpha):
      plt.plot(np.arange(1, self.iteration + 1), self.J_theta)
      plt.xlabel('Iteration')
      plt.ylabel('J(Theta)')
      plt.title(f'J(theta) as a function of #iteration for alpha{alpha}')
      plt.show()

   def de_j(self,th,j):
      res = 0
      if j == 0:
         for x_i, y_i in zip(self.x, self.y):
            res += (h(th[0, 0], th[1, 0], x_i) - y_i) * 1
      else:
         for x_i, y_i in zip(self.x, self.y):
            res += (h(th[0, 0], th[1, 0], x_i) - y_i) * x_i
      return res




if __name__ == "__main__":

   # part 1
   readdata = ReadData()
   readdata.plotdata()

   numOfalphas = 3
   randomalphas = np.random.normal(0, 0.0001, (numOfalphas,))

   GD = GradientDescent(readdata,theta0 = 2, theta1=0.5, iteration = 19, alpha = [0.0001] )
   # part 4
   rgs = [RegLinePlotter(GD.VOTCF.X_mat,GD.VOTCF.Y_vec, np.array(x['theta']).reshape(-1,1)) for x in GD.BATCH()]

   for rg in rgs :
      rg.plotdata()

      # List of x values
      xis = [87, 58, 38]

      # Print x and y values
      for x_value in xis:
         y_value = rg.find_y_value(x_value)
         print(f"x_value = {x_value}, y_value = {y_value}")