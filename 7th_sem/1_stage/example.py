import numpy as np
from Find_optim import Find_optim

def g_1(x):

  x = np.asarray(x).reshape(2,1)
  const = np.array([1,1]).reshape(2,1)

  return np.float(np.dot(const.T, x))
  
  
def g_2(x):

  x = np.asarray(x).reshape(2,1)
  const = np.array([1,-1]).reshape(2,1)

  return np.float(np.dot(const.T, x))
  

def f(x):

  x = np.asarray(x).reshape(2,1)
  const = np.array([10, 9]).reshape(2,1)

  x = x + const
  return np.float(np.dot(x.T, x))
  
x_0 = np.array([[2],[-1]])

x_optim = Find_optim(f, x_0, [g_1, g_2], eps=1e-6, r=1, logging=True)
