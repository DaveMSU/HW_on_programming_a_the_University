import numpy as np
import pandas as pd
from Find_optim import Find_optim

def J(x):
  res = 0
  x = np.asarray(x)
  for i in range(1,x.shape[0]):
    res += (x[i] - i*x[0])**4
  res = res**6 + (x[0] - 2)**4

  return np.float(res)


def g1(x):
  x = np.asarray(x)
  return np.float(-((x**2).sum()**6 - 363)) # >= 0


x_0 = np.array([[2.],[1.6],[3.5]])
x_optim, logging_df = Find_optim(J, x_0, [g1], eps=1e-9, p=2, r=0.01, beta=1, logging=False, iter_max=10, DFP_iter_max=100)
logging_df.to_excel("logging.xlsx")
