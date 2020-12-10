import numpy as np

def get_grad(functor, x_0):
  """
  Принимаем на вход функтор и
  координаты точки x_0,
  возвращаем градиент 
  (вектор столбец)
  в этой точке.
  """
  
  eps = 1e-9
  x_0 = np.asarray(x_0, dtype=float).reshape(-1)
  f_grad = np.zeros_like(x_0)

  for i in range(x_0.shape[0]):
    x_plus = x_0.copy()
    x_plus[i] += eps

    x_minus = x_0.copy()
    x_minus[i] -= eps

    f_grad[i] = functor(x_plus) - functor(x_minus)
    f_grad[i] /= 2*eps

  return f_grad.reshape(-1,1)
