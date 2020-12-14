import numpy as np
from get_grad import get_grad

def np_dot(a, b):
  return np.dot(a,b)

def norm(a):
  return np.linalg.norm(a)

def DFP_method(func, x, eps=1e-9, logging=False, iter_max=float("inf")):

  x = np.asarray(x).reshape(-1,1)
  lr = 0.001

  iter_num = 0
  
  f_grad_prev = get_grad(func, x)

  while norm(get_grad(func, x)) > eps and iter_num < iter_max:    

    if logging:
      print('%i-st step of DFP:' % (iter_num + 1))
    iter_num += 1

    if logging:
      print("f(x) = %.6f" % func(x))
      print("||f\'|| = %.6f" % norm(get_grad(func, x)))

    Q = np.diag([1.]*x.shape[0])

    x_prev = x

    if np_dot(f_grad_prev.T, get_grad(func, x_prev)) > 0: # Если направление движения не меняется увеличиваем шаг, иначе уменьшаем.
      #if i < 10000000:
      lr *= 1.2
    else:
      lr *= 0.6

    f_grad_prev = np.clip(get_grad(func, x_prev), -1, 1)

    d = -np_dot(Q, f_grad_prev)
    #lr = one_dim_min(func, x_prev, f_grad_prev) # Ищем длину шага.
    if logging:
      print('lr = %.5f' % lr)
    x_next = x_prev + lr * d  
    f_grad_next = get_grad(func, x_next)

    if logging:
      print('<f\'next, f\'prev> = %.9f' % np_dot(f_grad_next.T, f_grad_prev)) # Проверяем оротогональны ли напр-ия.

    r = x_next - x_prev
    s = f_grad_next - f_grad_prev

    Q_s = np_dot(Q, s)
    Q += np_dot(r, r.T) / np_dot(r.T, s)
    Q -= np_dot(Q_s, Q_s.T) / np_dot(Q_s, s.T)

    x = x_next    

    if logging:
      print('step has been taken', end='\n\n')
  
  if logging:
    print('DFP method converged.', end='\n\n')

  return x  
