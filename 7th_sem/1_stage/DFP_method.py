from get_grad import get_grad

def DFP_method(func, x, eps=1e-6, logging=False):

  # Help function for leraning rate search.
  #
  def one_dim_min(f, x_cur, d_cur):
    """
    One-dimensional minimization rule.

    Ищем lr с помощью градиентного спуска 
     с шагом lrlr, который уменьшается со временем.
    """

    lr = 0.5
    lrlr = 0.1
    f_from_lr = lambda lr: f(x_cur - lr * d_cur)
    fx_prev = np_dot(get_grad(f_from_lr, lr).T, lr) # <f',x>

    for _ in range(1000):          
      th = 1
      lr_d_cur = np.clip(get_grad(f_from_lr, lr), -th, th)[0,0]
      lr = lr - lrlr * lr_d_cur
      fx_next = np_dot(get_grad(f_from_lr, lr).T, lr)
      if fx_next * fx_prev > 0: # Если направление движения не меняется увеличиваем шаг, иначе уменьшаем.
        lrlr *= 1.2
      else:
        lrlr *= 0.6
      fx_prev = fx_next
      if norm(lr_d_cur) < eps:
        break

    return lr


  x = np.asarray(x).reshape(-1,1)

  if logging:
    i = 1

  while norm(get_grad(func, x)) > eps:    

    if logging:
      print('%i-st step of DFP:' % i)
    i+=1

    if logging:
      print("f(x) = %.6f" % func(x))
      #print("f\' = %.6f" % np_dot(get_grad(func, x).T, x))
      print("||f\'|| = %.6f" % norm(get_grad(func, x)))

    Q = np.diag([1.]*x.shape[0])

    x_prev = x
    f_grad_prev = get_grad(func, x_prev)

    d = -np_dot(Q, f_grad_prev)
    lr = one_dim_min(func, x_prev, f_grad_prev) # Ищем длину шага.
    if logging:
      print('lr = %.4f' % lr)
    x_next = x_prev + lr * d  
    f_grad_next = get_grad(func, x_next)

    if logging:
      print('<f\'next, f\'prev> = %.3f' % np_dot(f_grad_next.T, f_grad_prev)) # Проверяем оротогональны ли напр-ия.

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
