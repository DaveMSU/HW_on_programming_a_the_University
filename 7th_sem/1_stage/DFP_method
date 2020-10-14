def DFP_method(func, x, eps=1e-6):

  lr=0.2
  x = np.asarray(x).reshape(-1,1)
  while np.abs(np.dot(get_grad(func, x).T, x)) > eps:

    print("f\' = %.6f" % np.dot(get_grad(func, x).T, x))

    Q = np.diag([1.]*x.shape[0])

    x_prev = x
    f_grad_prev = get_grad(func, x_prev)

    d = - np.dot(Q, f_grad_prev)
    x_next = x_prev + lr * d  
    f_grad_next = get_grad(func, x_next)

    r = x_next - x_prev
    s = f_grad_next - f_grad_prev

    Q_s = np.dot(Q, s)
    Q += np.dot(r, r.T) / np.dot(r.T, s)
    Q -= np.dot(Q_s, Q_s.T) / np.dot(Q_s, s.T)

    x = x_next
    lr *= 0.99

  return x  
