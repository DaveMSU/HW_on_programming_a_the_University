def Z_func(x, f, G, r):

  x = np.asarray(x).reshape(-1,1)
  p = 10
  
  rest = 0
  for i in range(x.shape[0]):
    rest += np.max(0, -G[i](x))**p

  return f(x) + r * rest
