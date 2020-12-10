from DFP_method import DFP_method

def Find_optim(func, x, g_list = [], h_list = [], p=2, r=1, beta=2, eps=1e-6, logging=False):
  """
  x  - начальная точка.
  func - функция, минимум которой мы ищем.
  g_list - список функций-ограничений типа нер-во (>= 0).
  h_list - список функций-ограничений типа равенство.
  p - показатель степени (см. Z_func).
  r - начальное значение параметра штрафа.
  beta - коэф-нт в формуле вычисления параметра штрафа.
  eps  - условия останова, ограничение на ||f'||.
  logging - выводить логи работы или нет.
  return точку x_optim, в кот-ой достигается оптимум func.
  """


  # 2-nd part in Z_func, func - 1st.
  #
  def alpha_func(x, p=p): 

    alpha_func_val = 0    
    for g_func in g_list:
      alpha_func_val += max(0, - g_func(x)) ** p  

    for h_func in h_list:
      alpha_func_val += abs(h_list(x)) ** p
    
    return alpha_func_val

  
  # Main function for finding the optimum.
  #
  def Z_func(x):
    return func(x) + r * alpha_func(x)

  # Function for logging.
  #
  def barriers_logging(x, r):
    print('curr Z_val: %.5f, from' % Z_func(x))
    print('curr optim point: ', end='')
    for x_coor in x: print('%.3f  ' % x_coor, end='')
    print('\nr = %i' % r)
    print('alpha(x) = %.6f' % alpha_func(x))
    print('-'*30)

  if logging:
    print('Search has started!')
    print('-'*30)

  x = DFP_method(Z_func, x, logging=logging)

  if logging:
    barriers_logging(x, r)

  while r*alpha_func(x) > eps:

    r *= beta
    x = DFP_method(Z_func, x, eps, logging=logging)

    if logging:
      barriers_logging(x, r)
    
  if logging:
    print('Search has ended!')
  
  return x

