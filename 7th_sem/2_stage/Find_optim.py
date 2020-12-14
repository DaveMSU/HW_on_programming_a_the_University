import pandas as pd 
from DFP_method import DFP_method

def Find_optim(func, x, g_list = [], h_list = [], p=2, r=1, beta=2, eps=1e-6, logging=False, iter_max=float("inf"), DFP_iter_max=float("inf")):
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
  iter_max - максимальное кол-во итераций.
  DFP_iter_max - максимальное кол-во итераций для метода DFP.

  return точку x_optim, в кот-ой достигается условный оптимум func и pd.DataFrame с логами.
  """

  x_shape = len(x.reshape(-1))

  column_names = {'Z_value': [], 'F_value': [], 'alpha_value': [], 'r': []}
  column_names.update({'x['+str(i)+']': [] for i in range(x_shape)})

  logging_df = pd.DataFrame(column_names)

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
  def penalty_logging(x, r):
    print('current iter №%i' % (iter_num+1))
    print('curr Z_val: %.5f, from' % Z_func(x))
    print('curr optim point: ', end='')
    for x_coor in x: print('%.3f  ' % x_coor, end='')
    print('\nr =', r)
    print('alpha(x) = %.10f' % alpha_func(x))
    print('-'*30)
    #time.sleep(3)

  iter_num = 0
  if logging:
    print('Search has started!')
    print('-'*30)

  x = DFP_method(Z_func, x, eps, logging=logging, iter_max=DFP_iter_max)

  if logging:
    penalty_logging(x, r)

  new_line = {'Z_value': Z_func(x), 'F_value': func(x), 'alpha_value': alpha_func(x), 'r': r}
  new_line.update({'x['+str(i)+']': x.reshape(-1)[i] for i in range(x_shape)})
  logging_df = logging_df.append(pd.DataFrame(new_line, index=[iter_num]))
  

  while r*alpha_func(x) > eps and iter_num < iter_max:

    iter_num += 1
    r = (r**2 + beta)**0.5
    x = DFP_method(Z_func, x, eps, logging=logging, iter_max=DFP_iter_max)

    if logging:
      penalty_logging(x, r)

    new_line = {'Z_value': Z_func(x), 'F_value': func(x), 'alpha_value': alpha_func(x), 'r': r}
    new_line.update({'x['+str(i)+']': x.reshape(-1)[i] for i in range(x_shape)})
    logging_df = logging_df.append(pd.DataFrame(new_line, index=[iter_num]))

  if logging:
    print('Search has ended!')
  
  return x, logging_df
