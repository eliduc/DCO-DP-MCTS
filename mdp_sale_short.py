import numpy as np
import matplotlib.pyplot as plt

def mdp_sale_with_action(N, T, gamma, state, action_idx, fi, u_values, p_values,
                        constraints, k_fine, D_discr):
    """
    Один шаг MDP:
      state = (i, inv_i, D_i)  -- i: номер шага (0..T-1),
                                  inv_i: остаток квартир,
                                  D_i: накопленная выручка (без штрафа).
      action_idx -- индекс действия (u_j, p_j).
      
    Возвращает:
      next_state = (i+1, next_inv, D_(i+1 без штрафа)),  -- i+1, уменьшенный остаток, и накопленную без штрафа
      reward = immediate_revenue - fine,
      term = True, если i+1 >= T,
      D_next_no_fine = накопленная выручка без штрафа (неокруглённая),
      fine = штраф за этот шаг,
      u_j, p_j = фактическое число квартир и цена на этом шаге.
    """
    i, inv_i, D_i = state
    
    # Проверки корректности
    if u_values is None or p_values is None or len(u_values) != len(p_values):
        raise ValueError("Списки u_values и p_values должны быть непустыми и одинаковой длины")

    if action_idx < 0 or action_idx >= len(u_values):
        raise ValueError(f"action_idx должен быть в пределах [0, {len(u_values)-1}]")
    
    # Выбираем действие
    u_j = u_values[action_idx]
    p_j = p_values[action_idx]
    
    # Ограничиваем продажи имеющимися квартирами
    if u_j > inv_i:
        u_j = inv_i
    
    # Дисконтированная выручка на этом шаге (без штрафов)
    immediate_revenue = u_j * p_j * (gamma ** i)
    
    # Накопленная выручка без штрафов
    D_next_no_fine = D_i + immediate_revenue
    
    # Считаем штраф (если ограничение на момент i+1 не выполняется)
    fine = 0.0
    for t_req, D_req in constraints:
        if i + 1 == t_req and D_next_no_fine < D_req:
            fine = k_fine * (D_req - D_next_no_fine)
            break
    
    # Если включена дискретизация, округляем именно "без штрафа" (как было оговорено)
    if D_discr > 0:
        next_d_idx = int(round(D_next_no_fine / D_discr))
        next_d_val = next_d_idx * D_discr
    else:
        next_d_val = D_next_no_fine
    
    # Вознаграждение за шаг = immediate_revenue - fine
    reward = immediate_revenue - fine
    
    # Остаток квартир
    next_inv = max(0, inv_i - u_j)
    
    # Проверяем завершение
    term = (i + 1 >= T)
    
    # Новое состояние
    next_state = (i + 1, next_inv, next_d_val)
    
    return next_state, reward, term, D_next_no_fine, fine, u_j, p_j


def permissible_u(f_i, p_min, p_max, dt, T, a, b):
    """
    Генерирует списки (u, p) для каждого i = 0..T-1, используя модель:
      v = (a - p)/b, если p<a, иначе 0;
    затем умножаем на f_i(i) и округляем.
    """
    u_lists = []
    p_lists = []
    
    for i in range(T):
        u_dict = {}
        prices = np.arange(p_min, p_max + dt/2, dt)
        
        for p in prices:
            v = (a - p)/b if p < a else 0
            # Учитываем функцию f_i(i)
            demand = v * f_i(i)
            u_val = round(demand)
            
            # Если для u_val нашлись разные p, берём, например, максимум
            if u_val in u_dict:
                u_dict[u_val] = max(u_dict[u_val], p)
            else:
                u_dict[u_val] = p
        
        # Сортируем ключи
        u_sorted = sorted(u_dict.keys())
        p_sorted = [u_dict[u] for u in u_sorted]
        
        u_lists.append(u_sorted)
        p_lists.append(p_sorted)
    
    return u_lists, p_lists


def mdp_sale_strat_cl(N, T, gamma, constraints, fi, p_min, p_max, dp, d_D, u_strategy, k_fine=10):
    """
    Режим 2: расчёт полной траектории для заранее заданной стратегии (u_i).
    Возвращает:
      p_sequence, v_sequence, U_sequence,
      revenue_increments: прирост выручки без штрафа за каждый шаг,
      fines: штрафы за каждый шаг,
      D_constraints: выручка (без штрафа) на моментах, где есть ограничения.
    """
    
    # Параметры для линейной модели
    Bp = 3.0 * N / (100.0 * T)
    Ap = 4.0 * N / T
    a = Ap / Bp
    b = 1.0 / Bp
    
    # Генерируем списки допустимых (u, p)
    u_lists, p_lists = permissible_u(fi, p_min, p_max, dp, T, a, b)
    
    p_sequence = []
    v_sequence = []
    U_sequence = []
    revenue_increments = []
    fines = []
    D_constraints = {}
    
    # Начальное состояние
    state = (0, N, 0.0)
    
    # Корректируем длину стратегии
    if len(u_strategy)<T:
        print(f"Стратегия короче {T}, дополним нулями.")
        u_strategy += [0]*(T - len(u_strategy))
    elif len(u_strategy)>T:
        print(f"Стратегия длинее {T}, обрежем.")
        u_strategy = u_strategy[:T]
    
    for step_i in range(T):
        i, inv_i, D_i = state
        
        # Если квартиры закончились
        if inv_i<=0:
            print(f"На шаге {step_i} закончился инвентарь. Остальные продажи=0.")
            for jj in range(step_i, T):
                p_sequence.append(0)
                v_sequence.append(0)
                U_sequence.append(0)
                revenue_increments.append(0)
                fines.append(0)
            break
        
        desired_u = u_strategy[step_i]
        
        # Допустимые действия
        if step_i<len(u_lists):
            u_vals = u_lists[step_i]
            p_vals = p_lists[step_i]
        else:
            print(f"Не хватает (u,p) для шага {step_i}.")
            break
        
        # Ищем ближайшее допустимое
        if desired_u not in u_vals:
            closest_u = min(u_vals, key=lambda x: abs(x - desired_u))
            desired_u = closest_u
        
        desired_u = min(desired_u, inv_i)
        action_idx = u_vals.index(desired_u)
        
        # Выполняем шаг MDP
        next_state, reward, term, D_next_no_fine, fine, actual_u, actual_p = mdp_sale_with_action(
            N, T, gamma, state, action_idx, fi, u_vals, p_vals, constraints, k_fine, d_D
        )
        
        # Прирост выручки без штрафа
        increment = D_next_no_fine - D_i
        
        # Спрос (для информации)
        v_i = (a - actual_p)/b if actual_p<a else 0
        
        p_sequence.append(actual_p)
        v_sequence.append(v_i)
        U_sequence.append(actual_u)
        revenue_increments.append(increment)
        fines.append(fine)
        
        # Если (i+1) == t_req
        for (t_req, D_req) in constraints:
            if i+1 == t_req:
                D_constraints[t_req] = D_next_no_fine
        
        state = next_state
        if term:
            break
    
    return p_sequence, v_sequence, U_sequence, revenue_increments, fines, D_constraints

