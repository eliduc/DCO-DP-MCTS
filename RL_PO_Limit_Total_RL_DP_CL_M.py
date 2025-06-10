import numpy as np
import matplotlib.pyplot as plt
import math
import numba as nb
#from mdp_sale import mdp_sale, mdp_sale_with_action, permissible_u


#@nb.njit
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



@nb.njit
def compute_next_state(t, inv, d, u, p, gamma, d_discr, num_d_values, constraints, k_fine):
    immediate_revenue = u * p * (gamma ** t)
    next_d = d + immediate_revenue
    fine = 0.0

    for i in range(len(constraints)):
        if constraints[i, 0] == t + 1 and next_d < constraints[i, 1]:
            fine = k_fine * (constraints[i, 1] - next_d)
            break

    reward = immediate_revenue - fine
    next_d_with_fine = next_d - fine # Revenue after fine for state transition
    next_d_idx = min(int(round(next_d_with_fine / d_discr)), num_d_values - 1)
    next_d_val = next_d_idx * d_discr # This value isn't strictly needed by dp_core, but good for

    # --- ADD THIS LINE ---
    next_inv = max(0, inv - u) # Calculate the next inventory level
    # --- END ADDITION ---

    return next_inv, next_d_idx, next_d_val, reward

@nb.njit
def dp_core(T, N_int, gamma, constraints, u_arrays, p_arrays, action_counts, d_discr, num_d_values, k_fine):
    # Создаем массивы для значений и политики
    V_array = np.zeros((T+1, N_int+1, num_d_values))
    policy_array = np.full((T+1, N_int+1, num_d_values), -1, dtype=np.int32)

    # Заполняем снизу вверх
    for t in range(T-1, -1, -1):
        for inv in range(N_int+1):
            for d_idx in range(num_d_values):
                d = d_idx * d_discr
 
                best_value = -np.inf
                best_action = -1

                for j in range(action_counts[t]):
                    u = u_arrays[t, j]
                    if u > inv:
                        continue

                    p = p_arrays[t, j]

                    # --- MODIFICATION START ---
                    # Original line causing error:
                    # next_inv, next_d_idx, _, reward = compute_next_state(
                    #     t, inv, d, u, p, gamma, d_discr, num_d_values, constraints, k_fine)

                    # Modified code: Assign to a tuple first, then unpack
                    result_tuple = compute_next_state(
                        t, inv, d, u, p, gamma, d_discr, num_d_values, constraints, k_fine)
                    next_inv = result_tuple[0]
                    next_d_idx = result_tuple[1]
                    # We don't need result_tuple[2] (next_d_val) here
                    reward = result_tuple[3]
                    # --- MODIFICATION END ---

                    next_value = V_array[t+1, next_inv, next_d_idx]
                    total_value = reward + next_value

                    if total_value > best_value:
                        best_value = total_value
                        best_action = j

                if best_value == -np.inf:
                    # If no action was possible (e.g., inv=0 but actions require u>0)
                    # or all actions led to -inf (shouldn't happen with V=0 terminal)
                    # set value to 0 (or potentially handle differently if needed)
                    best_value = 0

                V_array[t, inv, d_idx] = best_value
                policy_array[t, inv, d_idx] = best_action

    return V_array, policy_array

def solve_mdp(N, T, gamma, constraints, u_lists, p_lists, fi, k_fine, D_discr):
    N_int = int(N)
    
    # Преобразование структур данных для Numba
    constraints_array = np.array([(t, D) for t, D in constraints])
    
    # Преобразование списков u_lists и p_lists для Numba
    max_actions = max(len(u) for u in u_lists)
    u_arrays = np.zeros((T, max_actions), dtype=np.int32)
    p_arrays = np.zeros((T, max_actions))
    action_counts = np.zeros(T, dtype=np.int32)
    
    for t in range(T):
        count = len(u_lists[t])
        action_counts[t] = count
        u_arrays[t, :count] = u_lists[t]
        p_arrays[t, :count] = p_lists[t]
    
    # Подготовка параметров для динамического программирования
    max_price = max(max(p_list) if p_list else 0 for p_list in p_lists)
    max_revenue = N * max_price * 1.5
    num_d_values = int(max_revenue / D_discr) + 1
    print(f"Using {num_d_values} discrete revenue values from 0 to {max_revenue:.2f}")
    
    print("Starting Numba-accelerated dynamic programming...")
    V_array, policy_array = dp_core(T, N_int, gamma, constraints_array, u_arrays, p_arrays, 
                                  action_counts, D_discr, num_d_values, k_fine)
    print("Dynamic programming completed!")
    
    # Преобразование массивов обратно в словари
    V = {}
    policy = {}
    
    for t in range(T+1):
        print(f"Processing time step {t} results...")
        for inv in range(N_int+1):
            for d_idx in range(num_d_values):
                d = d_idx * D_discr
                V[(t, inv, d)] = V_array[t, inv, d_idx]
                policy[(t, inv, d)] = int(policy_array[t, inv, d_idx])
    
    return policy, V, num_d_values

def extract_dp_trajectory_with_mdp_sale(policy, value_function, N, T, gamma, u_lists, p_lists, fi, D_discr, constraints, k_fine, num_d_values):
    """
    Извлекает оптимальную траекторию из политики DP, используя функцию mdp_sale.
    """
    v_optimal = np.zeros(T)
    u_optimal = np.zeros(T)
    p_optimal = np.zeros(T)

    V_cumulative = np.zeros(T + 1)
    U_cumulative = np.zeros(T + 1)
    D_cumulative = np.zeros(T + 1)
    D_cumulative_no_fine = np.zeros(T + 1)

    constraint_dict = {t: D for t, D in constraints}

    constraint_violations = {t: 0 for t, _ in constraints}

    t = 0
    inv = N
    # Use a continuous tracker for revenue state, similar to V1
    current_cumulative_revenue_net_fines = 0.0
    total_fines = 0

    while t < T:
        # Discretize the continuous tracker for policy lookup
        d_idx = int(round(current_cumulative_revenue_net_fines / D_discr))
        d_disc = d_idx * D_discr # Discrete revenue state for lookup

        # Lookup action
        action_idx = policy.get((t, int(inv), d_disc), -1)

        # Get u, p directly and calculate transitions like in V1's extract_dp_trajectory
        if action_idx < 0 or action_idx >= len(u_lists[t]):
            u = 0
            p = p_lists[t][0] if p_lists[t] else 0
        else:
            u = u_lists[t][action_idx]
            p = p_lists[t][action_idx]

        v = u / fi(t) if fi(t) > 0 else 0

        # --- Calculations like V1's extract_dp_trajectory ---
        v_optimal[t] = v
        u_optimal[t] = u
        p_optimal[t] = p

        revenue = p * u * (gamma ** t)
        D_cumulative_no_fine[t+1] = D_cumulative_no_fine[t] + revenue

        fine_this_step = 0
        if (t+1) in constraint_dict:
            if D_cumulative_no_fine[t+1] < constraint_dict[t+1]:
                shortfall = constraint_dict[t+1] - D_cumulative_no_fine[t+1]
                fine_this_step = k_fine * shortfall
                constraint_violations[t+1] = -shortfall
                total_fines += fine_this_step

        V_cumulative[t+1] = V_cumulative[t] + v
        U_cumulative[t+1] = U_cumulative[t] + u
        D_cumulative[t+1] = D_cumulative_no_fine[t+1] - total_fines # Tracks continuous cumulative revenue net of ALL fines

        # --- Update state for NEXT iteration ---
        t += 1
        inv = max(0, inv - u)
        # Update the continuous revenue tracker for the next lookup
        current_cumulative_revenue_net_fines = D_cumulative[t] # Use the value from D_cumulative array

    # --- ADD THIS SECTION BACK ---
    constraint_diffs = {}
    for t_constr, D_target in constraints:
        # Use the D_cumulative_no_fine array to check against target
        if t_constr < len(D_cumulative_no_fine):
             constraint_diffs[t_constr] = D_cumulative_no_fine[t_constr] - D_target
        else:
             # Handle case where constraint time might be T itself
             constraint_diffs[t_constr] = D_cumulative_no_fine[-1] - D_target
    # --- END ADDITION ---

    return v_optimal, u_optimal, p_optimal, V_cumulative, U_cumulative, D_cumulative, D_cumulative_no_fine, constraint_diffs, total_fines

def optimal_sales_strategy(N, T, gamma, t_0, V_0, fi, a, b):
    time_steps = np.arange(t_0, T)
    
    S1 = sum(fi(i) for i in time_steps)
    S2 = sum(fi(i) / (gamma ** i) for i in time_steps)
    
    lambda_opt = (a * S1 - 2 * b * (N - V_0)) / S2
    
    v_optimal = np.array([
        (a - lambda_opt / (gamma ** i)) / (2 * b)
        for i in time_steps
    ])
    
    u_optimal = np.array([
        v_optimal[i - t_0] * fi(i)
        for i in time_steps
    ])
    
    p_optimal = np.array([
        a - b * v_optimal[i - t_0]
        for i in time_steps
    ])
    
    V_cumulative = np.zeros(T - t_0 + 1)
    V_cumulative[0] = V_0
    for i in range(1, T - t_0 + 1):
        V_cumulative[i] = V_cumulative[i - 1] + u_optimal[i - 1]
    
    revenue_steps = np.array([
        (gamma ** i) * p_optimal[i - t_0] * u_optimal[i - t_0]
        for i in time_steps
    ])
    
    D_cumulative = np.zeros(T - t_0 + 1)
    for i in range(1, T - t_0 + 1):
        D_cumulative[i] = D_cumulative[i - 1] + revenue_steps[i - 1]
    
    return lambda_opt, v_optimal, u_optimal, p_optimal, V_cumulative, D_cumulative

def optimal_sales_strategy_demand(N, t_0, t_1, gamma, D_0, D_1, fi, a, b):
    time_steps = np.arange(t_0, t_1)
    
    S1 = sum(gamma**i * fi(i) for i in time_steps)
    S2 = sum(fi(i) / (gamma**i) for i in time_steps)
    
    denominator = 4*b*D_0 + a**2*S1 - 4*b*D_1
    
    if denominator <= 0:
        raise ValueError("No valid solution exists. The revenue target is too high for the given time frame.")
    
    lambda_opt = math.sqrt(denominator) / math.sqrt(S2)
    
    v_optimal = np.array([
        a/(2*b) - lambda_opt/(2*b*gamma**i)
        for i in time_steps
    ])
    
    u_optimal = np.array([
        v_optimal[i - t_0] * fi(i)
        for i in time_steps
    ])
    
    p_optimal = np.array([
        a - b * v_optimal[i - t_0]
        for i in time_steps
    ])
    
    V_cumulative = np.zeros(t_1 - t_0 + 1)
    for i in range(1, t_1 - t_0 + 1):
        V_cumulative[i] = V_cumulative[i - 1] + v_optimal[i - 1]
    
    U_cumulative = np.zeros(t_1 - t_0 + 1)
    for i in range(1, t_1 - t_0 + 1):
        U_cumulative[i] = U_cumulative[i - 1] + u_optimal[i - 1]
    
    revenue_steps = np.array([
        (gamma ** i) * p_optimal[i - t_0] * u_optimal[i - t_0]
        for i in time_steps
    ])
    
    D_cumulative = np.zeros(t_1 - t_0 + 1)
    D_cumulative[0] = D_0
    for i in range(1, t_1 - t_0 + 1):
        D_cumulative[i] = D_cumulative[i - 1] + revenue_steps[i - 1]
    
    return lambda_opt, v_optimal, u_optimal, p_optimal, V_cumulative, U_cumulative, D_cumulative

def optimal_sales_total(T, N, gamma, fi, constraints, a, b):
    v_optimal = np.zeros(T)
    u_optimal = np.zeros(T)
    p_optimal = np.zeros(T)
    V_cumulative = np.zeros(T + 1)
    U_cumulative = np.zeros(T + 1)
    D_cumulative = np.zeros(T + 1)
    
    constraints = sorted(constraints, key=lambda x: x[0])
    
    t = 0
    while t < T:
        print(f"\n--- Iteration starting at time t={t} ---")
        
        right_constraints = [(tj, Dj) for tj, Dj in constraints if tj > t]
        
        lambda_j_values = []
        for tj, Dj in right_constraints:
            try:
                lambda_j, _, _, _, _, _, _ = optimal_sales_strategy_demand(
                    N, t, tj, gamma, D_cumulative[t], Dj, fi, a, b
                )
                lambda_j_values.append((tj, lambda_j))
                print(f"For constraint (t^j={tj}, D^j={Dj:.2f}), lambda^j = {lambda_j:.6f}")
            except ValueError as e:
                print(f"Constraint at t^j={tj} is infeasible: {e}")
        
        try:
            lambda_d, _, _, _, _, _ = optimal_sales_strategy(
                N, T, gamma, t, U_cumulative[t], fi, a, b
            )
            print(f"Lambda^d for optimal revenue at T={T}: {lambda_d:.6f}")
        except ValueError as e:
            print(f"Optimal revenue calculation failed: {e}")
            return None
        
        if not lambda_j_values:
            min_lambda = lambda_d
            t_r = T
        else:
            lambda_values = [lj for _, lj in lambda_j_values] + [lambda_d]
            min_lambda = min(lambda_values)
            
            t_r = T
            for tj, lambda_j in lambda_j_values:
                if abs(lambda_j - min_lambda) < 1e-6:
                    t_r = tj
                    break
        
        lambda_r = min_lambda
        print(f"Selected t^r={t_r}, lambda^r={lambda_r:.6f}")
        
        if t_r < T and abs(lambda_r - lambda_d) > 1e-6:
            matching_constraint = None
            for tj, lambda_j in lambda_j_values:
                if tj == t_r and abs(lambda_j - lambda_r) < 1e-6:
                    matching_constraint = next((t, D) for t, D in right_constraints if t == tj)
                    break
            
            if matching_constraint:
                tj, Dj = matching_constraint
                _, v_opt, u_opt, p_opt, _, _, _ = optimal_sales_strategy_demand(
                    N, t, tj, gamma, D_cumulative[t], Dj, fi, a, b
                )
            else:
                print("Error: Matching constraint not found.")
                return None
        else:
            remaining_apartments = N - U_cumulative[t]
            _, v_opt, u_opt, p_opt, _, _ = optimal_sales_strategy(
                N, T, gamma, t, U_cumulative[t], fi, a, b
            )
            
            v_opt = v_opt[:(t_r-t)]
            u_opt = u_opt[:(t_r-t)]
            p_opt = p_opt[:(t_r-t)]
             
            if t_r == T:
                total_to_sell = sum(u_opt)
                if abs(total_to_sell - remaining_apartments) > 1e-3:
                    print(f"Warning: Expected to sell {remaining_apartments:.2f} more apartments, but will sell {total_to_sell:.2f}")
        
        for i in range(t, t_r):
            idx = i - t
            v_optimal[i] = v_opt[idx]
            u_optimal[i] = u_opt[idx]
            p_optimal[i] = p_opt[idx]
        
        for i in range(t+1, t_r+1):
            V_cumulative[i] = V_cumulative[i-1] + v_optimal[i-1]
            U_cumulative[i] = U_cumulative[i-1] + u_optimal[i-1]
            D_cumulative[i] = D_cumulative[i-1] + gamma**(i-1) * p_optimal[i-1] * u_optimal[i-1]
        
        print(f"Values at t^r={t_r}: V({t_r})={V_cumulative[t_r]:.2f}, U({t_r})={U_cumulative[t_r]:.2f}, D({t_r})={D_cumulative[t_r]:.2f}")
        
        if t_r == T:
            break
        
        t = t_r
    
    constraint_diffs = {}
    for t, D_target in constraints:
        constraint_diffs[t] = D_cumulative[t] - D_target
    
    return v_optimal, u_optimal, p_optimal, V_cumulative, U_cumulative, D_cumulative, D_cumulative, constraint_diffs, 0

def calculate_constraint_differences(D_cumulative, constraints):
    constraint_diffs = {}
    for t, D_target in constraints:
        if t < len(D_cumulative):
            constraint_diffs[t] = D_cumulative[t] - D_target
    return constraint_diffs

def main():
    try:
        N = float(input("Enter total number of apartments (N) [default=364]: ") or 364)
        T = int(input("Enter deadline for selling all apartments (T) [default=52]: ") or 52)
        gamma = float(input("Enter discount factor (gamma) [default=0.99]: ") or 0.99)
    except ValueError:
        print("Invalid input. Using default values.")
        N = 364
        T = 52
        gamma = 0.99
    
    T_diff = T
    Price_uniform = 100.0
    V_uniform = N / T_diff
    Bp = 3.0 * N / (Price_uniform * T_diff)
    Ap = 4.0 * N / T_diff
    a = Ap / Bp
    b = 1.0 / Bp
    
    print(f"Calculated parameters: a = {a:.4f}, b = {b:.4f}")
    
    def fi(t):
        return 1 + 0.2 * np.sin((t) * 2 * np.pi / T)
    
    constraints_t = []
    
    default_t = [T//4, T//2, 3*T//4]
    
    for i in range(3):
        try:
            t_val = int(input(f"Enter constraint time t^{i+1} [default={default_t[i]}]: ") or default_t[i])
            if t_val == 0:
                break
            constraints_t.append(t_val)
        except ValueError:
            print(f"Invalid input. Using default value: {default_t[i]}")
            constraints_t.append(default_t[i])
    
    while True:
        try:
            t_val = input(f"Enter constraint time t^{len(constraints_t)+1} (press Enter to stop): ")
            if not t_val:
                break
            t_val = int(t_val)
            if t_val == 0:
                break
            constraints_t.append(t_val)
        except ValueError:
            print("Invalid input. Please enter a valid integer.")
    
    constraints_t = sorted(constraints_t)
    
    lambda_uncon, v_optimal_uncon, u_optimal_uncon, p_optimal_uncon, V_cumulative_uncon, D_cumulative_uncon = optimal_sales_strategy(
        N, T, gamma, 0, 0, fi, a, b
    )
    
    print("\nRevenue at constraint times (unconstrained solution):")
    print("-" * 50)
    print(f"{'Time':^10} | {'Revenue':^15}")
    print("-" * 50)
    
    D_values_at_constraints = {}
    for t_val in constraints_t:
        D_val = D_cumulative_uncon[t_val]
        D_values_at_constraints[t_val] = D_val
        print(f"{t_val:^10} | {D_val:^15.2f}")
    
    constraints = []
    print("\nEnter target revenue values for each constraint time:")
    
    for i, t_val in enumerate(constraints_t):
        default_D = D_values_at_constraints[t_val]
        try:
            D_val = float(input(f"Enter target revenue D^{i+1} for t^{i+1}={t_val} [default={default_D:.2f}]: ") or default_D)
            constraints.append((t_val, D_val))
        except ValueError:
            print(f"Invalid input. Using default value: {default_D:.2f}")
            constraints.append((t_val, default_D))
    
    v_optimal, u_optimal, p_optimal, V_cumulative, U_cumulative, D_cumulative, D_cumulative_no_fine_ca, constraint_diffs_ca, total_fines_ca = optimal_sales_total(
        T, N, gamma, fi, constraints, a, b
    )
    
    if v_optimal is not None:
        time_steps = np.arange(0, T)
        time_steps_cumulative = np.arange(0, T + 1)
        
        u_optimal_discrete = np.round(u_optimal).astype(int)
        u_sum_before_last = np.sum(u_optimal_discrete[:-1])
        u_optimal_discrete[-1] = int(N - u_sum_before_last)
        
        v_optimal_discrete = np.zeros_like(v_optimal)
        for i in range(len(v_optimal)):
            v_optimal_discrete[i] = u_optimal_discrete[i] / fi(i) if fi(i) > 0 else 0
        
        p_optimal_discrete = np.array([
            a - b * v_optimal_discrete[i]
            for i in range(len(v_optimal_discrete))
        ])
        
        V_cumulative_discrete = np.zeros(T + 1)
        U_cumulative_discrete = np.zeros(T + 1)
        D_cumulative_discrete = np.zeros(T + 1)
        D_cumulative_discrete_no_fine = np.zeros(T + 1)
        
        for i in range(1, T + 1):
            V_cumulative_discrete[i] = V_cumulative_discrete[i-1] + v_optimal_discrete[i-1]
            U_cumulative_discrete[i] = U_cumulative_discrete[i-1] + u_optimal_discrete[i-1]
            revenue = gamma**(i-1) * p_optimal_discrete[i-1] * u_optimal_discrete[i-1]
            D_cumulative_discrete[i] = D_cumulative_discrete[i-1] + revenue
            D_cumulative_discrete_no_fine[i] = D_cumulative_discrete_no_fine[i-1] + revenue

        u_optimal_uncon_discrete = np.round(u_optimal_uncon).astype(int)
        u_sum_before_last_uncon = np.sum(u_optimal_uncon_discrete[:-1])
        u_optimal_uncon_discrete[-1] = int(N - u_sum_before_last_uncon)
        
        v_optimal_uncon_discrete = np.zeros_like(v_optimal_uncon)
        for i in range(len(v_optimal_uncon)):
            v_optimal_uncon_discrete[i] = u_optimal_uncon_discrete[i] / fi(i) if fi(i) > 0 else 0
        
        p_optimal_uncon_discrete = np.array([
            a - b * v_optimal_uncon_discrete[i]
            for i in range(len(v_optimal_uncon_discrete))
        ])
        
        V_cumulative_uncon_discrete = np.zeros(T + 1)
        U_cumulative_uncon_discrete = np.zeros(T + 1)
        D_cumulative_uncon_discrete = np.zeros(T + 1)
        
        for i in range(1, T + 1):
            V_cumulative_uncon_discrete[i] = V_cumulative_uncon_discrete[i-1] + v_optimal_uncon_discrete[i-1]
            U_cumulative_uncon_discrete[i] = U_cumulative_uncon_discrete[i-1] + u_optimal_uncon_discrete[i-1]
            D_cumulative_uncon_discrete[i] = D_cumulative_uncon_discrete[i-1] + gamma**(i-1) * p_optimal_uncon_discrete[i-1] * u_optimal_uncon_discrete[i-1]
        
        u_uniform = np.zeros(T)
        v_uniform = np.zeros(T)
        p_uniform = np.zeros(T)
        
        constraint_times = [0]
        constraint_revenues = [0]
        
        for t_j, D_j in sorted(constraints, key=lambda x: x[0]):
            if t_j > 0:
                constraint_times.append(t_j)
                constraint_revenues.append(D_j)
        
        if constraint_times[-1] < T:
            constraint_times.append(T)
        
        V_uniform_cumulative = np.zeros(T+1)
        U_uniform_cumulative = np.zeros(T+1)
        D_uniform_cumulative = np.zeros(T+1)
        D_uniform_cumulative_no_fine = np.zeros(T+1)
        
        for i in range(1, len(constraint_times)):
            t_start = constraint_times[i-1]
            t_end = constraint_times[i]
            
            interval_length = t_end - t_start
            
            if interval_length > 0:
                if t_end == T:
                    remaining_apartments = N - U_uniform_cumulative[t_start]
                    u_val = remaining_apartments / interval_length
                else:
                    target_revenue = constraint_revenues[i]
                    
                    def calculate_D_at_t_end(u_val):
                        V_temp = V_uniform_cumulative.copy()
                        U_temp = U_uniform_cumulative.copy()
                        D_temp = D_uniform_cumulative.copy()
                        
                        for t in range(t_start, t_end):
                            v_val = u_val / fi(t) if fi(t) > 0 else 0
                            p_val = a - b * v_val
                            
                            V_temp[t+1] = V_temp[t] + v_val
                            U_temp[t+1] = U_temp[t] + u_val
                            D_temp[t+1] = D_temp[t] + gamma**t * p_val * u_val
                        
                        return D_temp[t_end]
                    
                    u_low, u_high = 0, 20.0
                    
                    for _ in range(30):
                        u_mid = (u_low + u_high) / 2
                        D_at_t_end = calculate_D_at_t_end(u_mid)
                        
                        if abs(D_at_t_end - target_revenue) < 1e-3:
                            break
                            
                        if D_at_t_end < target_revenue:
                            u_low = u_mid
                        else:
                            u_high = u_mid
                    
                    u_val = (u_low + u_high) / 2
                
                for t in range(t_start, t_end):
                    u_uniform[t] = u_val
                    v_uniform[t] = u_uniform[t] / fi(t) if fi(t) > 0 else 0
                    p_uniform[t] = a - b * v_uniform[t]
                    
                    if t > 0:
                        V_uniform_cumulative[t] = V_uniform_cumulative[t-1] + v_uniform[t-1]
                        U_uniform_cumulative[t] = U_uniform_cumulative[t-1] + u_uniform[t-1]
                        revenue = gamma**(t-1) * p_uniform[t-1] * u_uniform[t-1]
                        D_uniform_cumulative[t] = D_uniform_cumulative[t-1] + revenue
                        D_uniform_cumulative_no_fine[t] = D_uniform_cumulative_no_fine[t-1] + revenue
                
                if t_end > 0:
                    V_uniform_cumulative[t_end] = V_uniform_cumulative[t_end-1] + v_uniform[t_end-1]
                    U_uniform_cumulative[t_end] = U_uniform_cumulative[t_end-1] + u_uniform[t_end-1]
                    revenue = gamma**(t_end-1) * p_uniform[t_end-1] * u_uniform[t_end-1]
                    D_uniform_cumulative[t_end] = D_uniform_cumulative[t_end-1] + revenue
                    D_uniform_cumulative_no_fine[t_end] = D_uniform_cumulative_no_fine[t_end-1] + revenue
        
        plt.figure(figsize=(18, 12))
        plt.suptitle('Comparison of All Algorithms', fontsize=16)

        # Анализ данных для настройки осей Y
        v_min = min(np.min(v_optimal), np.min(v_optimal_uncon), np.min(v_optimal_discrete), 
                   np.min(v_optimal_uncon_discrete), np.min(v_uniform)) - 1
        v_max = max(np.max(v_optimal), np.max(v_optimal_uncon), np.max(v_optimal_discrete), 
                   np.max(v_optimal_uncon_discrete), np.max(v_uniform)) + 1
        
        u_min = min(np.min(u_optimal), np.min(u_optimal_uncon), np.min(u_optimal_discrete), 
                   np.min(u_optimal_uncon_discrete), np.min(u_uniform)) - 1
        u_max = max(np.max(u_optimal), np.max(u_optimal_uncon), np.max(u_optimal_discrete), 
                   np.max(u_optimal_uncon_discrete), np.max(u_uniform)) + 1
        
        p_min = min(np.min(p_optimal), np.min(p_optimal_uncon), np.min(p_optimal_discrete), 
                   np.min(p_optimal_uncon_discrete), np.min(p_uniform)) - 1
        p_max = max(np.max(p_optimal), np.max(p_optimal_uncon), np.max(p_optimal_discrete), 
                   np.max(p_optimal_uncon_discrete), np.max(p_uniform)) + 1
        
        V_min = min(np.min(V_cumulative), np.min(V_cumulative_uncon), np.min(V_cumulative_discrete), 
                   np.min(V_cumulative_uncon_discrete), np.min(V_uniform_cumulative)) - 1
        V_max = max(np.max(V_cumulative), np.max(V_cumulative_uncon), np.max(V_cumulative_discrete), 
                   np.max(V_cumulative_uncon_discrete), np.max(V_uniform_cumulative)) + 1
        
        U_min = min(np.min(U_cumulative), np.min(V_cumulative_uncon), np.min(U_cumulative_discrete), 
                   np.min(U_cumulative_uncon_discrete), np.min(U_uniform_cumulative)) - 1
        U_max = max(np.max(U_cumulative), np.max(V_cumulative_uncon), np.max(U_cumulative_discrete), 
                   np.max(U_cumulative_uncon_discrete), np.max(U_uniform_cumulative)) + 1
        
        D_min = min(np.min(D_cumulative), np.min(D_cumulative_uncon), np.min(D_cumulative_discrete), 
                   np.min(D_cumulative_uncon_discrete), np.min(D_uniform_cumulative)) - 1
        D_max = max(np.max(D_cumulative), np.max(D_cumulative_uncon), np.max(D_cumulative_discrete), 
                   np.max(D_cumulative_uncon_discrete), np.max(D_uniform_cumulative)) + 1
        
        # Верхний ряд графиков: v_i, u_i, p_i
        plt.subplot(2, 3, 1)
        plt.plot(time_steps, v_optimal, 'b-', label='CA')
        plt.plot(time_steps, v_optimal_uncon, 'g--', label='NCA')
        plt.plot(time_steps, v_optimal_discrete, 'r-.', label='CAD')
        plt.plot(time_steps, v_optimal_uncon_discrete, 'c:', label='NCAD')
        plt.plot(time_steps, v_uniform, 'm-', lw=1.5, label='UD')
        plt.title('Optimal Demand (v_i)')
        plt.ylabel('Demand')
        plt.ylim(v_min, v_max)
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 3, 2)
        plt.plot(time_steps, u_optimal, 'b-', label='CA')
        plt.plot(time_steps, u_optimal_uncon, 'g--', label='NCA')
        plt.plot(time_steps, u_optimal_discrete, 'r-.', label='CAD')
        plt.plot(time_steps, u_optimal_uncon_discrete, 'c:', label='NCAD')
        plt.plot(time_steps, u_uniform, 'm-', lw=1.5, label='UD')
        plt.title('Optimal Adjusted Demand (u_i)')
        plt.ylabel('Adjusted Demand')
        plt.ylim(u_min, u_max)
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 3, 3)
        plt.plot(time_steps, p_optimal, 'b-', label='CA')
        plt.plot(time_steps, p_optimal_uncon, 'g--', label='NCA')
        plt.plot(time_steps, p_optimal_discrete, 'r-.', label='CAD')
        plt.plot(time_steps, p_optimal_uncon_discrete, 'c:', label='NCAD')
        plt.plot(time_steps, p_uniform, 'm-', lw=1.5, label='UD')
        plt.title('Optimal Prices (p_i)')
        plt.ylabel('Price')
        plt.ylim(p_min, p_max)
        plt.legend()
        plt.grid(True)
        
        # Нижний ряд графиков: V_i, U_i, D_i
        plt.subplot(2, 3, 4)
        plt.plot(time_steps_cumulative, V_cumulative, 'b-', label='CA')
        plt.plot(time_steps_cumulative, V_cumulative_uncon, 'g--', label='NCA')
        plt.plot(time_steps_cumulative, V_cumulative_discrete, 'r-.', label='CAD')
        plt.plot(time_steps_cumulative, V_cumulative_uncon_discrete, 'c:', label='NCAD')
        plt.plot(time_steps_cumulative, V_uniform_cumulative, 'm-', lw=1.5, label='UD')
        plt.title('Cumulative Sales (V_i)')
        plt.ylabel('Cumulative Sales')
        plt.ylim(V_min, V_max)
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 3, 5)
        plt.plot(time_steps_cumulative, U_cumulative, 'b-', label='CA')
        plt.plot(time_steps_cumulative, V_cumulative_uncon, 'g--', label='NCA')
        plt.plot(time_steps_cumulative, U_cumulative_discrete, 'r-.', label='CAD')
        plt.plot(time_steps_cumulative, U_cumulative_uncon_discrete, 'c:', label='NCAD')
        plt.plot(time_steps_cumulative, U_uniform_cumulative, 'm-', lw=1.5, label='UD')
        plt.title('Cumulative Adjusted Sales (U_i)')
        plt.ylabel('Cumulative Adjusted Sales')
        plt.ylim(U_min, U_max)
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 3, 6)
        plt.plot(time_steps_cumulative, D_cumulative, 'b-', label='CA')
        plt.plot(time_steps_cumulative, D_cumulative_uncon, 'g--', label='NCA')
        plt.plot(time_steps_cumulative, D_cumulative_discrete, 'r-.', label='CAD')
        plt.plot(time_steps_cumulative, D_cumulative_uncon_discrete, 'c:', label='NCAD')
        plt.plot(time_steps_cumulative, D_uniform_cumulative, 'm-', lw=1.5, label='UD')
        
        for tj, Dj in constraints:
            plt.plot(tj, Dj, 'ko', markersize=8)
        
        active_constraints = []
        t = 0
        while t < T:
            right_constraints = [(tj, Dj) for tj, Dj in constraints if tj > t]
            if not right_constraints:
                break
                
            lambda_values = []
            for tj, Dj in right_constraints:
                try:
                    lambda_j, _, _, _, _, _, _ = optimal_sales_strategy_demand(
                        N, t, tj, gamma, D_cumulative[t], Dj, fi, a, b
                    )
                    lambda_values.append((tj, lambda_j))
                except ValueError:
                    continue
            
            if lambda_values:
                min_lambda = min(lj for _, lj in lambda_values)
                t_r = T
                for tj, lambda_j in lambda_values:
                    if abs(lambda_j - min_lambda) < 1e-6:
                        t_r = tj
                        break
                
                if t_r < T:
                    matching_constraint = next((t_c, D_c) for t_c, D_c in right_constraints if t_c == t_r)
                    active_constraints.append(matching_constraint)
            
            t = t_r
        
        for tj, Dj in active_constraints:
            plt.plot(tj, Dj, 'm*', markersize=12)
        
        plt.title('Discounted Revenue (D_i)')
        plt.ylabel('Cumulative Revenue')
        plt.ylim(D_min, D_max)
        
        handles, labels = plt.gca().get_legend_handles_labels()
        
        if len(handles) < 5:
            custom_handles = [
                plt.Line2D([0], [0], color='b', linestyle='-', label='CA'),
                plt.Line2D([0], [0], color='g', linestyle='--', label='NCA'),
                plt.Line2D([0], [0], color='r', linestyle='-.', label='CAD'),
                plt.Line2D([0], [0], color='c', linestyle=':', label='NCAD'),
                plt.Line2D([0], [0], color='m', linestyle='-', linewidth=1.5, label='UD'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='k', markersize=8),
                plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='m', markersize=12)
            ]
        else:
            custom_handles = [
                handles[0],
                handles[1],
                handles[2],
                handles[3],
                handles[4],
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='k', markersize=8),
                plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='m', markersize=12)
            ]
            
        custom_labels = ['CA', 'NCA', 'CAD', 'NCAD', 'UD', 'Constraint', 'Active constraint']
        plt.legend(custom_handles, custom_labels, loc='best')
        
        plt.grid(True)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
        
        try:
            from tabulate import tabulate
            
            table_data = [
                ["CA", f"{V_cumulative[-1]:.2f}", f"{U_cumulative[-1]:.2f}", f"{D_cumulative[-1]:.2f}"],
                ["NCA", f"{V_cumulative_uncon[-1]:.2f}", f"{V_cumulative_uncon[-1]:.2f}", f"{D_cumulative_uncon[-1]:.2f}"],
                ["CAD", f"{V_cumulative_discrete[-1]:.2f}", f"{U_cumulative_discrete[-1]:.2f}", f"{D_cumulative_discrete[-1]:.2f}"],
                ["NCAD", f"{V_cumulative_uncon_discrete[-1]:.2f}", f"{U_cumulative_uncon_discrete[-1]:.2f}", f"{D_cumulative_uncon_discrete[-1]:.2f}"],
                ["UD", f"{V_uniform_cumulative[-1]:.2f}", f"{U_uniform_cumulative[-1]:.2f}", f"{D_uniform_cumulative[-1]:.2f}"]
            ]
            
            print("\nComparison of All Algorithms:")
            print(tabulate(table_data, headers=["Algorithm", "V(T)", "U(T)", "D(T)"], tablefmt="grid"))
            
        except ImportError:
            headers = ["Algorithm", "V(T)", "U(T)", "D(T)"]
            table_data = [
                ["CA", f"{V_cumulative[-1]:.2f}", f"{U_cumulative[-1]:.2f}", f"{D_cumulative[-1]:.2f}"],
                ["NCA", f"{V_cumulative_uncon[-1]:.2f}", f"{V_cumulative_uncon[-1]:.2f}", f"{D_cumulative_uncon[-1]:.2f}"],
                ["CAD", f"{V_cumulative_discrete[-1]:.2f}", f"{U_cumulative_discrete[-1]:.2f}", f"{D_cumulative_discrete[-1]:.2f}"],
                ["NCAD", f"{V_cumulative_uncon_discrete[-1]:.2f}", f"{U_cumulative_uncon_discrete[-1]:.2f}", f"{D_cumulative_uncon_discrete[-1]:.2f}"],
                ["UD", f"{V_uniform_cumulative[-1]:.2f}", f"{U_uniform_cumulative[-1]:.2f}", f"{D_uniform_cumulative[-1]:.2f}"]
            ]
            
            col_widths = [max(len(row[i]) for row in [headers] + table_data) for i in range(len(headers))]
            
            header_row = " | ".join(f"{headers[i]:^{col_widths[i]}}" for i in range(len(headers)))
            separator = "+" + "+".join("-" * (width + 2) for width in col_widths) + "+"
            
            print("\nComparison of All Algorithms:")
            print(separator)
            print("| " + header_row + " |")
            print(separator)
            
            for row in table_data:
                data_row = " | ".join(f"{row[i]:^{col_widths[i]}}" for i in range(len(row)))
                print("| " + data_row + " |")
                print(separator)
        
        print("\nFinal Results:")
        print(f"Total apartments sold V(T): {V_cumulative[-1]:.2f}")
        print(f"Total adjusted sales U(T): {U_cumulative[-1]:.2f} (target: {N:.2f})")
        print(f"Total discounted revenue D(T): {D_cumulative[-1]:.2f}")
        print(f"Unconstrained discounted revenue D(T): {D_cumulative_uncon[-1]:.2f}")

        print("\n=== Dynamic Programming Setup ===")
        try:
            p_min = float(input("Enter minimum permissible price (p_min) [default=85.0]: ") or 85.0)
            p_max = float(input("Enter maximum permissible price (p_max) [default=115.0]: ") or 115.0)
            dt = float(input("Enter price step (dt) [default=1.0]: ") or 1.0)
            k_fine = float(input("Enter penalty coefficient for constraint violations (k_fine) [default=10.0]: ") or 10.0)
            D_discr = float(input("Enter discretization step for revenue (D_discr) [default=50.0]: ") or 50.0)
            
            total_fines_cad = 0
            for t, D_target in constraints:
                if t < len(D_cumulative_discrete) and D_cumulative_discrete_no_fine[t] < D_target:
                    fine = k_fine * (D_target - D_cumulative_discrete_no_fine[t])
                    total_fines_cad += fine
                    for i in range(t, T + 1):
                        D_cumulative_discrete[i] -= fine
            
            constraint_diffs_cad = calculate_constraint_differences(D_cumulative_discrete_no_fine, constraints)
            
            total_fines_ud = 0
            for t, D_target in constraints:
                if t < len(D_uniform_cumulative) and D_uniform_cumulative_no_fine[t] < D_target:
                    fine = k_fine * (D_target - D_uniform_cumulative_no_fine[t])
                    total_fines_ud += fine
                    for i in range(t, T + 1):
                        D_uniform_cumulative[i] -= fine
            
            constraint_diffs_ud = calculate_constraint_differences(D_uniform_cumulative_no_fine, constraints)
            
            u_lists, p_lists = permissible_u(fi, p_min, p_max, dt, T, a, b)
            
            action_counts = [len(u_list) for u_list in u_lists]
            print("\nAction space statistics:")
            print(f"Min actions: {min(action_counts)}")
            print(f"Max actions: {max(action_counts)}")
            print(f"Avg actions: {sum(action_counts)/len(action_counts):.2f}")
            print(f"Total state-action pairs: {sum(action_counts)}")
            
            print("\n=== Running Dynamic Programming solver ===")
            print("This may take some time depending on the state space size...")
            dp_optimal_policy, dp_optimal_value, num_d_values = solve_mdp(N, T, gamma, constraints, u_lists, p_lists, fi, k_fine, D_discr)
            
            print("\n=== Extracting optimal trajectory ===")
            dp_v_optimal, dp_u_optimal, dp_p_optimal, dp_V_cumulative, dp_U_cumulative, dp_D_cumulative, dp_D_cumulative_no_fine, constraint_diffs_dp, total_fines_dp = extract_dp_trajectory_with_mdp_sale(
                dp_optimal_policy, dp_optimal_value, N, T, gamma, u_lists, p_lists, fi, D_discr, constraints, k_fine, num_d_values)
            
            plt.figure(figsize=(18, 12))
            plt.suptitle('Comparison of Optimization Algorithms: CAD, UD, DP', fontsize=16)
            
            # Изменение графиков после DP: Заменяем график V_i на график D_i с учетом штрафов
            plt.subplot(3, 2, 1)
            plt.plot(time_steps, v_optimal_discrete, 'r-', label='CAD')
            plt.plot(time_steps, v_uniform, 'm-', label='UD')
            plt.plot(time_steps, dp_v_optimal, 'k-', label='DP')
            plt.title('Optimal Demand (v_i)')
            plt.ylabel('Demand')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(3, 2, 2)
            plt.plot(time_steps, u_optimal_discrete, 'r-', label='CAD')
            plt.plot(time_steps, u_uniform, 'm-', label='UD')
            plt.plot(time_steps, dp_u_optimal, 'k-', label='DP')
            plt.title('Optimal Adjusted Demand (u_i)')
            plt.ylabel('Adjusted Demand')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(3, 2, 3)
            plt.plot(time_steps, p_optimal_discrete, 'r-', label='CAD')
            plt.plot(time_steps, p_uniform, 'm-', label='UD')
            plt.plot(time_steps, dp_p_optimal, 'k-', label='DP')
            plt.title('Optimal Prices (p_i)')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)
            
            # Заменяем график V_i на график D_i с учетом штрафов
            plt.subplot(3, 2, 4)
            # Используем D_cumulative с учетом штрафов
            plt.plot(time_steps_cumulative, D_cumulative_discrete, 'r-', label='CAD with fines')
            plt.plot(time_steps_cumulative, D_uniform_cumulative, 'm-', label='UD with fines')
            plt.plot(time_steps_cumulative, dp_D_cumulative, 'k-', label='DP with fines')
            
            for tj, Dj in constraints:
                plt.plot(tj, Dj, 'ko', markersize=8)
                
            plt.title('Discounted Revenue (D_i) with Fines')
            plt.ylabel('Cumulative Revenue')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(3, 2, 5)
            plt.plot(time_steps_cumulative, U_cumulative_discrete, 'r-', label='CAD')
            plt.plot(time_steps_cumulative, U_uniform_cumulative, 'm-', label='UD')
            plt.plot(time_steps_cumulative, dp_U_cumulative, 'k-', label='DP')
            plt.title('Cumulative Adjusted Sales (U_i)')
            plt.ylabel('Cumulative Adjusted Sales')
            plt.legend()
            plt.grid(True)
            
            # Изменяем D_i на D_i без учета штрафов
            plt.subplot(3, 2, 6)
            # Используем D_cumulative_no_fine без учета штрафов
            plt.plot(time_steps_cumulative, D_cumulative_discrete_no_fine, 'r-', label='CAD no fines')
            plt.plot(time_steps_cumulative, D_uniform_cumulative_no_fine, 'm-', label='UD no fines')
            plt.plot(time_steps_cumulative, dp_D_cumulative_no_fine, 'k-', label='DP no fines')
            
            for tj, Dj in constraints:
                plt.plot(tj, Dj, 'ko', markersize=8)
            
            plt.title('Discounted Revenue (D_i) without Fines')
            plt.ylabel('Cumulative Revenue')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()
            
            try:
                from tabulate import tabulate
                
                table_data = [
                    ["CA", f"{D_cumulative[-1]:.2f}", f"{D_cumulative_no_fine_ca[-1]:.2f}", f"{total_fines_ca:.2f}", f"{U_cumulative[-1]:.2f}", f"{V_cumulative[-1]:.2f}"] + 
                    [f"{constraint_diffs_ca.get(t, 'N/A'):.2f}" for t, _ in constraints],
                    
                    ["CAD", f"{D_cumulative_discrete[-1]:.2f}", f"{D_cumulative_discrete_no_fine[-1]:.2f}", f"{total_fines_cad:.2f}", f"{U_cumulative_discrete[-1]:.2f}", f"{V_cumulative_discrete[-1]:.2f}"] +
                    [f"{constraint_diffs_cad.get(t, 'N/A'):.2f}" for t, _ in constraints],
                    
                    ["UD", f"{D_uniform_cumulative[-1]:.2f}", f"{D_uniform_cumulative_no_fine[-1]:.2f}", f"{total_fines_ud:.2f}", f"{U_uniform_cumulative[-1]:.2f}", f"{V_uniform_cumulative[-1]:.2f}"] +
                    [f"{constraint_diffs_ud.get(t, 'N/A'):.2f}" for t, _ in constraints],
                    
                    ["DP", f"{dp_D_cumulative[-1]:.2f}", f"{dp_D_cumulative_no_fine[-1]:.2f}", f"{total_fines_dp:.2f}", f"{dp_U_cumulative[-1]:.2f}", f"{dp_V_cumulative[-1]:.2f}"] +
                    [f"{constraint_diffs_dp.get(t, 'N/A'):.2f}" for t, _ in constraints]
                ]
                
                constraint_headers = [f"D({t})-D^{i+1}" for i, (t, _) in enumerate(constraints)]
                headers = ["Algorithm", "D(T) with fines", "D(T) no fines", "Total fines", "U(T)", "V(T)"] + constraint_headers
                
                print("\nComprehensive Comparison of All Algorithms:")
                print(tabulate(table_data, headers=headers, tablefmt="grid"))
                
            except ImportError:
                constraint_headers = [f"D({t})-D^{i+1}" for i, (t, _) in enumerate(constraints)]
                headers = ["Algorithm", "D(T) with fines", "D(T) no fines", "Total fines", "U(T)", "V(T)"] + constraint_headers
                
                table_data = [
                    ["CA", f"{D_cumulative[-1]:.2f}", f"{D_cumulative_no_fine_ca[-1]:.2f}", f"{total_fines_ca:.2f}", f"{U_cumulative[-1]:.2f}", f"{V_cumulative[-1]:.2f}"] + 
                    [f"{constraint_diffs_ca.get(t, 'N/A'):.2f}" for t, _ in constraints],
                    
                    ["CAD", f"{D_cumulative_discrete[-1]:.2f}", f"{D_cumulative_discrete_no_fine[-1]:.2f}", f"{total_fines_cad:.2f}", f"{U_cumulative_discrete[-1]:.2f}", f"{V_cumulative_discrete[-1]:.2f}"] +
                    [f"{constraint_diffs_cad.get(t, 'N/A'):.2f}" for t, _ in constraints],
                    
                    ["UD", f"{D_uniform_cumulative[-1]:.2f}", f"{D_uniform_cumulative_no_fine[-1]:.2f}", f"{total_fines_ud:.2f}", f"{U_uniform_cumulative[-1]:.2f}", f"{V_uniform_cumulative[-1]:.2f}"] +
                    [f"{constraint_diffs_ud.get(t, 'N/A'):.2f}" for t, _ in constraints],
                    
                    ["DP", f"{dp_D_cumulative[-1]:.2f}", f"{dp_D_cumulative_no_fine[-1]:.2f}", f"{total_fines_dp:.2f}", f"{dp_U_cumulative[-1]:.2f}", f"{dp_V_cumulative[-1]:.2f}"] +
                    [f"{constraint_diffs_dp.get(t, 'N/A'):.2f}" for t, _ in constraints]
                ]
                
                col_widths = [max(len(str(row[i])) for row in [headers] + table_data) for i in range(len(headers))]
                
                header_row = " | ".join(f"{headers[i]:^{col_widths[i]}}" for i in range(len(headers)))
                separator = "+" + "+".join("-" * (width + 2) for width in col_widths) + "+"
                
                print("\nComprehensive Comparison of All Algorithms:")
                print(separator)
                print("| " + header_row + " |")
                print(separator)
                
                for row in table_data:
                    data_row = " | ".join(f"{row[i]:^{col_widths[i]}}" for i in range(len(row)))
                    print("| " + data_row + " |")
                    print(separator)
            
        except ValueError as e:
            print(f"Error in Dynamic Programming setup: {e}")
    else:
        print("Could not calculate optimal strategy with the given constraints.")

if __name__ == "__main__":
    main()