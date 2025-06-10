import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import matplotlib.pyplot as plt
import time
from mdp_sale_short import mdp_sale_with_action, permissible_u, mdp_sale_strat_cl

# Function for calculating f_i(t)
def fi(t, T=52):
    """
    Calculate the demand function at time t
    
    Args:
        t: Current time step
        T: Total time horizon
        
    Returns:
        Demand function value at time t
    """
    return 1 + 0.2 * np.sin((t) * 2 * np.pi / T)

class MCTSNode:
    """
    Node class for Monte Carlo Tree Search
    
    Attributes:
        state: MDP state (i, inventory, revenue)
        parent: Parent node
        action_idx: Action index that led to this node
        children: Dictionary of child nodes, keyed by action_idx
        visits: Number of visits to this node
        value: Accumulated value (sum of discounted rewards from simulations)
        untried_actions: List of actions not yet expanded from this node
        terminal: Whether this node represents a terminal state
    """
    def __init__(self, state, parent=None, action_idx=None):
        self.state = state
        self.parent = parent
        self.action_idx = action_idx
        self.children = {}
        self.visits = 0
        self.value = 0  # Stores sum of discounted rewards from simulations passing through this node
        self.untried_actions = []
        self.terminal = False

def ucb_score(node, parent, exploration_weight=1.0):
    """
    Calculate the Upper Confidence Bound score for node selection
    
    Args:
        node: Child node to evaluate
        parent: Parent node
        exploration_weight: Weight for exploration term
        
    Returns:
        UCB score for the node
    """
    if node.visits == 0:
        return float('inf')
    
    # UCB1 formula: exploitation + exploration
    exploitation = node.value / node.visits
    exploration = exploration_weight * np.sqrt(2 * np.log(parent.visits) / node.visits)
    
    return exploitation + exploration

def extract_best_actions_from_root(root, max_depth=None):
    """
    Extract the best action sequence by traversing from root down, 
    choosing child with highest Q-value at each step.
    
    Args:
        root: Root node of the MCTS tree
        max_depth: Maximum depth to search
        
    Returns:
        List of best actions from root
    """
    if max_depth is None:
        max_depth = float('inf')
    
    node = root
    best_actions = []
    depth = 0
    
    while node.children and depth < max_depth:
        # Choose child with maximum Q-value (average reward)
        best_action = max(node.children.items(), 
                          key=lambda kv: kv[1].value / max(1, kv[1].visits))[0]
        best_actions.append(best_action)
        node = node.children[best_action]
        depth += 1
        
        if node.terminal:
            break
            
    return best_actions

def po_mcts(N, T, gamma, constraints, p_min, p_max, dp, k_fine, D_discr, 
           num_simulations=1000, max_depth=None, exploration_weight=1.0):
    """
    Monte Carlo Tree Search for finding optimal demand function
    
    Args:
        N: Total number of apartments
        T: Deadline for selling all apartments
        gamma: Discount factor
        constraints: List of tuples (t_req, D_req) for revenue constraints
        p_min, p_max, dp: Price range and step
        k_fine: Penalty coefficient
        D_discr: Discretization step for revenue
        num_simulations: Number of MCTS simulations
        max_depth: Maximum simulation depth (default to T if None)
        exploration_weight: Weight for exploration term in UCB formula
        
    Returns:
        u_opt: Optimal demand values at each time step
        v_opt: Optimal demand control values
        p_opt: Optimal price values at each time step
        cumulative_reward_with_fines: Cumulative revenue with penalties over time
        final_reward_with_fines: Total revenue with penalties at time T
        final_revenue_no_fines: Total revenue without penalties at time T
        cumulative_revenue_no_fines: Cumulative revenue without penalties over time
    """
    if max_depth is None:
        max_depth = T
        
    # Parameters for linear model
    Bp = 3.0 * N / (100.0 * T)
    Ap = 4.0 * N / T
    a = Ap / Bp
    b = 1.0 / Bp
    
    # Generate permissible actions
    u_lists, p_lists = permissible_u(fi, p_min, p_max, dp, T, a, b)
    
    # Verify that permissible_u generated lists for all time steps
    if len(u_lists) != T or len(p_lists) != T:
        print(f"Warning: permissible_u generated {len(u_lists)} lists instead of {T}. "
              f"This might affect results.")
    
    # Initial state: (i, inventory, revenue)
    root_state = (0, N, 0.0)
    root = MCTSNode(root_state)
    
    # Check if root is terminal
    i, inv, _ = root_state
    if inv <= 0 or i >= T:
        root.terminal = True
    else:
        # Initialize untried actions for root node
        if len(u_lists) > 0 and len(u_lists[0]) > 0:
            root.untried_actions = list(range(len(u_lists[0])))
    
    best_reward = float('-inf')
    
    # ---- MCTS Main Loop ----
    for sim in range(num_simulations):
        # ----- Selection Phase -----
        # Start from root and traverse down the tree using UCB until we reach 
        # a node with untried actions or a terminal node
        node = root
        state = root_state
        
        # Track path to leaf and accumulate rewards along the path
        path = []  # List of (node, action) pairs along the path
        path_reward = 0.0  # Accumulated reward during Selection and Expansion
        
        # Selection until we reach a node with untried actions or a terminal node
        while not node.terminal and not node.untried_actions:
            # If no children, break (we're at a leaf)
            if not node.children:
                break
                
            # Choose best child according to UCB
            action_idx = max(node.children, key=lambda a: ucb_score(node.children[a], node, exploration_weight))
            
            # Record the path
            path.append((node, action_idx))
            
            # Move to the child
            node = node.children[action_idx]
            
            # Update state by executing the action
            i, inv, D_i = state
            
            # Safety check for action bounds
            if i < len(u_lists) and action_idx < len(u_lists[i]):
                u_values = u_lists[i]
                p_values = p_lists[i]
                
                # Execute the step
                next_state, reward, term, _, _, _, _ = mdp_sale_with_action(
                    N, T, gamma, state, action_idx, fi, u_values, p_values, 
                    constraints, k_fine, D_discr
                )
                
                # Accumulate reward from selection phase
                path_reward += reward
                
                state = next_state
                if term:
                    node.terminal = True
                    break
            else:
                # Out of bounds, mark terminal
                node.terminal = True
                break
        
        # ----- Expansion Phase -----
        # Add one new child if the current node is not terminal and has untried actions
        if not node.terminal and node.untried_actions:
            # Choose a random untried action
            action_idx = node.untried_actions.pop()
            
            i, inv, D_i = state
            
            # Safety check for bounds
            if i < len(u_lists) and action_idx < len(u_lists[i]):
                u_values = u_lists[i]
                p_values = p_lists[i]
                
                # Execute the step
                next_state, reward, term, D_next_no_fine, fine, u_j, p_j = mdp_sale_with_action(
                    N, T, gamma, state, action_idx, fi, u_values, p_values, 
                    constraints, k_fine, D_discr
                )
                
                # Accumulate reward from expansion phase
                path_reward += reward
                
                # Create new child node
                child = MCTSNode(next_state, parent=node, action_idx=action_idx)
                child.terminal = term
                
                # Initialize child's untried actions if not terminal
                if not term:
                    next_i, next_inv, _ = next_state
                    if next_i < len(u_lists) and next_inv > 0:
                        child.untried_actions = list(range(len(u_lists[next_i])))
                
                # Add child to parent's children
                node.children[action_idx] = child
                
                # Add this expansion step to the path
                path.append((node, action_idx))
                
                # Update state and node for simulation
                state = next_state
                node = child
            
        # ----- Simulation Phase -----
        # Continue from the selected/expanded node with random actions
        simulation_state = state
        simulation_reward = 0.0  # Reward accumulated during simulation
        depth = len(path)
        
        while not node.terminal and depth < max_depth:
            i, inv, D_i = simulation_state
            
            # End simulation if inventory is empty
            if inv <= 0:
                break
            
            # Get permissible actions for current time step
            if i < len(u_lists):
                u_values = u_lists[i]
                p_values = p_lists[i]
                
                # Choose random action
                if u_values:
                    random_action_idx = np.random.choice(len(u_values))
                    
                    # Execute the step
                    next_state, reward, term, _, _, _, _ = mdp_sale_with_action(
                        N, T, gamma, simulation_state, random_action_idx, fi, u_values, p_values, 
                        constraints, k_fine, D_discr
                    )
                    
                    simulation_reward += reward
                    simulation_state = next_state
                    depth += 1
                    
                    if term:
                        break
                else:
                    break
            else:
                break
        
        # Calculate full reward from path and simulation
        full_reward = path_reward + simulation_reward
        
        # ----- Backpropagation Phase -----
        # Start from the root, updating all nodes along the path
        node = root
        node.visits += 1
        node.value += full_reward  # Total discounted reward from entire episode
        
        for parent_node, action in path:
            child_node = parent_node.children[action]
            child_node.visits += 1
            child_node.value += full_reward
        
        # Track best reward for progress reporting
        if full_reward > best_reward:
            best_reward = full_reward
            
        # Print progress
        if (sim + 1) % max(1, num_simulations // 100) == 0 or sim == num_simulations - 1:
            progress = (sim + 1) / num_simulations * 100
            print(f"\rProgress: {progress:.1f}%, Best reward: {best_reward:.2f} (Path: {path_reward:.2f}, Simulation: {simulation_reward:.2f})", end="")
    
    print()  # New line after progress
    
    # Extract best actions from the MCTS tree based on mean Q-values
    best_sequence = extract_best_actions_from_root(root, max_depth=max_depth)
    
    # Reconstruct optimal strategy
    u_opt = []
    v_opt = []
    p_opt = []
    cumulative_reward_with_fines = [0]  # Start with initial revenue of 0 (with penalties)
    cumulative_revenue_no_fines = [0]  # Revenue without penalties
    
    state = root_state
    
    for i in range(T):
        if i < len(best_sequence) and i < len(u_lists):
            action_idx = best_sequence[i]
            
            # Safety check
            if i < len(u_lists) and action_idx < len(u_lists[i]):
                u_values = u_lists[i]
                p_values = p_lists[i]
                
                # Execute the step
                next_state, reward, term, D_next_no_fine, fine, u_j, p_j = mdp_sale_with_action(
                    N, T, gamma, state, action_idx, fi, u_values, p_values, 
                    constraints, k_fine, D_discr
                )
                
                # Calculate v_i
                v_i = (a - p_j) / b if p_j < a else 0
                
                u_opt.append(u_j)
                v_opt.append(v_i)
                p_opt.append(p_j)
                
                # Accumulate rewards:
                # reward = immediate_revenue - fine, so reward is penalized revenue
                cumulative_reward_with_fines.append(cumulative_reward_with_fines[-1] + reward)
                
                # reward + fine = immediate_revenue, so this is unpenalized revenue
                cumulative_revenue_no_fines.append(cumulative_revenue_no_fines[-1] + reward + fine)
                
                state = next_state
                
                # Check if all apartments sold
                _, inv_next, _ = next_state
                if inv_next <= 0:
                    # Fill the rest with zeros if inventory is depleted
                    for j in range(i+1, T):
                        u_opt.append(0)
                        v_opt.append(0)
                        p_opt.append(p_max)
                        cumulative_reward_with_fines.append(cumulative_reward_with_fines[-1])
                        cumulative_revenue_no_fines.append(cumulative_revenue_no_fines[-1])
                    break
            else:
                # Handle case where action index is out of bounds
                u_opt.append(0)
                v_opt.append(0)
                p_opt.append(p_max)
                cumulative_reward_with_fines.append(cumulative_reward_with_fines[-1])
                cumulative_revenue_no_fines.append(cumulative_revenue_no_fines[-1])
        else:
            # Fill remaining steps with zeros
            u_opt.append(0)
            v_opt.append(0)
            p_opt.append(p_max)
            cumulative_reward_with_fines.append(cumulative_reward_with_fines[-1])
            cumulative_revenue_no_fines.append(cumulative_revenue_no_fines[-1])
    
    # Make sure all arrays have length T
    while len(u_opt) < T:
        u_opt.append(0)
        v_opt.append(0)
        p_opt.append(p_max)
        
    while len(cumulative_reward_with_fines) < T+1:
        cumulative_reward_with_fines.append(cumulative_reward_with_fines[-1])
        cumulative_revenue_no_fines.append(cumulative_revenue_no_fines[-1])
    
    # Final revenue values
    final_reward_with_fines = cumulative_reward_with_fines[-1]
    final_revenue_no_fines = cumulative_revenue_no_fines[-1]
        
    # Return optimal strategy and values with clearer variable names
    return u_opt, v_opt, p_opt, cumulative_reward_with_fines, final_reward_with_fines, final_revenue_no_fines, cumulative_revenue_no_fines


def main():
    """
    Main function to run the apartment sales optimization using MCTS
    """
    # Default values
    default_N = 364
    default_T = 52
    default_gamma = 0.99
    default_constraint_times = [13, 26, 39]
    default_constraint_values = [11500, 21000, 27000]
    default_p_min = 85.0
    default_p_max = 115.0
    default_dp = 1.0
    default_k_fine = 2
    default_D_discr = 20
    
    # Get user inputs with defaults
    try:
        # Validate input for N and T
        while True:
            N_input = input(f"Enter total number of apartments (N) [default={default_N}]: ") or default_N
            N = int(N_input)
            if N <= 0:
                print("Error: N must be greater than 0. Please try again.")
            else:
                break
                
        while True:
            T_input = input(f"Enter deadline for selling all apartments (T) [default={default_T}]: ") or default_T
            T = int(T_input)
            if T <= 0:
                print("Error: T must be greater than 0. Please try again.")
            else:
                break
        
        while True:
            gamma_input = input(f"Enter discount factor (gamma) [default={default_gamma}]: ") or default_gamma
            gamma = float(gamma_input)
            if gamma <= 0 or gamma > 1:
                print("Error: gamma must be in (0, 1]. Please try again.")
            else:
                break
        
        # Constraint times
        constraint_times = []
        for i, default_time in enumerate(default_constraint_times, 1):
            if i <= 3:
                while True:
                    time_input = input(f"Enter constraint time t^{i} [default={default_time}]: ")
                    if not time_input:
                        constraint_times.append(default_time)
                        break
                    try:
                        t = int(time_input)
                        if t <= 0 or t > T:
                            print(f"Error: Constraint time must be between 1 and {T}. Please try again.")
                        else:
                            constraint_times.append(t)
                            break
                    except ValueError:
                        print("Error: Please enter a valid integer.")
            else:
                time_input = input(f"Enter constraint time t^{i} (press Enter to stop): ")
                if not time_input:
                    break
                try:
                    t = int(time_input)
                    if t <= 0 or t > T:
                        print(f"Error: Constraint time must be between 1 and {T}. Please try again.")
                    else:
                        constraint_times.append(t)
                except ValueError:
                    print("Error: Please enter a valid integer.")
        
        # Constraint revenue values
        print("\nEnter target revenue values for each constraint time:")
        constraint_values = []
        for i, t in enumerate(constraint_times):
            if i < len(default_constraint_values):
                default_value = default_constraint_values[i]
            else:
                default_value = int(default_constraint_values[-1] * (i + 1) / len(default_constraint_values))
            
            while True:
                value_input = input(f"Enter target revenue D^{i+1} for t^{i+1}={t} [default={default_value}]: ")
                if not value_input:
                    constraint_values.append(default_value)
                    break
                try:
                    val = float(value_input)
                    if val < 0:
                        print("Error: Revenue target must be non-negative. Please try again.")
                    else:
                        constraint_values.append(val)
                        break
                except ValueError:
                    print("Error: Please enter a valid number.")
        
        # Price parameters
        while True:
            p_min_input = input(f"Enter minimum permissible price (p_min) [default={default_p_min}]: ") or default_p_min
            p_min = float(p_min_input)
            if p_min <= 0:
                print("Error: p_min must be greater than 0. Please try again.")
            else:
                break
                
        while True:
            p_max_input = input(f"Enter maximum permissible price (p_max) [default={default_p_max}]: ") or default_p_max
            p_max = float(p_max_input)
            if p_max <= p_min:
                print(f"Error: p_max must be greater than p_min ({p_min}). Please try again.")
            else:
                break
                
        while True:
            dp_input = input(f"Enter price step (dp) [default={default_dp}]: ") or default_dp
            dp = float(dp_input)
            if dp <= 0:
                print("Error: dp must be greater than 0. Please try again.")
            else:
                break
        
        # Other parameters
        while True:
            k_fine_input = input(f"Enter penalty coefficient (k_fine) [default={default_k_fine}]: ") or default_k_fine
            k_fine = float(k_fine_input)
            if k_fine < 0:
                print("Error: k_fine must be non-negative. Please try again.")
            else:
                break
                
        while True:
            D_discr_input = input(f"Enter discretization step for revenue (D_discr) [default={default_D_discr}]: ") or default_D_discr
            D_discr = float(D_discr_input)
            if D_discr < 0:
                print("Error: D_discr must be non-negative. Please try again.")
            else:
                break
        
        # Form constraints list
        constraints = list(zip(constraint_times, constraint_values))
        
        # Calculate reasonable hyperparameters for MCTS
        default_num_simulations = min(5000, max(1000, int(T * 20)))
        default_max_depth = T
        default_exploration_weight = 1.0
        
        print("\nMCTS Hyperparameters:")
        num_simulations = int(input(f"Enter number of simulations [default={default_num_simulations}]: ") or default_num_simulations)
        max_depth = int(input(f"Enter maximum depth [default={default_max_depth}]: ") or default_max_depth)
        exploration_weight = float(input(f"Enter exploration weight [default={default_exploration_weight}]: ") or default_exploration_weight)
        
        print("\nRunning MCTS optimization...")
        start_time = time.time()
        
        # Run MCTS optimization
        u_opt, v_opt, p_opt, cumulative_reward_with_fines, final_reward_with_fines, final_revenue_no_fines, cumulative_revenue_no_fines = po_mcts(
            N, T, gamma, constraints, p_min, p_max, dp, k_fine, D_discr,
            num_simulations, max_depth, exploration_weight
        )
        
        end_time = time.time()
        print(f"\nMCTS completed in {end_time - start_time:.2f} seconds")
        
        # Calculate cumulative values
        U_opt = np.cumsum(u_opt)
        V_opt = np.cumsum(v_opt)
        
        # Print final results
        print(f"\nFinal Results:")
        print(f"Total apartments sold (U(T)): {U_opt[-1]}")
        print(f"Total demand (V(T)): {V_opt[-1]:.2f}")
        print(f"Total revenue with penalties (D(T)): {final_reward_with_fines:.2f}")
        print(f"Total revenue without penalties (Dn(T)): {final_revenue_no_fines:.2f}")
        
        # Print constraint violations
        print("\nConstraint satisfaction:")
        for idx, (t, D_req) in enumerate(constraints):
            actual_D = cumulative_revenue_no_fines[t] if t < len(cumulative_revenue_no_fines) else cumulative_revenue_no_fines[-1]
            diff = actual_D - D_req
            status = "Met" if diff >= 0 else "Not Met"
            print(f"At t={t}: Required={D_req:.2f}, Actual={actual_D:.2f}, Diff={diff:.2f}, Status: {status}")
        
        # Create plots
        plt.figure(figsize=(15, 10))
        
        # First row: v*, u*, p*
        plt.subplot(2, 3, 1)
        plt.plot(range(T), v_opt)
        plt.title('Optimal demand control (v*)')
        plt.xlabel('Time step')
        plt.ylabel('Demand control')
        plt.grid(True)
        
        plt.subplot(2, 3, 2)
        plt.plot(range(T), u_opt)
        plt.title('Optimal demand (u*)')
        plt.xlabel('Time step')
        plt.ylabel('Apartments sold')
        plt.grid(True)
        
        plt.subplot(2, 3, 3)
        plt.plot(range(T), p_opt)
        plt.title('Optimal price (p*)')
        plt.xlabel('Time step')
        plt.ylabel('Price')
        plt.grid(True)
        
        # Second row: V*, U*, D* & Dn*
        plt.subplot(2, 3, 4)
        plt.plot(range(T+1), np.append(V_opt, V_opt[-1]))
        plt.title('Cumulative demand control (V*)')
        plt.xlabel('Time step')
        plt.ylabel('Cumulative demand control')
        plt.grid(True)
        
        plt.subplot(2, 3, 5)
        plt.plot(range(T+1), np.append(U_opt, U_opt[-1]))
        plt.title('Cumulative demand (U*)')
        plt.xlabel('Time step')
        plt.ylabel('Cumulative apartments sold')
        plt.grid(True)
        
        plt.subplot(2, 3, 6)
        plt.plot(range(T+1), cumulative_reward_with_fines, label='With penalties')
        plt.plot(range(T+1), cumulative_revenue_no_fines, label='Without penalties')
        
        # Add constraint points
        for t, D_req in constraints:
            if t <= T:
                plt.plot(t, D_req, 'ro', markersize=8)
                plt.text(t, D_req, f'({t},{D_req})', verticalalignment='bottom')
        
        plt.title('Revenue (D* & Dn*)')
        plt.xlabel('Time step')
        plt.ylabel('Revenue')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3, wspace=0.3)
        plt.show()
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()