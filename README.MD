# Comparison of Reinforcement Learning Price Optimization Applications
## Real Estate Sales Optimization Systems

### License: MIT

---

## Executive Summary

Both applications implement reinforcement learning approaches for optimizing pricing strategies in real estate sales, specifically for selling apartments over a fixed time horizon. While they share a common foundation in dynamic programming and optimal control theory, they differ significantly in their implementation approaches and available algorithms.

---

## Application Overview

### Application 1: RL_PO_DP_MCTS_M.py
- **Full Name**: Reinforcement Learning Price Optimization with Dynamic Programming and Monte Carlo Tree Search
- **Key Features**: External MDP module integration, MCTS implementation
- **Algorithms**: Dynamic Programming (DP) + Monte Carlo Tree Search (MCTS)

### Application 2: RL_PO_Limit_Total_RL_DP_CL_M.py
- **Full Name**: Reinforcement Learning Price Optimization with Limited Total using RL and DP with Constraints
- **Key Features**: Self-contained implementation, constraint-focused
- **Algorithms**: Dynamic Programming (DP) only

---

## Detailed Comparison

### 1. **Architecture & Dependencies**

| Aspect | App 1 (DP+MCTS) | App 2 (DP Only) |
|--------|-----------------|-----------------|
| External Dependencies | `mdp_sale_short` module | None (self-contained) |
| Libraries | numpy, matplotlib, math, numba, random | numpy, matplotlib, math, numba |
| Architecture | Modular with external MDP | Monolithic implementation |

### 2. **MDP Implementation**

**App 1 (DP+MCTS):**
- Uses external `mdp_sale_with_action()` function from `mdp_sale_short` module
- Wrapper function `compute_next_state()` calls external MDP
- More flexible and reusable architecture
- Separation of concerns between MDP logic and optimization

**App 2 (DP Only):**
- Implements MDP logic directly in `compute_next_state()`
- Self-contained calculation of:
  - Immediate revenue
  - Constraint violations and fines
  - State transitions
- Tighter coupling but fewer dependencies

### 3. **Optimization Algorithms**

**App 1 (DP+MCTS):**

**Dynamic Programming:**
- Numba-accelerated backward induction
- Handles discrete state space (inventory, revenue)
- Optimal policy extraction

**Monte Carlo Tree Search (MCTS):**
- Tree-based search algorithm
- Configurable parameters:
  - Iterations (default: 1000)
  - Simulations per iteration (default: 20)
  - Exploration constant (default: 1.41)
- Balances exploration vs exploitation
- Can handle larger state spaces more efficiently

**App 2 (DP Only):**
- Only implements Dynamic Programming
- Same Numba acceleration
- No alternative algorithms for comparison

### 4. **Problem Formulation**

Both applications model the same problem:
- **State**: (time, inventory, cumulative_revenue)
- **Actions**: (units_to_sell, price)
- **Constraints**: Revenue targets at specific time points
- **Penalties**: Fines for constraint violations
- **Objective**: Maximize total discounted revenue

### 5. **Analytical Solutions**

Both implement the same analytical approaches:
1. **Unconstrained Algorithm (NCA)**: Optimal solution without constraints
2. **Constrained Algorithm (CA)**: Analytical solution with constraints
3. **Discretized versions (CAD, NCAD)**: Integer-valued sales quantities
4. **Uniform Distribution (UD)**: Baseline strategy

### 6. **State Space Handling**

| Aspect | App 1 (DP+MCTS) | App 2 (DP Only) |
|--------|-----------------|-----------------|
| Revenue Discretization | Yes | Yes |
| State Space Size | Configurable via `D_discr` | Configurable via `D_discr` |
| Memory Efficiency | MCTS more memory-efficient for large spaces | DP requires full state enumeration |

### 7. **Computational Performance**

**App 1 (DP+MCTS):**
- DP: O(T × N × D) where T=time, N=inventory, D=revenue states
- MCTS: O(iterations × simulations × depth)
- MCTS can be faster for large state spaces

**App 2 (DP Only):**
- Only DP complexity: O(T × N × D)
- No alternative for large state spaces

### 8. **Visualization & Output**

Both applications provide:
- Comprehensive plots comparing all strategies
- Tables with final metrics
- Constraint violation analysis

**App 1 Additional Features:**
- MCTS vs DP comparison plots
- MCTS convergence visualization
- Additional performance metrics

### 9. **User Interaction**

Both feature interactive input for:
- Total apartments (N)
- Time horizon (T)
- Discount factor (γ)
- Constraint times and revenue targets
- Price bounds and discretization

**App 1 Additional Inputs:**
- MCTS hyperparameters
- Choice between DP and MCTS

### 10. **Code Quality & Maintainability**

| Aspect | App 1 (DP+MCTS) | App 2 (DP Only) |
|--------|-----------------|-----------------|
| Modularity | High (external MDP) | Medium (self-contained) |
| Code Reuse | Better (shared MDP module) | Limited |
| Testing | Easier (isolated components) | Harder (monolithic) |
| Flexibility | Higher | Lower |

---

## Use Case Recommendations

### When to Use App 1 (DP+MCTS):
- Large state spaces where DP becomes computationally expensive
- Need for exploration of different solution approaches
- Research or experimentation with different algorithms
- When external MDP module is available
- Comparing performance of different RL algorithms

### When to Use App 2 (DP Only):
- Smaller, well-defined problems
- When guaranteed optimal solutions are required
- No external dependencies desired
- Simpler deployment requirements
- Educational purposes (clearer MDP implementation)

---

## Technical Innovations

### App 1 Innovations:
1. **MCTS Integration**: Novel application of MCTS to pricing optimization
2. **Hybrid Approach**: Combines exact (DP) and approximate (MCTS) methods
3. **Modular Design**: Reusable MDP components

### App 2 Innovations:
1. **Self-Contained**: Complete implementation without dependencies
2. **Efficient Numba Usage**: Optimized computational kernels
3. **Clear State Transitions**: Transparent MDP logic

---

## Performance Characteristics

### Memory Usage:
- **App 1**: Lower with MCTS option (builds tree incrementally)
- **App 2**: Higher (stores complete value function)

### Computation Time:
- **App 1**: Flexible (can trade accuracy for speed with MCTS)
- **App 2**: Fixed (depends on state space size)

### Solution Quality:
- **App 1**: DP gives optimal; MCTS gives near-optimal
- **App 2**: Always optimal (within discretization)

---

## Conclusion

Both applications represent sophisticated approaches to real estate pricing optimization:

- **App 1** offers flexibility and scalability through its dual-algorithm approach and modular architecture
- **App 2** provides a robust, self-contained solution with guaranteed optimality

The choice between them depends on:
- Problem size and complexity
- Computational resources
- Need for optimality vs. speed
- Development environment constraints

For production use, App 1's MCTS capability makes it more suitable for large-scale problems, while App 2's simplicity makes it ideal for smaller, well-defined scenarios or educational purposes.

---

## License

Both applications are released under the MIT License, allowing for:
- Commercial use
- Modification
- Distribution
- Private use

With requirements for:
- License and copyright notice inclusion

---

*Document prepared for comparative analysis of RL-based pricing optimization systems*