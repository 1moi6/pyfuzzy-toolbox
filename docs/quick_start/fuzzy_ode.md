# Fuzzy ODE Solver Quick Start Guide

## Overview

**FuzzyODESolver** solves ordinary differential equations (ODEs) with **fuzzy initial conditions and/or fuzzy parameters**, computing solutions as α-level sets that form fuzzy envelopes over time.

**Key Concept:**
```
dy/dt = f(t, y, parameters)

Where:
- y₀ can be fuzzy (e.g., "around 10")
- parameters can be fuzzy (e.g., "growth rate ≈ 0.5")
- Solution y(t) is fuzzy at each time point
```

**Applications:**
- Modeling uncertainty in initial conditions
- Systems with imprecise parameters
- Epidemiological models with uncertain rates
- Population dynamics with fuzzy carrying capacity
- Any ODE with incomplete/imprecise information

**Advantages:**
- Quantify uncertainty propagation
- Multiple scenarios in single solution
- α-level interpretation (confidence intervals)
- No need for probabilistic assumptions

---

## 1. Basic Concepts

### What is a Fuzzy ODE?

A **fuzzy ODE** is a differential equation where:
1. **Initial conditions** are fuzzy numbers (not single values)
2. **Parameters** can be fuzzy
3. **Solution** is a fuzzy-valued function

**Example:**
```
Regular ODE:  dy/dt = 0.5y(1 - y/100),  y(0) = 10
Fuzzy ODE:    dy/dt = r·y(1 - y/K),     y₀ ≈ 10 ± 2, r ≈ 0.5 ± 0.05
```

### Fuzzy Numbers

Fuzzy numbers represent imprecise values using membership functions:

```python
from fuzzy_systems.dynamics.fuzzy_ode import FuzzyNumber

# Triangular: "approximately 10, spread 2"
num1 = FuzzyNumber.triangular(center=10, spread=2)
# Support: [8, 12], peak at 10

# Gaussian: "around 50, sigma 5"
num2 = FuzzyNumber.gaussian(mean=50, sigma=5)

# Trapezoidal: "between 20-30, certainty 23-27"
num3 = FuzzyNumber.trapezoidal(a=20, b=23, c=27, d=30)
```

### α-Cuts (Alpha Levels)

An **α-cut** extracts an interval from a fuzzy number at confidence level α:

```
α = 1.0 → Core (most likely values)
α = 0.5 → Medium confidence
α = 0.0 → Support (all possible values)
```

The solver computes α-cuts for multiple levels, creating fuzzy envelopes.

---

## 2. Creating Fuzzy Numbers

### Triangular Fuzzy Numbers

```python
from fuzzy_systems.dynamics.fuzzy_ode import FuzzyNumber

# Symmetric triangular
y0 = FuzzyNumber.triangular(center=10, spread=2)
# Represents: "around 10, ±2"
# Support: [8, 12], peak at 10

# Check membership
print(y0.membership(10))    # 1.0 (core)
print(y0.membership(9))     # 0.5
print(y0.membership(8))     # 0.0 (boundary)

# Extract α-cut
interval = y0.alpha_cut(alpha=0.5)
print(interval)  # (9.0, 11.0)
```

### Gaussian Fuzzy Numbers

```python
# Gaussian: smoother, no hard boundaries
param = FuzzyNumber.gaussian(
    mean=0.5,
    sigma=0.1,
    n_sigmas=3  # Support extends 3 sigmas
)
# Support: [0.2, 0.8] (3σ from mean)
```

### Trapezoidal Fuzzy Numbers

```python
# Trapezoidal: plateau in the middle
capacity = FuzzyNumber.trapezoidal(
    a=80,   # Left boundary
    b=90,   # Start of plateau
    c=110,  # End of plateau
    d=120   # Right boundary
)
# Core (α=1.0): [90, 110]
# Support (α=0.0): [80, 120]
```

---

## 3. Defining the ODE Function

The ODE function must accept `(t, y, **params)`:

```python
def my_ode(t, y, r, K):
    """
    ODE: dy/dt = r * y * (1 - y/K)

    Args:
        t: Time (scalar)
        y: State vector (array)
        r: Parameter 1
        K: Parameter 2

    Returns:
        dy/dt (array, same shape as y)
    """
    return r * y * (1 - y / K)
```

**Important:**
- Signature must be `(t, y, **params)`
- `y` is always an array (even for 1D systems)
- Return array with same shape as `y`
- Parameters passed via `**params` (unpacked dict)

---

## 4. Creating the Solver

### Basic Setup

```python
from fuzzy_systems.dynamics.fuzzy_ode import FuzzyODESolver, FuzzyNumber

# Define ODE
def logistic(t, y, r, K):
    return r * y * (1 - y / K)

# Create fuzzy initial condition
y0_fuzzy = FuzzyNumber.triangular(center=10, spread=2)

# Create fuzzy parameter
r_fuzzy = FuzzyNumber.triangular(center=0.5, spread=0.05)

# Create solver
solver = FuzzyODESolver(
    ode_func=logistic,
    t_span=(0, 50),
    initial_condition=[y0_fuzzy],       # List of FuzzyNumber or float
    params={'r': r_fuzzy, 'K': 100},    # Dict with FuzzyNumber or float
    n_alpha_cuts=11,                    # Number of α-levels (2-50)
    method='RK45',                      # Integration method
    var_names=['population']            # Variable names (optional)
)
```

### Solver Parameters

- **`ode_func`**: ODE function with signature `(t, y, **params)`
- **`t_span`**: Time interval `(t_start, t_end)`
- **`initial_condition`**: List of initial values (FuzzyNumber or float)
- **`params`**: Dict of parameters (FuzzyNumber or float)
- **`n_alpha_cuts`**: Number of α-levels (default: 11)
  - More = smoother envelopes, slower computation
  - Typical range: 5-25
- **`method`**: Integration method (default: 'RK45')
  - Options: 'RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA'
  - RK45 = good default (Runge-Kutta 4th/5th order)
- **`t_eval`**: Time points for evaluation (optional)
  - If `None`, uses 100 points uniformly spaced
- **`n_jobs`**: Parallel workers (default: -1 = all cores)
- **`rtol`**, **`atol`**: Numerical tolerances (default: 1e-6, 1e-9)
- **`var_names`**: List of variable names for plots

---

## 5. Solving Fuzzy ODEs

### Method 1: Standard (Default)

```python
# Solve with standard method
solution = solver.solve(
    method='standard',
    n_grid_points=20,  # Points per dimension per α-level
    verbose=True
)
```

**Characteristics:**
- Most accurate
- Slowest (explores full grid)
- Best for low-dimensional problems (1-3 variables)
- Grid size: `n_grid_points^(n_vars + n_fuzzy_params)` per α

### Method 2: Monte Carlo (Recommended for High Dimensions)

```python
# Solve with Monte Carlo sampling
solution = solver.solve(
    method='monte_carlo',
    n_samples=1000,     # Number of random samples
    random_seed=42,     # For reproducibility
    verbose=True
)
```

**Characteristics:**
- Fastest for high-dimensional problems
- Scalable (10-400x faster in high dimensions)
- Stochastic (results vary slightly)
- Best for: many fuzzy parameters, multi-variable systems

### Method 3: Hierarchical

```python
# Solve with hierarchical optimization
solution = solver.solve(
    method='hierarchical',
    verbose=True
)
```

**Characteristics:**
- 3-5x faster than standard
- Deterministic (same results every time)
- Good middle ground
- Reuses computations from higher α-levels

---

## 6. Working with Solutions

### Solution Object

```python
# Solve
solution = solver.solve(method='monte_carlo')

# Access attributes
print(solution.t)            # Time points
print(solution.y_min.shape)  # (n_alpha, n_vars, n_time)
print(solution.y_max.shape)  # (n_alpha, n_vars, n_time)
print(solution.alphas)       # α-levels used
print(solution.var_names)    # Variable names
```

### Get Specific α-Level

```python
# Extract envelope for specific α
y_min_05, y_max_05 = solution.get_alpha_level(alpha=0.5)
# y_min_05.shape: (n_vars, n_time)
# y_max_05.shape: (n_vars, n_time)

# Plot specific α-level
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.fill_between(solution.t, y_min_05[0], y_max_05[0],
                 alpha=0.3, label='α=0.5')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()
```

---

## 7. Visualization

### Plot Fuzzy Solution

```python
# Plot with all α-levels
solution.plot(
    var_idx=0,                # Variable index (0 for first)
    alpha_levels=None,        # None = all, or list like [0.0, 0.5, 1.0]
    show=True
)
```

### Custom Multi-Variable Plot

```python
import matplotlib.pyplot as plt

# For multi-variable system
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for var_idx in range(2):
    ax = axes[var_idx]

    # Plot each α-level
    cmap = plt.cm.Blues
    for i, alpha in enumerate(solution.alphas):
        y_min, y_max = solution.get_alpha_level(alpha)
        color = cmap(0.3 + 0.7 * alpha)

        ax.fill_between(
            solution.t,
            y_min[var_idx],
            y_max[var_idx],
            alpha=0.3,
            color=color,
            label=f'α={alpha:.1f}' if i % 3 == 0 else None
        )

    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel(solution.var_names[var_idx], fontsize=12)
    ax.set_title(f'{solution.var_names[var_idx]} Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 8. Exporting Results

### Export to CSV

```python
# Export specific α-level to CSV
solution.to_csv('fuzzy_solution.csv', alpha=0.5)

# Export with European format
solution.to_csv('solution.csv', alpha=1.0, sep=';', decimal=',')
```

**CSV Format:**
```
time,population_min,population_max
0.000000,8.000000,12.000000
0.500000,8.234100,12.765900
1.000000,8.521234,13.478766
...
```

### Export to DataFrame

```python
# Convert to pandas DataFrame
df = solution.to_dataframe(alpha=0.5)
print(df.head())

# Access metadata
print(df.attrs['alpha_level'])
print(df.attrs['n_alpha_levels'])
print(df.attrs['var_names'])

# Further processing
df['y_mean'] = (df['y0_min'] + df['y0_max']) / 2
df['y_width'] = df['y0_max'] - df['y0_min']
```

---

## 9. Complete Example: Logistic Growth

```python
import numpy as np
import matplotlib.pyplot as plt
from fuzzy_systems.dynamics.fuzzy_ode import FuzzyODESolver, FuzzyNumber

# ============================================================================
# Define ODE: Logistic growth model
# ============================================================================
def logistic(t, y, r, K):
    """dy/dt = r * y * (1 - y/K)"""
    return r * y * (1 - y / K)

# ============================================================================
# Scenario A: Fuzzy Initial Condition Only
# ============================================================================
y0_fuzzy = FuzzyNumber.triangular(center=10, spread=4)

solver_a = FuzzyODESolver(
    ode_func=logistic,
    t_span=(0, 50),
    initial_condition=[y0_fuzzy],
    params={'r': 0.2, 'K': 100},
    n_alpha_cuts=15,
    var_names=['population']
)

sol_a = solver_a.solve(method='monte_carlo', n_samples=1000, verbose=True)
sol_a.plot()
plt.title('Scenario A: Fuzzy Initial Condition')

# ============================================================================
# Scenario B: Fuzzy Carrying Capacity
# ============================================================================
K_fuzzy = FuzzyNumber.triangular(center=100, spread=20)

solver_b = FuzzyODESolver(
    ode_func=logistic,
    t_span=(0, 50),
    initial_condition=[10.0],  # Crisp y0
    params={'r': 0.2, 'K': K_fuzzy},
    n_alpha_cuts=15,
    var_names=['population']
)

sol_b = solver_b.solve(method='monte_carlo', verbose=True)
sol_b.plot()
plt.title('Scenario B: Fuzzy Carrying Capacity')

# ============================================================================
# Scenario C: Fuzzy Growth Rate
# ============================================================================
r_fuzzy = FuzzyNumber.triangular(center=0.2, spread=0.04)

solver_c = FuzzyODESolver(
    ode_func=logistic,
    t_span=(0, 50),
    initial_condition=[10.0],
    params={'r': r_fuzzy, 'K': 100},
    n_alpha_cuts=15,
    var_names=['population']
)

sol_c = solver_c.solve(method='monte_carlo', verbose=True)
sol_c.plot()
plt.title('Scenario C: Fuzzy Growth Rate')

# ============================================================================
# Scenario D: Everything Fuzzy
# ============================================================================
y0_fuzzy_d = FuzzyNumber.triangular(center=10, spread=4)
r_fuzzy_d = FuzzyNumber.triangular(center=0.2, spread=0.04)
K_fuzzy_d = FuzzyNumber.triangular(center=100, spread=20)

solver_d = FuzzyODESolver(
    ode_func=logistic,
    t_span=(0, 50),
    initial_condition=[y0_fuzzy_d],
    params={'r': r_fuzzy_d, 'K': K_fuzzy_d},
    n_alpha_cuts=15,
    var_names=['population']
)

sol_d = solver_d.solve(method='monte_carlo', n_samples=2000, verbose=True)
sol_d.plot()
plt.title('Scenario D: All Parameters Fuzzy')

# ============================================================================
# Compare All Scenarios
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

scenarios = [
    (sol_a, 'Scenario A: y₀ Fuzzy'),
    (sol_b, 'Scenario B: K Fuzzy'),
    (sol_c, 'Scenario C: r Fuzzy'),
    (sol_d, 'Scenario D: All Fuzzy')
]

for ax, (sol, title) in zip(axes, scenarios):
    cmap = plt.cm.Blues
    for i, alpha in enumerate(sol.alphas):
        y_min, y_max = sol.get_alpha_level(alpha)
        color = cmap(0.3 + 0.7 * alpha)
        ax.fill_between(sol.t, y_min[0], y_max[0],
                       alpha=0.3, color=color)

    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Population', fontsize=11)
    ax.set_title(title, fontweight='bold')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Export results
sol_d.to_csv('logistic_fuzzy_all.csv', alpha=0.5)
```

---

## 10. Complete Example: SIR Epidemic Model

```python
import numpy as np
import matplotlib.pyplot as plt
from fuzzy_systems.dynamics.fuzzy_ode import FuzzyODESolver, FuzzyNumber

# ============================================================================
# SIR Model with Fuzzy Parameters
# ============================================================================
def sir_model(t, y, beta, gamma):
    """
    SIR epidemic model

    S: Susceptible
    I: Infected
    R: Recovered

    dS/dt = -beta * S * I
    dI/dt = beta * S * I - gamma * I
    dR/dt = gamma * I
    """
    S, I, R = y
    N = S + I + R  # Total population (constant)

    dS = -beta * S * I / N
    dI = beta * S * I / N - gamma * I
    dR = gamma * I

    return np.array([dS, dI, dR])

# ============================================================================
# Setup with Fuzzy Transmission Rate
# ============================================================================
# Initial conditions (crisp)
S0 = 990.0
I0 = 10.0
R0 = 0.0

# Fuzzy transmission rate: "beta around 0.5 ± 0.1"
beta_fuzzy = FuzzyNumber.triangular(center=0.5, spread=0.1)

# Recovery rate (crisp): 1/14 (14-day recovery)
gamma_crisp = 1.0 / 14.0

# Create solver
solver = FuzzyODESolver(
    ode_func=sir_model,
    t_span=(0, 160),
    initial_condition=[S0, I0, R0],
    params={'beta': beta_fuzzy, 'gamma': gamma_crisp},
    n_alpha_cuts=11,
    var_names=['Susceptible', 'Infected', 'Recovered']
)

# Solve
solution = solver.solve(method='monte_carlo', n_samples=1500, verbose=True)

# ============================================================================
# Visualize All Compartments
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
compartments = ['Susceptible', 'Infected', 'Recovered']
colors_map = [plt.cm.Reds, plt.cm.Oranges, plt.cm.Greens]

for idx, (ax, name, cmap) in enumerate(zip(axes, compartments, colors_map)):
    for i, alpha in enumerate(solution.alphas):
        y_min, y_max = solution.get_alpha_level(alpha)
        color = cmap(0.3 + 0.7 * alpha)
        ax.fill_between(
            solution.t,
            y_min[idx],
            y_max[idx],
            alpha=0.4,
            color=color,
            label=f'α={alpha:.1f}' if i % 3 == 0 else None
        )

    ax.set_xlabel('Days', fontsize=12)
    ax.set_ylabel('Population', fontsize=12)
    ax.set_title(f'{name} Population', fontweight='bold', fontsize=13)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================================
# Export Peak Infection Time (α=0.5)
# ============================================================================
y_min_05, y_max_05 = solution.get_alpha_level(alpha=0.5)
I_envelope = (y_min_05[1] + y_max_05[1]) / 2
peak_time = solution.t[np.argmax(I_envelope)]
peak_value = np.max(I_envelope)

print(f"\nPeak Infection (α=0.5):")
print(f"  Time: {peak_time:.1f} days")
print(f"  Value: {peak_value:.0f} infected")

# Export
solution.to_csv('sir_fuzzy_beta.csv', alpha=0.5)
```

---

## 11. Tips and Best Practices

### Choosing the Right Method

| Method | Best For | Speed | Accuracy | Stochastic |
|--------|----------|-------|----------|------------|
| **Standard** | 1-2 variables, few fuzzy params | Slow | Highest | No |
| **Monte Carlo** | High dimensions, many fuzzy params | Fast | Good | Yes |
| **Hierarchical** | Medium problems, deterministic | Medium | High | No |

**Recommendations:**
- 1-2 fuzzy inputs → `standard` (n_grid_points=20)
- 3+ fuzzy inputs → `monte_carlo` (n_samples=1000-5000)
- Need deterministic → `hierarchical`

### Choosing n_alpha_cuts

```python
# Coarse (fast, less smooth)
n_alpha_cuts=5   # [0.0, 0.25, 0.5, 0.75, 1.0]

# Standard (good balance)
n_alpha_cuts=11  # [0.0, 0.1, ..., 0.9, 1.0]

# Fine (smooth, slower)
n_alpha_cuts=25  # [0.0, 0.04, ..., 0.96, 1.0]
```

### Choosing Fuzzy Number Types

**Triangular:**
- Simple, easy to interpret
- Use when: "approximately X, ±Y"
- Example: `FuzzyNumber.triangular(center=10, spread=2)`

**Gaussian:**
- Smooth, no hard boundaries
- Use when: uncertainty is continuous
- Example: `FuzzyNumber.gaussian(mean=10, sigma=2)`

**Trapezoidal:**
- Plateau of maximum certainty
- Use when: "between X and Y, most likely Z to W"
- Example: `FuzzyNumber.trapezoidal(a=8, b=9, c=11, d=12)`

### Performance Optimization

```python
# Fast exploration (low accuracy)
sol = solver.solve(method='monte_carlo', n_samples=500)

# Standard (good balance)
sol = solver.solve(method='monte_carlo', n_samples=2000)

# High accuracy (slower)
sol = solver.solve(method='monte_carlo', n_samples=10000)

# Use parallel processing (default)
solver = FuzzyODESolver(..., n_jobs=-1)  # All cores
```

### Numerical Stability

```python
# Stiff systems: use appropriate solver
solver = FuzzyODESolver(
    ...,
    method='BDF',  # Better for stiff ODEs
    rtol=1e-5,
    atol=1e-8
)

# Non-stiff, high accuracy
solver = FuzzyODESolver(
    ...,
    method='DOP853',  # 8th order Runge-Kutta
    rtol=1e-8,
    atol=1e-10
)
```

---

## 12. Common Patterns

### Pattern 1: Uncertain Initial Conditions

```python
# Measurement uncertainty in starting population
y0 = FuzzyNumber.gaussian(mean=100, sigma=10)

solver = FuzzyODESolver(
    ode_func=my_ode,
    t_span=(0, 50),
    initial_condition=[y0],
    params={'rate': 0.1}  # Known parameter
)
```

### Pattern 2: Parameter Uncertainty

```python
# Unknown growth rate, measured as "around 0.5"
rate = FuzzyNumber.triangular(center=0.5, spread=0.1)

solver = FuzzyODESolver(
    ode_func=my_ode,
    t_span=(0, 50),
    initial_condition=[100.0],  # Known initial condition
    params={'rate': rate}
)
```

### Pattern 3: Multiple Uncertainties

```python
# Both initial condition and parameters are fuzzy
y0 = FuzzyNumber.triangular(center=100, spread=20)
rate = FuzzyNumber.triangular(center=0.5, spread=0.1)
capacity = FuzzyNumber.gaussian(mean=1000, sigma=100)

solver = FuzzyODESolver(
    ode_func=logistic,
    t_span=(0, 50),
    initial_condition=[y0],
    params={'r': rate, 'K': capacity}
)
```

---

## 13. Common Issues and Solutions

| Problem | Cause | Solution |
|---------|-------|----------|
| Very wide envelopes | Too much uncertainty | Reduce fuzzy spreads, check α-cuts |
| Solution doesn't match crisp | Numerical issues | Increase n_samples (MC) or n_grid_points (standard) |
| Slow computation | High dimensions | Use `method='monte_carlo'` with fewer samples |
| Envelopes cross | α-levels not monotonic | Check fuzzy number definitions |
| Memory error | Too many grid points | Reduce n_grid_points or use Monte Carlo |
| NaN values | ODE solver failed | Check ODE function, adjust rtol/atol |

### Debugging Tips

```python
# Test ODE function with crisp values
y_test = np.array([10.0])
dydt = my_ode(0, y_test, r=0.5, K=100)
print(f"dy/dt at t=0: {dydt}")

# Test fuzzy number
y0 = FuzzyNumber.triangular(center=10, spread=2)
print(f"Support: {y0.support}")
print(f"α=0.5 cut: {y0.alpha_cut(0.5)}")
print(f"Membership at 10: {y0.membership(10)}")

# Solve with verbose
sol = solver.solve(method='monte_carlo', verbose=True)
# Shows progress and statistics
```

---

## 14. Advanced: Multi-Variable Systems

```python
# Lotka-Volterra Predator-Prey with fuzzy parameters
def lotka_volterra(t, y, alpha, beta, delta, gamma):
    """
    Predator-prey model
    x: prey, y: predator
    """
    x, y_pred = y
    dx = alpha * x - beta * x * y_pred
    dy = delta * x * y_pred - gamma * y_pred
    return np.array([dx, dy])

# Fuzzy parameters
alpha_fuzzy = FuzzyNumber.triangular(center=1.0, spread=0.1)
beta_fuzzy = FuzzyNumber.triangular(center=0.1, spread=0.02)
delta_fuzzy = FuzzyNumber.triangular(center=0.075, spread=0.015)
gamma_fuzzy = FuzzyNumber.triangular(center=1.5, spread=0.15)

# Fuzzy initial conditions
x0_fuzzy = FuzzyNumber.triangular(center=40, spread=5)
y0_fuzzy = FuzzyNumber.triangular(center=9, spread=2)

solver = FuzzyODESolver(
    ode_func=lotka_volterra,
    t_span=(0, 50),
    initial_condition=[x0_fuzzy, y0_fuzzy],
    params={
        'alpha': alpha_fuzzy,
        'beta': beta_fuzzy,
        'delta': delta_fuzzy,
        'gamma': gamma_fuzzy
    },
    n_alpha_cuts=11,
    var_names=['Prey', 'Predator']
)

# Use Monte Carlo for high-dimensional problem
solution = solver.solve(method='monte_carlo', n_samples=5000, verbose=True)

# Plot both variables
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for idx, name in enumerate(['Prey', 'Predator']):
    solution.plot(var_idx=idx, ax=axes[idx], show=False)
    axes[idx].set_title(f'{name} Population')

plt.tight_layout()
plt.show()
```

---

## 15. Comparison with Other Approaches

### Fuzzy ODE vs Monte Carlo Simulation

| Aspect | Fuzzy ODE | Monte Carlo |
|--------|-----------|-------------|
| **Uncertainty** | Possibilistic | Probabilistic |
| **Input** | Fuzzy numbers | Probability distributions |
| **Output** | α-level envelopes | Confidence intervals |
| **Assumptions** | None (membership) | Statistical (distribution) |
| **Interpretation** | Possibility | Probability |

### When to Use Fuzzy ODE

✅ **Use Fuzzy ODE when:**
- Data is imprecise, not random
- Expert knowledge is qualitative ("around", "approximately")
- No probability distributions available
- Want worst-case/best-case scenarios

❌ **Use Monte Carlo when:**
- Have probability distributions
- Data is from random processes
- Need statistical confidence intervals

---

## References

- Barros, L. C., Bassanezi, R. C., & Lodwick, W. A. (2017). *A First Course in Fuzzy Logic, Fuzzy Dynamical Systems, and Biomathematics*. Springer.
- Kaleva, O. (1987). "Fuzzy differential equations." *Fuzzy Sets and Systems*, 24(3), 301-317.
- Bede, B., & Gal, S. G. (2005). "Generalizations of the differentiability of fuzzy-number-valued functions with applications to fuzzy differential equations." *Fuzzy Sets and Systems*, 151(3), 581-599.
