# User Guide: Fuzzy Dynamical Systems

This guide covers how to model and solve dynamical systems with fuzzy parameters and initial conditions.

## What are Fuzzy Dynamical Systems?

**Classical dynamical systems:**
- Crisp parameters (r = 0.5)
- Crisp initial conditions (y₀ = 10)
- Deterministic evolution

**Fuzzy dynamical systems:**
- Uncertain parameters (r ≈ "around 0.5")
- Uncertain initial conditions (y₀ ≈ "approximately 10")
- Prediction bands instead of single trajectories

**Why use fuzzy dynamics:**
- 📊 Model parameter uncertainty
- 🔬 Handle measurement errors
- 🎯 Propagate uncertainty through time
- 🌐 Capture expert knowledge about ranges

---

## Overview of Methods

| Method | Type | Best For | Output |
|--------|------|----------|--------|
| **Fuzzy ODE** | Continuous-time | ODEs with fuzzy IVPs | α-level trajectories |
| **p-Fuzzy (Discrete)** | Discrete-time | Difference equations | Fuzzy number sequences |
| **p-Fuzzy (Continuous)** | Continuous-time | Continuous dynamics | Fuzzy trajectories |

**Quick decision:**
- **Have a differential equation?** → Fuzzy ODE
- **Have a difference equation?** → p-Fuzzy Discrete
- **Need interactive dynamics?** → p-Fuzzy Continuous

---

## Fuzzy Numbers

Before solving fuzzy dynamical systems, we need to represent uncertainty.

### Creating Fuzzy Numbers

```python
from fuzzy_systems.dynamics import FuzzyNumber
import matplotlib.pyplot as plt

# Triangular fuzzy number: "approximately 10"
y0 = FuzzyNumber.triangular(center=10, spread=2)

# Trapezoidal: "between 8 and 12, most likely 9-11"
y0 = FuzzyNumber.trapezoidal(a=8, b=9, c=11, d=12)

# Gaussian: "around 10 with standard deviation 1"
y0 = FuzzyNumber.gaussian(center=10, sigma=1)

# Plot
y0.plot()
plt.xlabel('Value')
plt.ylabel('Membership')
plt.title('Fuzzy Initial Condition')
plt.show()
```

---

### Operations with Fuzzy Numbers

Fuzzy numbers support arithmetic operations:

```python
# Create fuzzy numbers
a = FuzzyNumber.triangular(center=5, spread=1)
b = FuzzyNumber.triangular(center=3, spread=0.5)

# Addition
c = a + b  # Approximately 8 ± 1.5

# Subtraction
d = a - b  # Approximately 2 ± 1.5

# Multiplication
e = a * b  # Approximately 15 ± ...

# Scalar operations
f = 2 * a  # Approximately 10 ± 2
g = a + 5  # Approximately 10 ± 1

# Plot results
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 3, figsize=(12, 6))
a.plot(ax=axes[0, 0], title='a')
b.plot(ax=axes[0, 1], title='b')
c.plot(ax=axes[0, 2], title='a + b')
d.plot(ax=axes[1, 0], title='a - b')
e.plot(ax=axes[1, 1], title='a * b')
f.plot(ax=axes[1, 2], title='2 * a')
plt.tight_layout()
plt.show()
```

**Operations use α-level arithmetic:**
- Addition: [a, b] + [c, d] = [a+c, b+d]
- Multiplication: [a, b] × [c, d] = [min(ac, ad, bc, bd), max(ac, ad, bc, bd)]

---

### α-levels

Access specific confidence intervals:

```python
y0 = FuzzyNumber.triangular(center=10, spread=2)

# Get 0.5-level (50% confidence)
lower, upper = y0.alpha_cut(alpha=0.5)
print(f"0.5-level: [{lower:.2f}, {upper:.2f}]")  # [9.0, 11.0]

# Get support (0-level)
lower, upper = y0.alpha_cut(alpha=0)
print(f"Support: [{lower:.2f}, {upper:.2f}]")  # [8.0, 12.0]

# Get core (1-level)
lower, upper = y0.alpha_cut(alpha=1)
print(f"Core: [{lower:.2f}, {upper:.2f}]")  # [10.0, 10.0]
```

---

## Fuzzy ODE Solver

Solve ordinary differential equations with fuzzy initial conditions using the **α-level method**.

### How It Works

1. **Choose α-levels**: 0, 0.25, 0.5, 0.75, 1.0
2. **For each α:**
   - Extract interval [y_lower(α), y_upper(α)]
   - Solve ODE twice: once with y_lower, once with y_upper
3. **Reconstruct fuzzy solution** from intervals at each time point

---

### Example 1: Logistic Growth

Model population growth with uncertain initial population:

$$\frac{dy}{dt} = r \cdot y \cdot \left(1 - \frac{y}{K}\right)$$

```python
from fuzzy_systems.dynamics import FuzzyODESolver, FuzzyNumber
import numpy as np
import matplotlib.pyplot as plt

# Define logistic equation
def logistic(t, y, r, K):
    """
    y: list of fuzzy numbers [y(t)]
    Returns: dy/dt
    """
    return r * y[0] * (1 - y[0] / K)

# Fuzzy initial condition: "approximately 10"
y0 = FuzzyNumber.triangular(center=10, spread=2)

# Solve
solver = FuzzyODESolver(
    f=logistic,
    t_span=(0, 20),
    y0_fuzzy=[y0],
    params={'r': 0.3, 'K': 100},
    n_alpha=11  # 11 α-levels
)

solution = solver.solve()

# Plot
solver.plot()
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('Logistic Growth with Fuzzy Initial Condition')
plt.show()
```

**Interpretation:**
- Dark region: High confidence (α = 1.0)
- Light region: Low confidence (α = 0.0)
- Uncertainty **increases** over time (characteristic of fuzzy ODEs)

---

### Example 2: Predator-Prey (Lotka-Volterra)

Two-dimensional system with fuzzy initial conditions:

$$\begin{aligned}
\frac{dx}{dt} &= \alpha x - \beta x y \\
\frac{dy}{dt} &= \delta x y - \gamma y
\end{aligned}$$

```python
def predator_prey(t, y, alpha, beta, delta, gamma):
    """
    y[0]: prey population
    y[1]: predator population
    """
    x, y_pred = y
    dx_dt = alpha * x - beta * x * y_pred
    dy_dt = delta * x * y_pred - gamma * y_pred
    return [dx_dt, dy_dt]

# Fuzzy initial conditions
x0 = FuzzyNumber.triangular(center=40, spread=5)   # Prey
y0 = FuzzyNumber.triangular(center=9, spread=1)    # Predator

# Solve
solver = FuzzyODESolver(
    f=predator_prey,
    t_span=(0, 30),
    y0_fuzzy=[x0, y0],
    params={'alpha': 0.1, 'beta': 0.02, 'delta': 0.01, 'gamma': 0.1},
    n_alpha=11
)

solution = solver.solve()

# Plot both populations
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

solver.plot(variable_index=0, ax=axes[0])
axes[0].set_ylabel('Prey Population')
axes[0].set_title('Prey')

solver.plot(variable_index=1, ax=axes[1])
axes[1].set_ylabel('Predator Population')
axes[1].set_title('Predator')

plt.tight_layout()
plt.show()

# Phase portrait
solver.plot_phase(variable_indices=(0, 1))
plt.xlabel('Prey')
plt.ylabel('Predator')
plt.title('Phase Portrait')
plt.show()
```

---

### Solver Parameters

```python
solver = FuzzyODESolver(
    f=equation,
    t_span=(t_start, t_end),
    y0_fuzzy=[y0_1, y0_2, ...],  # List of FuzzyNumber
    params={...},                 # Dictionary of crisp parameters
    n_alpha=11,                   # Number of α-levels (odd number)
    method='RK45',                # Integration method
    rtol=1e-6,                    # Relative tolerance
    atol=1e-9                     # Absolute tolerance
)
```

**Integration methods:**
- `'RK45'`: Runge-Kutta 4(5) (default, good balance)
- `'RK23'`: Runge-Kutta 2(3) (faster, less accurate)
- `'DOP853'`: Runge-Kutta 8 (slower, very accurate)
- `'BDF'`: Backward differentiation (for stiff problems)

**Number of α-levels:**
- 5-7: Fast, coarse uncertainty bands
- 11-21: Good balance (recommended)
- 51+: Slow, smooth bands

---

### Accessing Solutions

```python
solution = solver.solve()

# Time points
t = solution['t']

# Fuzzy solution at each time point
y_fuzzy = solution['y']  # List of lists of FuzzyNumber

# Get specific α-level trajectory
alpha = 0.5
y_lower, y_upper = solver.get_alpha_trajectory(alpha, variable_index=0)

# Plot custom α-levels
fig, ax = plt.subplots()
for alpha in [0, 0.25, 0.5, 0.75, 1.0]:
    lower, upper = solver.get_alpha_trajectory(alpha, 0)
    ax.fill_between(t, lower, upper, alpha=0.3, label=f'α={alpha}')
ax.legend()
ax.set_xlabel('Time')
ax.set_ylabel('y(t)')
plt.show()
```

---

### Fuzzy Parameters

Parameters can also be fuzzy:

```python
# Fuzzy growth rate: "approximately 0.3"
r_fuzzy = FuzzyNumber.triangular(center=0.3, spread=0.05)

# Convert to crisp samples for Monte Carlo
n_samples = 100
r_samples = [r_fuzzy.sample() for _ in range(n_samples)]

# Solve for each sample
trajectories = []
for r_val in r_samples:
    solver = FuzzyODESolver(
        f=logistic,
        t_span=(0, 20),
        y0_fuzzy=[y0],
        params={'r': r_val, 'K': 100},
        n_alpha=5  # Fewer α-levels for speed
    )
    solution = solver.solve()
    trajectories.append(solution['y'][0])  # First variable

# Plot envelope
import numpy as np
t = solution['t']
y_array = np.array([traj for traj in trajectories])
y_mean = y_array.mean(axis=0)
y_std = y_array.std(axis=0)

plt.fill_between(t, y_mean - 2*y_std, y_mean + 2*y_std, alpha=0.3)
plt.plot(t, y_mean, 'r-', linewidth=2)
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('Logistic Growth with Fuzzy Parameter')
plt.show()
```

---

## p-Fuzzy Systems (Discrete-Time)

**p-Fuzzy systems** use fuzzy rules to define dynamical systems.

### Basic Concept

Instead of equations, use **linguistic rules**:

```
IF x is LOW THEN x_next is MEDIUM
IF x is MEDIUM THEN x_next is HIGH
IF x is HIGH THEN x_next is LOW
```

---

### Example: Population Dynamics

```python
from fuzzy_systems.dynamics import PFuzzyDiscrete
import numpy as np
import matplotlib.pyplot as plt

# Create discrete p-fuzzy system
system = PFuzzyDiscrete(n_variables=1)

# Add input (current state)
system.add_input('x', (0, 100))
system.add_term('x', 'low', 'trapezoidal', (0, 0, 20, 40))
system.add_term('x', 'medium', 'triangular', (30, 50, 70))
system.add_term('x', 'high', 'trapezoidal', (60, 80, 100, 100))

# Add output (next state)
system.add_output('x_next', (0, 100))
system.add_term('x_next', 'low', 'trapezoidal', (0, 0, 20, 40))
system.add_term('x_next', 'medium', 'triangular', (30, 50, 70))
system.add_term('x_next', 'high', 'trapezoidal', (60, 80, 100, 100))

# Add rules (discrete map)
system.add_rules([
    {'x': 'low', 'x_next': 'medium'},      # Low pop → grows to medium
    {'x': 'medium', 'x_next': 'high'},     # Medium → grows to high
    {'x': 'high', 'x_next': 'low'}         # High → collapses to low
])

# Simulate
x0 = 10  # Initial population
trajectory = system.simulate(x0=x0, n_steps=20)

# Plot
plt.plot(trajectory['x'], 'o-')
plt.xlabel('Time Step')
plt.ylabel('Population')
plt.title('Discrete p-Fuzzy Population Dynamics')
plt.grid(True)
plt.show()

# Phase diagram (x_t vs x_{t+1})
plt.plot(trajectory['x'][:-1], trajectory['x'][1:], 'o-')
plt.plot([0, 100], [0, 100], 'k--', alpha=0.3)  # Identity line
plt.xlabel('x(t)')
plt.ylabel('x(t+1)')
plt.title('Discrete Map')
plt.grid(True)
plt.show()
```

---

### Example: Predator-Prey (Discrete)

Two-variable system:

```python
system = PFuzzyDiscrete(n_variables=2)

# Prey (x)
system.add_input('x', (0, 100))
system.add_term('x', 'low', 'triangular', (0, 0, 50))
system.add_term('x', 'high', 'triangular', (50, 100, 100))

system.add_output('x_next', (0, 100))
system.add_term('x_next', 'low', 'triangular', (0, 0, 50))
system.add_term('x_next', 'medium', 'triangular', (25, 50, 75))
system.add_term('x_next', 'high', 'triangular', (50, 100, 100))

# Predator (y)
system.add_input('y', (0, 50))
system.add_term('y', 'low', 'triangular', (0, 0, 25))
system.add_term('y', 'high', 'triangular', (25, 50, 50))

system.add_output('y_next', (0, 50))
system.add_term('y_next', 'low', 'triangular', (0, 0, 25))
system.add_term('y_next', 'medium', 'triangular', (12.5, 25, 37.5))
system.add_term('y_next', 'high', 'triangular', (25, 50, 50))

# Rules
system.add_rules([
    # When prey low, predator low → both grow
    {'x': 'low', 'y': 'low', 'x_next': 'medium', 'y_next': 'low'},

    # When prey low, predator high → prey recovers, predator declines
    {'x': 'low', 'y': 'high', 'x_next': 'medium', 'y_next': 'medium'},

    # When prey high, predator low → prey stays high, predator grows
    {'x': 'high', 'y': 'low', 'x_next': 'high', 'y_next': 'medium'},

    # When prey high, predator high → prey declines, predator stays high
    {'x': 'high', 'y': 'high', 'x_next': 'medium', 'y_next': 'high'},
])

# Simulate
trajectory = system.simulate(x0={'x': 40, 'y': 9}, n_steps=50)

# Plot time series
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

axes[0].plot(trajectory['x'], 'b-o', label='Prey')
axes[0].set_ylabel('Prey')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(trajectory['y'], 'r-o', label='Predator')
axes[1].set_ylabel('Predator')
axes[1].set_xlabel('Time Step')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

# Phase portrait
plt.plot(trajectory['x'], trajectory['y'], 'o-')
plt.plot(trajectory['x'][0], trajectory['y'][0], 'go', markersize=10, label='Start')
plt.plot(trajectory['x'][-1], trajectory['y'][-1], 'ro', markersize=10, label='End')
plt.xlabel('Prey')
plt.ylabel('Predator')
plt.title('Phase Portrait')
plt.legend()
plt.grid(True)
plt.show()
```

---

### Multiple Initial Conditions

Explore different starting points:

```python
initial_conditions = [
    {'x': 10, 'y': 5},
    {'x': 30, 'y': 15},
    {'x': 70, 'y': 30},
    {'x': 90, 'y': 45}
]

plt.figure(figsize=(10, 6))
for x0, y0 in initial_conditions:
    traj = system.simulate(x0={'x': x0, 'y': y0}, n_steps=30)
    plt.plot(traj['x'], traj['y'], 'o-', alpha=0.6)
    plt.plot(x0, y0, 'o', markersize=10)

plt.xlabel('Prey')
plt.ylabel('Predator')
plt.title('Phase Portrait from Multiple Initial Conditions')
plt.grid(True)
plt.show()
```

---

## p-Fuzzy Systems (Continuous-Time)

Continuous-time p-fuzzy systems use rules to define **derivatives**.

### Example: Logistic Growth

```python
from fuzzy_systems.dynamics import PFuzzyContinuous
import numpy as np
import matplotlib.pyplot as plt

# Create continuous p-fuzzy system
system = PFuzzyContinuous(n_variables=1)

# Add input (current population)
system.add_input('x', (0, 100))
system.add_term('x', 'low', 'trapezoidal', (0, 0, 20, 40))
system.add_term('x', 'medium', 'triangular', (30, 50, 70))
system.add_term('x', 'high', 'trapezoidal', (60, 80, 100, 100))

# Add output (growth rate dx/dt)
system.add_output('dx_dt', (-10, 10))
system.add_term('dx_dt', 'negative', 'trapezoidal', (-10, -10, -5, 0))
system.add_term('dx_dt', 'zero', 'triangular', (-2, 0, 2))
system.add_term('dx_dt', 'positive', 'trapezoidal', (0, 5, 10, 10))

# Add rules
system.add_rules([
    {'x': 'low', 'dx_dt': 'positive'},      # Low pop → grows
    {'x': 'medium', 'dx_dt': 'positive'},   # Medium → still grows
    {'x': 'high', 'dx_dt': 'negative'}      # High → declines (carrying capacity)
])

# Simulate
t_span = (0, 20)
x0 = 10
solution = system.simulate(x0=x0, t_span=t_span, method='RK45')

# Plot
plt.plot(solution['t'], solution['x'])
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('Continuous p-Fuzzy Logistic Growth')
plt.grid(True)
plt.show()
```

---

### Example: Predator-Prey (Continuous)

```python
system = PFuzzyContinuous(n_variables=2)

# Prey (x)
system.add_input('x', (0, 100))
system.add_term('x', 'low', 'triangular', (0, 0, 50))
system.add_term('x', 'medium', 'triangular', (25, 50, 75))
system.add_term('x', 'high', 'triangular', (50, 100, 100))

system.add_output('dx_dt', (-20, 20))
system.add_term('dx_dt', 'decrease', 'triangular', (-20, -20, 0))
system.add_term('dx_dt', 'stable', 'triangular', (-5, 0, 5))
system.add_term('dx_dt', 'increase', 'triangular', (0, 20, 20))

# Predator (y)
system.add_input('y', (0, 50))
system.add_term('y', 'low', 'triangular', (0, 0, 25))
system.add_term('y', 'medium', 'triangular', (12.5, 25, 37.5))
system.add_term('y', 'high', 'triangular', (25, 50, 50))

system.add_output('dy_dt', (-10, 10))
system.add_term('dy_dt', 'decrease', 'triangular', (-10, -10, 0))
system.add_term('dy_dt', 'stable', 'triangular', (-2, 0, 2))
system.add_term('dy_dt', 'increase', 'triangular', (0, 10, 10))

# Rules based on ecology
system.add_rules([
    # Low prey → prey can grow, predator declines
    {'x': 'low', 'y': 'low', 'dx_dt': 'increase', 'dy_dt': 'stable'},
    {'x': 'low', 'y': 'high', 'dx_dt': 'decrease', 'dy_dt': 'decrease'},

    # Medium prey → balanced
    {'x': 'medium', 'y': 'low', 'dx_dt': 'increase', 'dy_dt': 'increase'},
    {'x': 'medium', 'y': 'medium', 'dx_dt': 'stable', 'dy_dt': 'stable'},
    {'x': 'medium', 'y': 'high', 'dx_dt': 'decrease', 'dy_dt': 'increase'},

    # High prey → prey declines, predator grows
    {'x': 'high', 'y': 'low', 'dx_dt': 'stable', 'dy_dt': 'increase'},
    {'x': 'high', 'y': 'high', 'dx_dt': 'decrease', 'dy_dt': 'stable'},
])

# Simulate
solution = system.simulate(
    x0={'x': 40, 'y': 9},
    t_span=(0, 50),
    method='RK45'
)

# Plot time series
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

axes[0].plot(solution['t'], solution['x'], 'b-')
axes[0].set_ylabel('Prey')
axes[0].grid(True)

axes[1].plot(solution['t'], solution['y'], 'r-')
axes[1].set_ylabel('Predator')
axes[1].set_xlabel('Time')
axes[1].grid(True)

plt.tight_layout()
plt.show()

# Phase portrait
plt.plot(solution['x'], solution['y'])
plt.plot(solution['x'][0], solution['y'][0], 'go', markersize=10, label='Start')
plt.xlabel('Prey')
plt.ylabel('Predator')
plt.title('Continuous p-Fuzzy Phase Portrait')
plt.legend()
plt.grid(True)
plt.show()
```

---

## Comparing Methods

Let's compare all three methods on the same problem (logistic growth):

```python
import numpy as np
import matplotlib.pyplot as plt
from fuzzy_systems.dynamics import (
    FuzzyODESolver, FuzzyNumber,
    PFuzzyDiscrete, PFuzzyContinuous
)

# Parameters
r, K = 0.3, 100
y0_value = 10
t_max = 20

# Method 1: Fuzzy ODE
def logistic_ode(t, y, r, K):
    return r * y[0] * (1 - y[0] / K)

y0_fuzzy = FuzzyNumber.triangular(center=y0_value, spread=2)
solver_ode = FuzzyODESolver(
    f=logistic_ode,
    t_span=(0, t_max),
    y0_fuzzy=[y0_fuzzy],
    params={'r': r, 'K': K},
    n_alpha=11
)
solution_ode = solver_ode.solve()

# Method 2: p-Fuzzy Discrete
system_discrete = PFuzzyDiscrete(n_variables=1)
system_discrete.add_input('x', (0, K))
system_discrete.add_term('x', 'low', 'trapezoidal', (0, 0, K*0.3, K*0.5))
system_discrete.add_term('x', 'medium', 'triangular', (K*0.4, K*0.6, K*0.8))
system_discrete.add_term('x', 'high', 'trapezoidal', (K*0.7, K*0.9, K, K))

system_discrete.add_output('x_next', (0, K))
system_discrete.add_term('x_next', 'low', 'trapezoidal', (0, 0, K*0.3, K*0.5))
system_discrete.add_term('x_next', 'medium', 'triangular', (K*0.4, K*0.6, K*0.8))
system_discrete.add_term('x_next', 'high', 'trapezoidal', (K*0.7, K*0.9, K, K))

# Approximating continuous dynamics with discrete map
system_discrete.add_rules([
    {'x': 'low', 'x_next': 'medium'},
    {'x': 'medium', 'x_next': 'high'},
    {'x': 'high', 'x_next': 'high'}
])

traj_discrete = system_discrete.simulate(x0=y0_value, n_steps=int(t_max))

# Method 3: p-Fuzzy Continuous
system_continuous = PFuzzyContinuous(n_variables=1)
system_continuous.add_input('x', (0, K))
system_continuous.add_term('x', 'low', 'trapezoidal', (0, 0, K*0.3, K*0.5))
system_continuous.add_term('x', 'medium', 'triangular', (K*0.4, K*0.6, K*0.8))
system_continuous.add_term('x', 'high', 'trapezoidal', (K*0.7, K*0.9, K, K))

system_continuous.add_output('dx_dt', (-K, K))
system_continuous.add_term('dx_dt', 'negative', 'triangular', (-K, -K, 0))
system_continuous.add_term('dx_dt', 'zero', 'triangular', (-K*0.1, 0, K*0.1))
system_continuous.add_term('dx_dt', 'positive', 'triangular', (0, K, K))

system_continuous.add_rules([
    {'x': 'low', 'dx_dt': 'positive'},
    {'x': 'medium', 'dx_dt': 'positive'},
    {'x': 'high', 'dx_dt': 'zero'}
])

solution_continuous = system_continuous.simulate(
    x0=y0_value,
    t_span=(0, t_max),
    method='RK45'
)

# Compare
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Fuzzy ODE
solver_ode.plot(ax=axes[0])
axes[0].set_title('Fuzzy ODE (α-levels)')
axes[0].set_ylabel('Population')

# p-Fuzzy Discrete
axes[1].plot(range(len(traj_discrete['x'])), traj_discrete['x'], 'o-')
axes[1].set_title('p-Fuzzy Discrete')
axes[1].set_xlabel('Time Step')
axes[1].grid(True)

# p-Fuzzy Continuous
axes[2].plot(solution_continuous['t'], solution_continuous['x'])
axes[2].set_title('p-Fuzzy Continuous')
axes[2].set_xlabel('Time')
axes[2].grid(True)

plt.tight_layout()
plt.show()
```

**Observations:**
- **Fuzzy ODE**: Shows uncertainty bands growing over time
- **p-Fuzzy Discrete**: Step-wise evolution, good for discrete events
- **p-Fuzzy Continuous**: Smooth trajectories, rule-based dynamics

---

## Design Guidelines

### 1. Choosing α-levels

**Trade-off: accuracy vs speed**

```python
# Fast (3-5 levels)
solver = FuzzyODESolver(..., n_alpha=5)

# Balanced (11-21 levels)
solver = FuzzyODESolver(..., n_alpha=11)  # Recommended

# Smooth (51+ levels)
solver = FuzzyODESolver(..., n_alpha=51)  # Slow
```

**Use fewer α-levels when:**
- Prototyping or exploring
- Computational budget is limited
- Rough uncertainty estimates are sufficient

**Use more α-levels when:**
- Creating publication-quality figures
- Precise uncertainty quantification needed
- Computational resources available

---

### 2. Fuzzy Number Shapes

**Triangular vs Trapezoidal vs Gaussian:**

```python
# Triangular: "approximately X"
y0 = FuzzyNumber.triangular(center=10, spread=2)

# Trapezoidal: "between A and B"
y0 = FuzzyNumber.trapezoidal(a=8, b=9, c=11, d=12)

# Gaussian: "normally distributed"
y0 = FuzzyNumber.gaussian(center=10, sigma=1)
```

**Guidelines:**
- Use **triangular** for symmetric uncertainty
- Use **trapezoidal** for ranges with plateaus
- Use **gaussian** for measurement errors

---

### 3. Rule Design for p-Fuzzy

**Principle: Rules should reflect domain knowledge**

**Good rules (ecologically sound):**
```python
system.add_rules([
    {'prey': 'low', 'predator': 'high', 'dprey_dt': 'decrease'},
    {'prey': 'high', 'predator': 'low', 'dprey_dt': 'increase'}
])
```

**Bad rules (contradictory):**
```python
system.add_rules([
    {'prey': 'low', 'dprey_dt': 'increase'},
    {'prey': 'low', 'dprey_dt': 'decrease'}  # Conflict!
])
```

**Check rule coverage:**
```python
# Visualize rule activation
system.plot_rule_matrix()  # For 2D systems
```

---

### 4. Integration Method Selection

| Method | Speed | Accuracy | Best For |
|--------|-------|----------|----------|
| `'RK23'` | ⚡⚡⚡ | ⭐⭐ | Fast prototyping, smooth problems |
| `'RK45'` | ⚡⚡ | ⭐⭐⭐ | **Default**, most problems |
| `'DOP853'` | ⚡ | ⭐⭐⭐⭐⭐ | High precision needed |
| `'BDF'` | ⚡⚡ | ⭐⭐⭐ | Stiff equations |

**Stiff equations?** Try `'BDF'` or `'Radau'`:
```python
solver = FuzzyODESolver(..., method='BDF')
```

---

## Advanced Topics

### Sensitivity Analysis

How sensitive is the solution to initial conditions?

```python
from fuzzy_systems.dynamics import FuzzyODESolver, FuzzyNumber
import numpy as np
import matplotlib.pyplot as plt

def logistic(t, y, r, K):
    return r * y[0] * (1 - y[0] / K)

# Test different spreads
spreads = [1, 2, 5, 10]
fig, ax = plt.subplots(figsize=(10, 6))

for spread in spreads:
    y0 = FuzzyNumber.triangular(center=10, spread=spread)
    solver = FuzzyODESolver(
        f=logistic,
        t_span=(0, 20),
        y0_fuzzy=[y0],
        params={'r': 0.3, 'K': 100},
        n_alpha=11
    )
    solution = solver.solve()

    # Plot envelope
    t = solution['t']
    y_lower, y_upper = solver.get_alpha_trajectory(alpha=0, variable_index=0)
    ax.fill_between(t, y_lower, y_upper, alpha=0.3, label=f'spread={spread}')

ax.set_xlabel('Time')
ax.set_ylabel('Population')
ax.set_title('Sensitivity to Initial Uncertainty')
ax.legend()
ax.grid(True)
plt.show()
```

---

### Lyapunov Exponents (p-Fuzzy Discrete)

Measure chaos in discrete systems:

```python
def lyapunov_exponent(system, x0, n_steps=1000, epsilon=1e-8):
    """Estimate largest Lyapunov exponent."""
    traj1 = system.simulate(x0=x0, n_steps=n_steps)['x']

    # Perturbed trajectory
    x0_perturbed = x0 + epsilon
    traj2 = system.simulate(x0=x0_perturbed, n_steps=n_steps)['x']

    # Compute divergence
    divergence = [np.log(abs(traj2[i] - traj1[i]) / epsilon)
                  for i in range(1, n_steps)]

    lyapunov = np.mean(divergence)
    return lyapunov

# Test
system = PFuzzyDiscrete(n_variables=1)
# ... configure system ...

lambda_max = lyapunov_exponent(system, x0=10)
print(f"Lyapunov exponent: {lambda_max:.4f}")

if lambda_max > 0:
    print("System is chaotic!")
elif lambda_max < 0:
    print("System is stable.")
else:
    print("System is at the edge of chaos.")
```

---

### Bifurcation Diagrams (p-Fuzzy)

Explore parameter space:

```python
# Vary a parameter and observe long-term behavior
parameters = np.linspace(0.1, 0.5, 50)
final_states = []

for param in parameters:
    # Modify rule or parameter
    system = create_system_with_param(param)

    # Simulate and discard transient
    traj = system.simulate(x0=10, n_steps=500)
    final_states.append(traj['x'][-100:])  # Last 100 steps

# Plot bifurcation diagram
for i, param in enumerate(parameters):
    plt.plot([param]*len(final_states[i]), final_states[i],
             'k,', alpha=0.5)

plt.xlabel('Parameter')
plt.ylabel('Long-term Population')
plt.title('Bifurcation Diagram')
plt.show()
```

---

## Troubleshooting

### Problem: Fuzzy ODE solution "explodes"

**Symptoms:**
- Solution bands become extremely wide
- Values go to infinity

**Causes:**
- Unstable dynamics
- Tolerance too loose

**Solutions:**

```python
# Tighten tolerances
solver = FuzzyODESolver(..., rtol=1e-9, atol=1e-12)

# Use more stable integrator
solver = FuzzyODESolver(..., method='BDF')

# Check classical solution first
def check_stability(f, t_span, y0, params):
    from scipy.integrate import solve_ivp
    sol = solve_ivp(lambda t, y: f(t, [y[0]], **params),
                    t_span, [y0], method='RK45')
    plt.plot(sol.t, sol.y[0])
    plt.show()

check_stability(logistic, (0, 20), 10, {'r': 0.3, 'K': 100})
```

---

### Problem: p-Fuzzy system has no output

**Symptoms:**
- `simulate()` returns NaN or constant values
- No rules are activating

**Solutions:**

```python
# Debug: check fuzzification
x_test = 50
input_degrees = system.inputs['x'].fuzzify(x_test)
print(f"Input memberships at x={x_test}: {input_degrees}")
# Should be non-zero for at least one term

# Debug: check rule activations
details = system.evaluate_detailed(x=x_test)
print(f"Rule activations: {details['rule_activations']}")
# Should have at least one non-zero activation

# Fix: adjust term coverage
system.plot_variables()  # Visual check
```

---

### Problem: Discrete p-Fuzzy is stuck in a loop

**Symptoms:**
- Trajectory oscillates between same values
- Phase portrait shows closed loop

**Explanation:**
- This may be intentional (limit cycle)
- Or rules create an attractor

**To verify:**
```python
# Test multiple initial conditions
for x0 in [10, 30, 50, 70, 90]:
    traj = system.simulate(x0=x0, n_steps=50)
    plt.plot(traj['x'], alpha=0.6)
plt.xlabel('Time Step')
plt.ylabel('x')
plt.title('Trajectories from Different ICs')
plt.show()

# If all converge to same cycle → it's an attractor
```

---

### Problem: Continuous p-Fuzzy doesn't reach equilibrium

**Causes:**
- Rules don't allow convergence
- No "zero growth" rules

**Solutions:**

```python
# Add equilibrium rules
system.add_rules([
    {'x': 'medium', 'dx_dt': 'zero'},  # Equilibrium at medium
])

# Or increase tolerance
solution = system.simulate(..., method='RK45', rtol=1e-3)
```

---

## Next Steps

- **[Fundamentals](fundamentals.md)**: Review fuzzy logic basics
- **[API Reference: Dynamics](../api_reference/dynamics.md)**: Complete method documentation
- **[Examples: Dynamics](../examples/gallery.md#dynamics)**: Interactive notebooks

---

## Further Reading

- **Puri, M. L., & Ralescu, D. A. (1983)**: "Differentials of fuzzy functions". *Journal of Mathematical Analysis and Applications*, 91(2), 552-558.
- **Buckley, J. J., & Feuring, T. (2000)**: "Fuzzy differential equations". *Fuzzy Sets and Systems*, 110(1), 43-54.
- **Barros, L. C., Bassanezi, R. C., & Lodwick, W. A. (2017)**: *A First Course in Fuzzy Logic, Fuzzy Dynamical Systems, and Biomathematics*. Springer.
- **Jafelice, R. M., et al. (2015)**: "Fuzzy parameter in a prey-predator model". *Nonlinear Analysis: Real World Applications*, 16, 59-71.
