# Dynamics API Reference

The `fuzzy_systems.dynamics` module provides tools for dynamic systems with fuzzy uncertainty:

- **FuzzyODE**: Solve ODEs with fuzzy parameters/initial conditions (α-level method)
- **PFuzzyDiscrete**: Discrete dynamical systems with fuzzy rule-based evolution
- **PFuzzyContinuous**: Continuous dynamical systems with fuzzy rule-based evolution

**Reference:**
Barros, L. C., Bassanezi, R. C., & Lodwick, W. A. (2017). *A First Course in Fuzzy Logic, Fuzzy Dynamical Systems, and Biomathematics*.

---

## Fuzzy ODEs

### FuzzyNumber

Represent fuzzy numbers for use as initial conditions or parameters.

#### Class Methods

##### `.triangular(center, spread, name=None)`

Create a triangular fuzzy number.

**Parameters:**
- `center` (float): Center (peak) of the triangle
- `spread` (float): Half-width at base
- `name` (str, optional): Name of the fuzzy number

**Returns:** `FuzzyNumber`

**Example:**
```python
from fuzzy_systems.dynamics import FuzzyNumber

# Triangular: μ(x) peaks at 10, base from 8 to 12
y0_fuzzy = FuzzyNumber.triangular(center=10, spread=2)
```

---

##### `.trapezoidal(a, b, c, d, name=None)`

Create a trapezoidal fuzzy number.

**Parameters:**
- `a`, `b`, `c`, `d` (float): Trapezoidal parameters (a ≤ b ≤ c ≤ d)
- `name` (str, optional): Name of the fuzzy number

**Returns:** `FuzzyNumber`

---

##### `.gaussian(mean, sigma, name=None)`

Create a Gaussian fuzzy number.

**Parameters:**
- `mean` (float): Mean (center)
- `sigma` (float): Standard deviation
- `name` (str, optional): Name

**Returns:** `FuzzyNumber`

---

### FuzzyODESolver

Solve ordinary differential equations with fuzzy uncertainty.

#### Constructor

```python
FuzzyODESolver(ode_func, t_span, y0_fuzzy=None, params=None,
               alpha_levels=None, method='RK45', **options)
```

**Parameters:**

- `ode_func` (callable): ODE function `f(t, y, *params) -> dydt`
- `t_span` (tuple): Time interval `(t0, tf)`
- `y0_fuzzy` (list, optional): List of `FuzzyNumber` objects for initial conditions
- `params` (dict, optional): Crisp or fuzzy parameters: `{name: value_or_FuzzyNumber}`
- `alpha_levels` (list, optional): α-cut levels (default: `[0, 0.25, 0.5, 0.75, 1.0]`)
- `method` (str): Integration method: `'RK45'`, `'RK23'`, `'DOP853'`, `'Radau'`, `'BDF'`, `'LSODA'` (default: `'RK45'`)
- `**options`: Additional options for `scipy.integrate.solve_ivp`

**Example:**
```python
from fuzzy_systems.dynamics import FuzzyODESolver, FuzzyNumber
import numpy as np

# Define ODE: dy/dt = r*y*(1 - y/K)  (Logistic growth)
def logistic(t, y, r, K):
    return r * y[0] * (1 - y[0] / K)

# Fuzzy initial condition
y0 = FuzzyNumber.triangular(center=10, spread=2)

# Fuzzy parameters
r_fuzzy = FuzzyNumber.triangular(center=1.0, spread=0.2)
K_fuzzy = FuzzyNumber.triangular(center=100, spread=10)

# Create solver
solver = FuzzyODESolver(
    ode_func=logistic,
    t_span=(0, 20),
    y0_fuzzy=[y0],
    params={'r': r_fuzzy, 'K': K_fuzzy},
    alpha_levels=[0, 0.5, 1.0]
)
```

---

#### Methods

##### `.solve(n_points=100, parallel=True, n_jobs=-1)`

Solve the fuzzy ODE using α-level method.

**Parameters:**
- `n_points` (int): Number of time points (default: `100`)
- `parallel` (bool): Use parallel processing (default: `True`)
- `n_jobs` (int): Number of parallel jobs. `-1` uses all CPUs (default: `-1`)

**Returns:** `FuzzySolution` - Solution object

**Example:**
```python
solution = solver.solve(n_points=200, parallel=True)
```

---

##### `.plot_envelope(variables=None, alpha_colors=None, figsize=(12, 6), show=True)`

Plot fuzzy envelope showing uncertainty bands.

**Parameters:**
- `variables` (list, optional): Variable indices to plot. If None, plots all
- `alpha_colors` (dict, optional): Custom colors for α-levels: `{alpha: color}`
- `figsize` (tuple): Figure size (default: `(12, 6)`)
- `show` (bool): Whether to call `plt.show()` (default: `True`)

**Returns:** `tuple` - `(fig, axes)` matplotlib objects

**Example:**
```python
solver.plot_envelope(
    variables=[0],
    alpha_colors={0: 'lightblue', 0.5: 'blue', 1.0: 'darkblue'}
)
```

---

### FuzzySolution

Solution object returned by `FuzzyODESolver.solve()`.

#### Attributes

- `t` (ndarray): Time points
- `alpha_levels` (list): α-cut levels used
- `envelopes` (dict): Fuzzy envelopes: `{alpha: {'lower': array, 'upper': array}}`

#### Methods

##### `.plot(variables=None, **kwargs)`

Plot the fuzzy solution.

**Parameters:**
- `variables` (list, optional): Variables to plot
- `**kwargs`: Additional plotting options

---

### Complete Example: Fuzzy Logistic Growth

```python
import numpy as np
import matplotlib.pyplot as plt
from fuzzy_systems.dynamics import FuzzyODESolver, FuzzyNumber

# Define ODE: dy/dt = r*y*(1 - y/K)
def logistic(t, y, r, K):
    """Logistic growth model."""
    return r * y[0] * (1 - y[0] / K)

# Fuzzy initial condition: population around 10 ± 2
y0 = FuzzyNumber.triangular(center=10, spread=2)

# Fuzzy parameters
r = FuzzyNumber.triangular(center=0.5, spread=0.1)  # Growth rate
K = FuzzyNumber.triangular(center=100, spread=10)   # Carrying capacity

# Solve fuzzy ODE
solver = FuzzyODESolver(
    ode_func=logistic,
    t_span=(0, 30),
    y0_fuzzy=[y0],
    params={'r': r, 'K': K},
    alpha_levels=[0, 0.25, 0.5, 0.75, 1.0],
    method='RK45'
)

solution = solver.solve(n_points=200)

# Plot
solver.plot_envelope(
    variables=[0],
    figsize=(12, 6),
    alpha_colors={
        0: 'lightblue',
        0.5: 'blue',
        1.0: 'darkblue'
    }
)
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('Fuzzy Logistic Growth with Uncertain Parameters')
plt.show()
```

---

### Complete Example: Fuzzy Holling-Tanner (Predator-Prey)

```python
from fuzzy_systems.dynamics import FuzzyODESolver, FuzzyNumber

# Holling-Tanner predator-prey model
def holling_tanner(t, y, r, K, a, b, c, d):
    """
    Predator-prey with Holling Type II functional response.

    y[0] = prey (x)
    y[1] = predator (z)
    """
    x, z = y
    dx = r * x * (1 - x/K) - (a*x*z)/(b + x)
    dz = c * z * (1 - d*z/x) if x > 0 else 0
    return [dx, dz]

# Initial conditions (fuzzy)
x0 = FuzzyNumber.triangular(center=40, spread=5)
z0 = FuzzyNumber.triangular(center=15, spread=3)

# Parameters (some fuzzy, some crisp)
params = {
    'r': FuzzyNumber.triangular(1.0, 0.1),  # Fuzzy
    'K': 100,                                # Crisp
    'a': 1.0,
    'b': 10,
    'c': 0.5,
    'd': 0.1
}

# Solve
solver = FuzzyODESolver(
    ode_func=holling_tanner,
    t_span=(0, 100),
    y0_fuzzy=[x0, z0],
    params=params,
    alpha_levels=[0, 0.5, 1.0]
)

solution = solver.solve()

# Plot both variables
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

solver.plot_envelope(variables=[0], show=False)
ax1.set_title('Prey Population (x)')
ax1.set_xlabel('Time')
ax1.set_ylabel('Population')

solver.plot_envelope(variables=[1], show=False)
ax2.set_title('Predator Population (z)')
ax2.set_xlabel('Time')
ax2.set_ylabel('Population')

plt.tight_layout()
plt.show()

# Phase space plot
# (Would need to extract and plot lower/upper bounds in x-z plane)
```

---

## p-Fuzzy Systems

Systems where evolution is defined by fuzzy rules instead of equations.

### PFuzzyDiscrete

Discrete dynamical system with fuzzy rule-based evolution.

**Discrete evolution:**
- **Absolute mode**: x_{n+1} = x_n + f(x_n)
- **Relative mode**: x_{n+1} = x_n * (1 + f(x_n))

#### Constructor

```python
PFuzzyDiscrete(fis, mode='absolute', state_vars=None)
```

**Parameters:**

- `fis` (MamdaniSystem | SugenoSystem): Fuzzy inference system
- `mode` (str): Evolution mode: `'absolute'` or `'relative'` (default: `'absolute'`)
- `state_vars` (list, optional): State variable names. If None, uses all FIS inputs

**Example:**
```python
from fuzzy_systems import MamdaniSystem
from fuzzy_systems.dynamics import PFuzzyDiscrete

# Create FIS with rules
fis = MamdaniSystem()
fis.add_input('prey', (0, 100))
fis.add_input('predator', (0, 100))
fis.add_output('var_prey', (-5, 5))
fis.add_output('var_predator', (-5, 5))

# Add terms and rules...
# (See examples below)

# Create p-fuzzy system
pfuzzy = PFuzzyDiscrete(
    fis=fis,
    mode='absolute',
    state_vars=['prey', 'predator']
)
```

---

#### Methods

##### `.simulate(x0, n_steps, return_time=True)`

Simulate the discrete system.

**Parameters:**
- `x0` (dict | ndarray): Initial conditions: `{var_name: value}` or array
- `n_steps` (int): Number of time steps
- `return_time` (bool): If True, returns `(time, trajectory)`. Otherwise, only `trajectory` (default: `True`)

**Returns:** `tuple | ndarray` - Time array and trajectory matrix, or just trajectory

**Example:**
```python
# Using dictionary
time, trajectory = pfuzzy.simulate(
    x0={'prey': 50, 'predator': 40},
    n_steps=200
)

# Using array (order matches state_vars)
time, trajectory = pfuzzy.simulate(
    x0=[50, 40],
    n_steps=200
)
```

---

##### `.plot_trajectory(variables=None, figsize=(12, 6), show=True)`

Plot time evolution of state variables.

**Parameters:**
- `variables` (list, optional): Variable names to plot. If None, plots all
- `figsize` (tuple): Figure size
- `show` (bool): Whether to call `plt.show()`

**Returns:** `tuple` - `(fig, ax)`

**Example:**
```python
pfuzzy.plot_trajectory(variables=['prey', 'predator'])
```

---

##### `.plot_phase_space(var_x, var_y, figsize=(8, 8), show=True)`

Plot phase space (2D trajectory).

**Parameters:**
- `var_x` (str): Variable for x-axis
- `var_y` (str): Variable for y-axis
- `figsize` (tuple): Figure size
- `show` (bool): Whether to call `plt.show()`

**Returns:** `tuple` - `(fig, ax)`

**Example:**
```python
pfuzzy.plot_phase_space('prey', 'predator')
```

---

##### `.to_csv(filename, include_time=True)`

Export trajectory to CSV file.

**Parameters:**
- `filename` (str): Output file path
- `include_time` (bool): Include time column (default: `True`)

**Example:**
```python
pfuzzy.to_csv('trajectory.csv')
```

---

### PFuzzyContinuous

Continuous dynamical system with fuzzy rule-based evolution.

**Continuous evolution:**
- **Absolute mode**: dx/dt = f(x)
- **Relative mode**: dx/dt = x * f(x)

#### Constructor

```python
PFuzzyContinuous(fis, mode='absolute', state_vars=None)
```

**Parameters:** Same as `PFuzzyDiscrete`

---

#### Methods

##### `.simulate(x0, t_span, dt=0.1, method='RK4', return_time=True)`

Simulate the continuous system.

**Parameters:**
- `x0` (dict | ndarray): Initial conditions
- `t_span` (tuple): Time interval `(t0, tf)`
- `dt` (float): Time step for integration (default: `0.1`)
- `method` (str): Integration method: `'Euler'`, `'RK4'` (default: `'RK4'`)
- `return_time` (bool): Return time array (default: `True`)

**Returns:** `tuple | ndarray` - `(time, trajectory)` or just `trajectory`

**Example:**
```python
time, trajectory = pfuzzy.simulate(
    x0={'prey': 50, 'predator': 40},
    t_span=(0, 100),
    dt=0.05,
    method='RK4'
)
```

---

Other methods (`.plot_trajectory()`, `.plot_phase_space()`, `.to_csv()`) are identical to `PFuzzyDiscrete`.

---

### Complete Example: Discrete Predator-Prey

```python
from fuzzy_systems import MamdaniSystem
from fuzzy_systems.dynamics import PFuzzyDiscrete
import numpy as np
import matplotlib.pyplot as plt

# Create FIS
fis = MamdaniSystem(name="Predator-Prey Discrete")

# Define variables
fis.add_input('prey', (0, 100))
fis.add_input('predator', (0, 100))
fis.add_output('var_prey', (-2, 2))
fis.add_output('var_predator', (-2, 2))

# Add 4 linguistic terms per variable (Low, Medium-Low, Medium-High, High)
for var in ['prey', 'predator']:
    fis.add_term(var, 'B', 'gaussian', (0, 12))         # Low
    fis.add_term(var, 'MB', 'gaussian', (33, 12))       # Medium-Low
    fis.add_term(var, 'MA', 'gaussian', (67, 12))       # Medium-High
    fis.add_term(var, 'A', 'gaussian', (100, 12))       # High

# Add output terms (8 per variable: 4 positive, 4 negative)
lrg = 0.5
for out_var in ['var_prey', 'var_predator']:
    # Negative variations
    fis.add_term(out_var, 'A_n', 'trapezoidal', (-4*lrg, -4*lrg, -3*lrg, -2*lrg))
    fis.add_term(out_var, 'MA_n', 'triangular', (-3*lrg, -2*lrg, -lrg))
    fis.add_term(out_var, 'MB_n', 'triangular', (-2*lrg, -lrg, 0))
    fis.add_term(out_var, 'B_n', 'triangular', (-lrg, 0, 0))
    # Positive variations
    fis.add_term(out_var, 'B_p', 'triangular', (0, 0, lrg))
    fis.add_term(out_var, 'MB_p', 'triangular', (0, lrg, 2*lrg))
    fis.add_term(out_var, 'MA_p', 'triangular', (lrg, 2*lrg, 3*lrg))
    fis.add_term(out_var, 'A_p', 'trapezoidal', (2*lrg, 3*lrg, 4*lrg, 4*lrg))

# Define 16 rules (4x4 matrix)
rules = [
    # Prey=B (Low)
    ('B', 'B', 'MB_p', 'MB_n'),   # Few prey, few predators → prey increase
    ('B', 'MB', 'B_p', 'MB_n'),
    ('B', 'MA', 'B_n', 'MA_n'),
    ('B', 'A', 'MB_n', 'A_n'),

    # Prey=MB (Medium-Low)
    ('MB', 'B', 'MA_p', 'B_n'),
    ('MB', 'MB', 'MB_p', 'B_n'),
    ('MB', 'MA', 'B_n', 'MB_n'),
    ('MB', 'A', 'MB_n', 'MA_n'),

    # Prey=MA (Medium-High)
    ('MA', 'B', 'MB_p', 'MA_p'),
    ('MA', 'MB', 'B_p', 'MB_p'),
    ('MA', 'MA', 'MB_n', 'B_p'),
    ('MA', 'A', 'MA_n', 'B_p'),

    # Prey=A (High)
    ('A', 'B', 'B_n', 'A_p'),
    ('A', 'MB', 'MB_n', 'MA_p'),
    ('A', 'MA', 'MA_n', 'MB_p'),
    ('A', 'A', 'A_n', 'B_p')
]

fis.add_rules(rules)

# Create p-fuzzy system
pfuzzy = PFuzzyDiscrete(
    fis=fis,
    mode='absolute',
    state_vars=['prey', 'predator']
)

# Simulate
time, trajectory = pfuzzy.simulate(
    x0={'prey': 50, 'predator': 40},
    n_steps=250
)

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Time series
pfuzzy.plot_trajectory(show=False)
ax1 = plt.gca()
ax1.set_title('Population Dynamics')

# Phase space
pfuzzy.plot_phase_space('prey', 'predator', show=False)
ax2 = plt.gca()
ax2.set_title('Phase Space')

plt.tight_layout()
plt.show()

# Export
pfuzzy.to_csv('predator_prey_discrete.csv')
```

---

### Complete Example: Continuous Population Growth

```python
from fuzzy_systems import MamdaniSystem
from fuzzy_systems.dynamics import PFuzzyContinuous

# Create FIS for continuous population growth
fis = MamdaniSystem()

# Single variable: population
fis.add_input('population', (0, 150))
fis.add_output('growth_rate', (-5, 5))

# Terms: Low, Medium, High population
fis.add_term('population', 'low', 'triangular', (0, 0, 50))
fis.add_term('population', 'medium', 'triangular', (25, 75, 125))
fis.add_term('population', 'high', 'triangular', (100, 150, 150))

# Growth rates: negative, zero, positive
fis.add_term('growth_rate', 'negative', 'triangular', (-5, -2.5, 0))
fis.add_term('growth_rate', 'zero', 'triangular', (-1, 0, 1))
fis.add_term('growth_rate', 'positive', 'triangular', (0, 2.5, 5))

# Rules (logistic-like behavior)
fis.add_rules([
    ('low', 'positive'),      # Low population → growth
    ('medium', 'zero'),       # Medium population → equilibrium
    ('high', 'negative')      # High population → decline
])

# Create continuous p-fuzzy system
pfuzzy = PFuzzyContinuous(
    fis=fis,
    mode='absolute',
    state_vars=['population']
)

# Simulate
time, trajectory = pfuzzy.simulate(
    x0={'population': 10},
    t_span=(0, 50),
    dt=0.1,
    method='RK4'
)

# Plot
pfuzzy.plot_trajectory()
plt.axhline(y=75, color='r', linestyle='--', label='Equilibrium (~75)')
plt.legend()
plt.show()
```

---

## Comparison: Fuzzy ODE vs p-Fuzzy

| Feature | Fuzzy ODE | p-Fuzzy |
|---------|-----------|---------|
| **Evolution** | Mathematical equation: dy/dt = f(t, y) | Fuzzy rules: IF...THEN |
| **Uncertainty** | Parameters & initial conditions | Rule-based behavior |
| **Method** | α-level cuts + ODE solver | Direct FIS evaluation |
| **Output** | Fuzzy envelope (bands) | Deterministic trajectory |
| **Best for** | Models with known equations but uncertain params | Models defined by expert rules |
| **Interpretability** | Medium (equation-based) | High (linguistic rules) |

---

## See Also

- [Core API](core.md) - Fuzzy sets and membership functions
- [Inference API](inference.md) - Build fuzzy systems for p-fuzzy
- [Learning API](learning.md) - Learn fuzzy rules from data
- [User Guide: Dynamics](../user_guide/dynamics.md) - Detailed tutorials
- [Examples](../examples/gallery.md) - Interactive notebooks
