# p-Fuzzy Continuous System Quick Start Guide

## Overview

**p-Fuzzy Continuous** systems are continuous-time dynamical systems where the evolution is governed by ordinary differential equations (ODEs) with fuzzy inference rules.

**Key Concept:**
```
dx/dt = f(x)           [absolute mode]
dx/dt = x × f(x)       [relative mode]
```

Where `f(x)` is determined by a Fuzzy Inference System (FIS).

**Applications:**
- Population dynamics (continuous growth models)
- Chemical reactions and enzyme kinetics
- Temperature/cooling systems
- Ecological models
- Continuous control systems
- Physical processes (Newton's law of cooling, etc.)

**Advantages:**
- Model continuous processes with linguistic rules
- Natural representation of rate-based systems
- Smooth trajectories (no discretization artifacts)
- Integrate expert knowledge without explicit equations

---

## 1. Basic Concepts

### What is a p-Fuzzy Continuous System?

A **p-Fuzzy continuous system** models dynamics using differential equations where the derivative is defined by fuzzy rules:

**Mathematical Form:**
```
dx₁/dt = f₁(x₁, x₂, ..., xₙ)
dx₂/dt = f₂(x₁, x₂, ..., xₙ)
...
dxₙ/dt = fₙ(x₁, x₂, ..., xₙ)
```

Where each `fᵢ` is computed by the FIS.

### Evolution Modes

**Absolute Mode** (additive change):
```
dx/dt = f(x)
```
- Rate of change is independent of current state magnitude
- Example: Newton's law of cooling: dT/dt = -k(T - T_ambient)

**Relative Mode** (multiplicative change):
```
dx/dt = x × f(x)
```
- Rate of change is proportional to current state
- Example: Exponential growth: dN/dt = r × N

### Integration Methods

The system supports two numerical integration methods:

**Euler Method** (1st order):
- Simple and fast
- Less accurate (O(h²) error)
- Requires smaller time steps

**Runge-Kutta 4th Order (RK4)** (recommended):
- More accurate (O(h⁵) error)
- Allows larger time steps
- Better for smooth solutions

---

## 2. Creating a p-Fuzzy Continuous System

### Step 1: Create the Fuzzy Inference System

```python
import fuzzy_systems as fs

# Create FIS
fis = fs.MamdaniSystem(name="Cooling System")

# Input: current temperature
fis.add_input('temperature', (0, 100))
fis.add_term('temperature', 'cold', 'triangular', (0, 0, 50))
fis.add_term('temperature', 'warm', 'triangular', (0, 50, 100))
fis.add_term('temperature', 'hot', 'triangular', (50, 100, 100))

# Output: cooling rate (dT/dt)
fis.add_output('cooling_rate', (-10, 0))
fis.add_term('cooling_rate', 'fast', 'triangular', (-10, -10, -5))
fis.add_term('cooling_rate', 'medium', 'triangular', (-10, -5, 0))
fis.add_term('cooling_rate', 'slow', 'triangular', (-5, 0, 0))

# Rules
fis.add_rules([
    {'temperature': 'hot', 'cooling_rate': 'fast'},
    {'temperature': 'warm', 'cooling_rate': 'medium'},
    {'temperature': 'cold', 'cooling_rate': 'slow'}
])
```

### Step 2: Create p-Fuzzy Continuous System

```python
from fuzzy_systems.dynamics import PFuzzyContinuous

pfuzzy = PFuzzyContinuous(
    fis=fis,
    mode='absolute',               # 'absolute' or 'relative'
    state_vars=['temperature'],    # State variable names
    method='rk4'                   # Integration: 'euler' or 'rk4'
)
```

### Key Parameters

- **`fis`**: Fuzzy inference system (Mamdani or Sugeno)
- **`mode`**: Evolution mode
  - `'absolute'`: dx/dt = f(x)
  - `'relative'`: dx/dt = x × f(x)
- **`state_vars`**: List of state variable names (optional)
  - If `None`, uses all FIS input variables
- **`method`**: Numerical integration method
  - `'euler'`: Euler method (simpler, less accurate)
  - `'rk4'`: Runge-Kutta 4th order (recommended)

---

## 3. Simulating the System

### Fixed Time Step Simulation

```python
# Initial condition
x0 = {'temperature': 80.0}

# Time span: from t=0 to t=10
t_span = (0, 10)

# Fixed time step
dt = 0.1

# Simulate
time, trajectory = pfuzzy.simulate(
    x0=x0,
    t_span=t_span,
    dt=dt,
    adaptive=False,
    verbose=False
)

print(f"Time points: {len(time)}")
print(f"Final temperature: {trajectory[-1, 0]:.2f}")
```

### Adaptive Time Step Simulation (Recommended)

```python
# Adaptive simulation (automatically adjusts dt)
time, trajectory = pfuzzy.simulate(
    x0=x0,
    t_span=(0, 10),
    adaptive=True,              # Enable adaptive stepping
    dt=0.1,                     # Initial step size
    tolerance=1e-4,             # Error tolerance
    dt_min=1e-5,                # Minimum step size
    dt_max=1.0,                 # Maximum step size
    verbose=True                # Print statistics
)
```

### Simulation Parameters

**Basic Parameters:**
- **`x0`**: Initial condition (dict, list, or array)
- **`t_span`**: Tuple `(t_start, t_end)` defining time interval
- **`dt`**: Time step size (default: 0.05 for fixed, 0.1 for adaptive)
- **`verbose`**: Print progress information

**Adaptive Parameters:**
- **`adaptive`**: Enable adaptive time stepping (default: `False`)
- **`tolerance`**: Local error tolerance (default: 1e-4)
  - Smaller = more accurate, more steps
  - Larger = less accurate, fewer steps
- **`dt_min`**: Minimum allowed step size (default: 1e-5)
- **`dt_max`**: Maximum allowed step size (default: 1.0)
- **`max_steps`**: Maximum number of steps (safety limit, default: 100000)

**Returns:**
- `time`: Array of time points (variable length if adaptive)
- `trajectory`: Array of shape `(n_points, n_vars)` with states

---

## 4. Fixed vs Adaptive Stepping

### When to Use Fixed Stepping

```python
time, traj = pfuzzy.simulate(
    x0=x0,
    t_span=(0, 10),
    dt=0.01,
    adaptive=False
)
```

**Best for:**
- Simple systems with smooth behavior
- When you need regular time intervals
- Debugging and understanding system behavior
- Fast simulations with known dynamics

**Considerations:**
- Choose `dt` carefully (too large = inaccurate, too small = slow)
- RK4 allows larger `dt` than Euler

### When to Use Adaptive Stepping (Recommended)

```python
time, traj = pfuzzy.simulate(
    x0=x0,
    t_span=(0, 10),
    adaptive=True,
    tolerance=1e-4
)
```

**Best for:**
- Systems with regions of rapid change
- When accuracy is critical
- Long simulations
- Unknown system behavior

**Advantages:**
- Automatically finds optimal step size
- Faster than fixed stepping with same accuracy
- Fewer total steps in smooth regions
- Smaller steps only where needed

**Statistics Output (verbose=True):**
```
Método: RK4
Passos aceitos: 245
Passos rejeitados: 12
Taxa de aceitação: 95.3%
dt médio: 0.04892
dt mínimo: 0.00234
dt máximo: 0.15000
```

---

## 5. Initial Conditions - Multiple Formats

```python
# Format 1: Dictionary (recommended)
x0 = {'temperature': 80.0}

# Format 2: List
x0 = [80.0]

# Format 3: Tuple
x0 = (80.0,)

# Format 4: NumPy array
import numpy as np
x0 = np.array([80.0])

# Multi-variable systems:
x0_dict = {'prey': 50.0, 'predator': 30.0}
x0_list = [50.0, 30.0]
x0_array = np.array([50.0, 30.0])
```

---

## 6. Visualization

### Plot Trajectory (Time Series)

```python
# Plot all state variables over time
fig, ax = pfuzzy.plot_trajectory(
    variables=None,                  # None = all variables
    figsize=(12, 6),
    title='Temperature Cooling',
    xlabel='Time (s)',
    ylabel='Temperature (°C)'
)
```

### Plot Phase Space (2D Systems)

```python
# Phase portrait for 2-variable systems
fig, ax = pfuzzy.plot_phase_space(
    var_x='prey',
    var_y='predator',
    figsize=(8, 8),
    title='Predator-Prey Phase Portrait'
)
```

### Custom Plots

```python
import matplotlib.pyplot as plt

# Access stored results
time = pfuzzy.time
trajectory = pfuzzy.trajectory

# Custom plot with derivatives
dt_vals = np.diff(time)
dx_vals = np.diff(trajectory[:, 0]) / dt_vals

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(time, trajectory[:, 0], 'b-', linewidth=2)
plt.xlabel('Time')
plt.ylabel('State')
plt.title('State vs Time')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(time[:-1], dx_vals, 'r-', linewidth=2)
plt.xlabel('Time')
plt.ylabel('dx/dt')
plt.title('Rate of Change')
plt.grid(True)

plt.tight_layout()
plt.show()
```

---

## 7. Exporting Results

### Export to CSV

```python
# Standard format
pfuzzy.to_csv('results.csv')

# European/Brazilian format
pfuzzy.to_csv(
    'results.csv',
    sep=';',
    decimal=','
)
```

**CSV Format:**
```
time,temperature
0.000000,80.000000
0.100000,78.542100
0.200000,77.143521
...
```

---

## 8. Complete Example: Newton's Law of Cooling

```python
import numpy as np
import matplotlib.pyplot as plt
import fuzzy_systems as fs
from fuzzy_systems.dynamics import PFuzzyContinuous

# ============================================================================
# Newton's Law of Cooling: dT/dt = -k(T - T_ambient)
# Fuzzy version: cooling rate determined by rules
# ============================================================================

# Create FIS
fis = fs.MamdaniSystem(name="Newton Cooling")

# Input: temperature difference from ambient (20°C)
fis.add_input('temperature', (0, 100))
fis.add_term('temperature', 'ambient', 'gaussian', (20, 5))
fis.add_term('temperature', 'warm', 'gaussian', (50, 10))
fis.add_term('temperature', 'hot', 'gaussian', (80, 10))

# Output: cooling rate
fis.add_output('cooling_rate', (-15, 0))
fis.add_term('cooling_rate', 'fast', 'triangular', (-15, -15, -8))
fis.add_term('cooling_rate', 'medium', 'triangular', (-12, -6, -2))
fis.add_term('cooling_rate', 'slow', 'triangular', (-4, 0, 0))

# Rules: hotter → cools faster
fis.add_rules([
    {'temperature': 'hot', 'cooling_rate': 'fast'},
    {'temperature': 'warm', 'cooling_rate': 'medium'},
    {'temperature': 'ambient', 'cooling_rate': 'slow'}
])

# Create p-Fuzzy system
pfuzzy = PFuzzyContinuous(
    fis=fis,
    mode='absolute',
    state_vars=['temperature'],
    method='rk4'
)

# Simulate
x0 = {'temperature': 95.0}
time, trajectory = pfuzzy.simulate(
    x0=x0,
    t_span=(0, 20),
    adaptive=True,
    tolerance=1e-4,
    verbose=True
)

# Visualize
pfuzzy.plot_trajectory(
    title="Newton's Law of Cooling (Fuzzy)",
    xlabel='Time (minutes)',
    ylabel='Temperature (°C)'
)

# Export
pfuzzy.to_csv('cooling_curve.csv')
print(f"✅ Final temperature: {trajectory[-1, 0]:.2f}°C")
```

---

## 9. Complete Example: Lotka-Volterra Predator-Prey

```python
import numpy as np
import matplotlib.pyplot as plt
import fuzzy_systems as fs
from fuzzy_systems.dynamics import PFuzzyContinuous

# ============================================================================
# Lotka-Volterra Predator-Prey Model (Fuzzy version)
# ============================================================================

fis = fs.MamdaniSystem(name="Lotka-Volterra Fuzzy")

# Inputs: prey and predator populations
fis.add_input('prey', (0, 100))
fis.add_input('predator', (0, 100))

# Membership functions (4 levels each)
for var in ['prey', 'predator']:
    fis.add_term(var, 'low', 'gaussian', (10, 8))
    fis.add_term(var, 'medium_low', 'gaussian', (35, 8))
    fis.add_term(var, 'medium_high', 'gaussian', (65, 8))
    fis.add_term(var, 'high', 'gaussian', (90, 8))

# Outputs: growth/decline rates
fis.add_output('prey_rate', (-3, 3))
fis.add_output('predator_rate', (-3, 3))

for var in ['prey_rate', 'predator_rate']:
    fis.add_term(var, 'large_decrease', 'triangular', (-3, -3, -1.5))
    fis.add_term(var, 'small_decrease', 'triangular', (-2, -1, 0))
    fis.add_term(var, 'small_increase', 'triangular', (0, 1, 2))
    fis.add_term(var, 'large_increase', 'triangular', (1.5, 3, 3))

# Rules (example subset - add more for complete model)
fis.add_rules([
    # Many prey, few predators → prey increase, predators increase
    {'prey': 'high', 'predator': 'low',
     'prey_rate': 'small_increase', 'predator_rate': 'large_increase'},

    # Few prey, many predators → prey decrease, predators decrease
    {'prey': 'low', 'predator': 'high',
     'prey_rate': 'large_decrease', 'predator_rate': 'large_decrease'},

    # Balanced populations
    {'prey': 'medium_high', 'predator': 'medium_low',
     'prey_rate': 'small_increase', 'predator_rate': 'small_increase'},

    {'prey': 'medium_low', 'predator': 'medium_high',
     'prey_rate': 'small_decrease', 'predator_rate': 'small_decrease'},

    # Add more rules for complete coverage...
])

# Create p-Fuzzy system
pfuzzy = PFuzzyContinuous(
    fis=fis,
    mode='absolute',
    state_vars=['prey', 'predator'],
    method='rk4'
)

# Simulate multiple initial conditions
initial_conditions = [
    {'prey': 60, 'predator': 30},
    {'prey': 40, 'predator': 50},
    {'prey': 70, 'predator': 20}
]

plt.figure(figsize=(14, 6))

# Time series
plt.subplot(1, 2, 1)
for i, x0 in enumerate(initial_conditions):
    time, traj = pfuzzy.simulate(
        x0=x0,
        t_span=(0, 50),
        adaptive=True,
        tolerance=1e-4
    )
    plt.plot(time, traj[:, 0], '--', label=f'Prey IC{i+1}', alpha=0.7)
    plt.plot(time, traj[:, 1], '-', label=f'Predator IC{i+1}', alpha=0.7)

plt.xlabel('Time')
plt.ylabel('Population')
plt.title('Lotka-Volterra Dynamics')
plt.legend()
plt.grid(True)

# Phase portrait
plt.subplot(1, 2, 2)
for i, x0 in enumerate(initial_conditions):
    time, traj = pfuzzy.simulate(x0=x0, t_span=(0, 50), adaptive=True)
    plt.plot(traj[:, 0], traj[:, 1], linewidth=2, label=f'IC{i+1}')
    plt.plot(traj[0, 0], traj[0, 1], 'go', markersize=8)
    plt.plot(traj[-1, 0], traj[-1, 1], 'ro', markersize=8)

plt.xlabel('Prey Population')
plt.ylabel('Predator Population')
plt.title('Phase Portrait')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Export last simulation
pfuzzy.to_csv('lotka_volterra.csv')
```

---

## 10. Tips and Best Practices

### Choosing Integration Method

| Method | Accuracy | Speed | When to Use |
|--------|----------|-------|-------------|
| **Euler** | Low (O(h²)) | Fast | Quick tests, simple systems |
| **RK4** | High (O(h⁵)) | Slower | Production, accurate results |

**Recommendation:** Always use RK4 unless speed is critical.

### Choosing Time Step (Fixed Stepping)

```python
# Too large: inaccurate, unstable
dt = 1.0  # ❌ May miss dynamics

# Good balance
dt = 0.01  # ✅ Typical choice

# Too small: slow, unnecessary
dt = 1e-6  # ❌ Wasteful
```

**Rule of thumb:**
- Euler: dt ≈ 0.001 - 0.01
- RK4: dt ≈ 0.01 - 0.1

### Using Adaptive Stepping Effectively

```python
# High accuracy: scientific computing
pfuzzy.simulate(x0=x0, t_span=(0, 10), adaptive=True, tolerance=1e-6)

# Standard accuracy: most applications
pfuzzy.simulate(x0=x0, t_span=(0, 10), adaptive=True, tolerance=1e-4)

# Fast approximation: visualization
pfuzzy.simulate(x0=x0, t_span=(0, 10), adaptive=True, tolerance=1e-2)
```

### System Design

1. **Output ranges must match dynamics scale:**
   ```python
   # Temperature cooling: expect changes of ~10°C/min
   fis.add_output('cooling_rate', (-15, 5))  # ✅

   # Not: (-100, 100)  # ❌ Too large
   ```

2. **Absolute vs Relative mode:**
   - **Absolute** (`dx/dt = f(x)`): Most common, natural interpretation
   - **Relative** (`dx/dt = x·f(x)`): Use when rate proportional to state

3. **Domain boundaries:**
   - System stops if state exits input universe
   - Make universes large enough for expected trajectories

### Performance Optimization

```python
# Long simulations: use adaptive with relaxed tolerance
time, traj = pfuzzy.simulate(
    x0=x0,
    t_span=(0, 1000),
    adaptive=True,
    tolerance=1e-3,  # Relaxed
    dt_max=0.5       # Allow larger steps
)

# Short, accurate simulations: fixed step with RK4
time, traj = pfuzzy.simulate(
    x0=x0,
    t_span=(0, 10),
    dt=0.01,
    adaptive=False,
    method='rk4'
)
```

---

## 11. Common Patterns

### Pattern 1: Exponential Decay

```python
# dx/dt = -k·x  (k > 0)
fis.add_rules([
    {'state': 'high', 'rate': 'large_negative'},
    {'state': 'medium', 'rate': 'medium_negative'},
    {'state': 'low', 'rate': 'small_negative'}
])
```

### Pattern 2: Logistic Growth

```python
# dx/dt = r·x·(1 - x/K)
fis.add_rules([
    {'state': 'low', 'rate': 'positive'},     # Below capacity: grow
    {'state': 'medium', 'rate': 'positive'},  # Near capacity: slow growth
    {'state': 'high', 'rate': 'negative'}     # Above capacity: decline
])
```

### Pattern 3: Oscillator (Limit Cycle)

```python
# Two variables: dx/dt = f(x,y), dy/dt = g(x,y)
fis.add_rules([
    {'x': 'positive', 'y': 'low', 'x_rate': 'positive', 'y_rate': 'positive'},
    {'x': 'high', 'y': 'positive', 'x_rate': 'negative', 'y_rate': 'positive'},
    {'x': 'negative', 'y': 'high', 'x_rate': 'negative', 'y_rate': 'negative'},
    {'x': 'low', 'y': 'negative', 'x_rate': 'positive', 'y_rate': 'negative'}
])
```

---

## 12. Common Issues and Solutions

| Problem | Cause | Solution |
|---------|-------|----------|
| Simulation stops early | State exits domain | Increase input universe range |
| Oscillations/instability | dt too large (Euler) | Use RK4 or smaller dt |
| Slow simulation | dt too small | Use adaptive stepping or increase dt |
| Flat trajectory | Output ranges too small | Scale up output universe |
| Inaccurate results | Tolerance too large | Decrease tolerance (adaptive) |
| dt hits minimum | Stiff system or tolerance too tight | Relax tolerance or check FIS |

### Debugging Unstable Simulations

```python
# Check FIS behavior at test points
test_state = {'temperature': 50}
output = fis.evaluate(test_state)
print(f"At T=50: dT/dt = {output['cooling_rate']}")

# Use verbose mode
time, traj = pfuzzy.simulate(
    x0=x0,
    t_span=(0, 10),
    adaptive=True,
    verbose=True  # Shows step acceptance/rejection
)

# Try smaller time step
time, traj = pfuzzy.simulate(
    x0=x0,
    t_span=(0, 10),
    dt=0.001,  # Very small
    adaptive=False
)
```

---

## 13. Absolute vs Relative Mode

### Absolute Mode: dx/dt = f(x)

```python
mode='absolute'
```

**Characteristics:**
- Rate of change independent of current value
- Most intuitive and common
- Natural for: cooling, chemical reactions, oscillators

**Example: Cooling**
```
dT/dt = -5  # Temperature decreases by 5°C/min regardless of T
```

### Relative Mode: dx/dt = x × f(x)

```python
mode='relative'
```

**Characteristics:**
- Rate proportional to current state
- Natural for growth processes
- Output interpreted as relative rate

**Example: Population Growth**
```
dN/dt = N × 0.1  # Population grows at 10% per unit time
```

**Output Guidelines:**
- `f(x) = 0`: No change
- `f(x) > 0`: Growth
- `f(x) < 0`: Decline
- Typical range: (-1, 1) for ±100% rates

---

## 14. Comparison: Discrete vs Continuous

| Aspect | Discrete | Continuous |
|--------|----------|------------|
| **Time** | Integer steps (n) | Real time (t∈ℝ) |
| **Equation** | x_{n+1} = x_n + f(x_n) | dx/dt = f(x) |
| **Simulation** | Simple iteration | Numerical integration |
| **Smoothness** | Discrete jumps | Smooth curves |
| **Speed** | Faster | Slower (integration) |
| **Accuracy** | Exact (per iteration) | Depends on dt, method |
| **Best for** | Generations, events | Physical processes |

---

## 15. Advanced: Vector Field Analysis

Visualize the system dynamics with vector fields:

```python
import numpy as np
import matplotlib.pyplot as plt

# Create grid
x_grid = np.linspace(0, 100, 20)
y_grid = np.linspace(0, 100, 20)
X, Y = np.meshgrid(x_grid, y_grid)

# Compute vector field
DX = np.zeros_like(X)
DY = np.zeros_like(Y)

for i in range(len(x_grid)):
    for j in range(len(y_grid)):
        state = {'prey': X[j, i], 'predator': Y[j, i]}
        output = fis.evaluate(state)
        DX[j, i] = output['prey_rate']
        DY[j, i] = output['predator_rate']

# Plot vector field + trajectory
fig, ax = plt.subplots(figsize=(10, 10))

# Vector field
ax.quiver(X, Y, DX, DY, alpha=0.5, color='gray')

# Simulate and overlay trajectory
time, traj = pfuzzy.simulate(
    x0={'prey': 60, 'predator': 30},
    t_span=(0, 50),
    adaptive=True
)
ax.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=3, label='Trajectory')
ax.plot(traj[0, 0], traj[0, 1], 'go', markersize=12, label='Start')
ax.plot(traj[-1, 0], traj[-1, 1], 'ro', markersize=12, label='End')

ax.set_xlabel('Prey')
ax.set_ylabel('Predator')
ax.set_title('Phase Portrait with Vector Field')
ax.legend()
ax.grid(True)
plt.show()
```

---

## 16. Advanced: Comparing Methods

```python
import time as timer

methods = ['euler', 'rk4']
results = {}

for method in methods:
    pfuzzy = PFuzzyContinuous(fis=fis, mode='absolute', method=method)

    # Time the simulation
    start = timer.time()
    t, traj = pfuzzy.simulate(x0=x0, t_span=(0, 10), dt=0.01)
    elapsed = timer.time() - start

    results[method] = {
        'time': elapsed,
        'final_state': traj[-1],
        'trajectory': traj
    }

    print(f"{method.upper()}: {elapsed:.4f}s, final={traj[-1]}")

# Plot comparison
plt.figure(figsize=(12, 5))
for method, data in results.items():
    plt.plot(pfuzzy.time, data['trajectory'], linewidth=2, label=method.upper())

plt.xlabel('Time')
plt.ylabel('State')
plt.title('Euler vs RK4 Comparison')
plt.legend()
plt.grid(True)
plt.show()
```

---

## 17. Advanced: Stiff Systems

For stiff systems (very different time scales), adaptive stepping helps:

```python
# Fast and slow dynamics
fis = fs.MamdaniSystem()
fis.add_input('fast', (0, 10))
fis.add_input('slow', (0, 10))
fis.add_output('fast_rate', (-100, 100))  # Fast dynamics
fis.add_output('slow_rate', (-1, 1))       # Slow dynamics

# ... add terms and rules ...

pfuzzy = PFuzzyContinuous(fis=fis, method='rk4')

# Adaptive stepping handles different scales automatically
time, traj = pfuzzy.simulate(
    x0=[5, 5],
    t_span=(0, 10),
    adaptive=True,
    tolerance=1e-4,
    verbose=True
)

# Check step size variation
import matplotlib.pyplot as plt
dt_vals = np.diff(time)
plt.semilogy(time[:-1], dt_vals)
plt.xlabel('Time')
plt.ylabel('Step Size (log scale)')
plt.title('Adaptive Step Size')
plt.grid(True)
plt.show()
```

---

## References

- Barros, L. C., Bassanezi, R. C., & Lodwick, W. A. (2017). *A First Course in Fuzzy Logic, Fuzzy Dynamical Systems, and Biomathematics*. Springer.
- Butcher, J. C. (2016). *Numerical Methods for Ordinary Differential Equations*. John Wiley & Sons.
- Hairer, E., Nørsett, S. P., & Wanner, G. (1993). *Solving Ordinary Differential Equations I: Nonstiff Problems*. Springer.
