# p-Fuzzy Discrete System Quick Start Guide

## Overview

**p-Fuzzy Discrete** systems are discrete-time dynamical systems where the evolution function is defined by fuzzy inference rules instead of explicit mathematical equations.

**Key Concept:**
```
x_{n+1} = x_n + f(x_n)     [absolute mode]
x_{n+1} = x_n × f(x_n)     [relative mode]
```

Where `f(x_n)` is determined by a Fuzzy Inference System (FIS).

**Applications:**
- Population dynamics (predator-prey, epidemiology)
- Economic models with qualitative rules
- Discrete control systems
- Time-series modeling with expert knowledge

**Advantages:**
- Model complex dynamics using linguistic rules
- Incorporate expert knowledge without mathematical equations
- Natural representation of qualitative relationships
- Interpretable system behavior

---

## 1. Basic Concepts

### What is a p-Fuzzy System?

A **p-Fuzzy system** combines:
1. **State variables** (x₁, x₂, ..., xₙ): System states that evolve over time
2. **Fuzzy Inference System**: Maps current state to rate of change
3. **Evolution mode**: How changes are applied (absolute or relative)

### Evolution Modes

**Absolute Mode** (additive change):
```
x_{n+1} = x_n + f(x_n)
```
- Change is **added** to current state
- Example: population growth = current + birth_rate

**Relative Mode** (multiplicative change):
```
x_{n+1} = x_n × f(x_n)
```
- Change is **proportional** to current state
- Example: population growth = current × growth_factor

---

## 2. Creating a p-Fuzzy Discrete System

### Step 1: Create the Fuzzy Inference System (FIS)

The FIS defines how state variables evolve based on fuzzy rules.

**Important:**
- **Inputs**: Current state variables
- **Outputs**: Rate of change for each state variable (same order as inputs)

```python
import fuzzy_systems as fs

# Create Mamdani or Sugeno FIS
fis = fs.MamdaniSystem(name="Population Growth")

# Add input: current population
fis.add_input('population', (0, 100))
fis.add_term('population', 'low', 'triangular', (0, 0, 50))
fis.add_term('population', 'medium', 'triangular', (0, 50, 100))
fis.add_term('population', 'high', 'triangular', (50, 100, 100))

# Add output: population change rate
fis.add_output('growth_rate', (-10, 10))
fis.add_term('growth_rate', 'negative', 'triangular', (-10, -10, 0))
fis.add_term('growth_rate', 'zero', 'triangular', (-5, 0, 5))
fis.add_term('growth_rate', 'positive', 'triangular', (0, 10, 10))

# Add rules
fis.add_rules([
    {'population': 'low', 'growth_rate': 'positive'},     # Low pop → grow
    {'population': 'medium', 'growth_rate': 'zero'},      # Medium → stable
    {'population': 'high', 'growth_rate': 'negative'}     # High → decline
])
```

### Step 2: Create p-Fuzzy Discrete System

```python
from fuzzy_systems.dynamics import PFuzzyDiscrete

pfuzzy = PFuzzyDiscrete(
    fis=fis,                          # Fuzzy inference system
    mode='absolute',                  # 'absolute' or 'relative'
    state_vars=['population']         # State variable names (optional)
)
```

### Key Parameters

- **`fis`**: Fuzzy inference system (Mamdani or Sugeno)
- **`mode`**: Evolution mode
  - `'absolute'`: Additive change (x_{n+1} = x_n + f(x_n))
  - `'relative'`: Multiplicative change (x_{n+1} = x_n × f(x_n))
- **`state_vars`**: List of state variable names (optional)
  - If `None`, uses all FIS input variables as state variables
  - Must match FIS input variable names

**Important:** Number of FIS outputs must equal number of state variables!

---

## 3. Simulating the System

### Basic Simulation

```python
# Define initial conditions
x0 = {'population': 10.0}

# Simulate for 100 time steps
time, trajectory = pfuzzy.simulate(
    x0=x0,
    n_steps=100,
    verbose=False
)

# Access results
print(f"Time points: {time}")          # [0, 1, 2, ..., 100]
print(f"Trajectory shape: {trajectory.shape}")  # (101, n_vars)
print(f"Final state: {trajectory[-1]}")
```

### Simulation Parameters

```python
time, trajectory = pfuzzy.simulate(
    x0=x0,              # Initial condition (dict, list, or array)
    n_steps=100,        # Number of iterations
    verbose=True        # Print progress information
)
```

**Parameters:**
- **`x0`**: Initial condition (multiple formats):
  - Dictionary: `{'pop': 10.0, 'resource': 50.0}`
  - List/Tuple: `[10.0, 50.0]` (order matches state_vars)
  - NumPy array: `np.array([10.0, 50.0])`
- **`n_steps`**: Number of discrete time steps to simulate
- **`verbose`**: If `True`, prints simulation progress

**Returns:**
- `time`: Array of iteration numbers `[0, 1, 2, ..., n_steps]`
- `trajectory`: Array of shape `(n_steps+1, n_vars)` with state history

---

## 4. Initial Conditions - Multiple Formats

```python
# Format 1: Dictionary (recommended - most readable)
x0 = {'population': 10.0}

# Format 2: List (order must match state_vars)
x0 = [10.0]

# Format 3: Tuple
x0 = (10.0,)

# Format 4: NumPy array
import numpy as np
x0 = np.array([10.0])

# For multi-variable systems:
x0_dict = {'prey': 50.0, 'predator': 30.0}
x0_list = [50.0, 30.0]  # Order: prey, predator
x0_array = np.array([50.0, 30.0])
```

---

## 5. Visualization

### Plot Trajectory (Time Series)

```python
# Plot all state variables over time
fig, ax = pfuzzy.plot_trajectory(
    variables=None,              # None = all variables
    figsize=(12, 6),
    title='Population Dynamics',
    xlabel='Time (steps)',
    ylabel='Population'
)
```

**Parameters:**
- `variables`: List of variables to plot (None = all)
  - Example: `['population']` or `['prey', 'predator']`
- `figsize`: Figure size (width, height)
- `title`: Plot title
- `xlabel`, `ylabel`: Axis labels

### Plot Phase Space (2D Systems)

```python
# Phase space plot for 2-variable systems
fig, ax = pfuzzy.plot_phase_space(
    var_x='prey',
    var_y='predator',
    figsize=(8, 8),
    title='Predator-Prey Phase Space'
)
```

Shows trajectory in state space with initial (green) and final (red) points.

### Custom Plots

```python
import matplotlib.pyplot as plt

# Access stored results
time = pfuzzy.time
trajectory = pfuzzy.trajectory

# Custom plot
plt.figure(figsize=(10, 6))
plt.plot(time, trajectory[:, 0], 'b-', linewidth=2, label='Prey')
plt.plot(time, trajectory[:, 1], 'r-', linewidth=2, label='Predator')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.grid(True)
plt.show()
```

---

## 6. Exporting Results

### Export to CSV

```python
# Export trajectory to CSV file
pfuzzy.to_csv('results.csv')

# Custom format (Brazilian/European style)
pfuzzy.to_csv(
    'results.csv',
    sep=';',        # Semicolon separator
    decimal=','     # Comma as decimal separator
)
```

**Parameters:**
- `filename`: Output file path
- `sep`: Column separator (default: `,`)
- `decimal`: Decimal separator (default: `.`)
  - Use `','` for European/Brazilian format

**CSV Format:**
```
time,population
0.000000,10.000000
1.000000,12.500000
2.000000,14.800000
...
```

---

## 7. Single Step Execution

### Manual Stepping

Useful for debugging or custom integration:

```python
# Execute one iteration manually
x_current = np.array([10.0])
x_next = pfuzzy.step(x_current)

print(f"Current: {x_current}")
print(f"Next: {x_next}")

# Chain multiple steps
x0 = np.array([10.0])
x1 = pfuzzy.step(x0)
x2 = pfuzzy.step(x1)
x3 = pfuzzy.step(x2)
```

---

## 8. Complete Example: Simple Population

```python
import numpy as np
import matplotlib.pyplot as plt
import fuzzy_systems as fs
from fuzzy_systems.dynamics import PFuzzyDiscrete

# ============================================================================
# Step 1: Create FIS
# ============================================================================
fis = fs.MamdaniSystem(name="Logistic Growth")

# Input: current population
fis.add_input('population', (0, 100))
fis.add_term('population', 'low', 'gaussian', (10, 10))
fis.add_term('population', 'medium', 'gaussian', (50, 10))
fis.add_term('population', 'high', 'gaussian', (90, 10))

# Output: growth rate
fis.add_output('growth_rate', (-5, 5))
fis.add_term('growth_rate', 'negative', 'triangular', (-5, -5, 0))
fis.add_term('growth_rate', 'zero', 'triangular', (-2, 0, 2))
fis.add_term('growth_rate', 'positive', 'triangular', (0, 5, 5))

# Rules: logistic-like behavior
fis.add_rules([
    {'population': 'low', 'growth_rate': 'positive'},     # Low → grow fast
    {'population': 'medium', 'growth_rate': 'zero'},      # Medium → stable
    {'population': 'high', 'growth_rate': 'negative'}     # High → decline
])

# ============================================================================
# Step 2: Create p-Fuzzy System
# ============================================================================
pfuzzy = PFuzzyDiscrete(
    fis=fis,
    mode='absolute',
    state_vars=['population']
)

# ============================================================================
# Step 3: Simulate
# ============================================================================
# Initial condition
x0 = {'population': 5.0}

# Run simulation
time, trajectory = pfuzzy.simulate(
    x0=x0,
    n_steps=100,
    verbose=True
)

# ============================================================================
# Step 4: Visualize
# ============================================================================
pfuzzy.plot_trajectory(
    title='Logistic Population Growth',
    xlabel='Time (generations)',
    ylabel='Population'
)

# ============================================================================
# Step 5: Export Results
# ============================================================================
pfuzzy.to_csv('population_growth.csv')
print("✅ Results saved to population_growth.csv")
```

---

## 9. Complete Example: Predator-Prey System

```python
import numpy as np
import matplotlib.pyplot as plt
import fuzzy_systems as fs
from fuzzy_systems.dynamics import PFuzzyDiscrete

# ============================================================================
# Step 1: Create FIS for Predator-Prey
# ============================================================================
fis = fs.MamdaniSystem(name="Predator-Prey")

# Inputs: prey and predator populations
fis.add_input('prey', (0, 100))
fis.add_input('predator', (0, 100))

# Add terms for both variables (4 levels: Low, Medium-Low, Medium-High, High)
for var in ['prey', 'predator']:
    fis.add_term(var, 'low', 'gaussian', (0, 12))
    fis.add_term(var, 'med_low', 'gaussian', (33, 12))
    fis.add_term(var, 'med_high', 'gaussian', (67, 12))
    fis.add_term(var, 'high', 'gaussian', (100, 12))

# Outputs: change rates
fis.add_output('prey_change', (-10, 10))
fis.add_output('predator_change', (-10, 10))

# Add terms for outputs
for var in ['prey_change', 'predator_change']:
    fis.add_term(var, 'large_decrease', 'triangular', (-10, -10, -5))
    fis.add_term(var, 'small_decrease', 'triangular', (-7, -3, 0))
    fis.add_term(var, 'small_increase', 'triangular', (0, 3, 7))
    fis.add_term(var, 'large_increase', 'triangular', (5, 10, 10))

# Rules (simplified example - add more for realistic behavior)
fis.add_rules([
    # Few predators, many prey → prey increase, predators increase
    {'prey': 'high', 'predator': 'low',
     'prey_change': 'small_increase', 'predator_change': 'large_increase'},

    # Many predators, few prey → prey decrease, predators decrease
    {'prey': 'low', 'predator': 'high',
     'prey_change': 'large_decrease', 'predator_change': 'large_decrease'},

    # Balanced populations → small changes
    {'prey': 'med_high', 'predator': 'med_low',
     'prey_change': 'small_increase', 'predator_change': 'small_increase'},

    # Add more rules for complete coverage...
])

# ============================================================================
# Step 2: Create p-Fuzzy System
# ============================================================================
pfuzzy = PFuzzyDiscrete(
    fis=fis,
    mode='absolute',
    state_vars=['prey', 'predator']
)

# ============================================================================
# Step 3: Simulate Multiple Initial Conditions
# ============================================================================
initial_conditions = [
    {'prey': 60, 'predator': 30},
    {'prey': 40, 'predator': 50},
    {'prey': 70, 'predator': 20}
]

plt.figure(figsize=(14, 6))

# Time series plot
plt.subplot(1, 2, 1)
for i, x0 in enumerate(initial_conditions):
    time, traj = pfuzzy.simulate(x0=x0, n_steps=200)
    plt.plot(time, traj[:, 0], '--', label=f'Prey IC{i+1}', alpha=0.7)
    plt.plot(time, traj[:, 1], '-', label=f'Predator IC{i+1}', alpha=0.7)

plt.xlabel('Time (steps)')
plt.ylabel('Population')
plt.title('Predator-Prey Dynamics')
plt.legend()
plt.grid(True)

# Phase space plot
plt.subplot(1, 2, 2)
for i, x0 in enumerate(initial_conditions):
    time, traj = pfuzzy.simulate(x0=x0, n_steps=200)
    plt.plot(traj[:, 0], traj[:, 1], linewidth=2, label=f'IC{i+1}')
    plt.plot(traj[0, 0], traj[0, 1], 'go', markersize=8)  # Start
    plt.plot(traj[-1, 0], traj[-1, 1], 'ro', markersize=8)  # End

plt.xlabel('Prey Population')
plt.ylabel('Predator Population')
plt.title('Phase Space')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# ============================================================================
# Step 4: Export
# ============================================================================
pfuzzy.to_csv('predator_prey.csv')
```

---

## 10. Tips and Best Practices

### System Design

1. **Choose appropriate universe of discourse**:
   - Must contain all reachable states
   - Simulation stops if state exits domain

2. **Select evolution mode carefully**:
   - **Absolute**: Natural for additive processes (birth/death)
   - **Relative**: Natural for multiplicative processes (growth rates)

3. **Output ranges**:
   - Absolute mode: output = change magnitude
   - Relative mode: output = multiplication factor (1.0 = no change)

4. **Rule coverage**:
   - Ensure rules cover expected state space
   - Missing rules → unexpected behavior

### Performance Optimization

```python
# For long simulations, monitor memory
import sys
time, traj = pfuzzy.simulate(x0=x0, n_steps=10000)
print(f"Memory: {sys.getsizeof(traj) / 1e6:.2f} MB")

# For very long simulations, consider periodic saving
for i in range(0, 10000, 1000):
    t, tr = pfuzzy.simulate(x0=x0, n_steps=1000)
    pfuzzy.to_csv(f'results_batch_{i}.csv')
    x0 = tr[-1]  # Continue from last state
```

### Debugging

```python
# Check FIS behavior at specific states
test_state = {'prey': 50, 'predator': 30}
output = fis.evaluate(test_state)
print(f"State: {test_state}")
print(f"Change rates: {output}")

# Manually verify first step
x0 = np.array([50.0, 30.0])
x1_manual = x0 + np.array([output['prey_change'], output['predator_change']])
x1_auto = pfuzzy.step(x0)
print(f"Manual: {x1_manual}")
print(f"Auto: {x1_auto}")
```

---

## 11. Common Patterns

### Pattern 1: Carrying Capacity (Logistic Growth)

```python
# Rules for logistic growth
rules = [
    {'population': 'low', 'growth_rate': 'high_positive'},
    {'population': 'medium', 'growth_rate': 'low_positive'},
    {'population': 'high', 'growth_rate': 'negative'}
]
```

### Pattern 2: Allee Effect (Minimum Viable Population)

```python
# Population declines if too small
rules = [
    {'population': 'very_low', 'growth_rate': 'negative'},  # Extinction
    {'population': 'low', 'growth_rate': 'positive'},       # Recovery
    {'population': 'high', 'growth_rate': 'negative'}       # Overpopulation
]
```

### Pattern 3: Oscillatory Behavior (Predator-Prey)

```python
# Creates cycles
rules = [
    {'prey': 'high', 'predator': 'low',
     'prey_change': 'positive', 'predator_change': 'positive'},
    {'prey': 'low', 'predator': 'high',
     'prey_change': 'negative', 'predator_change': 'negative'}
]
```

---

## 12. Common Issues and Solutions

| Problem | Cause | Solution |
|---------|-------|----------|
| Simulation stops early | State exits domain | Increase input universe range |
| No dynamics (flat line) | All rules output zero | Check rule consequents |
| Explosive growth | Output ranges too large | Reduce output universe |
| Unexpected behavior | Sparse rule coverage | Add more rules |
| Oscillations too large | Output ranges too large | Scale down outputs |

### Domain Exit Handling

```python
# System warns and stops when state exits domain
try:
    time, traj = pfuzzy.simulate(x0={'pop': 95}, n_steps=100)
except Exception as e:
    print(f"Simulation error: {e}")

# Check if simulation completed
if len(time) < 101:
    print(f"⚠️ Stopped at step {len(time)-1}")
```

---

## 13. Absolute vs Relative Mode

### When to Use Absolute Mode

```python
mode='absolute'  # x_{n+1} = x_n + f(x_n)
```

**Best for:**
- Additive processes (birth - death)
- Fixed changes (add/subtract constants)
- When change doesn't depend on current value

**Example:** Population with constant birth/death rates
```
population_next = population_current + (births - deaths)
```

### When to Use Relative Mode

```python
mode='relative'  # x_{n+1} = x_n × f(x_n)
```

**Best for:**
- Multiplicative processes (growth factors)
- Percentage changes
- When change is proportional to current state

**Example:** Population with growth rate
```
population_next = population_current × (1 + growth_rate)
```

**Important:** In relative mode, outputs should be around 1.0:
- Output = 1.0 → no change
- Output = 1.1 → 10% increase
- Output = 0.9 → 10% decrease

---

## 14. Comparison: Discrete vs Continuous

| Aspect | Discrete | Continuous |
|--------|----------|------------|
| **Time** | Integer steps (n=0,1,2,...) | Real time (t∈ℝ) |
| **Evolution** | x_{n+1} = F(x_n) | dx/dt = f(x) |
| **Simulation** | Iteration | Numerical integration |
| **Speed** | Fast | Slower |
| **Best for** | Generations, cycles | Physical processes |

---

## 15. Advanced: Multi-Variable Systems

```python
# 3-variable system: SIR epidemic model
fis = fs.MamdaniSystem(name="SIR Model")

# Inputs
fis.add_input('susceptible', (0, 1000))
fis.add_input('infected', (0, 1000))
fis.add_input('recovered', (0, 1000))

# Outputs (changes)
fis.add_output('susceptible_change', (-50, 50))
fis.add_output('infected_change', (-50, 50))
fis.add_output('recovered_change', (-50, 50))

# ... add terms and rules ...

# Create p-fuzzy system
pfuzzy = PFuzzyDiscrete(
    fis=fis,
    mode='absolute',
    state_vars=['susceptible', 'infected', 'recovered']
)

# Simulate
x0 = {'susceptible': 990, 'infected': 10, 'recovered': 0}
time, traj = pfuzzy.simulate(x0=x0, n_steps=100)

# Plot all variables
pfuzzy.plot_trajectory(title='SIR Epidemic Model')
```

---

## References

- Barros, L. C., Bassanezi, R. C., & Lodwick, W. A. (2017). *A First Course in Fuzzy Logic, Fuzzy Dynamical Systems, and Biomathematics*. Springer.
- Bassanezi, R. C., & Barros, L. C. (2015). *Tópicos de Lógica Fuzzy e Biomatemática*. UNICAMP.
