# Mamdani Learning Quick Start Guide

## Overview

MamdaniLearning is a class for **learning optimal fuzzy rules** in Mamdani systems using **metaheuristic optimization**. Unlike Wang-Mendel (which extracts rules from data in a single pass), MamdaniLearning iteratively searches for the best rule combinations using evolutionary algorithms.

**Key Features:**
- Optimizes rule consequents (outputs) for existing membership functions
- Supports multiple metaheuristic algorithms (SA, GA, PSO, DE)
- Pre-computes rule activations for efficiency
- Works with any pre-configured Mamdani system
- No gradient computation required

---

## Algorithm Overview

### What MamdaniLearning Does

1. **Takes a Mamdani system** with defined membership functions for inputs and outputs
2. **Generates all possible rules** from the Cartesian product of input MFs
3. **Optimizes which output MF** each rule should use as its consequent
4. **Uses metaheuristics** to search the discrete space of possible rule combinations

### What It Optimizes

- **Input**: Pre-defined membership functions (triangular, trapezoidal, etc.)
- **Output**: Assignment of output terms to rules (e.g., Rule 1 → "high", Rule 2 → "low")
- **Objective**: Minimize prediction error (RMSE) on training data

### Difference from Other Methods

| Method | What it learns | Algorithm | Speed |
|--------|----------------|-----------|-------|
| **Wang-Mendel** | Selects rules from data | One-pass extraction | Very fast |
| **MamdaniLearning** | Optimizes rule consequents | Metaheuristics | Moderate |
| **ANFIS** | Adjusts MF parameters + consequents | Gradient descent + LSE | Slow |

---

## 1. Setup: Create and Configure Mamdani System

Before using MamdaniLearning, you must create a Mamdani system with **all membership functions defined**.

```python
from fuzzy_systems.inference import MamdaniSystem
from fuzzy_systems.learning import MamdaniLearning
import numpy as np

# Create Mamdani system
system = MamdaniSystem()

# Add input variables
system.add_input('temperature', (0, 40))
system.add_input('humidity', (0, 100))

# Add output variable
system.add_output('fan_speed', (0, 100))

# Define INPUT membership functions
# Temperature: cold, warm, hot
system.add_term('temperature', 'cold', 'trapezoidal', (0, 0, 10, 20))
system.add_term('temperature', 'warm', 'triangular', (15, 25, 35))
system.add_term('temperature', 'hot', 'trapezoidal', (30, 35, 40, 40))

# Humidity: dry, normal, wet
system.add_term('humidity', 'dry', 'trapezoidal', (0, 0, 20, 40))
system.add_term('humidity', 'normal', 'triangular', (30, 50, 70))
system.add_term('humidity', 'wet', 'trapezoidal', (60, 80, 100, 100))

# Define OUTPUT membership functions
# Fan speed: low, medium, high
system.add_term('fan_speed', 'low', 'triangular', (0, 0, 50))
system.add_term('fan_speed', 'medium', 'triangular', (25, 50, 75))
system.add_term('fan_speed', 'high', 'triangular', (50, 100, 100))
```

### Important Notes

- **All membership functions must be defined** before learning
- The learner will create **all possible rules** (Cartesian product)
- For 3 MFs per input × 2 inputs = 3×3 = 9 rules total
- Each rule can have any of the 3 output terms (low/medium/high)
- MamdaniLearning finds the best assignment of outputs to rules

---

## 2. Instantiate MamdaniLearning Class

```python
# Create learner
learner = MamdaniLearning(
    fis=system,           # Pre-configured Mamdani system
    num_points=1000,      # Discretization points for output universe
    verbose=True          # Print progress information
)
```

### Parameters

- **`fis`**: MamdaniSystem
  - Must have all input/output variables and terms defined
  - Rules will be created/optimized automatically

- **`num_points`**: int, default=1000
  - Number of discretization points for output domain
  - Higher = more accurate defuzzification, but slower
  - Typical range: 500-2000

- **`verbose`**: bool, default=True
  - Print detailed progress during optimization
  - Shows iteration count, cost evolution, temperature (SA), etc.

---

## 3. Training with `fit_rules()`

The `fit_rules()` method optimizes the fuzzy rules using metaheuristic algorithms:

```python
# Prepare data
X_train = ...  # shape: (n_samples, n_features)
y_train = ...  # shape: (n_samples,)

# Train with Simulated Annealing (default)
learner.fit_rules(
    X_train=X_train,
    y_train=y_train,
    optimizer='sa',                      # Algorithm: 'sa', 'ga', 'pso', 'de'
    optimizer_params=None,               # Optional: algorithm-specific parameters
    initial_solution_method='random'     # Initialization: 'random', 'uniform', 'gradient'
)
```

### Parameters

- **`X_train`**: np.ndarray, shape (n_samples, n_features)
  - Training input data
  - Must match number of input variables in system

- **`y_train`**: np.ndarray, shape (n_samples,)
  - Training target values
  - Currently supports single-output systems

- **`optimizer`**: str, default='sa'
  - Optimization algorithm:
    - `'sa'`: Simulated Annealing (recommended for small-medium problems)
    - `'ga'`: Genetic Algorithm (good for discrete optimization)
    - `'pso'`: Particle Swarm Optimization (fast convergence)
    - `'de'`: Differential Evolution (robust)

- **`optimizer_params`**: dict, optional
  - Algorithm-specific hyperparameters (see section 4)

- **`initial_solution_method`**: str, default='random'
  - How to initialize the rule consequents:
    - `'random'`: Random assignment of output terms
    - `'uniform'`: All rules start with middle output term
    - `'gradient'`: Data-driven initialization (recommended for faster convergence)

### Return Value

Returns `self` for method chaining:
```python
learner = MamdaniLearning(system).fit_rules(X_train, y_train)
```

---

## 4. Optimizer-Specific Parameters

### 4.1 Simulated Annealing (`optimizer='sa'`)

**Best for:** Small-to-medium rule bases, guarantees convergence

```python
learner.fit_rules(
    X_train, y_train,
    optimizer='sa',
    optimizer_params={
        'temperature_init': 100.0,      # Initial temperature
        'temperature_min': 0.01,        # Minimum temperature (stopping criterion)
        'cooling_rate': 0.95,           # Temperature decay (0.9-0.99)
        'max_iterations': 5000,         # Maximum iterations
        'plateau_iterations': 1000,     # Stop if no improvement for N iterations
        'cooling_schedule': 'exponential'  # 'exponential', 'linear', or 'logarithmic'
    }
)
```

**Parameters:**
- **`temperature_init`**: Starting temperature (higher = more exploration)
- **`cooling_rate`**: Decay factor per iteration (closer to 1 = slower cooling)
- **`cooling_schedule`**: How temperature decreases
  - `'exponential'`: T *= cooling_rate (default, balanced)
  - `'linear'`: T -= (T_init - T_min) / max_iter (fast cooling)
  - `'logarithmic'`: T = T_init / log(1 + iteration) (slow cooling)

**Tips:**
- Start with high temperature (50-200) for good exploration
- Use cooling_rate 0.90-0.99 (0.95 is good default)
- Increase max_iterations if solution hasn't converged

### 4.2 Genetic Algorithm (`optimizer='ga'`)

**Best for:** Large rule bases, diverse solution exploration

```python
learner.fit_rules(
    X_train, y_train,
    optimizer='ga',
    optimizer_params={
        'pop_size': 100,                # Population size
        'max_gen': 500,                 # Maximum generations
        'elite_ratio': 0.15,            # Fraction of elites to keep (0.1-0.2)
        'crossover_rate': 0.8,          # Probability of crossover (0.7-0.9)
        'crossover_type': 'uniform',    # 'uniform' or 'single_point'
        'mutation_rate': 0.05,          # Probability of mutation (0.01-0.1)
        'tournament_size': 5,           # Tournament selection size (3-7)
        'adaptive_mutation': True,      # Increase mutation when stagnant
        'plateau_generations': 50,      # Trigger adaptive mutation after N gens
        'mutation_boost_factor': 2.0    # Mutation rate multiplier when stagnant
    }
)
```

**Parameters:**
- **`pop_size`**: Number of individuals (50-200, larger for complex problems)
- **`elite_ratio`**: Fraction of best individuals preserved (typical: 0.1-0.2)
- **`crossover_rate`**: Higher = more recombination (0.7-0.9)
- **`mutation_rate`**: Higher = more exploration (0.01-0.1)
- **`adaptive_mutation`**: Automatically increases mutation when stuck

**Tips:**
- Use `crossover_type='uniform'` for better exploration
- Enable `adaptive_mutation=True` to escape local optima
- Balance: high crossover (0.8) + low mutation (0.05)

### 4.3 Particle Swarm Optimization (`optimizer='pso'`)

**Best for:** Fast convergence, continuous-like exploration

```python
learner.fit_rules(
    X_train, y_train,
    optimizer='pso',
    optimizer_params={
        'n_particles': 30,          # Number of particles (20-50)
        'n_iterations': 100,        # Number of iterations (50-200)
        'w_max': 0.9,               # Initial inertia weight (exploration)
        'w_min': 0.4,               # Final inertia weight (exploitation)
        'c1': 1.49618,              # Cognitive parameter (personal best attraction)
        'c2': 1.49618               # Social parameter (global best attraction)
    }
)
```

**Parameters:**
- **`w_max/w_min`**: Inertia weight (linearly decreases from w_max to w_min)
  - High w = exploration, low w = exploitation
- **`c1`**: Attraction to personal best (typical: 1.5-2.0)
- **`c2`**: Attraction to global best (typical: 1.5-2.0)

**Tips:**
- Use c1 ≈ c2 for balanced exploration/exploitation
- Increase c1 for more individual exploration
- Increase c2 for faster convergence (may get stuck)

### 4.4 Differential Evolution (`optimizer='de'`)

**Best for:** Robust global optimization, fewer parameters

```python
learner.fit_rules(
    X_train, y_train,
    optimizer='de',
    optimizer_params={
        'pop_size': 50,             # Population size (30-100)
        'max_iter': 100,            # Maximum iterations (50-200)
        'F': 0.8,                   # Differential weight (0.5-1.0)
        'CR': 0.9                   # Crossover probability (0.7-0.95)
    }
)
```

**Parameters:**
- **`F`**: Differential weight (controls mutation scale)
  - Low F (0.5-0.7) = conservative
  - High F (0.8-1.0) = aggressive exploration
- **`CR`**: Crossover rate
  - High CR (0.8-0.95) = more information exchange

**Tips:**
- Use F=0.8, CR=0.9 as good defaults
- Increase pop_size for complex problems
- More stable than GA, fewer hyperparameters

---

## 5. Making Predictions

After training, use the learned rules to make predictions:

```python
# Make predictions
X_test = ...  # shape: (n_samples, n_features)
y_pred = learner.predict(X_test)
```

### Prediction Methods

```python
# Basic prediction
y_pred = learner.predict(X_test)

# Calculate RMSE on test set
rmse = learner.score(X_test, y_test)
print(f"Test RMSE: {rmse:.4f}")
```

---

## 6. Accessing Learned Rules and History

### 6.1 Get Learned Rules

```python
# Get best rule consequent indices
rules = learner.get_rules()
print(f"Optimized rules: {rules}")
# Example output: [0, 2, 1, 0, 2, 1, 0, 1, 2]
# This means: Rule 0 uses output term 0 (low)
#             Rule 1 uses output term 2 (high)
#             Rule 2 uses output term 1 (medium), etc.
```

### 6.2 Get Best Cost

```python
# Get final optimization cost (RMSE)
best_cost = learner.get_cost()
print(f"Best RMSE: {best_cost:.6f}")
```

### 6.3 Get Optimization History

```python
# Get convergence history
history = learner.get_history()

# Plot convergence
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(history['costs'])
plt.xlabel('Iteration')
plt.ylabel('RMSE')
plt.title('Optimization Convergence')
plt.grid(True)
plt.show()
```

**History dictionary contains:**
```python
{
    'costs': list,           # Cost at each iteration
    'temperatures': list,    # Temperature schedule (SA only)
    'acceptances': list,     # Acceptance rate (SA only)
    'best_fitnesses': list,  # Best fitness per generation (GA only)
    'avg_fitnesses': list,   # Average fitness per generation (GA only)
    # ... optimizer-specific metrics
}
```

---

## 7. Complete Example

```python
import numpy as np
from fuzzy_systems.inference import MamdaniSystem
from fuzzy_systems.learning import MamdaniLearning
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ============================================================================
# 1. Generate synthetic data
# ============================================================================
np.random.seed(42)
X = np.random.uniform(0, 10, (500, 2))
y = np.sin(X[:, 0]) * np.cos(X[:, 1]) + np.random.normal(0, 0.1, 500)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================================================
# 2. Create and configure Mamdani system
# ============================================================================
system = MamdaniSystem()

# Inputs
system.add_input('x1', (0, 10))
system.add_input('x2', (0, 10))

# Add 3 terms per input
for var in ['x1', 'x2']:
    system.add_term(var, 'low', 'triangular', (0, 0, 5))
    system.add_term(var, 'medium', 'triangular', (2.5, 5, 7.5))
    system.add_term(var, 'high', 'triangular', (5, 10, 10))

# Output
system.add_output('y', (-2, 2))
system.add_term('y', 'negative', 'triangular', (-2, -2, 0))
system.add_term('y', 'zero', 'triangular', (-1, 0, 1))
system.add_term('y', 'positive', 'triangular', (0, 2, 2))

print(f"Total possible rules: 3 × 3 = 9")
print(f"Each rule can have 3 different consequents (negative/zero/positive)")

# ============================================================================
# 3. Create learner and optimize rules
# ============================================================================
learner = MamdaniLearning(system, num_points=1000, verbose=True)

# Train with Simulated Annealing
learner.fit_rules(
    X_train, y_train,
    optimizer='sa',
    optimizer_params={
        'temperature_init': 100.0,
        'cooling_rate': 0.95,
        'max_iterations': 3000,
        'plateau_iterations': 500
    },
    initial_solution_method='gradient'
)

# ============================================================================
# 4. Evaluate
# ============================================================================
y_pred_train = learner.predict(X_train)
y_pred_test = learner.predict(X_test)

rmse_train = learner.score(X_train, y_train)
rmse_test = learner.score(X_test, y_test)

print(f"\n{'='*70}")
print("RESULTS")
print(f"{'='*70}")
print(f"Train RMSE: {rmse_train:.6f}")
print(f"Test RMSE:  {rmse_test:.6f}")
print(f"Best rules: {learner.get_rules()}")

# ============================================================================
# 5. Plot convergence
# ============================================================================
history = learner.get_history()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Cost evolution
ax1.plot(history['costs'])
ax1.set_xlabel('Iteration')
ax1.set_ylabel('RMSE')
ax1.set_title('Cost Evolution')
ax1.grid(True)

# Temperature evolution (SA only)
if 'temperatures' in history:
    ax2.plot(history['temperatures'])
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Temperature')
    ax2.set_title('Temperature Schedule')
    ax2.grid(True)

plt.tight_layout()
plt.show()

# ============================================================================
# 6. Plot predictions
# ============================================================================
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred_test, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title(f'Test Set Predictions (RMSE: {rmse_test:.4f})')
plt.grid(True)
plt.show()
```

---

## 8. Choosing the Right Optimizer

### Decision Guide

```
┌─────────────────────────────────────┐
│  How many rules in your system?     │
└──────────────┬──────────────────────┘
               │
       ┌───────┴───────┐
       │               │
    < 20 rules     > 20 rules
       │               │
       ▼               ▼
  Use SA/PSO      Use GA/DE
  (faster)        (better exploration)
       │               │
       └───────┬───────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Do you need guaranteed convergence?│
└──────────────┬──────────────────────┘
               │
       ┌───────┴───────┐
       │               │
      YES              NO
       │               │
       ▼               ▼
     Use SA      Use GA/PSO/DE
   (proven)       (faster, stochastic)
```

### Recommendations by Problem Size

| Rules | Best Algorithm | Typical Time | Parameters |
|-------|---------------|--------------|------------|
| < 10 | SA | Fast (seconds) | temperature_init=100, cooling_rate=0.95 |
| 10-30 | SA or PSO | Medium (minutes) | SA: cooling_rate=0.97, PSO: n_particles=30 |
| 30-100 | GA or DE | Slow (minutes-hours) | GA: pop_size=100, DE: pop_size=50 |
| > 100 | GA | Very slow | pop_size=200, max_gen=1000 |

### Algorithm Characteristics

**Simulated Annealing (SA):**
- ✅ Theoretical convergence guarantee
- ✅ Few hyperparameters
- ✅ Good for small-medium problems
- ❌ Slow for large problems
- ❌ Sequential (no parallelization)

**Genetic Algorithm (GA):**
- ✅ Excellent exploration
- ✅ Handles large search spaces
- ✅ Can be parallelized
- ❌ Many hyperparameters to tune
- ❌ Can be slow to converge

**Particle Swarm Optimization (PSO):**
- ✅ Fast convergence
- ✅ Few parameters
- ✅ Good balance exploration/exploitation
- ❌ Can get stuck in local optima
- ❌ Adapted for discrete spaces (may be suboptimal)

**Differential Evolution (DE):**
- ✅ Robust and reliable
- ✅ Very few parameters
- ✅ Good for difficult landscapes
- ❌ Slower than PSO
- ❌ Adapted for discrete spaces

---

## 9. Tips and Best Practices

### Membership Function Design

```python
# Good: 3-5 MFs with 50% overlap
system.add_term('temp', 'low', 'triangular', (0, 0, 5))
system.add_term('temp', 'med', 'triangular', (2.5, 5, 7.5))
system.add_term('temp', 'high', 'triangular', (5, 10, 10))

# Bad: Too many MFs = exponential rule growth
# 5 MFs × 5 MFs × 5 MFs = 125 rules!
```

### Initial Solution

```python
# Random: Good default
learner.fit_rules(X, y, initial_solution_method='random')

# Gradient: Best for faster convergence (uses data)
learner.fit_rules(X, y, initial_solution_method='gradient')

# Uniform: All rules start with middle term (conservative)
learner.fit_rules(X, y, initial_solution_method='uniform')
```

### Monitoring Convergence

```python
# Check if optimization converged
history = learner.get_history()
costs = history['costs']

# Plateau detection
last_100 = costs[-100:]
if max(last_100) - min(last_100) < 1e-4:
    print("✓ Converged (cost plateau)")
else:
    print("⚠️ Not converged, increase max_iterations")
```

### Overfitting Prevention

```python
# Use validation set for early stopping (manual)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

learner.fit_rules(X_train, y_train, optimizer='sa')

rmse_train = learner.score(X_train, y_train)
rmse_val = learner.score(X_val, y_val)

if rmse_val > rmse_train * 1.5:
    print("⚠️ Overfitting detected")
```

### Computational Efficiency

```python
# Pre-computation is automatic, but you can control discretization
learner = MamdaniLearning(
    system,
    num_points=500    # Lower = faster, less accurate defuzzification
)

# For very large datasets, consider sampling
if len(X_train) > 1000:
    indices = np.random.choice(len(X_train), 1000, replace=False)
    X_sample = X_train[indices]
    y_sample = y_train[indices]
    learner.fit_rules(X_sample, y_sample)
```

---

## 10. Troubleshooting

| Problem | Possible Cause | Solution |
|---------|---------------|----------|
| Slow convergence | Too many rules | Reduce MFs per variable |
| Stuck in local optimum | SA cooling too fast | Decrease cooling_rate (0.95→0.98) |
| Poor accuracy | Bad MF placement | Adjust MF centers/shapes |
| Overfitting | Too many rules | Reduce MFs, add regularization |
| No improvement | Bad initialization | Try `initial_solution_method='gradient'` |
| NaN values | Defuzzification issues | Check MF coverage, increase num_points |

---

## 11. Comparison with Other Methods

### vs Wang-Mendel

**Wang-Mendel:**
- ✅ Very fast (single pass)
- ✅ No hyperparameters
- ❌ No optimization (may be suboptimal)

**MamdaniLearning:**
- ✅ Optimizes for best accuracy
- ✅ Flexible (multiple algorithms)
- ❌ Slower (iterative)

**When to use which:**
- Use **Wang-Mendel** for quick baseline
- Use **MamdaniLearning** to improve accuracy

### vs ANFIS

**ANFIS:**
- ✅ Learns MF parameters + consequents
- ✅ Gradient-based (efficient for continuous)
- ❌ Complex implementation
- ❌ Requires differentiable MFs

**MamdaniLearning:**
- ✅ Simpler (only learns consequents)
- ✅ Works with any MF type
- ✅ Interpretable rules
- ❌ Doesn't adjust MF shapes

**When to use which:**
- Use **MamdaniLearning** for interpretability, fixed MFs
- Use **ANFIS** for maximum accuracy, flexible MFs

---

## 12. Advanced Usage

### Sequential Optimization

```python
# Step 1: Quick initialization with Wang-Mendel
from fuzzy_systems.learning import WangMendelLearning
wm = WangMendelLearning(system, X_train, y_train)
wm.fit()

# Step 2: Fine-tune with MamdaniLearning
learner = MamdaniLearning(system)
learner.fit_rules(X_train, y_train, optimizer='pso',
                 initial_solution_method='random')
```

### Comparing Multiple Optimizers

```python
results = {}

for opt in ['sa', 'ga', 'pso', 'de']:
    learner = MamdaniLearning(system, verbose=False)
    learner.fit_rules(X_train, y_train, optimizer=opt)
    rmse = learner.score(X_test, y_test)
    results[opt] = rmse
    print(f"{opt.upper()}: RMSE = {rmse:.6f}")

best_opt = min(results, key=results.get)
print(f"\nBest optimizer: {best_opt.upper()}")
```

---

## References

- Mamdani, E. H., & Assilian, S. (1975). "An experiment in linguistic synthesis with a fuzzy logic controller." International Journal of Man-Machine Studies, 7(1), 1-13.
- Kirkpatrick, S., et al. (1983). "Optimization by simulated annealing." Science, 220(4598), 671-680.
- Holland, J. H. (1992). "Genetic algorithms." Scientific American, 267(1), 66-73.
- Kennedy, J., & Eberhart, R. (1995). "Particle swarm optimization." IEEE International Conference on Neural Networks.
- Storn, R., & Price, K. (1997). "Differential evolution–a simple and efficient heuristic for global optimization." Journal of Global Optimization, 11(4), 341-359.
