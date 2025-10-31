# User Guide: Learning Systems

This guide covers how to automatically generate fuzzy systems from data using various learning algorithms.

## Why Learn Fuzzy Systems from Data?

**Manual approach:**
- Define membership functions by hand
- Write rules based on expert knowledge
- Time-consuming and subjective

**Learning approach:**
- Automatically extract rules from data
- Optimize membership function parameters
- Data-driven and objective

**When to use learning:**
- You have training data (input-output pairs)
- Expert knowledge is incomplete or unavailable
- You need to tune an existing system
- The system needs to adapt over time

---

## Overview of Learning Methods

| Method | Type | Speed | Interpretability | Best For |
|--------|------|-------|------------------|----------|
| **Wang-Mendel** | Rule extraction | ⚡⚡⚡ Fast | ⭐⭐⭐ High | Quick prototyping, simple datasets |
| **ANFIS** | Neuro-fuzzy | ⚡⚡ Moderate | ⭐⭐ Moderate | Function approximation, regression |
| **Mamdani Learning** | Metaheuristics | ⚡ Slow | ⭐⭐⭐ High | Complex optimization, interpretable rules |

**Quick decision guide:**
- **Need interpretable rules fast?** → Wang-Mendel
- **Need precise predictions?** → ANFIS
- **Need custom optimization?** → Mamdani Learning + PSO/DE/GA

---

## Wang-Mendel Algorithm

The Wang-Mendel algorithm generates fuzzy rules directly from data in **one pass**.

### How It Works

1. **Partition input/output spaces** into fuzzy sets
2. **Generate candidate rules** from each data point
3. **Resolve conflicts** by keeping rules with highest degree
4. **Create rule base** from non-conflicting rules

### Basic Example: Regression

```python
from fuzzy_systems.learning import WangMendelLearning
import numpy as np

# Generate training data
X = np.linspace(0, 10, 50).reshape(-1, 1)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, 50)

# Create learner
learner = WangMendelLearning(
    n_inputs=1,
    n_outputs=1,
    n_terms=5,  # 5 fuzzy sets per variable
    input_ranges=[(0, 10)],
    output_ranges=[(-1.5, 1.5)]
)

# Learn from data
system = learner.fit(X, y)

# Predict
X_test = np.linspace(0, 10, 100).reshape(-1, 1)
y_pred = learner.predict(X_test)

# Evaluate
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y, learner.predict(X))
print(f"MSE: {mse:.4f}")
```

**What happened:**
1. Algorithm partitioned [0, 10] into 5 fuzzy sets: very_low, low, medium, high, very_high
2. For each data point, it created a rule like: "IF x is medium THEN y is medium"
3. Conflicting rules were resolved by keeping the one with highest membership degree
4. Result: A Mamdani system with ~15-25 rules (fewer than 5¹=5 maximum)

---

### Classification Example

Wang-Mendel also works for classification:

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42
)

# Learn classifier
learner = WangMendelLearning(
    n_inputs=4,
    n_outputs=1,
    n_terms=3,
    input_ranges=[(X_train[:, i].min(), X_train[:, i].max()) for i in range(4)],
    output_ranges=[(0, 2)]  # 3 classes: 0, 1, 2
)

system = learner.fit(X_train, y_train)

# Predict
y_pred = learner.predict(X_test).round().astype(int)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2%}")
```

---

### Wang-Mendel Parameters

#### Number of Terms (`n_terms`)

Controls granularity of fuzzy partitions:

```python
# Coarse (faster, fewer rules)
learner = WangMendelLearning(n_terms=3, ...)

# Fine (slower, more rules, more precise)
learner = WangMendelLearning(n_terms=7, ...)
```

**Guidelines:**
- **3 terms**: Simple problems, fast prototyping
- **5 terms**: Good default for most problems
- **7-9 terms**: Complex, nonlinear relationships

#### Conflict Resolution

When multiple rules have the same antecedent:

```python
learner = WangMendelLearning(
    conflict_resolution='degree',  # Default: keep rule with highest degree
    # conflict_resolution='first'  # Keep first rule encountered
)
```

---

### Strengths and Limitations

**Strengths:**
- ⚡ Very fast (single pass over data)
- 🧠 Generates interpretable rules
- 📊 Works with small datasets
- 🎯 Good for quick prototyping

**Limitations:**
- 🔧 No parameter optimization (MF shapes are fixed)
- 📉 May not achieve best accuracy
- 🎲 Sensitive to initial partitioning
- 🗑️ May generate redundant rules

**When to use:**
- You need a baseline quickly
- Interpretability is more important than accuracy
- Data is limited or expensive
- You want to understand the problem structure

---

## ANFIS (Adaptive Neuro-Fuzzy Inference System)

ANFIS combines neural networks with fuzzy logic to learn both **rule structure** and **parameters**.

### Architecture

ANFIS is a **Sugeno system** trained like a neural network:

```
Input → Fuzzification → Rules → Normalization → Defuzzification → Output
          (Layer 1)    (Layer 2)   (Layer 3)       (Layer 4)
```

**Learnable parameters:**
- **Premise parameters**: Membership function shapes (c, σ for gaussian)
- **Consequent parameters**: Linear function coefficients (a, b, c in y = ax₁ + bx₂ + c)

---

### Basic Example

```python
from fuzzy_systems.learning import ANFIS
import numpy as np

# Generate training data
X = np.random.rand(200, 2) * 10
y = X[:, 0]**2 + 2*X[:, 1] + np.random.normal(0, 0.5, 200)

# Create ANFIS
anfis = ANFIS(
    n_inputs=2,
    n_terms=3,  # 3 MFs per input → 3² = 9 rules
    mf_type='gaussian'
)

# Train
history = anfis.fit(
    X, y,
    epochs=50,
    learning_rate=0.01,
    batch_size=32,
    validation_split=0.2,
    verbose=True
)

# Predict
X_test = np.random.rand(50, 2) * 10
y_pred = anfis.predict(X_test)

# Check convergence
import matplotlib.pyplot as plt
plt.plot(history['loss'], label='Train')
plt.plot(history['val_loss'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.show()
```

---

### Training Parameters

#### Learning Rate

Controls step size of gradient descent:

```python
# Too high: unstable, oscillates
anfis = ANFIS(n_inputs=2, n_terms=3)
anfis.fit(X, y, learning_rate=0.1)  # ⚠️ May diverge

# Too low: slow convergence
anfis.fit(X, y, learning_rate=0.0001)  # 🐌 Takes forever

# Good range: 0.001 - 0.01
anfis.fit(X, y, learning_rate=0.005)  # ✓ Usually works well
```

#### Number of Epochs

```python
# Monitor validation loss to avoid overfitting
history = anfis.fit(
    X, y,
    epochs=100,
    validation_split=0.2,
    early_stopping=True,  # Stop if val_loss doesn't improve
    patience=10
)
```

#### Batch Size

```python
# Small batches: more updates, noisier gradients
anfis.fit(X, y, batch_size=16)

# Large batches: fewer updates, smoother gradients
anfis.fit(X, y, batch_size=128)

# Rule of thumb: 32 or 64 for most problems
```

---

### Hybrid Learning (Advanced)

ANFIS supports **hybrid learning**: gradient descent for premise parameters + least squares for consequent parameters.

```python
anfis = ANFIS(
    n_inputs=2,
    n_terms=3,
    learning_method='hybrid'  # Faster convergence
)

anfis.fit(X, y, epochs=30)  # Needs fewer epochs
```

**Comparison:**

| Method | Speed | Stability | When to use |
|--------|-------|-----------|-------------|
| `'gradient'` | ⚡⚡ Moderate | ⭐⭐ Can be unstable | Small datasets, simple problems |
| `'hybrid'` | ⚡⚡⚡ Fast | ⭐⭐⭐ Stable | Large datasets, complex problems |

---

### Extracting Rules

After training, inspect learned rules:

```python
# Get Sugeno system
system = anfis.get_system()

# Print rules
for i, rule in enumerate(system.rules):
    print(f"Rule {i+1}:")
    print(f"  IF x1 is {rule['x1']} AND x2 is {rule['x2']}")
    print(f"  THEN y = {rule['a']}*x1 + {rule['b']}*x2 + {rule['c']}")
    print()

# Visualize membership functions
system.plot_variables(['x1', 'x2'])
```

---

### Strengths and Limitations

**Strengths:**
- 🎯 High accuracy on regression tasks
- 🔧 Optimizes both structure and parameters
- 📊 Handles nonlinear relationships well
- 🧮 Efficient gradient-based learning

**Limitations:**
- 🤔 Less interpretable than Wang-Mendel
- 📈 Can overfit on small datasets
- 🔢 Requires tuning hyperparameters
- 🏗️ Sugeno output (functions, not linguistic terms)

**When to use:**
- Accuracy is the priority
- You have enough training data (100+ samples)
- Regression or function approximation task
- You can accept Sugeno-style outputs

---

## Mamdani Learning with Metaheuristics

For **highly interpretable** systems, learn Mamdani rules using metaheuristic optimization.

### Why Metaheuristics?

Mamdani systems are hard to optimize with gradients because:
- Defuzzification (centroid) is non-differentiable
- Rule structure is discrete
- MF parameters interact in complex ways

**Metaheuristics** (PSO, DE, GA) are gradient-free and handle this naturally.

---

### Particle Swarm Optimization (PSO)

PSO simulates a swarm of particles searching for optimal parameters.

```python
from fuzzy_systems.learning import MamdaniLearning
import numpy as np

# Generate data
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = np.sin(X).ravel()

# Create learner
learner = MamdaniLearning(
    n_inputs=1,
    n_outputs=1,
    n_terms=5,
    input_ranges=[(0, 10)],
    output_ranges=[(-1, 1)],
    mf_type='triangular'
)

# Initialize random system
system = learner.initialize_system()

# Optimize with PSO
optimized_system, history = learner.optimize(
    X, y,
    method='pso',
    n_particles=30,
    n_iterations=100,
    inertia=0.7,
    cognitive=1.5,
    social=1.5,
    verbose=True
)

# Predict
y_pred = optimized_system.evaluate_batch(X)
```

**PSO Parameters:**

```python
learner.optimize(
    X, y,
    method='pso',
    n_particles=30,    # Population size (20-50 typical)
    n_iterations=100,  # Generations (50-200 typical)
    inertia=0.7,       # Velocity decay (0.4-0.9)
    cognitive=1.5,     # Personal best influence
    social=1.5         # Global best influence
)
```

**Tuning guide:**
- **Exploration** (diverse search): High inertia (0.8-0.9), low social (1.0-1.5)
- **Exploitation** (refine best): Low inertia (0.4-0.6), high social (2.0-2.5)

---

### Differential Evolution (DE)

DE uses difference vectors to mutate solutions.

```python
optimized_system, history = learner.optimize(
    X, y,
    method='de',
    population_size=40,
    n_iterations=100,
    mutation_factor=0.8,   # F: controls mutation strength
    crossover_prob=0.7,    # CR: controls recombination
    strategy='best1bin',   # Mutation strategy
    verbose=True
)
```

**DE Parameters:**

| Parameter | Range | Effect |
|-----------|-------|--------|
| `mutation_factor` (F) | 0.4-1.0 | Higher → more exploration |
| `crossover_prob` (CR) | 0.5-0.9 | Higher → faster convergence |
| `strategy` | 'best1bin', 'rand1bin', 'best2bin' | Mutation scheme |

**Strategies:**
- `'best1bin'`: Exploits best solution (fast convergence)
- `'rand1bin'`: More exploration (avoid local optima)
- `'best2bin'`: Balanced (good default)

---

### Genetic Algorithm (GA)

GA uses selection, crossover, and mutation.

```python
optimized_system, history = learner.optimize(
    X, y,
    method='ga',
    population_size=50,
    n_iterations=100,
    crossover_prob=0.8,
    mutation_prob=0.1,
    selection_method='tournament',
    tournament_size=3,
    elitism=True,  # Keep best individuals
    verbose=True
)
```

**GA Parameters:**

```python
learner.optimize(
    X, y,
    method='ga',
    crossover_prob=0.8,     # High crossover (0.7-0.9)
    mutation_prob=0.1,      # Low mutation (0.01-0.1)
    selection_method='tournament',  # or 'roulette', 'rank'
    tournament_size=3,      # Tournament selection size
    elitism=True           # Preserve best solutions
)
```

---

### Comparing Metaheuristics

```python
methods = ['pso', 'de', 'ga']
results = {}

for method in methods:
    system, history = learner.optimize(
        X, y,
        method=method,
        n_iterations=100,
        verbose=False
    )

    y_pred = system.evaluate_batch(X)
    mse = np.mean((y - y_pred['output'])**2)
    results[method] = mse

    print(f"{method.upper()}: MSE = {mse:.4f}")

# Plot convergence
import matplotlib.pyplot as plt
for method in methods:
    _, history = learner.optimize(X, y, method=method, n_iterations=100, verbose=False)
    plt.plot(history['fitness'], label=method.upper())

plt.xlabel('Iteration')
plt.ylabel('Fitness (MSE)')
plt.legend()
plt.yscale('log')
plt.show()
```

**Performance comparison:**

| Method | Speed | Exploration | Stability | Best for |
|--------|-------|-------------|-----------|----------|
| **PSO** | ⚡⚡⚡ Fast | ⭐⭐ Moderate | ⭐⭐⭐ Very stable | Continuous optimization, fast results |
| **DE** | ⚡⚡ Moderate | ⭐⭐⭐ High | ⭐⭐ Moderate | Complex landscapes, avoiding local optima |
| **GA** | ⚡ Slow | ⭐⭐⭐ High | ⭐⭐ Moderate | Discrete+continuous, diverse solutions |

**Rule of thumb:**
- **Start with PSO** (fastest, stable)
- **Switch to DE** if PSO gets stuck
- **Use GA** for discrete optimization (e.g., rule selection)

---

### What Gets Optimized?

You can control what parameters are optimized:

```python
learner = MamdaniLearning(
    n_inputs=1,
    n_outputs=1,
    n_terms=5,
    optimize_mf=True,      # Optimize membership function parameters
    optimize_rules=True,   # Optimize rule weights
    optimize_defuzz=False  # Keep defuzzification method fixed
)
```

**Typical configurations:**

**Configuration 1: MF parameters only**
```python
optimize_mf=True, optimize_rules=False
```
- Fastest
- Good if rule structure is already known
- Fine-tunes MF shapes

**Configuration 2: Full optimization**
```python
optimize_mf=True, optimize_rules=True
```
- Slowest but most flexible
- Optimizes everything
- Best accuracy potential

**Configuration 3: Rules only**
```python
optimize_mf=False, optimize_rules=True
```
- Medium speed
- Good if MFs are well-designed
- Tunes rule weights and operators

---

### Strengths and Limitations

**Strengths:**
- ⭐⭐⭐ Highly interpretable (Mamdani output)
- 🔧 No gradients needed
- 🎯 Can optimize discrete and continuous parameters
- 🌐 Global search (avoids local optima)

**Limitations:**
- ⏱️ Very slow (minutes to hours)
- 🎲 Stochastic (results vary between runs)
- 🔢 Many hyperparameters to tune
- 💾 Memory-intensive for large populations

**When to use:**
- Interpretability is critical
- You have time for optimization
- Gradient-based methods don't work
- You need linguistic outputs (Mamdani)

---

## Choosing a Learning Method

### Decision Tree

```
Do you have labeled data?
├─ No → Use expert knowledge (manual design)
└─ Yes → Continue

How important is interpretability?
├─ Critical → Wang-Mendel or Mamdani Learning
└─ Less important → ANFIS

How much data do you have?
├─ Small (<100 samples) → Wang-Mendel
├─ Medium (100-1000) → ANFIS
└─ Large (>1000) → ANFIS or Mamdani + PSO

How much time can you spend?
├─ Minutes → Wang-Mendel
├─ Hours → ANFIS
└─ Hours to days → Mamdani Learning
```

### Practical Comparison

```python
from fuzzy_systems.learning import WangMendelLearning, ANFIS, MamdaniLearning
import time

# Prepare data
X_train, y_train = ...  # Your data here

# Method 1: Wang-Mendel
start = time.time()
wm = WangMendelLearning(n_inputs=2, n_outputs=1, n_terms=5)
wm_system = wm.fit(X_train, y_train)
wm_time = time.time() - start
wm_pred = wm.predict(X_test)
wm_mse = np.mean((y_test - wm_pred)**2)
print(f"Wang-Mendel: MSE={wm_mse:.4f}, Time={wm_time:.2f}s")

# Method 2: ANFIS
start = time.time()
anfis = ANFIS(n_inputs=2, n_terms=3)
anfis.fit(X_train, y_train, epochs=50, verbose=False)
anfis_time = time.time() - start
anfis_pred = anfis.predict(X_test)
anfis_mse = np.mean((y_test - anfis_pred)**2)
print(f"ANFIS: MSE={anfis_mse:.4f}, Time={anfis_time:.2f}s")

# Method 3: Mamdani + PSO
start = time.time()
ml = MamdaniLearning(n_inputs=2, n_outputs=1, n_terms=5)
ml_system, _ = ml.optimize(X_train, y_train, method='pso',
                           n_iterations=50, verbose=False)
ml_time = time.time() - start
ml_pred = ml_system.evaluate_batch(X_test)['output']
ml_mse = np.mean((y_test - ml_pred)**2)
print(f"Mamdani+PSO: MSE={ml_mse:.4f}, Time={ml_time:.2f}s")
```

---

## Advanced Topics

### Transfer Learning

Start from a pre-trained system:

```python
# Load pre-trained system
base_system = MamdaniSystem.load('pretrained.pkl')

# Fine-tune with new data
learner = MamdaniLearning.from_system(base_system)
optimized_system, _ = learner.optimize(
    X_new, y_new,
    method='pso',
    n_iterations=50  # Fewer iterations needed
)
```

---

### Ensemble Learning

Combine multiple fuzzy systems:

```python
from fuzzy_systems.learning import FuzzyEnsemble

# Train multiple systems
systems = []
for seed in range(5):
    np.random.seed(seed)
    learner = MamdaniLearning(n_inputs=2, n_outputs=1, n_terms=5)
    system, _ = learner.optimize(X_train, y_train, method='pso',
                                  n_iterations=50, verbose=False)
    systems.append(system)

# Create ensemble
ensemble = FuzzyEnsemble(systems, method='average')  # or 'weighted', 'voting'

# Predict
y_pred = ensemble.predict(X_test)
```

---

### Cross-Validation

Evaluate generalization performance:

```python
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []

for train_idx, val_idx in kfold.split(X):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Train ANFIS
    anfis = ANFIS(n_inputs=2, n_terms=3)
    anfis.fit(X_train, y_train, epochs=50, verbose=False)

    # Evaluate
    y_pred = anfis.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    scores.append(mse)

print(f"Cross-validation MSE: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
```

---

### Regularization

Prevent overfitting in ANFIS:

```python
anfis = ANFIS(n_inputs=2, n_terms=3)
anfis.fit(
    X, y,
    epochs=100,
    learning_rate=0.01,
    l2_penalty=0.001,  # L2 regularization on consequent parameters
    dropout=0.1,       # Dropout on rule activations
    validation_split=0.2
)
```

---

## Design Guidelines

### 1. Data Preparation

**Normalization:**
```python
from sklearn.preprocessing import StandardScaler

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# Train on scaled data
anfis.fit(X_scaled, y_scaled, ...)

# Predict and inverse transform
y_pred_scaled = anfis.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
```

**Why normalize:**
- Improves convergence speed
- Balances influence of different features
- Prevents numerical instability

---

### 2. Number of Terms vs Dataset Size

| Dataset Size | Recommended n_terms | Total Rules (2 inputs) |
|--------------|---------------------|------------------------|
| < 50 samples | 3 | 9 |
| 50-200 samples | 3-5 | 9-25 |
| 200-1000 samples | 5-7 | 25-49 |
| > 1000 samples | 7-9 | 49-81 |

**Rule of thumb:** Total rules should be ≤ N/10 where N is dataset size.

---

### 3. Avoiding Overfitting

**Symptoms:**
- Training error very low, test error high
- Validation loss starts increasing after some epochs
- System output is jagged or oscillates

**Solutions:**

**Reduce model complexity:**
```python
# Fewer terms
anfis = ANFIS(n_inputs=2, n_terms=3)  # instead of 5 or 7

# Simpler MF types
learner = MamdaniLearning(mf_type='triangular')  # instead of gaussian
```

**Early stopping:**
```python
anfis.fit(X, y, epochs=200, early_stopping=True, patience=15)
```

**Regularization:**
```python
anfis.fit(X, y, l2_penalty=0.01)
```

**Get more data:**
- Collect more samples
- Use data augmentation (carefully!)
- Use cross-validation to detect overfitting

---

### 4. Hyperparameter Tuning

Use grid search or random search:

```python
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error

# Define parameter grid
param_grid = {
    'n_terms': [3, 5, 7],
    'learning_rate': [0.001, 0.005, 0.01],
    'l2_penalty': [0, 0.001, 0.01]
}

best_score = float('inf')
best_params = None

for params in ParameterGrid(param_grid):
    anfis = ANFIS(n_inputs=2, n_terms=params['n_terms'])
    anfis.fit(X_train, y_train,
              epochs=50,
              learning_rate=params['learning_rate'],
              l2_penalty=params['l2_penalty'],
              verbose=False)

    y_pred = anfis.predict(X_val)
    score = mean_squared_error(y_val, y_pred)

    if score < best_score:
        best_score = score
        best_params = params

print(f"Best params: {best_params}")
print(f"Best MSE: {best_score:.4f}")
```

---

## Troubleshooting

### Problem: ANFIS loss is NaN

**Causes:**
- Learning rate too high
- Numerical overflow

**Solutions:**
```python
# Reduce learning rate
anfis.fit(X, y, learning_rate=0.001)

# Normalize data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Use gradient clipping
anfis.fit(X, y, clip_gradients=True, max_grad_norm=1.0)
```

---

### Problem: Metaheuristic optimization is stuck

**Symptoms:**
- Fitness doesn't improve after many iterations
- All particles/individuals converge to same solution

**Solutions:**

**Increase diversity:**
```python
# PSO: increase inertia
learner.optimize(X, y, method='pso', inertia=0.9)

# DE: increase mutation factor
learner.optimize(X, y, method='de', mutation_factor=0.9)

# GA: increase mutation probability
learner.optimize(X, y, method='ga', mutation_prob=0.2)
```

**Increase population:**
```python
learner.optimize(X, y, method='pso', n_particles=50)  # instead of 30
```

**Try different method:**
```python
# If PSO stuck, try DE
learner.optimize(X, y, method='de')
```

---

### Problem: Wang-Mendel generates too many rules

**Cause:** Too many terms or sparse data distribution.

**Solutions:**

**Reduce n_terms:**
```python
learner = WangMendelLearning(n_terms=3)  # instead of 5 or 7
```

**Prune rules after learning:**
```python
system = learner.fit(X, y)

# Remove rules with low activation
learner.prune_rules(min_activation=0.1)

# Or keep only top K rules
learner.keep_top_rules(k=20)
```

---

### Problem: Learning is too slow

**For ANFIS:**
```python
# Reduce epochs
anfis.fit(X, y, epochs=30)  # instead of 100

# Increase batch size
anfis.fit(X, y, batch_size=128)  # instead of 32

# Use hybrid learning
anfis = ANFIS(learning_method='hybrid')
```

**For metaheuristics:**
```python
# Reduce population and iterations
learner.optimize(X, y, method='pso',
                 n_particles=20, n_iterations=50)

# Parallelize (if available)
learner.optimize(X, y, method='pso', n_jobs=-1)
```

---

## Next Steps

- **[Inference Systems](inference_systems.md)**: Build systems manually before learning
- **[API Reference: Learning](../api_reference/learning.md)**: Complete method documentation
- **[Examples: Learning](../examples/gallery.md#learning)**: Interactive notebooks

---

## Further Reading

- **Wang, L. X., & Mendel, J. M. (1992)**: "Generating fuzzy rules by learning from examples". *IEEE Transactions on Systems, Man, and Cybernetics*, 22(6), 1414-1427.
- **Jang, J. S. (1993)**: "ANFIS: adaptive-network-based fuzzy inference system". *IEEE Transactions on Systems, Man, and Cybernetics*, 23(3), 665-685.
- **Eberhart, R., & Kennedy, J. (1995)**: "Particle swarm optimization". *Proceedings of ICNN'95*, Vol. 4, 1942-1948.
- **Storn, R., & Price, K. (1997)**: "Differential evolution–a simple and efficient heuristic for global optimization over continuous spaces". *Journal of Global Optimization*, 11(4), 341-359.
