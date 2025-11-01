# ANFIS Quick Start Guide

## Overview

ANFIS (Adaptive Neuro-Fuzzy Inference System) is a hybrid learning system that combines neural networks and fuzzy logic for supervised learning tasks.

---

## 1. Instantiate ANFIS Class

```python
from fuzzy_systems.learning import ANFIS

# Create ANFIS instance
anfis = ANFIS(
    n_inputs=2,                    # Number of input variables
    n_mfs=[3, 3],                  # Number of membership functions per input
    mf_type='gaussmf',             # Type: 'gaussmf', 'gbellmf', or 'sigmf'
    learning_rate=0.01,            # Learning rate for gradient descent
    input_ranges=[(-5, 5), (-5, 5)],  # Optional: input bounds (default: (-8, 8))
    lambda_l1=0.0,                 # L1 regularization (Lasso) on MF widths
    lambda_l2=0.01,                # L2 regularization (Ridge) on MF widths
    batch_size=32,                 # Batch size (None=batch, 1=SGD, 16-128=minibatch)
    use_adaptive_lr=False,         # Use adaptive learning rate (slower but guaranteed convergence)
    classification=False           # Set True for classification tasks
)
```

### Key Parameters

- **`n_inputs`**: Number of input features
- **`n_mfs`**: Number of membership functions per input (int or list)
- **`mf_type`**:
  - `'gaussmf'`: Gaussian (parameters: center, sigma)
  - `'gbellmf'`: Generalized bell (parameters: width, slope, center)
  - `'sigmf'`: Sigmoid (parameters: slope, center)
- **`learning_rate`**: Step size for gradient descent (typical: 0.001-0.1)
- **`lambda_l1/lambda_l2`**: Regularization coefficients to prevent overfitting
- **`batch_size`**:
  - `None`: Full batch gradient descent
  - `1`: Stochastic gradient descent
  - `16-128`: Mini-batch (recommended)

---

## 2. Training with `fit()` - Hybrid Learning

The `fit()` method uses **hybrid learning**:
- **Least Squares Estimation (LSE)** for consequent parameters (fast, analytical)
- **Gradient Descent** for premise parameters (membership functions)

```python
# Prepare data
X_train = ...  # shape: (n_samples, n_inputs)
y_train = ...  # shape: (n_samples,)
X_val = ...    # Optional validation set
y_val = ...

# Train with hybrid learning
anfis.fit(
    X=X_train,
    y=y_train,
    epochs=100,                        # Number of training epochs
    verbose=True,                      # Print progress
    train_premises=True,               # Adjust membership functions (True) or only consequents (False)
    X_val=X_val,                       # Optional: validation data
    y_val=y_val,
    early_stopping_patience=20,        # Stop if no improvement for N epochs
    restore_best_weights=True          # Restore best model when early stopping
)

# Make predictions
y_pred = anfis.predict(X_test)
```

### Key Parameters

- **`epochs`**: Number of training iterations
- **`train_premises`**:
  - `True`: Adjust both MFs and consequents (full hybrid learning)
  - `False`: Only adjust consequents via LSE (faster, simpler)
- **`early_stopping_patience`**: Stop training if validation loss doesn't improve for N epochs
- **`restore_best_weights`**: Automatically restore best model (requires validation data)

### When to Use `fit()`

✅ **Good for:**
- Fast convergence with gradient-based optimization
- When you have good initialization
- Computational efficiency (especially with `train_premises=False`)

⚠️ **Limitations:**
- May converge to local minima
- Requires tuning learning rate
- Gradient vanishing/exploding in deep networks

---

## 3. Training with `fit_metaheuristic()` - Evolutionary Optimization

The `fit_metaheuristic()` method uses **metaheuristic algorithms** to optimize all parameters simultaneously:
- **PSO** (Particle Swarm Optimization)
- **DE** (Differential Evolution)
- **GA** (Genetic Algorithm)

```python
# Train with metaheuristic optimization
anfis.fit_metaheuristic(
    X=X_train,
    y=y_train,
    optimizer='pso',                   # Optimizer: 'pso', 'de', or 'ga'
    n_particles=30,                    # Population size
    n_iterations=100,                  # Number of optimization iterations
    verbose=True,                      # Print progress
    X_val=X_val,                       # Optional: validation data
    y_val=y_val,
    early_stopping_patience=20,        # Stop if no improvement for N iterations
    restore_best_weights=True,         # Restore best parameters
    # PSO-specific parameters
    w=0.7,                             # Inertia weight
    c1=1.5,                            # Cognitive parameter
    c2=1.5,                            # Social parameter
    # DE-specific parameters
    # F=0.8,                           # Differential weight
    # CR=0.9,                          # Crossover probability
    # GA-specific parameters
    # crossover_rate=0.8,
    # mutation_rate=0.1
)

# Make predictions
y_pred = anfis.predict(X_test)
```

### Key Parameters

- **`optimizer`**: Choose optimization algorithm:
  - `'pso'`: Particle Swarm Optimization (recommended)
  - `'de'`: Differential Evolution (robust)
  - `'ga'`: Genetic Algorithm (exploratory)
- **`n_particles`**: Population size (typical: 20-50)
- **`n_iterations`**: Number of optimization iterations (typical: 50-200)

### Optimizer-Specific Parameters

**PSO:**
- `w`: Inertia weight (0.4-0.9, balances exploration/exploitation)
- `c1`: Cognitive parameter (1.0-2.0, personal best attraction)
- `c2`: Social parameter (1.0-2.0, global best attraction)

**DE:**
- `F`: Differential weight (0.5-1.0, mutation scale)
- `CR`: Crossover probability (0.7-0.95, recombination rate)

**GA:**
- `crossover_rate`: Crossover probability (0.7-0.9)
- `mutation_rate`: Mutation probability (0.01-0.1)

### When to Use `fit_metaheuristic()`

✅ **Good for:**
- Global optimization (avoids local minima)
- No need to tune learning rates
- Robust to initialization
- Optimizes all parameters simultaneously

⚠️ **Limitations:**
- Slower than gradient descent
- Requires more function evaluations
- Stochastic (results may vary between runs)

---

## 4. Comparison: `fit()` vs `fit_metaheuristic()`

| Aspect | `fit()` | `fit_metaheuristic()` |
|--------|---------|----------------------|
| **Algorithm** | Gradient Descent + LSE | PSO/DE/GA |
| **Convergence** | Local optimum | Global optimum |
| **Speed** | Fast | Slower |
| **Stability** | Requires LR tuning | Self-adaptive |
| **Consequents** | LSE (analytical) | Evolutionary |
| **Premises** | Gradient (analytical) | Evolutionary |
| **Regularization** | Widths only | All parameters |
| **Best for** | Quick training, good init | Global search, robust |

---

## 5. Complete Example

```python
import numpy as np
from fuzzy_systems.learning import ANFIS
from sklearn.model_selection import train_test_split

# Generate synthetic data
X = np.random.uniform(-5, 5, (500, 2))
y = np.sin(X[:, 0]) + np.cos(X[:, 1]) + np.random.normal(0, 0.1, 500)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Create ANFIS
anfis = ANFIS(
    n_inputs=2,
    n_mfs=[3, 3],
    mf_type='gaussmf',
    learning_rate=0.01,
    lambda_l2=0.01,
    batch_size=32
)

# Option 1: Train with hybrid learning
anfis.fit(
    X_train, y_train,
    epochs=100,
    X_val=X_val, y_val=y_val,
    early_stopping_patience=20,
    verbose=True
)

# Option 2: Train with metaheuristics (uncomment to use)
# anfis.fit_metaheuristic(
#     X_train, y_train,
#     optimizer='pso',
#     n_particles=30,
#     n_iterations=100,
#     X_val=X_val, y_val=y_val,
#     verbose=True
# )

# Evaluate
y_pred = anfis.predict(X_test)
mse = np.mean((y_test - y_pred) ** 2)
print(f"Test MSE: {mse:.4f}")

# Visualize (if 2D input)
anfis.visualizar_mfs()  # Plot membership functions
```

---

## 6. Tips and Best Practices

### Initialization
- Start with 2-5 membership functions per input
- Use Gaussian MFs (`gaussmf`) for smooth, general-purpose modeling
- Set appropriate `input_ranges` if you know the data bounds

### Training
- **Use validation data** to prevent overfitting
- **Enable early stopping** to save computation time
- Start with `train_premises=False` for quick baseline

### Hyperparameter Tuning
- **Learning rate**: Start with 0.01, decrease if unstable
- **Regularization**: Use `lambda_l2=0.001-0.1` to prevent overfitting
- **Batch size**: 32-64 is usually optimal

### When Results Differ
If `fit()` and `fit_metaheuristic()` give very different results:
- `fit()` likely found a local minimum
- `fit_metaheuristic()` likely found a better global minimum
- Try multiple random seeds for `fit_metaheuristic()` to verify consistency

### Performance
- **Speed**: `fit()` with `train_premises=False` > `fit()` > `fit_metaheuristic()`
- **Accuracy**: `fit_metaheuristic()` ≥ `fit()` (but stochastic)

---

## 7. Troubleshooting

| Problem | Solution |
|---------|----------|
| Training unstable | Reduce learning rate, increase regularization |
| Overfitting | Add L2 regularization, use early stopping, reduce n_mfs |
| Slow convergence | Increase learning rate, use adaptive LR, try metaheuristics |
| Poor performance | Increase n_mfs, try different mf_type, use metaheuristics |
| NaN values | Check input ranges, reduce learning rate, add regularization |

---

## References

- Jang, J. S. (1993). "ANFIS: adaptive-network-based fuzzy inference system." IEEE Transactions on Systems, Man, and Cybernetics, 23(3), 665-685.
