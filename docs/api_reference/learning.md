# Learning API Reference

The `fuzzy_systems.learning` module provides algorithms for automatic rule generation and system optimization:

- **WangMendelLearning**: Automatic rule generation from data (single-pass algorithm)
- **ANFIS**: Adaptive Neuro-Fuzzy Inference System (gradient-based learning)
- **MamdaniLearning**: Mamdani system optimization with gradients and metaheuristics
- **Metaheuristics**: PSO, Differential Evolution, Genetic Algorithms

---

## WangMendelLearning

Automatic fuzzy rule generation using the Wang-Mendel algorithm (1992).

**Reference:**
Wang, L. X., & Mendel, J. M. (1992). "Generating fuzzy rules by learning from examples." *IEEE Transactions on Systems, Man, and Cybernetics*, 22(6), 1414-1427.

### Algorithm Steps

1. **Partition** variable domains (use existing MFs)
2. **Generate** candidate rules from each data sample
3. **Assign** degree to each rule based on membership strengths
4. **Resolve** conflicts (keep rule with highest degree)
5. **Create** final fuzzy system with learned rules

### Constructor

```python
WangMendelLearning(system, X, y, task='auto',
                   scale_classification=True, verbose_init=False)
```

**Parameters:**

- `system` (MamdaniSystem): Pre-configured system with variables and terms (NO rules yet)
- `X` (ndarray): Input data, shape `(n_samples, n_features)`
- `y` (ndarray): Output data, shape `(n_samples,)` or `(n_samples, n_outputs)`
- `task` (str): `'auto'` (detect), `'regression'`, or `'classification'` (default: `'auto'`)
- `scale_classification` (bool): Scale classification outputs to [0, 1] (default: `True`)
- `verbose_init` (bool): Print initialization info (default: `False`)

**Example:**
```python
import numpy as np
from fuzzy_systems import MamdaniSystem
from fuzzy_systems.learning import WangMendelLearning

# Prepare data
X_train = np.random.uniform(0, 10, (100, 2))
y_train = np.sin(X_train[:, 0]) + np.cos(X_train[:, 1])

# Create base system (with variables and terms, NO rules)
system = MamdaniSystem()
system.add_input('x1', (0, 10))
system.add_input('x2', (0, 10))
system.add_output('y', (-2, 2))

# Add partitions (e.g., 5 terms per variable)
for var in ['x1', 'x2']:
    for i in range(5):
        center = i * 2.5
        system.add_term(var, f'term_{i}', 'triangular',
                       (max(0, center-2.5), center, min(10, center+2.5)))

# Similar for output
for i in range(5):
    center = -2 + i * 1.0
    system.add_term('y', f'out_{i}', 'triangular',
                   (max(-2, center-1), center, min(2, center+1)))

# Learn rules from data
wm = WangMendelLearning(system, X_train, y_train)
wm.fit(verbose=True)
```

---

### Methods

#### `.fit(verbose=False)`

Generate fuzzy rules from the training data.

**Parameters:**
- `verbose` (bool): Print progress information (default: `False`)

**Returns:** `MamdaniSystem` - The trained fuzzy system

**Example:**
```python
trained_system = wm.fit(verbose=True)
```

**Output (verbose=True):**
```
🔄 Wang-Mendel Algorithm Starting...
   ✓ Generated 100 candidate rules
   ✓ Resolved 23 conflicts
   ✓ Final rule base: 77 rules
✅ Wang-Mendel training complete!
```

---

#### `.predict(X)`

Predict outputs for new inputs.

**Parameters:**
- `X` (ndarray): Input data, shape `(n_samples, n_features)`

**Returns:** `ndarray` - Predicted outputs

**For Regression:**
```python
y_pred = wm.predict(X_test)  # Shape: (n_samples, n_outputs)
```

**For Classification:**
```python
y_pred_classes = wm.predict(X_test)  # Shape: (n_samples,) - class indices
```

---

#### `.predict_proba(X)` (Classification only)

Predict class probabilities.

**Parameters:**
- `X` (ndarray): Input data

**Returns:** `ndarray` - Probability matrix, shape `(n_samples, n_classes)`

**Example:**
```python
proba = wm.predict_proba(X_test)
print(proba[0])  # [0.1, 0.7, 0.2] for 3 classes
```

---

#### `.get_training_stats()`

Get statistics about the training process.

**Returns:** `dict` - Training statistics:
```python
{
    'candidate_rules': 100,      # Rules generated from data
    'final_rules': 77,           # Rules after conflict resolution
    'conflicts_resolved': 23,    # Number of conflicts
    'task': 'regression'         # Task type
}
```

**Example:**
```python
stats = wm.get_training_stats()
print(f"Generated {stats['candidate_rules']} rules")
print(f"Final: {stats['final_rules']} rules")
```

---

### Complete Example: Regression

```python
import numpy as np
from fuzzy_systems import MamdaniSystem
from fuzzy_systems.learning import WangMendelLearning

# Generate nonlinear data
X_train = np.linspace(0, 2*np.pi, 50).reshape(-1, 1)
y_train = np.sin(X_train) + 0.1*X_train

# Create system with 11 partitions
system = MamdaniSystem()
system.add_input('x', (0, 2*np.pi))
system.add_output('y', (-2, 2))

# Add 11 triangular terms to input and output
n_terms = 11
for i in range(n_terms):
    # Input terms
    center_x = i * (2*np.pi) / (n_terms - 1)
    width = (2*np.pi) / (n_terms - 1)
    system.add_term('x', f'x_{i}', 'triangular',
                   (max(0, center_x - width),
                    center_x,
                    min(2*np.pi, center_x + width)))

    # Output terms
    center_y = -2 + i * 4 / (n_terms - 1)
    width_y = 4 / (n_terms - 1)
    system.add_term('y', f'y_{i}', 'triangular',
                   (max(-2, center_y - width_y),
                    center_y,
                    min(2, center_y + width_y)))

# Train Wang-Mendel
wm = WangMendelLearning(system, X_train, y_train)
wm.fit(verbose=True)

# Predict
X_test = np.linspace(0, 2*np.pi, 200).reshape(-1, 1)
y_pred = wm.predict(X_test)

# Evaluate
from sklearn.metrics import mean_squared_error, r2_score
y_true = np.sin(X_test) + 0.1*X_test
mse = mean_squared_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")
```

---

### Complete Example: Classification

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from fuzzy_systems import MamdaniSystem
from fuzzy_systems.learning import WangMendelLearning

# Load Iris dataset
iris = load_iris()
X = iris.data[:, [2, 3]]  # Use petal length and width
y = iris.target

# One-hot encode targets
encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y.reshape(-1, 1))

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_onehot, test_size=0.3, random_state=42
)

# Create system with 3 terms per input, 1 output per class
system = MamdaniSystem()
system.add_input('petal_length', (X[:, 0].min(), X[:, 0].max()))
system.add_input('petal_width', (X[:, 1].min(), X[:, 1].max()))

# Add 3 binary outputs (one per class)
for i in range(3):
    system.add_output(f'class_{i}', (0, 1))

# Add terms (3 per variable)
for var in ['petal_length', 'petal_width']:
    universe = system.input_variables[var].universe
    for i in range(3):
        center = universe[0] + i * (universe[1] - universe[0]) / 2
        width = (universe[1] - universe[0]) / 3
        system.add_term(var, f'term_{i}', 'triangular',
                       (max(universe[0], center - width),
                        center,
                        min(universe[1], center + width)))

# Add output terms (2 per class: 0 and 1)
for i in range(3):
    system.add_term(f'class_{i}', 'no', 'triangular', (0, 0, 0.5))
    system.add_term(f'class_{i}', 'yes', 'triangular', (0.5, 1, 1))

# Train
wm = WangMendelLearning(system, X_train, y_train, task='classification')
wm.fit(verbose=True)

# Predict
y_pred_classes = wm.predict(X_test)
y_pred_proba = wm.predict_proba(X_test)

# Evaluate
from sklearn.metrics import accuracy_score, classification_report
y_test_classes = y_test.argmax(axis=1)
accuracy = accuracy_score(y_test_classes, y_pred_classes)

print(f"Accuracy: {accuracy:.2%}")
print("\nClassification Report:")
print(classification_report(y_test_classes, y_pred_classes,
                           target_names=iris.target_names))
```

---

## ANFIS

Adaptive Neuro-Fuzzy Inference System with gradient-based learning.

### Constructor

```python
ANFIS(n_inputs, n_terms, n_outputs=1, mf_type='gaussian')
```

**Parameters:**

- `n_inputs` (int): Number of input variables
- `n_terms` (int): Number of membership functions per input
- `n_outputs` (int): Number of outputs (default: `1`)
- `mf_type` (str): Membership function type: `'gaussian'`, `'bell'` (default: `'gaussian'`)

**Example:**
```python
from fuzzy_systems.learning import ANFIS

anfis = ANFIS(n_inputs=2, n_terms=3, n_outputs=1)
```

---

### Methods

#### `.fit(X, y, epochs=100, learning_rate=0.01, batch_size=None, validation_split=0.0, early_stopping=False, patience=10, lyapunov_check=True, verbose=True)`

Train ANFIS using gradient descent with backpropagation.

**Parameters:**

- `X` (ndarray): Input data, shape `(n_samples, n_inputs)`
- `y` (ndarray): Output data, shape `(n_samples, n_outputs)` or `(n_samples,)`
- `epochs` (int): Number of training epochs (default: `100`)
- `learning_rate` (float): Learning rate (default: `0.01`)
- `batch_size` (int, optional): Batch size for mini-batch gradient descent. If None, uses full batch
- `validation_split` (float): Fraction of data for validation (default: `0.0`)
- `early_stopping` (bool): Stop if validation loss doesn't improve (default: `False`)
- `patience` (int): Epochs to wait before early stopping (default: `10`)
- `lyapunov_check` (bool): Monitor Lyapunov stability (default: `True`)
- `verbose` (bool): Print training progress (default: `True`)

**Returns:** `dict` - Training history

**Example:**
```python
history = anfis.fit(
    X_train, y_train,
    epochs=50,
    learning_rate=0.01,
    validation_split=0.2,
    early_stopping=True,
    verbose=True
)
```

**Output (verbose=True):**
```
Epoch 10/50 - Loss: 0.0234 - Val Loss: 0.0251 - Lyapunov: 0.98
Epoch 20/50 - Loss: 0.0156 - Val Loss: 0.0178 - Lyapunov: 0.99
...
✅ Training complete!
```

---

#### `.predict(X)`

Predict outputs for new inputs.

**Parameters:**
- `X` (ndarray): Input data, shape `(n_samples, n_inputs)`

**Returns:** `ndarray` - Predictions, shape `(n_samples, n_outputs)`

**Example:**
```python
y_pred = anfis.predict(X_test)
```

---

#### `.get_training_history()`

Get complete training history.

**Returns:** `dict` - History with keys:
```python
{
    'epochs': [1, 2, 3, ...],
    'loss': [0.5, 0.3, 0.2, ...],
    'val_loss': [0.6, 0.4, 0.25, ...],  # If validation_split > 0
    'lyapunov': [0.95, 0.97, 0.99, ...]  # If lyapunov_check=True
}
```

**Example:**
```python
history = anfis.get_training_history()

import matplotlib.pyplot as plt
plt.plot(history['epochs'], history['loss'], label='Training')
plt.plot(history['epochs'], history['val_loss'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

---

### Complete Example: ANFIS Regression

```python
import numpy as np
from fuzzy_systems.learning import ANFIS

# Generate data
X_train = np.random.uniform(0, 10, (200, 2))
y_train = np.sin(X_train[:, 0]) + np.cos(X_train[:, 1])

X_test = np.random.uniform(0, 10, (50, 2))
y_true = np.sin(X_test[:, 0]) + np.cos(X_test[:, 1])

# Create and train ANFIS
anfis = ANFIS(n_inputs=2, n_terms=5, n_outputs=1, mf_type='gaussian')

history = anfis.fit(
    X_train, y_train,
    epochs=100,
    learning_rate=0.01,
    validation_split=0.2,
    early_stopping=True,
    patience=10,
    verbose=True
)

# Predict
y_pred = anfis.predict(X_test)

# Evaluate
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")

# Plot training curve
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history['epochs'], history['loss'], label='Train')
plt.plot(history['epochs'], history['val_loss'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Learning Curve')

plt.subplot(1, 2, 2)
plt.plot(history['epochs'], history['lyapunov'])
plt.xlabel('Epoch')
plt.ylabel('Lyapunov Stability')
plt.title('Stability Monitoring')
plt.tight_layout()
plt.show()
```

---

## MamdaniLearning

Optimize Mamdani systems using gradients or metaheuristics.

### Constructor

```python
MamdaniLearning(system=None, X=None, y=None)
```

**Parameters:**

- `system` (MamdaniSystem, optional): Existing Mamdani system to optimize
- `X` (ndarray, optional): Training input data
- `y` (ndarray, optional): Training output data

**Example:**
```python
from fuzzy_systems.learning import MamdaniLearning

# Create from existing system
learner = MamdaniLearning.from_mamdani(system, X_train, y_train)

# Or create new
learner = MamdaniLearning()
# ... configure ...
```

---

### Class Methods

#### `.from_mamdani(system, X, y)`

Create MamdaniLearning from existing MamdaniSystem.

**Parameters:**
- `system` (MamdaniSystem): Existing fuzzy system
- `X` (ndarray): Training inputs
- `y` (ndarray): Training outputs

**Returns:** `MamdaniLearning` - Learner instance

**Example:**
```python
learner = MamdaniLearning.from_mamdani(system, X_train, y_train)
```

---

### Methods

#### `.fit(X, y, method='gradient', epochs=100, learning_rate=0.01, **kwargs)`

Optimize the fuzzy system.

**Parameters:**

- `X` (ndarray): Training input data
- `y` (ndarray): Training output data
- `method` (str): Optimization method:
    - `'gradient'`: Gradient descent
    - `'pso'`: Particle Swarm Optimization
    - `'de'`: Differential Evolution
    - `'ga'`: Genetic Algorithm
- `epochs` (int): Number of iterations (default: `100`)
- `learning_rate` (float): Learning rate for gradient (default: `0.01`)
- `**kwargs`: Method-specific parameters

**Gradient-specific kwargs:**
- `batch_size` (int): Batch size for mini-batch
- `momentum` (float): Momentum factor

**PSO-specific kwargs:**
- `n_particles` (int): Number of particles (default: `30`)
- `inertia` (float): Inertia weight (default: `0.7`)
- `cognitive` (float): Cognitive parameter (default: `1.5`)
- `social` (float): Social parameter (default: `1.5`)

**DE-specific kwargs:**
- `population_size` (int): Population size (default: `50`)
- `mutation_factor` (float): Mutation factor F (default: `0.8`)
- `crossover_prob` (float): Crossover probability (default: `0.9`)

**GA-specific kwargs:**
- `population_size` (int): Population size (default: `50`)
- `mutation_rate` (float): Mutation rate (default: `0.1`)
- `crossover_rate` (float): Crossover rate (default: `0.8`)

**Example:**
```python
# Gradient descent
learner.fit(X_train, y_train, method='gradient',
           epochs=100, learning_rate=0.01)

# PSO
learner.fit(X_train, y_train, method='pso',
           epochs=50, n_particles=30)

# Differential Evolution
learner.fit(X_train, y_train, method='de',
           epochs=100, population_size=50)
```

---

#### `.predict(X)`

Predict outputs using the optimized system.

**Parameters:**
- `X` (ndarray): Input data

**Returns:** `ndarray` - Predictions

---

#### `.to_mamdani()`

Convert back to MamdaniSystem.

**Returns:** `MamdaniSystem` - Optimized fuzzy system

**Example:**
```python
optimized_system = learner.to_mamdani()
optimized_system.plot_variables()
```

---

### Complete Example: Optimization with PSO

```python
import numpy as np
from fuzzy_systems import MamdaniSystem
from fuzzy_systems.learning import MamdaniLearning

# Generate data
X_train = np.linspace(-5, 5, 100).reshape(-1, 1)
y_train = -2 * X_train + 5 + np.random.normal(0, 0.5, X_train.shape)

# Create initial system
system = MamdaniSystem()
system.add_input('x', (-5, 5))
system.add_output('y', (-15, 15))

# Add terms with suboptimal initial parameters
system.add_term('x', 'low', 'triangular', (-5, -2, 1))
system.add_term('x', 'high', 'triangular', (-1, 2, 5))
system.add_term('y', 'low', 'triangular', (-15, -7, 1))
system.add_term('y', 'high', 'triangular', (-1, 7, 15))

# Initial rules
system.add_rules([('low', 'high'), ('high', 'low')])

# Create learner
learner = MamdaniLearning.from_mamdani(system, X_train, y_train)

# Optimize with PSO
history = learner.fit(
    X_train, y_train,
    method='pso',
    epochs=100,
    n_particles=30,
    verbose=True
)

# Predict
y_pred = learner.predict(X_train)

# Evaluate
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_train, y_pred)
print(f"MSE after optimization: {mse:.4f}")

# Get optimized system
optimized_system = learner.to_mamdani()
optimized_system.save('optimized_system.pkl')
```

---

## Metaheuristics

Direct access to optimization algorithms.

### PSO

Particle Swarm Optimization.

```python
from fuzzy_systems.learning import PSO

optimizer = PSO(
    objective_func,
    bounds,
    n_particles=30,
    max_iter=100,
    inertia=0.7,
    cognitive=1.5,
    social=1.5
)

best_params, best_cost = optimizer.optimize()
```

---

### DE

Differential Evolution.

```python
from fuzzy_systems.learning import DE

optimizer = DE(
    objective_func,
    bounds,
    population_size=50,
    max_iter=100,
    mutation_factor=0.8,
    crossover_prob=0.9
)

best_params, best_cost = optimizer.optimize()
```

---

### GA

Genetic Algorithm.

```python
from fuzzy_systems.learning import GA

optimizer = GA(
    objective_func,
    bounds,
    population_size=50,
    max_iter=100,
    mutation_rate=0.1,
    crossover_rate=0.8
)

best_params, best_cost = optimizer.optimize()
```

---

## Comparison Table

| Method | Type | Speed | Accuracy | Best For |
|--------|------|-------|----------|----------|
| **Wang-Mendel** | Rule generation | ⚡⚡⚡ Fast | ⭐⭐ Good | Quick prototyping, interpretable rules |
| **ANFIS** | Neuro-fuzzy | ⚡⚡ Medium | ⭐⭐⭐ Excellent | Precise approximation, differentiable problems |
| **MamdaniLearning (Gradient)** | Gradient | ⚡⚡ Medium | ⭐⭐⭐ Excellent | Fine-tuning existing systems |
| **MamdaniLearning (PSO)** | Metaheuristic | ⚡ Slow | ⭐⭐⭐ Excellent | Non-differentiable, global search |
| **MamdaniLearning (DE)** | Metaheuristic | ⚡ Slow | ⭐⭐⭐ Excellent | Robust optimization, fewer parameters |
| **MamdaniLearning (GA)** | Metaheuristic | ⚡ Slow | ⭐⭐ Good | Discrete/combinatorial optimization |

---

## See Also

- [Core API](core.md) - Fuzzy sets and operators
- [Inference API](inference.md) - Mamdani and Sugeno systems
- [Dynamics API](dynamics.md) - Dynamic fuzzy systems
- [User Guide: Learning](../user_guide/learning.md) - Detailed tutorials
- [Examples](../examples/gallery.md) - Interactive notebooks
