# 03. Learning and Optimization

Automatic rule generation and fuzzy system optimization using `fuzzy_systems.learning`.

## Notebooks

### 01. Wang-Mendel Nonlinear
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1moi6/pyfuzzy-toolbox/blob/main/notebooks_colab/03_learning/01_wang_mendel_nonlinear.ipynb)

**Topics:**
- Wang-Mendel algorithm for automatic rule generation
- Function approximation: f(x) = sin(x) + 0.1x
- Single-pass learning from data
- Rule conflict resolution

**Key Classes:**
```python
from fuzzy_systems.learning import WangMendelLearning
from fuzzy_systems import MamdaniSystem
```

**Example:**
```python
# Create base system
system = MamdaniSystem()
system.add_input('x', (0, 2*np.pi))
system.add_output('y', (-2, 2))

# Add partitions (11 terms for input/output)
# ... partition definitions ...

# Learn rules from data
wm = WangMendelLearning(system, X_train, y_train)
trained_system = wm.fit(verbose=True)

# Predict
y_pred = wm.predict(X_test)

# Get statistics
stats = wm.get_training_stats()
# {'candidate_rules': 50, 'final_rules': 32, 'conflicts_resolved': 18}
```

---

### 02. Wang-Mendel Linear
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1moi6/pyfuzzy-toolbox/blob/main/notebooks_colab/03_learning/02_wang_mendel_linear.ipynb)

**Topics:**
- Wang-Mendel for linear functions
- Simpler case study
- Performance metrics (MSE, RMSE, R²)

---

### 03. Wang-Mendel Iris Classification
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1moi6/pyfuzzy-toolbox/blob/main/notebooks_colab/03_learning/03_wang_mendel_iris.ipynb)

**Topics:**
- Classification with Wang-Mendel
- Multi-class fuzzy classification
- Iris dataset example

---

### 04. ANFIS Iris
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1moi6/pyfuzzy-toolbox/blob/main/notebooks_colab/03_learning/04_anfis_iris.ipynb)

**Topics:**
- ANFIS (Adaptive Neuro-Fuzzy Inference System)
- Gradient-based learning
- Lyapunov stability monitoring
- Classification with ANFIS

**Key Classes:**
```python
from fuzzy_systems.learning import ANFIS
```

**Example:**
```python
# Create ANFIS
anfis = ANFIS(n_inputs=2, n_terms=3)

# Train with gradient descent
anfis.fit(X_train, y_train, epochs=50, learning_rate=0.01, verbose=True)

# Predict
y_pred = anfis.predict(X_test)

# Access training history
history = anfis.get_training_history()
# {'epochs': [...], 'loss': [...], 'lyapunov': [...]}
```

---

### 05. Rules Optimization
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1moi6/pyfuzzy-toolbox/blob/main/notebooks_colab/03_learning/05_rules_optimization.ipynb)

**Topics:**
- Optimize Mamdani systems with metaheuristics
- PSO (Particle Swarm Optimization)
- Differential Evolution (DE)
- Genetic Algorithms (GA)

**Key Classes:**
```python
from fuzzy_systems.learning import MamdaniLearning
```

**Example:**
```python
# Create Mamdani system
system = MamdaniSystem()
# ... define variables and initial rules ...

# Convert to learning system
learner = MamdaniLearning.from_mamdani(system)

# Optimize with PSO
learner.fit(
    X_train, y_train,
    method='pso',
    n_particles=30,
    max_iter=100
)

# Predict
y_pred = learner.predict(X_test)

# Convert back to Mamdani
optimized_system = learner.to_mamdani()
```

---

### 06. ANFIS Regression
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1moi6/pyfuzzy-toolbox/blob/main/notebooks_colab/03_learning/06_anfis_regression.ipynb)

**Topics:**
- ANFIS for regression problems
- Nonlinear function approximation
- Comparison with other methods

---

### 07. Rules Optimization Iris
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1moi6/pyfuzzy-toolbox/blob/main/notebooks_colab/03_learning/07_rules_optimization_iris.ipynb)

**Topics:**
- Classification with optimized rules
- Metaheuristic comparison (PSO vs DE vs GA)
- Iris dataset

---

## What You'll Learn

- ✅ Generate rules automatically with Wang-Mendel
- ✅ Train ANFIS with gradient descent
- ✅ Optimize fuzzy systems with metaheuristics (PSO, DE, GA)
- ✅ Monitor Lyapunov stability during training
- ✅ Convert between MamdaniSystem ↔ MamdaniLearning
- ✅ Apply to regression and classification problems
- ✅ Compare learning methods

## Prerequisites

```bash
pip install pyfuzzy-toolbox[ml]
```

## Key Algorithms

| Algorithm | Type | Best For |
|-----------|------|----------|
| **Wang-Mendel** | Single-pass | Quick rule generation, interpretable rules |
| **ANFIS** | Gradient descent | Precise approximation, differentiable problems |
| **PSO** | Metaheuristic | Global optimization, non-differentiable |
| **DE** | Metaheuristic | Robust optimization, fewer parameters |
| **GA** | Metaheuristic | Discrete/combinatorial optimization |

## Next Steps

- **[04_dynamics](../04_dynamics/)**: Apply learning to dynamic systems
- **[01_fundamentals](../01_fundamentals/)**: Review fuzzy basics
