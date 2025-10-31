# 02. Inference Systems

Complete fuzzy inference systems using `fuzzy_systems.inference`.

## Notebooks

### 01. Mamdani Tipping System
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1moi6/pyfuzzy-toolbox/blob/main/notebooks_colab/02_inference/01_mamdani_tipping.ipynb)

**Topics:**
- Complete Mamdani inference system
- 5 Mamdani steps: fuzzification, rule application, implication, aggregation, defuzzification
- Multiple inputs (service, food quality) → output (tip)
- 3D control surfaces

**Key Classes:**
```python
from fuzzy_systems import MamdaniSystem
```

**Example:**
```python
# Create Mamdani system
system = MamdaniSystem()

# Add inputs
system.add_input('service', (0, 10))
system.add_term('service', 'poor', 'triangular', (0, 0, 5))
system.add_term('service', 'acceptable', 'triangular', (0, 5, 10))
system.add_term('service', 'excellent', 'triangular', (5, 10, 10))

# Add output
system.add_output('tip', (0, 25))
system.add_term('tip', 'low', 'triangular', (0, 0, 13))
system.add_term('tip', 'medium', 'triangular', (0, 13, 25))
system.add_term('tip', 'high', 'triangular', (13, 25, 25))

# Add rules
system.add_rules([
    ('poor', 'low'),
    ('acceptable', 'medium'),
    ('excellent', 'high')
])

# Evaluate
result = system.evaluate({'service': 7.5})
```

---

### 02. Sugeno Zero-Order
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1moi6/pyfuzzy-toolbox/blob/main/notebooks_colab/02_inference/02_sugeno_zero_order.ipynb)

**Topics:**
- Sugeno/TSK system with constant outputs (order 0)
- Difference between Mamdani and Sugeno
- Weighted average defuzzification

**Key Classes:**
```python
from fuzzy_systems import SugenoSystem
```

**Example:**
```python
# Create Sugeno system
system = SugenoSystem()
system.add_input('x', (0, 10))
system.add_term('x', 'low', 'triangular', (0, 0, 5))
system.add_term('x', 'high', 'triangular', (5, 10, 10))

# Add output (order 0 = constants)
system.add_output('y', order=0)

# Rules with constant outputs
system.add_rules([
    ('low', 2.0),   # IF x is low THEN y = 2.0
    ('high', 8.0)   # IF x is high THEN y = 8.0
])

result = system.evaluate({'x': 6})
```

---

### 03. Sugeno First-Order
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1moi6/pyfuzzy-toolbox/blob/main/notebooks_colab/02_inference/03_sugeno_first_order.ipynb)

**Topics:**
- Sugeno system with linear outputs (order 1)
- Output = f(inputs) as linear function
- Function approximation

**Example:**
```python
# Create Sugeno first-order system
system = SugenoSystem()
system.add_input('x', (0, 10))
system.add_term('x', 'low', 'triangular', (0, 0, 5))
system.add_term('x', 'high', 'triangular', (5, 10, 10))

# Output = linear function: a*x + b
system.add_output('y', order=1)

# Rules with linear functions
system.add_rules([
    ('low', 2.0, 1.0),   # IF x is low THEN y = 2*x + 1
    ('high', 0.5, 3.0)   # IF x is high THEN y = 0.5*x + 3
])

result = system.evaluate({'x': 7})
```

---

### 04. Voting Prediction
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1moi6/pyfuzzy-toolbox/blob/main/notebooks_colab/02_inference/04_voting_prediction.ipynb)

**Topics:**
- Real-world application: voting prediction
- Multiple inputs (economic indicators, polls)
- Complex rule base

---

## What You'll Learn

- ✅ Build complete Mamdani systems
- ✅ Understand the 5 Mamdani inference steps
- ✅ Use Sugeno/TSK systems (order 0 and 1)
- ✅ Compare Mamdani vs Sugeno approaches
- ✅ Create 3D control surfaces
- ✅ Apply to real-world problems

## Prerequisites

```bash
pip install pyfuzzy-toolbox
```

## Next Steps

- **[03_learning](../03_learning/)**: Automatic rule generation and optimization
- **[04_dynamics](../04_dynamics/)**: Dynamic fuzzy systems
