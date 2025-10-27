# pyfuzzy-toolbox

[![PyPI version](https://badge.fury.io/py/pyfuzzy-toolbox.svg)](https://badge.fury.io/py/pyfuzzy-toolbox)
[![Python Versions](https://img.shields.io/pypi/pyversions/pyfuzzy-toolbox.svg)](https://pypi.org/project/pyfuzzy-toolbox/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://pepy.tech/badge/pyfuzzy-toolbox)](https://pepy.tech/project/pyfuzzy-toolbox)

**Fuzzy Systems** is a comprehensive Python library for Fuzzy Systems, developed with a focus on educational and professional applications. It goes beyond basic inference, including learning, fuzzy differential equations, and p-fuzzy systems.

## 🚀 Quick Start

### 📓 Interactive Notebooks (Recommended for Learning)

Explore **11 Colab-ready notebooks** covering fundamentals, inference systems, and machine learning:

**👉 [Access Interactive Notebooks](notebooks_colab/)**

Topics include:
- 🔰 Fundamentals: Membership functions, fuzzy sets, operators
- 🎛️ Inference Systems: Mamdani, Sugeno (order 0 and 1)
- 🧠 Learning: Wang-Mendel, ANFIS, PSO optimization

All notebooks can be opened directly in Google Colab with a single click!

---

### ⚡ Quick Code Example

```python
import fuzzy_systems as fs

# Create a Mamdani system in just 3 lines!
system = fs.MamdaniSystem()
system.add_input('temperature', (0, 40))
system.add_output('fan_speed', (0, 100))

# Add linguistic terms
system.add_term('temperature', 'cold', 'triangular', (0, 0, 20))
system.add_term('temperature', 'hot', 'triangular', (20, 40, 40))
system.add_term('fan_speed', 'slow', 'triangular', (0, 0, 50))
system.add_term('fan_speed', 'fast', 'triangular', (50, 100, 100))

# Add rules (flat tuple syntax)
system.add_rules([
    ('cold', 'slow'),
    ('hot', 'fast')
])

# Evaluate!
result = system.evaluate(temperature=25)
print(f"Fan speed: {result['fan_speed']:.1f}%")
```

## Features

### ✅ Implemented (v1.0.0)

**Fuzzy Inference Systems:**
- ✅ **Mamdani Systems**: Classic fuzzy inference with defuzzification
- ✅ **Sugeno/TSK Systems**: Inference with functional outputs (order 0 and 1)
- ✅ **Membership Functions**: Triangular, trapezoidal, gaussian, bell, sigmoid, singleton
- ✅ **Fuzzy Operators**: Multiple t-norms and s-norms (min, max, product, etc.)
- ✅ **Defuzzification**: Centroid, bisector, mean of maximum, and more
- ✅ **Ultra Simplified API**: 60-80% less code than traditional libraries

**Learning and Optimization:**
- ✅ **ANFIS**: Adaptive Neuro-Fuzzy Inference System with Lyapunov stability
- ✅ **Wang-Mendel**: Automatic rule generation from data
- ✅ **MamdaniLearning**: Mamdani system optimization with gradient and metaheuristics
- ✅ **Metaheuristics**: PSO, Differential Evolution, Genetic Algorithms
- ✅ **Rule Import/Export**: Save and load rules in JSON, CSV, TXT

**Dynamic Systems:**
- ✅ **Fuzzy ODE Solver**: Solve ODEs with fuzzy uncertainty (α-level method)
- ✅ **p-Fuzzy Systems**: Discrete and continuous dynamic systems

**Integration:**
- ✅ **MamdaniSystem ↔ MamdaniLearning**: Bidirectional conversion between systems

**Examples and Documentation:**
- ✅ **16 examples** organized by complexity
- ✅ **Interactive Jupyter notebooks**
- ✅ **Complete documentation** with learning guides

## 📦 Installation

### From PyPI (Recommended)

```bash
# Basic installation
pip install pyfuzzy-toolbox

# With machine learning dependencies (ANFIS, Wang-Mendel, etc.)
pip install pyfuzzy-toolbox[ml]

# With development dependencies (testing, formatting, etc.)
pip install pyfuzzy-toolbox[dev]

# Complete installation (all dependencies)
pip install pyfuzzy-toolbox[all]
```

**Note:** The package name on PyPI is `pyfuzzy-toolbox`, but the import is `fuzzy_systems`:

```python
import fuzzy_systems as fs  # Import name
```

### From source (development)

```bash
git clone https://github.com/1moi6/pyfuzzy-toolbox.git
cd pyfuzzy-toolbox
pip install -e .          # Editable installation
pip install -e .[dev]     # With development dependencies
```

## 💡 Usage Examples

### Basic Example: Fan Control

```python
import fuzzy_systems as fs

# Create system
system = fs.MamdaniSystem()

# Add input variable
system.add_input('temperature', (0, 40))
system.add_term('temperature', 'cold', 'triangular', (0, 0, 20))
system.add_term('temperature', 'warm', 'triangular', (10, 20, 30))
system.add_term('temperature', 'hot', 'triangular', (20, 40, 40))

# Add output variable
system.add_output('fan_speed', (0, 100))
system.add_term('fan_speed', 'slow', 'triangular', (0, 0, 50))
system.add_term('fan_speed', 'medium', 'triangular', (25, 50, 75))
system.add_term('fan_speed', 'fast', 'triangular', (50, 100, 100))

# Add rules (flat tuple syntax)
system.add_rules([
    ('cold', 'slow'),
    ('warm', 'medium'),
    ('hot', 'fast')
])

# Evaluate
result = system.evaluate({'temperature': 25})
print(f"Speed: {result['fan_speed']:.1f}%")
```

### Sugeno/TSK System

```python
import fuzzy_systems as fs

# Sugeno system with functional outputs
system = fs.SugenoSystem()

system.add_input('x', (0, 10))
system.add_term('x', 'low', 'triangular', (0, 0, 5))
system.add_term('x', 'high', 'triangular', (5, 10, 10))

# Output = linear function: a*x + b
system.add_output('y', order=1)  # Order 1 (linear)

# Rules with coefficients (flat tuple syntax)
system.add_rules([
    ('low', 2.0, 1.0),   # y = 2*x + 1
    ('high', 0.5, 3.0)   # y = 0.5*x + 3
])

result = system.evaluate({'x': 7})
```

### Learning with ANFIS

```python
import fuzzy_systems as fs
import numpy as np

# Training data
X_train = np.random.uniform(0, 10, (100, 2))
y_train = np.sin(X_train[:, 0]) + np.cos(X_train[:, 1])

# Create and train ANFIS
anfis = fs.learning.ANFIS(n_inputs=2, n_terms=3)
anfis.fit(X_train, y_train, epochs=50)

# Predict
X_test = np.array([[5.0, 3.0]])
y_pred = anfis.predict(X_test)
```

## 📚 Documentation and Examples

### Examples Organized by Level

```
examples/
├── 01_inference/      # ⭐ Beginner - Basic systems
├── 02_learning/       # ⭐⭐ Intermediate - Learning
├── 03_dynamics/       # ⭐⭐⭐ Advanced - Dynamic systems
└── 04_complete/       # ⭐⭐⭐ Professional - Complete applications
```

See `examples/README.md` for complete guide.

## 🎯 Main Features

- **Fuzzy Inference**: Mamdani and Sugeno/TSK
- **Learning**: ANFIS, Wang-Mendel, optimization with PSO/DE/GA
- **Dynamic Systems**: Fuzzy ODEs, p-fuzzy
- **Simple API**: Less code, more productivity
- **Well Documented**: 16 examples + interactive notebooks

## 📝 Citation

```bibtex
@software{pyfuzzy_toolbox,
  title = {pyfuzzy-toolbox: A Comprehensive Python Library for Fuzzy Systems},
  author = {Cecconello, Moiseis},
  year = {2025},
  url = {https://github.com/1moi6/pyfuzzy-toolbox},
  note = {Includes inference, learning, fuzzy differential equations, and p-fuzzy systems}
}
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

**PyPI**: https://pypi.org/project/pyfuzzy-toolbox/  
**GitHub**: https://github.com/1moi6/pyfuzzy-toolbox
