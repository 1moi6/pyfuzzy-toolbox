# pyfuzzy-toolbox

[![PyPI version](https://badge.fury.io/py/pyfuzzy-toolbox.svg)](https://badge.fury.io/py/pyfuzzy-toolbox)
[![Python Versions](https://img.shields.io/pypi/pyversions/pyfuzzy-toolbox.svg)](https://pypi.org/project/pyfuzzy-toolbox/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://pepy.tech/badge/pyfuzzy-toolbox)](https://pepy.tech/project/pyfuzzy-toolbox)

A comprehensive Python library for Fuzzy Systems with focus on education and professional applications. Includes inference, learning, fuzzy differential equations, and p-fuzzy systems.

## 📦 Installation

```bash
pip install pyfuzzy-toolbox
```

**Note:** Package name is `pyfuzzy-toolbox`, import as `fuzzy_systems`:
```python
import fuzzy_systems as fs
```

## 🧩 Core Modules

### `fuzzy_systems.core`
Fundamental fuzzy logic components
- **Membership functions**: `triangular`, `trapezoidal`, `gaussian`, `sigmoid`, `generalized_bell`
- **Classes**: `FuzzySet`, `LinguisticVariable`
- **Operators**: `fuzzy_and_min`, `fuzzy_or_max`, `fuzzy_not`

### `fuzzy_systems.inference`
Fuzzy inference systems
- **MamdaniSystem**: Classic fuzzy inference with defuzzification (COG, MOM, etc.)
- **SugenoSystem**: TSK systems with functional outputs (order 0 and 1)

### `fuzzy_systems.learning`
Learning and optimization
- **ANFIS**: Adaptive Neuro-Fuzzy Inference System
- **WangMendel**: Automatic rule generation from data
- **MamdaniLearning**: Gradient descent and metaheuristics (PSO, DE, GA)

### `fuzzy_systems.dynamics`
Fuzzy dynamic systems
- **FuzzyODE**: Solve ODEs with fuzzy uncertainty (α-level method)
- **PFuzzySystem**: Discrete and continuous p-fuzzy systems

## 📓 Interactive Notebooks

Explore hands-on examples organized by topic:

| Topic | Notebooks | Description |
|-------|-----------|-------------|
| **[01_fundamentals](notebooks_colab/01_fundamentals/)** | 2 notebooks | Membership functions, fuzzy sets, operators, fuzzification |
| **[02_inference](notebooks_colab/02_inference/)** | 4 notebooks | Mamdani and Sugeno systems |
| **[03_learning](notebooks_colab/03_learning/)** | 7 notebooks | Wang-Mendel, ANFIS, optimization |
| **[04_dynamics](notebooks_colab/04_dynamics/)** | 5 notebooks | Fuzzy ODEs, p-fuzzy systems |

All notebooks can be opened directly in Google Colab!

## ⚡ Quick Start

```python
import fuzzy_systems as fs

# Create Mamdani system
system = fs.MamdaniSystem()
system.add_input('temperature', (0, 40))
system.add_output('fan_speed', (0, 100))

# Add terms
system.add_term('temperature', 'cold', 'triangular', (0, 0, 20))
system.add_term('temperature', 'hot', 'triangular', (20, 40, 40))
system.add_term('fan_speed', 'slow', 'triangular', (0, 0, 50))
system.add_term('fan_speed', 'fast', 'triangular', (50, 100, 100))

# Add rules
system.add_rules([('cold', 'slow'), ('hot', 'fast')])

# Evaluate
result = system.evaluate(temperature=25)
print(f"Fan speed: {result['fan_speed']:.1f}%")
```

## 🔗 Links

- **PyPI**: https://pypi.org/project/pyfuzzy-toolbox/
- **GitHub**: https://github.com/1moi6/pyfuzzy-toolbox

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
