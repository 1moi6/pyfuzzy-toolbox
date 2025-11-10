# pyfuzzy-toolbox

[![PyPI version](https://badge.fury.io/py/pyfuzzy-toolbox.svg)](https://badge.fury.io/py/pyfuzzy-toolbox)
[![Python Versions](https://img.shields.io/pypi/pyversions/pyfuzzy-toolbox.svg)](https://pypi.org/project/pyfuzzy-toolbox/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://pepy.tech/badge/pyfuzzy-toolbox)](https://pepy.tech/project/pyfuzzy-toolbox)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://1moi6.github.io/pyfuzzy-toolbox/)

A comprehensive Python library for Fuzzy Systems with focus on education and professional applications. Includes inference, learning, fuzzy differential equations, and p-fuzzy systems.

## 📚 Documentation

**[Read the full documentation →](https://1moi6.github.io/pyfuzzy-toolbox/)**

- **Getting Started**: Installation and quick tutorials
- **User Guides**: In-depth guides for each module
- **API Reference**: Complete method documentation
- **Examples**: 18+ interactive Colab notebooks

## 📦 Installation

### Basic Installation (Library only)

```bash
pip install pyfuzzy-toolbox
```

### Full Installation (with Web Interface)

For the complete experience including the interactive web interface:

**macOS/Linux (Zsh/Bash):**
```bash
pip install 'pyfuzzy-toolbox[ui]'
```

**Windows (PowerShell/CMD):**
```bash
pip install pyfuzzy-toolbox[ui]
```

**Note:** Package name is `pyfuzzy-toolbox`, import as `fuzzy_systems`:
```python
import fuzzy_systems as fs
```

## 🖥️ Web Interface (NEW!)

Launch the interactive web interface with a single command:

```bash
pyfuzzy interface
```

This opens a modern Streamlit-based interface featuring:
- 🎯 **ANFIS**: Complete workflow (data → training → evaluation → prediction)
- 📊 **Interactive visualizations**: Membership functions, decision surfaces, training curves
- 🔮 **Real-time predictions**: Manual input or batch CSV upload
- 📈 **Model analysis**: Rules visualization, feature importance, sensitivity analysis
- 💾 **Export capabilities**: Models, predictions, and results

### CLI Commands

```bash
# Launch web interface (browser opens automatically)
pyfuzzy interface

# Custom port and dark theme
pyfuzzy interface --port 8080 --dark-theme

# Headless mode (no browser)
pyfuzzy interface --no-browser

# Show version
pyfuzzy version

# Help
pyfuzzy --help
```

### Programmatic API

Launch the interface from Python code or Jupyter notebooks:

```python
from fuzzy_systems import launch_interface

# Simple launch
launch_interface()

# Custom configuration
launch_interface(
    port=8080,
    theme='dark',
    open_browser=True
)
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

## 📖 Quick Start Guides

Comprehensive guides for each module with theory, examples, and best practices:

### 🎛️ Inference Systems
Build fuzzy control systems and decision-making tools.

<table>
<tr>
<td width="50%">

**[Mamdani System](https://1moi6.github.io/pyfuzzy-toolbox/quick_start/mamdani_system/)**

Linguistic fuzzy inference with interpretable rules.

- ✅ Intuitive rule creation
- ✅ Multiple defuzzification methods
- ✅ Visualization tools
- 📓 [Tipping Example](notebooks_colab/02_inference/01_mamdani_tipping.ipynb)

</td>
<td width="50%">

**[Sugeno System](https://1moi6.github.io/pyfuzzy-toolbox/quick_start/sugeno_system/)**

Efficient inference with mathematical consequents.

- ✅ Order 0 (constant) or Order 1 (linear)
- ✅ Fast computation
- ✅ Ideal for optimization
- 📓 [Zero-Order Example](notebooks_colab/02_inference/03_sugeno_zero_order.ipynb)

</td>
</tr>
</table>

### 🧠 Learning & Optimization
Automatic rule generation and parameter tuning from data.

<table>
<tr>
<td width="33%">

**[Wang-Mendel](https://1moi6.github.io/pyfuzzy-toolbox/quick_start/wang_mendel/)**

Single-pass rule extraction.

- ✅ Fast learning
- ✅ Auto task detection
- ✅ Interpretable rules
- 📓 [Nonlinear Example](notebooks_colab/03_learning/wang_mendel_nonlinear.ipynb)

</td>
<td width="33%">

**[ANFIS](https://1moi6.github.io/pyfuzzy-toolbox/quick_start/anfis/)**

Neuro-fuzzy hybrid learning.

- ✅ Gradient descent
- ✅ Metaheuristics (PSO/DE/GA)
- ✅ High accuracy
- 📓 [Classification Example](notebooks_colab/03_learning/anfis_iris.ipynb)

</td>
<td width="33%">

**[Mamdani Learning](https://1moi6.github.io/pyfuzzy-toolbox/quick_start/mamdani_learning/)**

Optimize existing systems.

- ✅ SA, GA, PSO, DE
- ✅ Preserve interpretability
- ✅ Fine-tune consequents
- 📓 [Optimization Example](notebooks_colab/03_learning/rules_optimization.ipynb)

</td>
</tr>
</table>

### 🌊 Dynamic Systems
Model temporal evolution with fuzzy uncertainty.

<table>
<tr>
<td width="33%">

**[p-Fuzzy Discrete](https://1moi6.github.io/pyfuzzy-toolbox/quick_start/pfuzzy_discrete/)**

Discrete-time dynamics.

- ✅ x_{n+1} = x_n + f(x_n)
- ✅ Absolute/relative modes
- ✅ Population models
- 📓 [Predator-Prey Example](notebooks_colab/04_dynamics/pfuzzy_discrete_predator_prey.ipynb)

</td>
<td width="33%">

**[p-Fuzzy Continuous](https://1moi6.github.io/pyfuzzy-toolbox/quick_start/pfuzzy_continuous/)**

Continuous-time dynamics.

- ✅ dx/dt = f(x)
- ✅ Euler or RK4
- ✅ Adaptive stepping
- 📓 [Continuous Example](notebooks_colab/04_dynamics/pfuzzy_continuous_predator_prey.ipynb)

</td>
<td width="33%">

**[Fuzzy ODE](https://1moi6.github.io/pyfuzzy-toolbox/quick_start/fuzzy_ode/)**

ODEs with fuzzy uncertainty.

- ✅ α-level method
- ✅ Fuzzy parameters/ICs
- ✅ Monte Carlo option
- 📓 [Logistic Example](notebooks_colab/04_dynamics/fuzzy_ode_logistic.ipynb)

</td>
</tr>
</table>

**[📚 View All Guides](https://1moi6.github.io/pyfuzzy-toolbox/quick_start/quickstart_index/)**

---

## ⚡ Quick Example

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

- **Documentation**: https://1moi6.github.io/pyfuzzy-toolbox/
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
