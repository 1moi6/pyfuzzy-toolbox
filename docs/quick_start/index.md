# pyfuzzy-toolbox Quick Start Guides

Complete collection of quick start guides for all modules in the pyfuzzy-toolbox library.

---

## 📚 Available Guides

### 🧠 Learning Algorithms

Learn fuzzy systems from data using various learning techniques.

#### 1. [ANFIS - Adaptive Neuro-Fuzzy Inference System](ANFIS_QUICKSTART.md)
**Hybrid learning combining neural networks and fuzzy logic**

- ✅ Supervised learning for regression/classification
- ✅ Two training methods: `fit()` (gradient descent) and `fit_metaheuristic()` (PSO/DE/GA)
- ✅ Automatic parameter optimization
- 📓 **Notebooks**: `03_learning/01_anfis_regression.ipynb`, `03_learning/02_anfis_classification.ipynb`

**Key Features:**
- Gaussian, Bell, and Sigmoid membership functions
- Hybrid learning: LSE for consequents + gradient descent for premises
- Metaheuristic optimization: PSO, DE, GA
- Early stopping and adaptive learning rate

**When to use:** Complex nonlinear regression/classification with automatic rule extraction.

---

#### 2. [Wang-Mendel Learning](WANG_MENDEL_QUICKSTART.md)
**Single-pass fuzzy rule extraction from data**

- ✅ Fast rule generation from training data
- ✅ Automatic task detection (regression/classification)
- ✅ No iterative optimization needed
- 📓 **Notebooks**: `03_learning/03_wang_mendel_regression.ipynb`, `03_learning/04_wang_mendel_classification.ipynb`

**Key Features:**
- One-pass algorithm (very fast)
- Automatic membership function generation
- Rule conflict resolution
- Handles multi-output systems

**When to use:** Quick fuzzy model from data, interpretable rules, baseline models.

---

#### 3. [Mamdani Learning](MAMDANI_LEARNING_QUICKSTART.md)
**Optimize Mamdani fuzzy system consequents with metaheuristics**

- ✅ Optimize rule consequents for existing Mamdani FIS
- ✅ Four metaheuristic algorithms: SA, GA, PSO, DE
- ✅ Preserves linguistic interpretability
- 📓 **Notebooks**: `03_learning/05_mamdani_learning_optimization.ipynb`

**Key Features:**
- Simulated Annealing (SA): local search with probabilistic acceptance
- Genetic Algorithm (GA): population-based with crossover/mutation
- Particle Swarm Optimization (PSO): swarm intelligence
- Differential Evolution (DE): mutation-based evolution

**When to use:** Fine-tune existing Mamdani systems, optimize rule consequents while keeping antecedents.

---

### 🎛️ Fuzzy Inference Systems

Build and use fuzzy inference systems for control and decision-making.

#### 4. [Mamdani System](MAMDANI_SYSTEM_QUICKSTART.md)
**Linguistic fuzzy inference with fuzzy outputs**

- ✅ Manual or automatic membership function generation
- ✅ Intuitive rule creation (dictionaries, lists, indices)
- ✅ Multiple defuzzification methods
- 📓 **Notebooks**: `02_inference/01_mamdani_tipping.ipynb`, `02_inference/04_voting_prediction.ipynb`

**Key Features:**
- `add_auto_mfs()`: automatic MF generation with linguistic labels
- Multiple rule formats: dict, list, tuple
- Visualization: `plot_variables()`, `plot_output()`, `plot_rule_matrix()`
- Save/load: `save()`, `load()`, `export_rules()`, `import_rules()`

**When to use:** Human-interpretable control systems, linguistic rules, fuzzy decision-making.

---

#### 5. [Sugeno System](SUGENO_SYSTEM_QUICKSTART.md)
**Efficient fuzzy inference with mathematical consequents**

- ✅ Order 0 (constant) or Order 1 (linear) consequents
- ✅ No defuzzification needed (weighted average)
- ✅ Ideal for ANFIS and optimization
- 📓 **Notebooks**: `02_inference/02_sugeno_zero_order.ipynb`, `02_inference/03_sugeno_first_order.ipynb`

**Key Features:**
- **Order 0**: IF-THEN rules with constant outputs (singletons)
- **Order 1**: IF-THEN rules with linear functions of inputs
- Faster computation than Mamdani
- Better for learning algorithms (ANFIS compatible)

**When to use:** Optimization tasks, ANFIS learning, smooth approximations, computational efficiency.

---

### 🔄 Dynamical Systems

Model temporal evolution with fuzzy rules.

#### 6. [p-Fuzzy Discrete](PFUZZY_DISCRETE_QUICKSTART.md)
**Discrete-time fuzzy dynamical systems**

- ✅ Model discrete-time evolution: x_{n+1} = x_n + f(x_n)
- ✅ Absolute or relative modes
- ✅ Single-step execution for analysis
- 📓 **Notebooks**: `04_dynamics/01_pfuzzy_discrete_predator_prey.ipynb`, `04_dynamics/03_pfuzzy_discrete_population.ipynb`

**Key Features:**
- Absolute mode: x_{n+1} = x_n + f(x_n) (additive change)
- Relative mode: x_{n+1} = x_n × f(x_n) (multiplicative change)
- `simulate()`: run n_steps iterations
- `step()`: single iteration for manual control
- Export: `to_csv()` with international/Brazilian formats

**When to use:** Population dynamics (generations), discrete events, time-series with discrete steps.

---

#### 7. [p-Fuzzy Continuous](PFUZZY_CONTINUOUS_QUICKSTART.md)
**Continuous-time fuzzy dynamical systems with ODEs**

- ✅ Model continuous evolution: dx/dt = f(x)
- ✅ Euler or RK4 integration methods
- ✅ Fixed or adaptive time stepping
- 📓 **Notebooks**: `04_dynamics/02_pfuzzy_continuous_predator_prey.ipynb`

**Key Features:**
- Absolute mode: dx/dt = f(x) (rate independent of state)
- Relative mode: dx/dt = x·f(x) (rate proportional to state)
- Integration methods: `'euler'` (fast), `'rk4'` (accurate)
- Adaptive stepping: automatically adjusts dt for accuracy
- Verbose mode: prints statistics (accepted/rejected steps, dt range)

**When to use:** Physical processes, continuous growth, temperature/cooling systems, smooth dynamics.

---

#### 8. [Fuzzy ODE Solver](FUZZY_ODE_QUICKSTART.md)
**Solve ODEs with fuzzy initial conditions and parameters**

- ✅ Propagate uncertainty through differential equations
- ✅ Fuzzy numbers for initial conditions and parameters
- ✅ Three solution methods: standard, Monte Carlo, hierarchical
- 📓 **Notebooks**: `04_dynamics/04_fuzzy_ode_logistic.ipynb`, `04_dynamics/05_fuzzy_ode_holling_tanner.ipynb`

**Key Features:**
- **Fuzzy numbers**: triangular, gaussian, trapezoidal
- **α-cuts**: confidence levels for uncertainty quantification
- **Solution methods**:
  - `'standard'`: full grid (most accurate)
  - `'monte_carlo'`: sampling (10-400x faster for high dimensions)
  - `'hierarchical'`: optimization (3-5x faster)
- Visualization: plot α-level envelopes
- Export: `to_csv()`, `to_dataframe()`

**When to use:** Uncertain initial conditions, imprecise parameters, possibilistic uncertainty (vs probabilistic).

---

## 🗂️ Quick Reference Table

| Module | Type | Input | Output | Best For | Notebooks |
|--------|------|-------|--------|----------|-----------|
| **ANFIS** | Learning | Data (X, y) | Sugeno FIS | Regression, classification | 01, 02 |
| **Wang-Mendel** | Learning | Data (X, y) | Mamdani FIS | Fast rule extraction | 03, 04 |
| **Mamdani Learning** | Learning | FIS + Data | Optimized FIS | Fine-tuning consequents | 05 |
| **Mamdani System** | Inference | Variables + Rules | FIS | Linguistic control | 01, 04 |
| **Sugeno System** | Inference | Variables + Rules | FIS | Efficient inference | 02, 03 |
| **p-Fuzzy Discrete** | Dynamics | FIS + x₀ | Trajectory | Discrete-time evolution | 01, 03 |
| **p-Fuzzy Continuous** | Dynamics | FIS + x₀ | Trajectory | Continuous-time evolution | 02 |
| **Fuzzy ODE** | Dynamics | ODE + Fuzzy params | Fuzzy trajectory | Uncertainty propagation | 04, 05 |

---

## 📂 Notebook Organization

All notebooks are available in the `notebooks_colab/` directory:

```
notebooks_colab/
├── 01_fundamentals/          # Fuzzy logic basics
│   ├── 01_membership_functions.ipynb
│   ├── 02_fuzzy_operations.ipynb
│   ├── 03_linguistic_variables.ipynb
│   └── 04_fuzzy_relations.ipynb
│
├── 02_inference/              # Inference systems
│   ├── 01_mamdani_tipping.ipynb
│   ├── 02_sugeno_zero_order.ipynb
│   ├── 03_sugeno_first_order.ipynb
│   └── 04_voting_prediction.ipynb
│
├── 03_learning/               # Learning algorithms
│   ├── 01_anfis_regression.ipynb
│   ├── 02_anfis_classification.ipynb
│   ├── 03_wang_mendel_regression.ipynb
│   ├── 04_wang_mendel_classification.ipynb
│   └── 05_mamdani_learning_optimization.ipynb
│
└── 04_dynamics/               # Dynamical systems
    ├── 01_pfuzzy_discrete_predator_prey.ipynb
    ├── 02_pfuzzy_continuous_predator_prey.ipynb
    ├── 03_pfuzzy_discrete_population.ipynb
    ├── 04_fuzzy_ode_logistic.ipynb
    └── 05_fuzzy_ode_holling_tanner.ipynb
```

---

## 🚀 Getting Started

### Installation

```bash
pip install pyfuzzy-toolbox
```

### Choose Your Path

#### 1️⃣ **I want to learn from data**
→ Start with [Wang-Mendel](WANG_MENDEL_QUICKSTART.md) for quick rules, or [ANFIS](ANFIS_QUICKSTART.md) for accurate models

#### 2️⃣ **I want to build a control system**
→ Start with [Mamdani System](MAMDANI_SYSTEM_QUICKSTART.md) for interpretability, or [Sugeno System](SUGENO_SYSTEM_QUICKSTART.md) for efficiency

#### 3️⃣ **I want to model temporal dynamics**
→ Start with [p-Fuzzy Discrete](PFUZZY_DISCRETE_QUICKSTART.md) for discrete-time, or [p-Fuzzy Continuous](PFUZZY_CONTINUOUS_QUICKSTART.md) for continuous-time

#### 4️⃣ **I have uncertain parameters**
→ Start with [Fuzzy ODE Solver](FUZZY_ODE_QUICKSTART.md)

---

## 📖 Learning Path

### Beginner Path
1. **Fundamentals** → Notebooks `01_fundamentals/`
2. **Inference** → [Mamdani System](MAMDANI_SYSTEM_QUICKSTART.md)
3. **Learning** → [Wang-Mendel](WANG_MENDEL_QUICKSTART.md)

### Intermediate Path
1. **Advanced Inference** → [Sugeno System](SUGENO_SYSTEM_QUICKSTART.md)
2. **Learning** → [ANFIS](ANFIS_QUICKSTART.md)
3. **Dynamics** → [p-Fuzzy Discrete](PFUZZY_DISCRETE_QUICKSTART.md)

### Advanced Path
1. **Optimization** → [Mamdani Learning](MAMDANI_LEARNING_QUICKSTART.md)
2. **Continuous Dynamics** → [p-Fuzzy Continuous](PFUZZY_CONTINUOUS_QUICKSTART.md)
3. **Uncertainty** → [Fuzzy ODE Solver](FUZZY_ODE_QUICKSTART.md)

---

## 💡 Common Use Cases

### Control Systems
- **Tipping problem**: [Mamdani System](MAMDANI_SYSTEM_QUICKSTART.md) → Notebook `02_inference/01_mamdani_tipping.ipynb`
- **Temperature control**: [Sugeno System](SUGENO_SYSTEM_QUICKSTART.md) → Notebook `02_inference/02_sugeno_zero_order.ipynb`

### Machine Learning
- **Regression**: [ANFIS](ANFIS_QUICKSTART.md) → Notebook `03_learning/01_anfis_regression.ipynb`
- **Classification**: [Wang-Mendel](WANG_MENDEL_QUICKSTART.md) → Notebook `03_learning/04_wang_mendel_classification.ipynb`

### Population Dynamics
- **Predator-prey (discrete)**: [p-Fuzzy Discrete](PFUZZY_DISCRETE_QUICKSTART.md) → Notebook `04_dynamics/01_pfuzzy_discrete_predator_prey.ipynb`
- **Predator-prey (continuous)**: [p-Fuzzy Continuous](PFUZZY_CONTINUOUS_QUICKSTART.md) → Notebook `04_dynamics/02_pfuzzy_continuous_predator_prey.ipynb`

### Uncertainty Modeling
- **Logistic growth with fuzzy parameters**: [Fuzzy ODE Solver](FUZZY_ODE_QUICKSTART.md) → Notebook `04_dynamics/04_fuzzy_ode_logistic.ipynb`
- **Epidemic model with uncertain transmission**: [Fuzzy ODE Solver](FUZZY_ODE_QUICKSTART.md) → Notebook `04_dynamics/05_fuzzy_ode_holling_tanner.ipynb`

---

## 🔗 Additional Resources

- **Main Documentation**: https://1moi6.github.io/pyfuzzy-toolbox/
- **GitHub Repository**: https://github.com/1moi6/pyfuzzy-toolbox
- **PyPI Package**: https://pypi.org/project/pyfuzzy-toolbox/
- **Issue Tracker**: https://github.com/1moi6/pyfuzzy-toolbox/issues

---

## 📝 Document Structure

Each quickstart guide follows the same structure:

1. **Overview** - What is it and why use it
2. **Basic Concepts** - Key ideas and terminology
3. **Getting Started** - Minimal working example
4. **Parameters** - Detailed parameter descriptions
5. **Methods** - Available methods and their uses
6. **Visualization** - How to plot results
7. **Export** - Saving results
8. **Complete Examples** - Real-world applications
9. **Tips & Best Practices** - Expert recommendations
10. **Common Issues** - Troubleshooting guide
11. **Advanced Features** - Power-user techniques
12. **References** - Academic citations

---

## 🎯 Quick Navigation

- [ANFIS](ANFIS_QUICKSTART.md) | [Wang-Mendel](WANG_MENDEL_QUICKSTART.md) | [Mamdani Learning](MAMDANI_LEARNING_QUICKSTART.md)
- [Mamdani System](MAMDANI_SYSTEM_QUICKSTART.md) | [Sugeno System](SUGENO_SYSTEM_QUICKSTART.md)
- [p-Fuzzy Discrete](PFUZZY_DISCRETE_QUICKSTART.md) | [p-Fuzzy Continuous](PFUZZY_CONTINUOUS_QUICKSTART.md) | [Fuzzy ODE](FUZZY_ODE_QUICKSTART.md)

---

*Generated with pyfuzzy-toolbox documentation system*
