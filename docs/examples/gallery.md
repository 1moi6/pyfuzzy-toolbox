# Examples Gallery

Explore practical examples through interactive Colab notebooks organized by topic and difficulty.

## 🔰 Fundamentals (Beginner)

Learn the basics of fuzzy logic.

### Membership Functions
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1moi6/pyfuzzy-toolbox/blob/main/notebooks_colab/01_fundamentals/01_membership_functions.ipynb)

**What you'll learn:**
- Triangular, trapezoidal, gaussian, sigmoid functions
- `FuzzySet` and `LinguisticVariable` classes
- Fuzzification process
- Fuzzy operators (AND, OR, NOT)

**Estimated time:** 45-60 min

---

### Thermal Comfort System
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1moi6/pyfuzzy-toolbox/blob/main/notebooks_colab/01_fundamentals/02_thermal_comfort.ipynb)

**What you'll learn:**
- Model multiple variables (temperature + humidity)
- Combine variables with fuzzy operators
- Implement simple IF-THEN rules
- Create 2D comfort maps

**Estimated time:** 40-50 min

---

## 🎛️ Inference Systems (Intermediate)

Build complete fuzzy inference systems.

### Mamdani Tipping System
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1moi6/pyfuzzy-toolbox/blob/main/notebooks_colab/02_inference/01_mamdani_tipping.ipynb)

**What you'll learn:**
- Complete Mamdani inference system
- 5 Mamdani steps: fuzzification → rules → implication → aggregation → defuzzification
- Multiple inputs (service + food quality)
- 3D control surfaces

**Estimated time:** 60-75 min

---

### Sugeno Zero-Order System
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1moi6/pyfuzzy-toolbox/blob/main/notebooks_colab/02_inference/03_sugeno_zero_order.ipynb)

**What you'll learn:**
- Sugeno system with constant outputs
- Difference between Mamdani and Sugeno
- Weighted average defuzzification

**Estimated time:** 45-60 min

---

### Sugeno First-Order System
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1moi6/pyfuzzy-toolbox/blob/main/notebooks_colab/02_inference/04_sugeno_first_order.ipynb)

**What you'll learn:**
- Sugeno with linear output functions: y = ax + b
- Function approximation
- Comparison with zero-order

**Estimated time:** 40-50 min

---

### Voting Prediction
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1moi6/pyfuzzy-toolbox/blob/main/notebooks_colab/02_inference/02_voting_prediction.ipynb)

**What you'll learn:**
- Real-world application
- Complex rule base
- Multiple inputs (income + education)

**Estimated time:** 50-70 min

---

## 🧠 Learning & Optimization (Advanced)

Automatic rule generation and system optimization.

### Wang-Mendel: Nonlinear Approximation
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1moi6/pyfuzzy-toolbox/blob/main/notebooks_colab/03_learning/wang_mendel_nonlinear.ipynb)

**What you'll learn:**
- Automatic rule generation from data
- Single-pass learning algorithm
- Function approximation: f(x) = sin(x) + 0.1x
- Rule conflict resolution

**Estimated time:** 60-75 min

---

### Wang-Mendel: Linear Function
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1moi6/pyfuzzy-toolbox/blob/main/notebooks_colab/03_learning/wang_mendel_linear.ipynb)

**What you'll learn:**
- Simple case study
- Effect of number of partitions
- Performance metrics (MSE, RMSE, R²)

**Estimated time:** 40-50 min

---

### Wang-Mendel: Iris Classification
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1moi6/pyfuzzy-toolbox/blob/main/notebooks_colab/03_learning/wang_mendel_iris.ipynb)

**What you'll learn:**
- Classification with Wang-Mendel
- Multi-class fuzzy classification
- Interpretable fuzzy rules

**Estimated time:** 50-65 min

---

### ANFIS: Iris Classification
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1moi6/pyfuzzy-toolbox/blob/main/notebooks_colab/03_learning/anfis_iris.ipynb)

**What you'll learn:**
- Adaptive Neuro-Fuzzy Inference System
- Gradient-based learning (backpropagation)
- Membership function refinement
- Lyapunov stability monitoring

**Estimated time:** 60-75 min

---

### ANFIS: Regression
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1moi6/pyfuzzy-toolbox/blob/main/notebooks_colab/03_learning/anfis_regression.ipynb)

**What you'll learn:**
- ANFIS for regression problems
- Nonlinear function approximation
- Comparison with neural networks

**Estimated time:** 50-65 min

---

### Rules Optimization with PSO
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1moi6/pyfuzzy-toolbox/blob/main/notebooks_colab/03_learning/rules_optimization.ipynb)

**What you'll learn:**
- Particle Swarm Optimization (PSO)
- Metaheuristic optimization
- Optimize membership function parameters

**Estimated time:** 50-65 min

---

### Rules Optimization: Iris
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1moi6/pyfuzzy-toolbox/blob/main/notebooks_colab/03_learning/rules_optimization_iris.ipynb)

**What you'll learn:**
- Comparison: PSO vs DE vs GA
- Classification optimization
- Best practices

**Estimated time:** 55-70 min

---

## 🌊 Dynamic Systems (Advanced)

Fuzzy systems with time evolution.

### p-Fuzzy Discrete: Predator-Prey
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1moi6/pyfuzzy-toolbox/blob/main/notebooks_colab/04_dynamics/pfuzzy_discrete_predator_prey.ipynb)

**What you'll learn:**
- Discrete p-fuzzy systems: x_{n+1} = x_n + f(x_n)
- Population dynamics with fuzzy rules
- Phase space analysis
- Multiple initial conditions

**Estimated time:** 50-65 min

---

### p-Fuzzy Continuous: Predator-Prey
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1moi6/pyfuzzy-toolbox/blob/main/notebooks_colab/04_dynamics/pfuzzy_continuous_predator_prey.ipynb)

**What you'll learn:**
- Continuous p-fuzzy: dx/dt = f(x)
- ODE integration (Euler, RK4)
- Oscillatory dynamics
- Vector fields

**Estimated time:** 60-75 min

---

### p-Fuzzy Discrete: Population Growth
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1moi6/pyfuzzy-toolbox/blob/main/notebooks_colab/04_dynamics/pfuzzy_population.ipynb)

**What you'll learn:**
- Single population model
- Logistic-like fuzzy dynamics
- Bifurcation analysis

**Estimated time:** 45-60 min

---

### Fuzzy ODE: Logistic Growth
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1moi6/pyfuzzy-toolbox/blob/main/notebooks_colab/04_dynamics/fuzzy_ode_logistic.ipynb)

**What you'll learn:**
- ODEs with fuzzy parameters/initial conditions
- α-level method for uncertainty propagation
- Fuzzy envelopes

**Estimated time:** 55-70 min

---

### Fuzzy ODE: Holling-Tanner
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1moi6/pyfuzzy-toolbox/blob/main/notebooks_colab/04_dynamics/fuzzy_ode_holling_tanner.ipynb)

**What you'll learn:**
- System of ODEs with fuzzy uncertainty
- Multi-dimensional envelopes
- Phase space with uncertainty

**Estimated time:** 60-75 min

---

## By Difficulty Level

### 🟢 Beginner (0-2 notebooks recommended)
- Membership Functions
- Thermal Comfort

### 🟡 Intermediate (After fundamentals)
- All Inference Systems (Mamdani, Sugeno, Voting)

### 🔴 Advanced (Requires ML/math background)
- All Learning notebooks (Wang-Mendel, ANFIS, PSO)
- All Dynamics notebooks (p-fuzzy, Fuzzy ODEs)

## Running the Examples

### On Google Colab (Recommended)
1. Click any "Open in Colab" badge
2. Run the first cell to install: `!pip install pyfuzzy-toolbox`
3. Execute cells sequentially

### Locally
```bash
# Clone repository
git clone https://github.com/1moi6/pyfuzzy-toolbox.git
cd pyfuzzy-toolbox/notebooks_colab

# Install dependencies
pip install pyfuzzy-toolbox jupyter

# Launch Jupyter
jupyter notebook
```

## Need Help?

- **API Reference**: Detailed documentation of all methods
- **User Guide**: Conceptual explanations and tutorials
- **GitHub Issues**: Report problems or ask questions
