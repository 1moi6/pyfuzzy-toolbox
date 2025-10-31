# 04. Fuzzy Dynamic Systems

Dynamic systems with fuzzy uncertainty using `fuzzy_systems.dynamics`.

## Notebooks

### 01. p-Fuzzy Discrete: Predator-Prey
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1moi6/pyfuzzy-toolbox/blob/main/notebooks_colab/04_dynamics/01_pfuzzy_discrete_predator_prey.ipynb)

**Topics:**
- p-Fuzzy discrete systems: x_{n+1} = x_n + f(x_n)
- Evolution function f defined by fuzzy rules
- Predator-prey population dynamics
- Phase space analysis

**Key Classes:**
```python
from fuzzy_systems.dynamics import PFuzzyDiscrete
from fuzzy_systems import MamdaniSystem
```

**Example:**
```python
# Create Mamdani FIS with rules
fis = MamdaniSystem()
fis.add_input('prey', (0, 100))
fis.add_input('predator', (0, 100))
fis.add_output('var_prey', (-2, 2))
fis.add_output('var_predator', (-2, 2))

# Add 16 rules for prey-predator interaction
fis.add_rules([...])

# Create p-fuzzy system
pfuzzy = PFuzzyDiscrete(
    fis=fis,
    mode='absolute',  # x_{n+1} = x_n + f(x_n)
    state_vars=['prey', 'predator']
)

# Simulate
x0 = {'prey': 50, 'predator': 40}
time, trajectory = pfuzzy.simulate(x0=x0, n_steps=250)

# Plot
pfuzzy.plot_trajectory()
pfuzzy.plot_phase_space('prey', 'predator')
```

---

### 02. p-Fuzzy Continuous: Predator-Prey
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1moi6/pyfuzzy-toolbox/blob/main/notebooks_colab/04_dynamics/02_pfuzzy_continuous_predator_prey.ipynb)

**Topics:**
- p-Fuzzy continuous systems: dx/dt = f(x)
- ODE integration with fuzzy evolution
- Continuous predator-prey dynamics
- Limit cycles

**Key Classes:**
```python
from fuzzy_systems.dynamics import PFuzzyContinuous
```

**Example:**
```python
# Create continuous p-fuzzy system
pfuzzy = PFuzzyContinuous(
    fis=fis,
    state_vars=['prey', 'predator']
)

# Simulate with time span
x0 = {'prey': 50, 'predator': 40}
time, trajectory = pfuzzy.simulate(
    x0=x0,
    t_span=(0, 50),
    dt=0.1
)
```

---

### 03. p-Fuzzy Discrete: Population
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1moi6/pyfuzzy-toolbox/blob/main/notebooks_colab/04_dynamics/03_pfuzzy_discrete_population.ipynb)

**Topics:**
- Single population growth model
- Discrete logistic-like dynamics with fuzzy rules
- Bifurcation analysis

---

### 04. Fuzzy ODE: Logistic Growth
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1moi6/pyfuzzy-toolbox/blob/main/notebooks_colab/04_dynamics/04_fuzzy_ode_logistic.ipynb)

**Topics:**
- ODEs with fuzzy parameters/initial conditions
- α-level method for uncertainty propagation
- Fuzzy logistic growth: dx/dt = r*x*(1 - x/K)
- Envelope solutions for different α-cuts

**Key Classes:**
```python
from fuzzy_systems.dynamics import FuzzyODE
from fuzzy_systems.core import FuzzySet
```

**Example:**
```python
# Define ODE with fuzzy parameters
def logistic(t, x, r, K):
    return r * x * (1 - x / K)

# Create fuzzy parameters
r_fuzzy = FuzzySet(name='r', mf_type='triangular', params=(0.8, 1.0, 1.2))
K_fuzzy = FuzzySet(name='K', mf_type='triangular', params=(90, 100, 110))

# Solve with α-levels
solver = FuzzyODE(
    ode_func=logistic,
    t_span=(0, 20),
    x0=10,  # crisp initial condition
    fuzzy_params={'r': r_fuzzy, 'K': K_fuzzy},
    alpha_levels=[0, 0.25, 0.5, 0.75, 1.0]
)

# Solve
results = solver.solve()

# Plot envelope
solver.plot_envelope()
```

---

### 05. Fuzzy ODE: Holling-Tanner
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1moi6/pyfuzzy-toolbox/blob/main/notebooks_colab/04_dynamics/05_fuzzy_ode_holling_tanner.ipynb)

**Topics:**
- System of ODEs with fuzzy uncertainty
- Holling-Tanner predator-prey model
- Multi-dimensional fuzzy envelopes
- Phase space with uncertainty

**Example:**
```python
# Holling-Tanner system
def holling_tanner(t, y, r, K, a, b, c, d):
    x, z = y
    dx = r * x * (1 - x/K) - (a*x*z)/(b + x)
    dz = c * z * (1 - d*z/x)
    return [dx, dz]

# Fuzzy parameters
fuzzy_params = {
    'r': FuzzySet('r', 'gaussian', (1.0, 0.1)),
    'a': FuzzySet('a', 'triangular', (0.8, 1.0, 1.2))
}

# Solve system
solver = FuzzyODE(
    ode_func=holling_tanner,
    t_span=(0, 50),
    x0=[40, 15],
    fuzzy_params=fuzzy_params,
    alpha_levels=[0, 0.5, 1.0]
)

results = solver.solve()
solver.plot_envelope(variables=['x', 'z'])
```

---

## What You'll Learn

- ✅ Build p-fuzzy discrete systems (x_{n+1} = x_n + f(x_n))
- ✅ Build p-fuzzy continuous systems (dx/dt = f(x))
- ✅ Solve ODEs with fuzzy parameters/initial conditions
- ✅ Use α-level method for uncertainty propagation
- ✅ Analyze phase space and trajectories
- ✅ Visualize fuzzy envelopes
- ✅ Apply to population dynamics and ecology

## Prerequisites

```bash
pip install pyfuzzy-toolbox
```

## Key Concepts

### p-Fuzzy Systems
Evolution function defined by **fuzzy rules** instead of equations:
- **Discrete**: x_{n+1} = x_n + FIS(x_n)
- **Continuous**: dx/dt = FIS(x)

### Fuzzy ODEs
ODEs with **fuzzy uncertainty** in parameters or initial conditions:
- Uses **α-level method** to propagate uncertainty
- Results in **fuzzy envelopes** (bands of possible solutions)
- Each α-level gives interval bounds

## Applications

- Population dynamics (predator-prey, logistic growth)
- Epidemiology (SIR models with uncertainty)
- Chemical kinetics
- Economic models
- Climate models with uncertain parameters

## Next Steps

- **[03_learning](../03_learning/)**: Optimize dynamic systems
- **[02_inference](../02_inference/)**: Learn FIS construction
