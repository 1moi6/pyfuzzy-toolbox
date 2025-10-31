# 01. Fundamentals

Introduction to fuzzy logic basics using `fuzzy_systems.core`.

## Notebooks

### 01. Membership Functions
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1moi6/pyfuzzy-toolbox/blob/main/notebooks_colab/01_fundamentals/01_membership_functions.ipynb)

**Topics:**
- Membership function types: `triangular`, `trapezoidal`, `gaussian`, `sigmoid`
- Creating fuzzy sets with `FuzzySet`
- Linguistic variables with `LinguisticVariable`
- Fuzzy operators: `fuzzy_and_min`, `fuzzy_or_max`, `fuzzy_not`

**Key Classes/Functions:**
```python
from fuzzy_systems.core import (
    triangular, trapezoidal, gaussian, sigmoid,
    FuzzySet, LinguisticVariable,
    fuzzy_and_min, fuzzy_or_max, fuzzy_not
)
```

**Example:**
```python
# Create linguistic variable
temperature = LinguisticVariable(name="temperature", universe=(0, 50))
temperature.add_term("cold", "trapezoidal", (0, 0, 10, 20))
temperature.add_term("warm", "triangular", (15, 25, 35))
temperature.add_term("hot", "trapezoidal", (30, 40, 50, 50))

# Fuzzify value
degrees = temperature.fuzzify(28)  # {'cold': 0.0, 'warm': 0.143, 'hot': 0.333}

# Plot
temperature.plot()
```

---

### 02. Thermal Comfort
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1moi6/pyfuzzy-toolbox/blob/main/notebooks_colab/01_fundamentals/02_thermal_comfort.ipynb)

**Topics:**
- Modeling multiple linguistic variables (temperature, humidity)
- Combining variables with fuzzy operators
- Implementing simple fuzzy rules (IF-THEN)
- Creating comfort maps

**Application:**
Complete thermal comfort evaluation system combining temperature and humidity.

**Example:**
```python
# Create variables
temperature = LinguisticVariable(name="temperature", universe=(0, 40))
temperature.add_term("cold", "trapezoidal", (0, 0, 10, 20))
temperature.add_term("warm", "triangular", (15, 22, 29))
temperature.add_term("hot", "trapezoidal", (26, 32, 40, 40))

humidity = LinguisticVariable(name="humidity", universe=(0, 100))
humidity.add_term("low", "trapezoidal", (0, 0, 20, 40))
humidity.add_term("normal", "triangular", (30, 50, 70))
humidity.add_term("high", "trapezoidal", (60, 80, 100, 100))

# Fuzzify inputs
mu_temp = temperature.fuzzify(22)
mu_humid = humidity.fuzzify(50)

# Apply rules (e.g., comfort = warm AND normal)
comfort = min(mu_temp['warm'], mu_humid['normal'])  # 1.0
```

---

## What You'll Learn

- ✅ Create and visualize membership functions
- ✅ Use `FuzzySet` and `LinguisticVariable` classes
- ✅ Perform fuzzification with `.fuzzify()`
- ✅ Apply fuzzy operators (AND, OR, NOT)
- ✅ Build simple rule-based systems

## Prerequisites

```bash
pip install pyfuzzy-toolbox
```

## Next Steps

After completing fundamentals:
- **[02_inference](../02_inference/)**: Complete fuzzy inference systems (Mamdani, Sugeno)
- **[03_learning](../03_learning/)**: Machine learning with ANFIS and Wang-Mendel
