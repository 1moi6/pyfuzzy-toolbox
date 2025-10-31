# Core API Reference

The `fuzzy_systems.core` module provides fundamental components for fuzzy logic:

- **Membership functions**: Define fuzzy set shapes
- **Fuzzy sets**: `FuzzySet` and `LinguisticVariable` classes
- **Operators**: AND, OR, NOT operations
- **Defuzzification**: Convert fuzzy to crisp values

---

## Membership Functions

### `triangular(x, params)`

Triangular membership function.

**Parameters:**

- `x` (float | ndarray): Input value(s)
- `params` (tuple): `(a, b, c)` where:
    - `a`: Left foot
    - `b`: Peak (μ = 1)
    - `c`: Right foot

**Returns:** `float | ndarray` - Membership degree(s) in [0, 1]

**Example:**
```python
import numpy as np
from fuzzy_systems.core import triangular

x = np.linspace(0, 10, 100)
mu = triangular(x, (2, 5, 8))
```

---

### `trapezoidal(x, params)`

Trapezoidal membership function.

**Parameters:**

- `x` (float | ndarray): Input value(s)
- `params` (tuple): `(a, b, c, d)` where:
    - `a`: Left foot
    - `b`: Left shoulder (μ = 1 starts)
    - `c`: Right shoulder (μ = 1 ends)
    - `d`: Right foot

**Returns:** `float | ndarray` - Membership degree(s) in [0, 1]

**Example:**
```python
from fuzzy_systems.core import trapezoidal

mu = trapezoidal(x, (1, 3, 7, 9))
```

---

### `gaussian(x, params)`

Gaussian (bell-shaped) membership function.

**Parameters:**

- `x` (float | ndarray): Input value(s)
- `params` (tuple): `(mean, sigma)` where:
    - `mean`: Center of the curve
    - `sigma`: Standard deviation (controls width)

**Returns:** `float | ndarray` - Membership degree(s) in [0, 1]

**Example:**
```python
from fuzzy_systems.core import gaussian

mu = gaussian(x, (5, 1.5))
```

---

### `sigmoid(x, params)`

Sigmoid membership function.

**Parameters:**

- `x` (float | ndarray): Input value(s)
- `params` (tuple): `(a, c)` where:
    - `a`: Slope parameter
    - `c`: Inflection point (where μ = 0.5)

**Returns:** `float | ndarray` - Membership degree(s) in [0, 1]

**Example:**
```python
from fuzzy_systems.core import sigmoid

mu = sigmoid(x, (1, 5))
```

---

### `generalized_bell(x, params)`

Generalized bell-shaped membership function.

**Parameters:**

- `x` (float | ndarray): Input value(s)
- `params` (tuple): `(a, b, c)` where:
    - `a`: Width parameter
    - `b`: Slope parameter
    - `c`: Center

**Returns:** `float | ndarray` - Membership degree(s) in [0, 1]

---

## Classes

### `FuzzySet`

Represents a fuzzy set with its membership function.

#### Constructor

```python
FuzzySet(name, mf_type, params, mf_func=None)
```

**Parameters:**

- `name` (str): Name of the fuzzy set (e.g., "low", "medium", "high")
- `mf_type` (str): Membership function type (`"triangular"`, `"trapezoidal"`, `"gaussian"`, etc.)
- `params` (tuple): Parameters for the membership function
- `mf_func` (callable, optional): Custom membership function

**Example:**
```python
from fuzzy_systems.core import FuzzySet

fs = FuzzySet(
    name="warm",
    mf_type="triangular",
    params=(15, 22.5, 30)
)
```

#### Methods

##### `.membership(x)`

Calculate membership degree of value(s) in this fuzzy set.

**Parameters:**
- `x` (float | ndarray): Input value(s)

**Returns:** `float | ndarray` - Membership degree(s)

**Example:**
```python
mu = fs.membership(20)  # Returns: 0.727...
```

---

### `LinguisticVariable`

Represents a linguistic variable with multiple fuzzy terms.

#### Constructor

```python
LinguisticVariable(name, universe)
```

**Parameters:**

- `name` (str): Variable name (e.g., "temperature", "speed")
- `universe` (tuple): Range `(min, max)` of the variable

**Example:**
```python
from fuzzy_systems.core import LinguisticVariable

temperature = LinguisticVariable(
    name="temperature",
    universe=(0, 50)
)
```

#### Methods

##### `.add_term(name, mf_type, params, mf_func=None)`

Add a fuzzy term to the variable.

**Parameters:**
- `name` (str): Term name (e.g., "cold", "warm", "hot")
- `mf_type` (str): Membership function type
- `params` (tuple): Function parameters
- `mf_func` (callable, optional): Custom function

**Example:**
```python
temperature.add_term("cold", "trapezoidal", (0, 0, 10, 20))
temperature.add_term("warm", "triangular", (15, 25, 35))
temperature.add_term("hot", "trapezoidal", (30, 40, 50, 50))
```

**Alternative (pass FuzzySet):**
```python
from fuzzy_systems.core import FuzzySet

cold_set = FuzzySet("cold", "triangular", (0, 0, 20))
temperature.add_term(cold_set)
```

---

##### `.fuzzify(value)`

Convert a crisp value to fuzzy membership degrees.

**Parameters:**
- `value` (float): Crisp input value

**Returns:** `dict` - Membership degrees for all terms: `{term_name: degree}`

**Example:**
```python
degrees = temperature.fuzzify(28)
# Returns: {'cold': 0.0, 'warm': 0.143, 'hot': 0.333}
```

---

##### `.plot(ax=None, show=True, figsize=(10, 6), **kwargs)`

Plot all fuzzy terms of the variable.

**Parameters:**
- `ax` (matplotlib.axes.Axes, optional): Axes to plot on
- `show` (bool): Whether to call `plt.show()`
- `figsize` (tuple): Figure size if creating new figure
- `**kwargs`: Additional matplotlib styling options

**Returns:** `tuple` - `(fig, ax)` matplotlib objects

**Example:**
```python
temperature.plot()
```

---

## Fuzzy Operators

### AND Operators (T-norms)

#### `fuzzy_and_min(a, b)`

Minimum t-norm (standard fuzzy AND).

**Parameters:**
- `a`, `b` (float | ndarray): Membership degrees

**Returns:** `float | ndarray` - `min(a, b)`

**Example:**
```python
from fuzzy_systems.core import fuzzy_and_min

result = fuzzy_and_min(0.7, 0.5)  # Returns: 0.5
```

---

#### `fuzzy_and_product(a, b)`

Product t-norm.

**Returns:** `float | ndarray` - `a * b`

---

### OR Operators (S-norms)

#### `fuzzy_or_max(a, b)`

Maximum s-norm (standard fuzzy OR).

**Parameters:**
- `a`, `b` (float | ndarray): Membership degrees

**Returns:** `float | ndarray` - `max(a, b)`

**Example:**
```python
from fuzzy_systems.core import fuzzy_or_max

result = fuzzy_or_max(0.7, 0.5)  # Returns: 0.7
```

---

#### `fuzzy_or_probabilistic(a, b)`

Probabilistic s-norm.

**Returns:** `float | ndarray` - `a + b - a*b`

---

### NOT Operators

#### `fuzzy_not(a)`

Standard fuzzy negation.

**Parameters:**
- `a` (float | ndarray): Membership degree(s)

**Returns:** `float | ndarray` - `1 - a`

**Example:**
```python
from fuzzy_systems.core import fuzzy_not

result = fuzzy_not(0.7)  # Returns: 0.3
```

---

## Defuzzification

### `centroid(x, mu)`

Centroid (center of gravity) defuzzification method.

**Parameters:**
- `x` (ndarray): Universe of discourse values
- `mu` (ndarray): Aggregated membership degrees

**Returns:** `float` - Crisp output value

**Formula:** ∫ x·μ(x)dx / ∫ μ(x)dx

**Example:**
```python
from fuzzy_systems.core import centroid
import numpy as np

x = np.linspace(0, 100, 500)
mu = np.maximum(0.5 * triangular(x, (0, 0, 50)),
                0.8 * triangular(x, (50, 100, 100)))

crisp_value = centroid(x, mu)
```

---

### `bisector(x, mu)`

Bisector defuzzification method (divides area in half).

**Parameters:**
- `x` (ndarray): Universe of discourse values
- `mu` (ndarray): Aggregated membership degrees

**Returns:** `float` - Crisp output value

---

### `mean_of_maximum(x, mu)`

Mean of Maximum (MOM) defuzzification method.

**Parameters:**
- `x` (ndarray): Universe of discourse values
- `mu` (ndarray): Aggregated membership degrees

**Returns:** `float` - Mean of x values where μ is maximum

---

## Complete Example

```python
import numpy as np
from fuzzy_systems.core import (
    LinguisticVariable,
    triangular,
    fuzzy_and_min,
    fuzzy_or_max,
    fuzzy_not
)

# Create linguistic variable
temperature = LinguisticVariable("temperature", (0, 50))
temperature.add_term("cold", "trapezoidal", (0, 0, 10, 20))
temperature.add_term("warm", "triangular", (15, 25, 35))
temperature.add_term("hot", "trapezoidal", (30, 40, 50, 50))

# Fuzzify a value
current_temp = 28
degrees = temperature.fuzzify(current_temp)
print(degrees)  # {'cold': 0.0, 'warm': 0.143, 'hot': 0.333}

# Apply fuzzy operations
mu_warm = degrees['warm']
mu_hot = degrees['hot']

comfort = fuzzy_and_min(mu_warm, fuzzy_not(mu_hot))
print(f"Comfort level: {comfort:.3f}")

# Plot the variable
temperature.plot()
```

---

## See Also

- [Inference API](inference.md) - Build complete fuzzy inference systems
- [User Guide: Fundamentals](../user_guide/fundamentals.md) - Learn fuzzy logic concepts
- [Examples](../examples/gallery.md) - Practical examples
