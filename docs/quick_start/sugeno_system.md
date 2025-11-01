# Sugeno FIS Quick Start Guide

## Overview

SugenoSystem (Takagi-Sugeno-Kang) is a fuzzy inference system that uses fuzzy antecedents with mathematical consequents (constants or linear functions) for efficient modeling and control.

**Key Features:**
- Fuzzy inputs with crisp mathematical outputs
- Two orders: 0 (constant) and 1 (linear functions)
- No defuzzification needed (weighted average)
- Computationally efficient
- Ideal for optimization and learning
- Rule export/import and system persistence

**Advantages over Mamdani:**
- Faster computation (no defuzzification)
- Easier to optimize (linear consequents)
- Better for ANFIS and learning algorithms
- More compact representation

---

## 1. Creating a Sugeno FIS

### Basic Instantiation

```python
from fuzzy_systems import SugenoSystem
from fuzzy_systems.core.membership_functions import TNorm, SNorm

# Create Sugeno system
fis = SugenoSystem(
    name="My Sugeno FIS",
    and_method=TNorm.MIN,      # T-norm for AND operator
    or_method=SNorm.MAX,       # S-norm for OR operator
    order=0                    # 0=constant, 1=linear
)
```

### Key Parameters

- **`name`**: System identifier
- **`and_method`**: T-norm for AND operations
  - `TNorm.MIN`: Minimum (default, most common)
  - `TNorm.PRODUCT`: Product
  - `TNorm.LUKASIEWICZ`: Lukasiewicz
- **`or_method`**: S-norm for OR operations
  - `SNorm.MAX`: Maximum (default)
  - `SNorm.PROBOR`: Probabilistic OR
  - `SNorm.LUKASIEWICZ`: Lukasiewicz
- **`order`**: System order (determines output type)
  - `0`: Zero-order (singleton/constant outputs)
  - `1`: First-order (linear function outputs)

---

## 2. System Orders Explained

### Order 0: Zero-Order Sugeno (Singletons)

**Consequents are constants:**

```
IF temperature is cold THEN output = 2.5
IF temperature is hot THEN output = 8.0
```

**Output calculation:**
```
y = (w1 × c1 + w2 × c2 + ...) / (w1 + w2 + ...)
```

Where:
- `wi` = firing strength of rule i
- `ci` = constant output of rule i

**Best for:**
- Classification tasks
- Simple control systems
- Fast computation needs

### Order 1: First-Order Sugeno (Linear Functions)

**Consequents are linear equations:**

```
IF temperature is cold THEN output = 0.5×temp + 1.0
IF temperature is hot THEN output = 0.8×temp + 2.0
```

**Output calculation:**
```
y = (w1 × f1(x) + w2 × f2(x) + ...) / (w1 + w2 + ...)
```

Where:
- `wi` = firing strength of rule i
- `fi(x)` = linear function of inputs for rule i
- For 2 inputs: `f(x1, x2) = p1×x1 + p2×x2 + p0`

**Best for:**
- Regression and approximation
- ANFIS learning
- Complex nonlinear systems
- Smooth control surfaces

---

## 3. Adding Variables

### Input Variables

```python
# Add input variables with universe of discourse
fis.add_input('temperature', (0, 100))
fis.add_input('humidity', (0, 100))
```

### Output Variables

```python
# Add output variable (no terms needed for Sugeno)
fis.add_output('fan_speed')

# Can optionally specify range for visualization
fis.add_output('fan_speed', (0, 100))
```

**Note:** Unlike Mamdani, Sugeno outputs don't need membership functions since outputs are computed mathematically.

---

## 4. Adding Membership Functions (Inputs Only)

### Manual MF Definition

```python
# Temperature MFs
fis.add_term('temperature', 'cold', 'trapezoidal', (0, 0, 20, 40))
fis.add_term('temperature', 'warm', 'triangular', (20, 50, 80))
fis.add_term('temperature', 'hot', 'trapezoidal', (60, 80, 100, 100))

# Humidity MFs
fis.add_term('humidity', 'dry', 'triangular', (0, 0, 50))
fis.add_term('humidity', 'humid', 'triangular', (0, 50, 100))
fis.add_term('humidity', 'wet', 'triangular', (50, 100, 100))
```

### Automatic MF Generation

```python
# Generate evenly spaced MFs automatically
fis.add_auto_mfs('temperature', n_mfs=3, mf_type='triangular')
fis.add_auto_mfs('humidity', n_mfs=3, mf_type='gaussian')

# This creates:
# temperature: very low, medium, very high
# humidity: very low, medium, very high
```

### Available MF Types

| Type | Parameters | Example |
|------|------------|---------|
| `'triangular'` | (a, b, c) | `(0, 50, 100)` |
| `'trapezoidal'` | (a, b, c, d) | `(0, 20, 80, 100)` |
| `'gaussian'` | (center, sigma) | `(50, 15)` |
| `'bell'` | (center, width, slope) | `(50, 20, 2)` |

---

## 5. Adding Rules

### Order 0: Rules with Constants

```python
# Method 1: Tuples (term_name, constant_value)
fis.add_rules([
    ('cold', 20.0),      # IF temp is cold THEN fan = 20
    ('warm', 50.0),      # IF temp is warm THEN fan = 50
    ('hot', 80.0)        # IF temp is hot THEN fan = 80
])

# Method 2: Dictionary format
fis.add_rule({
    'temperature': 'cold',
    'humidity': 'dry',
    'fan_speed': 25.0,
    'operator': 'AND'
})

# Method 3: List format (by variable order)
fis.add_rule(['cold', 'dry', 25.0])
```

### Order 1: Rules with Linear Functions

For order 1, consequent is a list of coefficients: `[p1, p2, ..., pn, p0]`

**Output function:** `y = p1×x1 + p2×x2 + ... + pn×xn + p0`

```python
# Single input example
# y = p1×x + p0
fis.add_rules([
    ('cold', [0.2, 10.0]),    # y = 0.2×temp + 10.0
    ('warm', [0.5, 20.0]),    # y = 0.5×temp + 20.0
    ('hot', [0.8, 30.0])      # y = 0.8×temp + 30.0
])

# Two inputs example
# y = p1×x1 + p2×x2 + p0
fis.add_rules([
    ['cold', 'dry', [0.2, 0.1, 10.0]],      # y = 0.2×temp + 0.1×hum + 10
    ['warm', 'humid', [0.5, 0.3, 30.0]],    # y = 0.5×temp + 0.3×hum + 30
    ['hot', 'wet', [0.8, 0.5, 50.0]]        # y = 0.8×temp + 0.5×hum + 50
])

# Dictionary format (order 1)
fis.add_rule({
    'temperature': 'cold',
    'humidity': 'dry',
    'fan_speed': [0.2, 0.1, 10.0],  # Linear function
    'operator': 'AND'
})
```

### Rule Parameters

- **`operator`**: Rule connector
  - `'AND'`: All antecedents must be true (default)
  - `'OR'`: At least one antecedent must be true
- **`weight`**: Rule importance (0.0 to 1.0, default: 1.0)

---

## 6. Evaluating the System

### Basic Evaluation

```python
# Dictionary input
output = fis.evaluate({'temperature': 75, 'humidity': 60})
print(f"Fan speed: {output['fan_speed']:.2f}")

# Keyword arguments
output = fis.evaluate(temperature=75, humidity=60)

# Positional arguments (follows variable order)
output = fis.evaluate(75, 60)
```

### Detailed Evaluation

```python
# Get detailed inference information
result = fis.evaluate_detailed(temperature=75, humidity=60)

print(f"Inputs: {result['inputs']}")
print(f"Fuzzified: {result['fuzzified']}")
print(f"Outputs: {result['outputs']}")
print(f"Rule weights: {result['rule_weights']}")
print(f"Rule outputs: {result['rule_outputs']}")
```

---

## 7. Visualization

### Plot Input Variables

```python
# Plot all input membership functions
fis.plot_variables()

# Plot specific variables
fis.plot_variables(variables=['temperature', 'humidity'])
```

### Plot Output Surface (2D Input Systems)

```python
# 3D surface plot (requires 2 inputs, 1 output)
fis.plot_output(
    output_var='fan_speed',
    resolution=50,              # Grid resolution
    figsize=(10, 8),
    colormap='viridis'
)
```

### Plot System Response (1D Input)

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate response curve
x_vals = np.linspace(0, 100, 100)
y_vals = [fis.evaluate({'temperature': x})['fan_speed'] for x in x_vals]

plt.plot(x_vals, y_vals, linewidth=2)
plt.xlabel('Temperature')
plt.ylabel('Fan Speed')
plt.title('System Response')
plt.grid(True)
plt.show()
```

---

## 8. Rule Management

### Export Rules

```python
# Export rules to JSON
rules_json = fis.export_rules()

# Save to file
import json
with open('sugeno_rules.json', 'w') as f:
    json.dump(rules_json, f, indent=2)
```

### Import Rules

```python
# Order 0 rules
rules_order0 = [
    {
        "antecedents": {"temperature": "cold"},
        "consequents": {"fan_speed": 20.0},
        "operator": "AND",
        "weight": 1.0
    }
]

# Order 1 rules
rules_order1 = [
    {
        "antecedents": {"temperature": "cold", "humidity": "dry"},
        "consequents": {"fan_speed": [0.2, 0.1, 10.0]},
        "operator": "AND",
        "weight": 1.0
    }
]

# Import rules
fis.import_rules(rules_order0)  # or rules_order1
```

### Clear Rules

```python
# Remove all rules
fis.rule_base.rules.clear()
```

---

## 9. System Persistence

### Save System

```python
# Save complete system (variables, MFs, rules, order)
fis.save('sugeno_controller.fis')
```

### Load System

```python
from fuzzy_systems import SugenoSystem

# Load saved system
fis = SugenoSystem.load('sugeno_controller.fis')

# Use immediately
output = fis.evaluate(temperature=75, humidity=60)
```

### Export to JSON

```python
# Export to JSON format
fis_json = fis.to_json()

with open('sugeno_fis.json', 'w') as f:
    json.dump(fis_json, f, indent=2)
```

### Import from JSON

```python
# Load from JSON
with open('sugeno_fis.json', 'r') as f:
    fis_json = json.load(f)

fis = SugenoSystem.from_json(fis_json)
```

---

## 10. Complete Examples

### Example 1: Order 0 System (Student Grading)

```python
import numpy as np
import matplotlib.pyplot as plt
from fuzzy_systems import SugenoSystem

# ============================================================================
# Create Order 0 Sugeno System
# ============================================================================
fis = SugenoSystem(name="Student Grading", order=0)

# Add input
fis.add_input('grade', (0, 10))
fis.add_term('grade', 'low', 'triangular', (0, 0, 5))
fis.add_term('grade', 'medium', 'triangular', (0, 5, 10))
fis.add_term('grade', 'high', 'triangular', (5, 10, 10))

# Add output
fis.add_output('performance', (0, 10))

# Add rules with constant outputs
fis.add_rules([
    ('low', 2.0),      # IF grade is low THEN performance = 2.0
    ('medium', 6.0),   # IF grade is medium THEN performance = 6.0
    ('high', 9.0)      # IF grade is high THEN performance = 9.0
])

# Evaluate
result = fis.evaluate({'grade': 6.5})
print(f"Grade: 6.5 → Performance: {result['performance']:.2f}")

# Plot response curve
grades = np.linspace(0, 10, 100)
performances = [fis.evaluate({'grade': g})['performance'] for g in grades]

plt.figure(figsize=(10, 6))
plt.plot(grades, performances, 'b-', linewidth=3)
plt.xlabel('Grade')
plt.ylabel('Performance')
plt.title('Order 0 Sugeno: Student Grading')
plt.grid(True)
plt.show()

# Save system
fis.save('grading_system.fis')
```

### Example 2: Order 1 System (Temperature Control)

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from fuzzy_systems import SugenoSystem

# ============================================================================
# Create Order 1 Sugeno System
# ============================================================================
fis = SugenoSystem(name="Temperature Controller", order=1)

# Add inputs
fis.add_input('temperature', (0, 100))
fis.add_input('humidity', (0, 100))

# Generate MFs automatically
fis.add_auto_mfs('temperature', n_mfs=3, mf_type='triangular')
fis.add_auto_mfs('humidity', n_mfs=3, mf_type='triangular')

# Add output
fis.add_output('fan_speed', (0, 100))

# Add rules with linear functions
# Format: [coef_temp, coef_humidity, constant]
fis.add_rules([
    ['very low', 'very low', [0.1, 0.05, 10.0]],    # y = 0.1×T + 0.05×H + 10
    ['very low', 'medium', [0.2, 0.1, 15.0]],
    ['very low', 'very high', [0.3, 0.15, 20.0]],
    ['medium', 'very low', [0.3, 0.1, 25.0]],
    ['medium', 'medium', [0.4, 0.2, 35.0]],
    ['medium', 'very high', [0.5, 0.3, 45.0]],
    ['very high', 'very low', [0.5, 0.2, 40.0]],
    ['very high', 'medium', [0.6, 0.3, 55.0]],
    ['very high', 'very high', [0.7, 0.4, 65.0]]     # y = 0.7×T + 0.4×H + 65
])

# Evaluate single point
output = fis.evaluate(temperature=75, humidity=60)
print(f"Temperature: 75, Humidity: 60 → Fan Speed: {output['fan_speed']:.2f}%")

# Create control surface
temp_range = np.linspace(0, 100, 40)
hum_range = np.linspace(0, 100, 40)
T, H = np.meshgrid(temp_range, hum_range)

SPEED = np.zeros_like(T)
for i in range(T.shape[0]):
    for j in range(T.shape[1]):
        result = fis.evaluate({'temperature': T[i, j], 'humidity': H[i, j]})
        SPEED[i, j] = result['fan_speed']

# Plot 3D surface
fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(121, projection='3d')
surf = ax1.plot_surface(T, H, SPEED, cmap='viridis', alpha=0.9)
ax1.set_xlabel('Temperature (°C)')
ax1.set_ylabel('Humidity (%)')
ax1.set_zlabel('Fan Speed (%)')
ax1.set_title('Order 1 Sugeno Control Surface')
ax1.view_init(elev=25, azim=135)
fig.colorbar(surf, ax=ax1, shrink=0.5)

# Plot contour
ax2 = fig.add_subplot(122)
contour = ax2.contourf(T, H, SPEED, levels=15, cmap='viridis')
ax2.set_xlabel('Temperature (°C)')
ax2.set_ylabel('Humidity (%)')
ax2.set_title('Control Surface (Contour)')
fig.colorbar(contour, ax=ax2)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Save system
fis.save('temperature_controller.fis')

# Export rules
rules = fis.export_rules()
with open('temp_rules.json', 'w') as f:
    json.dump(rules, f, indent=2)
```

---

## 11. Order 0 vs Order 1 Comparison

```python
import numpy as np
import matplotlib.pyplot as plt
from fuzzy_systems import SugenoSystem

# ============================================================================
# Order 0 System
# ============================================================================
fis_o0 = SugenoSystem(order=0)
fis_o0.add_input('x', (0, 10))
fis_o0.add_term('x', 'low', 'trapezoidal', (0, 0, 3, 5))
fis_o0.add_term('x', 'medium', 'triangular', (3, 5, 7))
fis_o0.add_term('x', 'high', 'trapezoidal', (5, 7, 10, 10))
fis_o0.add_output('y')
fis_o0.add_rules([
    ('low', 2.0),
    ('medium', 5.0),
    ('high', 8.0)
])

# ============================================================================
# Order 1 System
# ============================================================================
fis_o1 = SugenoSystem(order=1)
fis_o1.add_input('x', (0, 10))
fis_o1.add_term('x', 'low', 'trapezoidal', (0, 0, 3, 5))
fis_o1.add_term('x', 'medium', 'triangular', (3, 5, 7))
fis_o1.add_term('x', 'high', 'trapezoidal', (5, 7, 10, 10))
fis_o1.add_output('y')
fis_o1.add_rules([
    ('low', [0.3, 1.0]),      # y = 0.3x + 1.0
    ('medium', [0.5, 2.5]),   # y = 0.5x + 2.5
    ('high', [0.7, 4.0])      # y = 0.7x + 4.0
])

# Compare outputs
x_vals = np.linspace(0, 10, 100)
y_o0 = [fis_o0.evaluate({'x': x})['y'] for x in x_vals]
y_o1 = [fis_o1.evaluate({'x': x})['y'] for x in x_vals]

plt.figure(figsize=(12, 6))
plt.plot(x_vals, y_o0, 'b-', linewidth=3, label='Order 0 (Constant)')
plt.plot(x_vals, y_o1, 'r-', linewidth=3, label='Order 1 (Linear)')
plt.xlabel('Input (x)', fontsize=12)
plt.ylabel('Output (y)', fontsize=12)
plt.title('Sugeno Order 0 vs Order 1', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.show()

print("Order 0: Piecewise constant (step-like)")
print("Order 1: Smooth, continuous (linear transitions)")
```

---

## 12. Tips and Best Practices

### Choosing System Order

| Aspect | Order 0 | Order 1 |
|--------|---------|---------|
| **Output Type** | Constants | Linear functions |
| **Smoothness** | Step-like | Continuous |
| **Complexity** | Simple | Moderate |
| **Parameters** | Few (n_rules) | Many (n_rules × n_inputs) |
| **Learning** | Easier | More powerful |
| **Best for** | Classification | Regression, ANFIS |

### Design Guidelines

1. **Start with Order 0**:
   - Faster to design and test
   - Fewer parameters to tune
   - Easier to interpret

2. **Use Order 1 when**:
   - Need smooth outputs
   - Using ANFIS or learning algorithms
   - Modeling complex nonlinear systems
   - Need better approximation accuracy

3. **Input MFs**:
   - Use 2-5 MFs per input
   - Triangular or Gaussian are most common
   - Ensure adequate overlap between MFs

4. **Rule Design**:
   - Cover all important input regions
   - For Order 1, start with simple coefficients (0.1, 0.5, 1.0)
   - Validate with test data

### Performance Optimization

- **Reduce evaluation points**: Not applicable (no defuzzification)
- **Minimize rules**: Use only necessary rules
- **Use triangular MFs**: Fastest computation
- **Order 0 vs 1**: Order 0 is ~20% faster

### Common Patterns

**Order 0 - Linear approximation:**
```python
# Approximate y = 2x with 3 rules
fis.add_rules([
    ('low', 2.0),      # x ≈ 1 → y ≈ 2
    ('medium', 10.0),  # x ≈ 5 → y ≈ 10
    ('high', 18.0)     # x ≈ 9 → y ≈ 18
])
```

**Order 1 - Exact linear function:**
```python
# Exact y = 2x
fis.add_rules([
    ('low', [2.0, 0.0]),
    ('medium', [2.0, 0.0]),
    ('high', [2.0, 0.0])
])
```

---

## 13. Common Issues and Solutions

| Problem | Cause | Solution |
|---------|-------|----------|
| Output always constant | Single rule firing | Add more rules, check MF overlap |
| Unexpected output values | Wrong consequent format | Check order 0 vs 1 format |
| Output out of expected range | Consequent values too large | Adjust constants/coefficients |
| Non-smooth output | Too few rules | Add more rules or use Order 1 |
| System too complex | Too many parameters | Use Order 0 or reduce n_rules |

---

## 14. Mamdani vs Sugeno

| Aspect | Mamdani | Sugeno |
|--------|---------|--------|
| **Output** | Fuzzy sets | Constants/functions |
| **Defuzzification** | Required | Not needed |
| **Computation** | Slower | Faster |
| **Interpretability** | High (linguistic) | Medium (numeric) |
| **Learning** | Harder | Easier (especially Order 1) |
| **ANFIS Compatible** | No | Yes |
| **Smoothness** | Depends on MFs | Order 0: step-like, Order 1: smooth |
| **Best for** | Control, interpretation | Optimization, learning |

---

## 15. Advanced Features

### Custom T-norms and S-norms

```python
from fuzzy_systems.core.membership_functions import TNorm, SNorm

fis = SugenoSystem(
    and_method=TNorm.PRODUCT,
    or_method=SNorm.PROBOR,
    order=1
)
```

### Weighted Rules

```python
# Add rule with reduced importance
fis.add_rule({
    'temperature': 'medium',
    'humidity': 'medium',
    'fan_speed': [0.4, 0.2, 30.0],
    'weight': 0.5  # Half importance
})
```

### Hybrid Rules (Mixed Operators)

```python
# Some rules with AND, others with OR
fis.add_rule({'temperature': 'cold', 'humidity': 'dry', 'fan_speed': 20.0, 'operator': 'AND'})
fis.add_rule({'temperature': 'hot', 'humidity': 'wet', 'fan_speed': 80.0, 'operator': 'OR'})
```

---

## 16. Integration with ANFIS

Sugeno Order 1 systems are ideal for ANFIS learning:

```python
from fuzzy_systems.learning import ANFIS

# Create ANFIS with Sugeno structure
anfis = ANFIS(
    n_inputs=2,
    n_mfs=[3, 3],
    mf_type='gaussian',
    learning_rate=0.01
)

# Train on data (automatically uses Sugeno Order 1)
anfis.fit(X_train, y_train, epochs=100)

# Convert to SugenoSystem for inference
# (ANFIS internally uses Sugeno Order 1 structure)
```

---

## References

- Takagi, T., & Sugeno, M. (1985). "Fuzzy identification of systems and its applications to modeling and control." IEEE Transactions on Systems, Man, and Cybernetics, (1), 116-132.
- Sugeno, M., & Kang, G. T. (1988). "Structure identification of fuzzy model." Fuzzy Sets and Systems, 28(1), 15-33.
- Jang, J. S. (1993). "ANFIS: adaptive-network-based fuzzy inference system." IEEE Transactions on Systems, Man, and Cybernetics, 23(3), 665-685.
