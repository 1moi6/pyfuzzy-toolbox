# Mamdani FIS Quick Start Guide

## Overview

MamdaniSystem is a fuzzy inference system that uses linguistic rules with fuzzy outputs for decision-making and control applications.

**Key Features:**
- Flexible input/output variable creation
- Manual or automatic membership function generation
- Intuitive rule creation
- Multiple visualization tools
- Rule export/import capabilities
- System persistence (save/load)

---

## 1. Creating a Mamdani FIS

### Basic Instantiation

```python
from fuzzy_systems import MamdaniSystem
from fuzzy_systems.core.membership_functions import TNorm, SNorm, DefuzzMethod

# Create Mamdani system
fis = MamdaniSystem(
    name="My FIS",
    and_method=TNorm.MIN,              # T-norm for AND operator
    or_method=SNorm.MAX,               # S-norm for OR operator
    implication_method='min',          # Implication: 'min' or 'product'
    aggregation_method='max',          # Aggregation: 'max', 'sum', 'probabilistic'
    defuzzification_method=DefuzzMethod.CENTROID  # Defuzzification method
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
- **`implication_method`**: Rule implication
  - `'min'`: Minimum (default, Mamdani)
  - `'product'`: Product (Larsen)
- **`aggregation_method`**: Aggregate multiple rules
  - `'max'`: Maximum (default)
  - `'sum'`: Bounded sum
  - `'probabilistic'`: Probabilistic OR
- **`defuzzification_method`**: Convert fuzzy output to crisp
  - `DefuzzMethod.CENTROID`: Center of area (most common)
  - `DefuzzMethod.BISECTOR`: Bisector of area
  - `DefuzzMethod.MOM`: Mean of maximum
  - `DefuzzMethod.SOM`: Smallest of maximum
  - `DefuzzMethod.LOM`: Largest of maximum

---

## 2. Adding Variables

### Input Variables

```python
# Add input variable with universe of discourse
fis.add_input('temperature', (0, 100))  # Temperature from 0 to 100
fis.add_input('humidity', (0, 100))     # Humidity from 0 to 100
```

### Output Variables

```python
# Add output variable
fis.add_output('fan_speed', (0, 100))   # Fan speed from 0 to 100
```

---

## 3. Adding Membership Functions (Manual)

### Method 1: Using `add_term()`

```python
# Add membership functions manually
# Syntax: add_term(variable_name, term_name, mf_type, parameters)

# Temperature: cold, warm, hot
fis.add_term('temperature', 'cold', 'triangular', (0, 0, 50))
fis.add_term('temperature', 'warm', 'triangular', (0, 50, 100))
fis.add_term('temperature', 'hot', 'triangular', (50, 100, 100))

# Humidity: low, medium, high
fis.add_term('humidity', 'low', 'trapezoidal', (0, 0, 30, 50))
fis.add_term('humidity', 'medium', 'triangular', (30, 50, 70))
fis.add_term('humidity', 'high', 'trapezoidal', (50, 70, 100, 100))

# Fan speed: slow, medium, fast
fis.add_term('fan_speed', 'slow', 'gaussian', (25, 10))
fis.add_term('fan_speed', 'medium', 'gaussian', (50, 10))
fis.add_term('fan_speed', 'fast', 'gaussian', (75, 10))
```

### Available Membership Function Types

| Type | Parameters | Example |
|------|------------|---------|
| `'triangular'` | (a, b, c) | `(0, 50, 100)` |
| `'trapezoidal'` | (a, b, c, d) | `(0, 20, 80, 100)` |
| `'gaussian'` | (center, sigma) | `(50, 15)` |
| `'bell'` | (center, width, slope) | `(50, 20, 2)` |
| `'sigmoid'` | (center, slope) | `(50, 0.1)` |

---

## 4. Adding Membership Functions (Automatic)

### Using `add_auto_mfs()` - Quick Setup

```python
# Automatically generate evenly spaced membership functions
fis.add_input('temperature', (0, 100))
fis.add_auto_mfs(
    variable_name='temperature',
    n_mfs=3,                          # Number of MFs (minimum 2)
    mf_type='triangular',             # MF type
    label_prefix=None,                # Custom labels (None = linguistic)
    overlap_strategy='standard'       # 'standard' or 'perfect'
)

# This creates 3 triangular MFs with linguistic labels:
# - 'very low' at left
# - 'medium' at center
# - 'very high' at right
```

### Key Parameters

- **`n_mfs`**: Number of membership functions (≥ 2)
- **`mf_type`**: Type of membership function
  - `'triangular'`: Triangle-shaped (default)
  - `'gaussian'`: Gaussian/bell-shaped
  - `'trapezoidal'`: Trapezoid-shaped
  - `'bell'`: Generalized bell
- **`label_prefix`**: Custom label prefix
  - `None`: Uses linguistic labels (very low, low, medium, high, very high, etc.)
  - `'temp_'`: Creates labels like temp_1, temp_2, temp_3
- **`overlap_strategy`**: How MFs overlap
  - `'standard'`: Standard overlap (default)
  - `'perfect'`: Perfect overlap (sum of memberships = 1.0 at any point)

### Linguistic Labels (when `label_prefix=None`)

For different `n_mfs`, automatic linguistic labels are generated:

- **n_mfs=2**: very low, very high
- **n_mfs=3**: very low, medium, very high
- **n_mfs=4**: very low, low, high, very high
- **n_mfs=5**: very low, low, medium, high, very high
- **n_mfs=7**: very low, low, somewhat low, medium, somewhat high, high, very high

### Example: Complete System with Auto MFs

```python
from fuzzy_systems import MamdaniSystem

# Create system
fis = MamdaniSystem(name="Temperature Controller")

# Add variables
fis.add_input('temperature', (0, 100))
fis.add_input('humidity', (0, 100))
fis.add_output('fan_speed', (0, 100))

# Generate MFs automatically
fis.add_auto_mfs('temperature', n_mfs=3, mf_type='triangular')
fis.add_auto_mfs('humidity', n_mfs=3, mf_type='triangular')
fis.add_auto_mfs('fan_speed', n_mfs=3, mf_type='triangular')

# Now you have:
# temperature: very low, medium, very high
# humidity: very low, medium, very high
# fan_speed: very low, medium, very high
```

---

## 5. Adding Rules

### Method 1: Using Dictionaries (Most Readable)

```python
# Add rules using variable and term names
fis.add_rule({
    'temperature': 'cold',
    'humidity': 'high',
    'fan_speed': 'slow',
    'operator': 'AND',
    'weight': 1.0
})

# Operator and weight are optional
fis.add_rule({
    'temperature': 'hot',
    'humidity': 'high',
    'fan_speed': 'fast'
})
```

### Method 2: Using Lists (Ordered by Variable)

```python
# Rules as lists: [input1_term, input2_term, ..., output1_term, ...]
# Order follows the order variables were added
fis.add_rule(['cold', 'low', 'slow'])
fis.add_rule(['warm', 'medium', 'medium'])
fis.add_rule(['hot', 'high', 'fast'])
```

### Method 3: Batch Adding with `add_rules()`

```python
# Add multiple rules at once
fis.add_rules([
    {'temperature': 'cold', 'humidity': 'low', 'fan_speed': 'slow'},
    {'temperature': 'cold', 'humidity': 'high', 'fan_speed': 'medium'},
    {'temperature': 'warm', 'humidity': 'low', 'fan_speed': 'medium'},
    {'temperature': 'warm', 'humidity': 'high', 'fan_speed': 'medium'},
    {'temperature': 'hot', 'humidity': 'low', 'fan_speed': 'medium'},
    {'temperature': 'hot', 'humidity': 'high', 'fan_speed': 'fast'}
])
```

### Rule Parameters

- **`operator`**: Rule connector
  - `'AND'`: All conditions must be true (default)
  - `'OR'`: At least one condition must be true
- **`weight`**: Rule importance (0.0 to 1.0, default: 1.0)
  - Used to reduce rule influence without removing it
  - Example: `weight=0.5` reduces rule firing strength by half

---

## 6. Evaluating the System

### Basic Evaluation

```python
# Evaluate with dictionary
output = fis.evaluate({'temperature': 75, 'humidity': 80})
print(f"Fan speed: {output['fan_speed']:.2f}")

# Evaluate with keyword arguments
output = fis.evaluate(temperature=75, humidity=80)

# Evaluate with positional arguments (follows variable order)
output = fis.evaluate(75, 80)
```

### Detailed Evaluation (Debugging)

```python
# Get detailed information about inference process
result = fis.evaluate_detailed(temperature=75, humidity=80)

print(f"Inputs: {result['inputs']}")
print(f"Fuzzified: {result['fuzzified']}")
print(f"Outputs: {result['outputs']}")
print(f"Rule activations: {result['rule_activations']}")
```

---

## 7. Visualization

### Plot Input/Output Variables

```python
# Plot all variables (inputs and outputs)
fis.plot_variables()

# Plot specific variable
fis.plot_variables(variables=['temperature'])

# Customize plot
fis.plot_variables(
    variables=['temperature', 'humidity'],
    figsize=(12, 6),
    grid=True
)
```

### Plot Output Surface (2D Input Only)

```python
# Plot 3D surface for 2-input, 1-output systems
fis.plot_output(
    output_var='fan_speed',
    resolution=50,              # Points per dimension
    figsize=(10, 8),
    colormap='viridis'
)
```

### Plot Rule Matrix

```python
# Visualize rule firing strength matrix (2D systems)
fis.plot_rule_matrix(
    input1='temperature',
    input2='humidity',
    output='fan_speed',
    resolution=30
)

# Alternative 2D visualization
fis.plot_rule_matrix_2d(
    output_var='fan_speed',
    resolution=30
)
```

---

## 8. Rule Management

### Export Rules

```python
# Export rules to JSON format
rules_json = fis.export_rules()
print(rules_json)

# Save rules to file
import json
with open('rules.json', 'w') as f:
    json.dump(rules_json, f, indent=2)
```

### Import Rules

```python
# Import rules from JSON
rules_json = [
    {
        "antecedents": {"temperature": "cold", "humidity": "low"},
        "consequents": {"fan_speed": "slow"},
        "operator": "AND",
        "weight": 1.0
    },
    {
        "antecedents": {"temperature": "hot", "humidity": "high"},
        "consequents": {"fan_speed": "fast"},
        "operator": "AND",
        "weight": 1.0
    }
]

fis.import_rules(rules_json)
```

### Clear Rules

```python
# Remove all rules
fis.rule_base.rules.clear()
```

---

## 9. System Persistence

### Save System to File

```python
# Save entire system (variables, MFs, rules)
fis.save('temperature_controller.fis')
```

### Load System from File

```python
from fuzzy_systems import MamdaniSystem

# Load saved system
fis = MamdaniSystem.load('temperature_controller.fis')

# Use immediately
output = fis.evaluate(temperature=75, humidity=80)
```

### Export to JSON

```python
# Export system to JSON format
fis_json = fis.to_json()

# Save JSON to file
with open('fis.json', 'w') as f:
    json.dump(fis_json, f, indent=2)
```

### Import from JSON

```python
# Load from JSON
with open('fis.json', 'r') as f:
    fis_json = json.load(f)

fis = MamdaniSystem.from_json(fis_json)
```

---

## 10. Complete Example

```python
import numpy as np
import matplotlib.pyplot as plt
from fuzzy_systems import MamdaniSystem
from fuzzy_systems.core.membership_functions import DefuzzMethod

# ============================================================================
# Step 1: Create System
# ============================================================================
fis = MamdaniSystem(
    name="Temperature Controller",
    defuzzification_method=DefuzzMethod.CENTROID
)

# ============================================================================
# Step 2: Add Variables
# ============================================================================
fis.add_input('temperature', (0, 100))
fis.add_input('humidity', (0, 100))
fis.add_output('fan_speed', (0, 100))

# ============================================================================
# Step 3: Generate Membership Functions Automatically
# ============================================================================
fis.add_auto_mfs('temperature', n_mfs=3, mf_type='triangular')
fis.add_auto_mfs('humidity', n_mfs=3, mf_type='triangular')
fis.add_auto_mfs('fan_speed', n_mfs=3, mf_type='triangular')

# ============================================================================
# Step 4: Add Rules
# ============================================================================
fis.add_rules([
    {'temperature': 'very low', 'humidity': 'very low', 'fan_speed': 'very low'},
    {'temperature': 'very low', 'humidity': 'medium', 'fan_speed': 'very low'},
    {'temperature': 'very low', 'humidity': 'very high', 'fan_speed': 'medium'},
    {'temperature': 'medium', 'humidity': 'very low', 'fan_speed': 'very low'},
    {'temperature': 'medium', 'humidity': 'medium', 'fan_speed': 'medium'},
    {'temperature': 'medium', 'humidity': 'very high', 'fan_speed': 'medium'},
    {'temperature': 'very high', 'humidity': 'very low', 'fan_speed': 'medium'},
    {'temperature': 'very high', 'humidity': 'medium', 'fan_speed': 'very high'},
    {'temperature': 'very high', 'humidity': 'very high', 'fan_speed': 'very high'}
])

# ============================================================================
# Step 5: Evaluate System
# ============================================================================
# Single evaluation
output = fis.evaluate(temperature=75, humidity=60)
print(f"Temperature: 75°C, Humidity: 60%")
print(f"Fan Speed: {output['fan_speed']:.2f}%")
print()

# Multiple evaluations
print("System Response Table:")
print(f"{'Temperature':<12} {'Humidity':<12} {'Fan Speed':<12}")
print("-" * 40)

for temp in [20, 40, 60, 80]:
    for hum in [30, 60, 90]:
        result = fis.evaluate(temperature=temp, humidity=hum)
        print(f"{temp:<12} {hum:<12} {result['fan_speed']:<12.2f}")

# ============================================================================
# Step 6: Visualize
# ============================================================================
# Plot membership functions
fis.plot_variables()

# Plot control surface
fis.plot_output('fan_speed', resolution=50)

# Plot rule matrix
fis.plot_rule_matrix('temperature', 'humidity', 'fan_speed')

# ============================================================================
# Step 7: Save System
# ============================================================================
fis.save('temperature_controller.fis')
print("System saved successfully!")

# Export rules
rules = fis.export_rules()
with open('rules.json', 'w') as f:
    json.dump(rules, f, indent=2)
print("Rules exported successfully!")
```

---

## 11. Tips and Best Practices

### System Design

- **Start simple**: Begin with 2-3 membership functions per variable
- **Use `add_auto_mfs()`**: Faster than manual definition for symmetric distributions
- **Triangular MFs**: Good default choice (simple, efficient)
- **Gaussian MFs**: Better for smooth control surfaces
- **Rule coverage**: Ensure rules cover all important input combinations

### Membership Function Selection

| MF Type | Best For | Advantages | Disadvantages |
|---------|----------|------------|---------------|
| Triangular | General-purpose, control | Simple, fast, interpretable | Sharp peaks |
| Gaussian | Smooth control, modeling | Smooth, differentiable | More parameters |
| Trapezoidal | Flat regions, classification | Stable plateau | More parameters |
| Bell | Smooth transitions | Very smooth | Computationally expensive |

### Defuzzification Methods

- **Centroid**: Most common, balanced (default choice)
- **Bisector**: Similar to centroid but may be faster
- **MOM/SOM/LOM**: Use when you need extreme values
- **For control**: Use centroid or bisector
- **For classification**: Consider MOM (mean of maximum)

### Performance Optimization

- **Reduce `num_points`**: Use 100-500 for evaluation (default: 1000)
- **Minimize rules**: More rules = slower inference
- **Use triangular MFs**: Fastest computation
- **Cache evaluations**: Store results if inputs repeat

### Rule Design

- **Complete rule base**: Cover all critical regions
- **Avoid contradictions**: Don't assign different outputs to same inputs
- **Use weights**: Instead of removing rules, reduce weight (0.1-0.5)
- **Test edge cases**: Verify behavior at universe boundaries

### Visualization Best Practices

- **Plot early**: Visualize MFs before adding rules
- **Check surface**: Use `plot_output()` to verify system behavior
- **Rule matrix**: Useful for debugging 2-input systems
- **Document decisions**: Save plots with system versions

### Debugging

```python
# Use detailed evaluation to debug
result = fis.evaluate_detailed(temperature=50, humidity=50)

# Check which rules fired
for i, activation in enumerate(result['rule_activations']):
    if activation > 0.01:
        rule = fis.rule_base.rules[i]
        print(f"Rule {i}: activation = {activation:.3f}")
        print(f"  {rule}")
```

---

## 12. Common Issues and Solutions

| Problem | Cause | Solution |
|---------|-------|----------|
| Output always constant | Only one rule fires | Add more rules, check MF overlap |
| Output at universe boundary | Defuzzification issue | Check rule consequents, verify MF coverage |
| Slow evaluation | Too many points | Reduce `num_points` parameter |
| Unexpected behavior | Rule contradictions | Review rule base with `export_rules()` |
| NaN output | No rules fire | Check input ranges, MF coverage |
| Jerky control surface | Sharp MF transitions | Use Gaussian instead of triangular |

---

## 13. Comparison with Sugeno Systems

| Aspect | Mamdani | Sugeno |
|--------|---------|--------|
| **Output** | Fuzzy sets | Mathematical functions |
| **Interpretability** | High (linguistic) | Medium (numeric) |
| **Computation** | Slower (defuzzification) | Faster (weighted average) |
| **Learning** | Harder to optimize | Easier (linear consequents) |
| **Control** | Better for complex | Better for simple |
| **Best for** | Human interpretation | Mathematical optimization |

---

## 14. Advanced Features

### Custom T-norms and S-norms

```python
from fuzzy_systems.core.membership_functions import TNorm, SNorm

# Use different operators
fis = MamdaniSystem(
    and_method=TNorm.PRODUCT,      # Product T-norm
    or_method=SNorm.PROBOR         # Probabilistic OR
)
```

### Custom Defuzzification Points

```python
# Use more points for smoother defuzzification
output = fis.evaluate(temperature=75, humidity=60, num_points=2000)
```

### Weighted Rules

```python
# Add rules with different importance
fis.add_rule({
    'temperature': 'very high',
    'humidity': 'very high',
    'fan_speed': 'very high',
    'weight': 1.0  # Critical rule
})

fis.add_rule({
    'temperature': 'medium',
    'humidity': 'medium',
    'fan_speed': 'medium',
    'weight': 0.5  # Less important
})
```

---

## References

- Mamdani, E. H., & Assilian, S. (1975). "An experiment in linguistic synthesis with a fuzzy logic controller." International Journal of Man-Machine Studies, 7(1), 1-13.
- Zadeh, L. A. (1965). "Fuzzy sets." Information and Control, 8(3), 338-353.
