# Inference API Reference

The `fuzzy_systems.inference` module provides complete fuzzy inference systems:

- **MamdaniSystem**: Classic fuzzy inference with linguistic outputs
- **SugenoSystem**: TSK systems with functional outputs (order 0 and 1)

---

## MamdaniSystem

Classic Mamdani fuzzy inference system with linguistic rule base.

### Constructor

```python
MamdaniSystem(name="Mamdani FIS", t_norm='min', s_norm='max',
              implication='min', aggregation='max', defuzz_method='centroid')
```

**Parameters:**

- `name` (str): System name (default: `"Mamdani FIS"`)
- `t_norm` (str): T-norm for AND operation: `'min'`, `'product'`, etc. (default: `'min'`)
- `s_norm` (str): S-norm for OR operation: `'max'`, `'probabilistic'`, etc. (default: `'max'`)
- `implication` (str): Implication method: `'min'` (Mamdani), `'product'` (Larsen) (default: `'min'`)
- `aggregation` (str): Aggregation method: `'max'`, `'sum'`, `'probabilistic'` (default: `'max'`)
- `defuzz_method` (str): Defuzzification: `'centroid'`, `'bisector'`, `'mom'`, `'som'`, `'lom'` (default: `'centroid'`)

**Example:**
```python
from fuzzy_systems import MamdaniSystem

system = MamdaniSystem(name="Temperature Control")
```

---

### Methods

#### `.add_input(name, universe)`

Add an input variable to the system.

**Parameters:**
- `name` (str): Variable name (e.g., `"temperature"`)
- `universe` (tuple): Range `(min, max)` of the variable

**Returns:** `LinguisticVariable` - The created variable

**Example:**
```python
system.add_input('temperature', (0, 40))
system.add_input('humidity', (0, 100))
```

**Alternative (pass LinguisticVariable):**
```python
from fuzzy_systems.core import LinguisticVariable

temp_var = LinguisticVariable('temperature', (0, 40))
system.add_input(temp_var)
```

---

#### `.add_output(name, universe)`

Add an output variable to the system.

**Parameters:**
- `name` (str): Variable name (e.g., `"fan_speed"`)
- `universe` (tuple): Range `(min, max)` of the variable

**Returns:** `LinguisticVariable` - The created variable

**Example:**
```python
system.add_output('fan_speed', (0, 100))
```

---

#### `.add_term(variable_name, term_name, mf_type, params, mf_func=None)`

Add a fuzzy term to an input or output variable.

**Parameters:**
- `variable_name` (str): Name of the variable (input or output)
- `term_name` (str): Name of the term (e.g., `"cold"`, `"hot"`)
- `mf_type` (str): Membership function type: `'triangular'`, `'trapezoidal'`, `'gaussian'`, etc.
- `params` (tuple): Parameters for the membership function
- `mf_func` (callable, optional): Custom membership function

**Example:**
```python
# Add terms to input
system.add_term('temperature', 'cold', 'triangular', (0, 0, 20))
system.add_term('temperature', 'warm', 'triangular', (10, 20, 30))
system.add_term('temperature', 'hot', 'triangular', (20, 40, 40))

# Add terms to output
system.add_term('fan_speed', 'slow', 'triangular', (0, 0, 50))
system.add_term('fan_speed', 'fast', 'triangular', (50, 100, 100))
```

---

#### `.add_rule(rule_dict, operator='AND', weight=1.0)`

Add a single fuzzy rule to the system.

**Parameters:**
- `rule_dict` (dict | list | tuple): Rule specification
- `operator` (str): `'AND'` or `'OR'` (default: `'AND'`)
- `weight` (float): Rule weight in [0, 1] (default: `1.0`)

**Rule Formats:**

**Format 1 - Dictionary (Recommended):**
```python
system.add_rule({
    'temperature': 'cold',
    'humidity': 'high',
    'fan_speed': 'slow'
})

# With operator and weight
system.add_rule({
    'temperature': 'hot',
    'humidity': 'low',
    'fan_speed': 'fast',
    'operator': 'OR',
    'weight': 0.9
})
```

**Format 2 - Tuple (Compact):**
```python
# (input1_term, input2_term, ..., output1_term, ...)
system.add_rule(('cold', 'high', 'slow'))
system.add_rule(('hot', 'low', 'fast'))
```

**Format 3 - Tuple with indices:**
```python
# Use term indices instead of names
system.add_rule((0, 2, 0))  # First term of each variable
```

---

#### `.add_rules(rules_list, operator='AND', weight=1.0)`

Add multiple rules at once.

**Parameters:**
- `rules_list` (list): List of rules in any supported format
- `operator` (str): Default operator for all rules
- `weight` (float): Default weight for all rules

**Example:**
```python
# Using tuples (simple)
system.add_rules([
    ('cold', 'slow'),
    ('warm', 'medium'),
    ('hot', 'fast')
])

# Using dictionaries (explicit)
system.add_rules([
    {'temperature': 'cold', 'fan_speed': 'slow'},
    {'temperature': 'hot', 'fan_speed': 'fast', 'operator': 'OR'}
])

# Mixed formats
system.add_rules([
    ('cold', 'slow'),
    {'temperature': 'hot', 'fan_speed': 'fast', 'weight': 0.8}
])
```

---

#### `.evaluate(inputs, **kwargs)`

Evaluate the fuzzy system for given inputs.

**Parameters:**
- `inputs` (dict | list | tuple | scalar): Input values in various formats
- `**kwargs`: Alternative way to pass inputs as keyword arguments

**Returns:** `dict` - Output values: `{output_name: crisp_value}`

**Input Formats:**

**Format 1 - Dictionary:**
```python
result = system.evaluate({'temperature': 25, 'humidity': 60})
```

**Format 2 - Keyword arguments:**
```python
result = system.evaluate(temperature=25, humidity=60)
```

**Format 3 - List/tuple (order matches variable addition order):**
```python
result = system.evaluate([25, 60])
```

**Format 4 - Scalar (for single input):**
```python
result = system.evaluate(25)
```

**Example:**
```python
# Evaluate
result = system.evaluate(temperature=25)
print(f"Fan speed: {result['fan_speed']:.1f}%")
# Output: Fan speed: 62.5%
```

---

#### `.evaluate_detailed(inputs, **kwargs)`

Evaluate with detailed intermediate results.

**Parameters:**
- `inputs` (dict | list | tuple | scalar): Input values

**Returns:** `dict` - Detailed results:
```python
{
    'inputs': {...},           # Fuzzified inputs
    'rule_activations': [...], # Activation level of each rule
    'aggregated': {...},       # Aggregated output MFs
    'outputs': {...}           # Final crisp outputs
}
```

**Example:**
```python
details = system.evaluate_detailed(temperature=25)

print("Input fuzzification:")
print(details['inputs'])
# {'temperature': {'cold': 0.25, 'warm': 0.5, 'hot': 0.25}}

print("\nRule activations:")
for i, activation in enumerate(details['rule_activations']):
    print(f"  Rule {i+1}: {activation:.3f}")

print("\nFinal output:")
print(details['outputs'])
# {'fan_speed': 62.5}
```

---

#### `.plot_variables(var_names=None, figsize=(12, 8), show=True)`

Plot membership functions of variables.

**Parameters:**
- `var_names` (list, optional): List of variable names to plot. If None, plots all.
- `figsize` (tuple): Figure size (default: `(12, 8)`)
- `show` (bool): Whether to call `plt.show()` (default: `True`)

**Returns:** `tuple` - `(fig, axes)` matplotlib objects

**Example:**
```python
# Plot all variables
system.plot_variables()

# Plot specific variables
system.plot_variables(['temperature', 'fan_speed'])

# Get figure for customization
fig, axes = system.plot_variables(show=False)
axes[0].set_title("My Custom Title")
fig.savefig('variables.png')
```

---

#### `.plot_rule_matrix(figsize=(10, 8), cmap='RdYlGn', show=True)`

Plot rule matrix as a heatmap (for 2-input systems).

**Parameters:**
- `figsize` (tuple): Figure size (default: `(10, 8)`)
- `cmap` (str): Colormap name (default: `'RdYlGn'`)
- `show` (bool): Whether to call `plt.show()` (default: `True`)

**Returns:** `tuple` - `(fig, ax)` matplotlib objects

**Example:**
```python
system.plot_rule_matrix()
```

---

#### `.export_rules(filename, format='txt')`

Export rules to a file.

**Parameters:**
- `filename` (str): Output file path
- `format` (str): Format: `'txt'`, `'json'`, `'csv'` (default: `'txt'`)

**Example:**
```python
system.export_rules('rules.txt', format='txt')
system.export_rules('rules.json', format='json')
system.export_rules('rules.csv', format='csv')
```

---

#### `.import_rules(filename, format='txt')`

Import rules from a file.

**Parameters:**
- `filename` (str): Input file path
- `format` (str): Format: `'txt'`, `'json'`, `'csv'` (default: `'txt'`)

**Example:**
```python
system.import_rules('rules.json', format='json')
```

---

#### `.save(filename)`

Save complete system (variables + rules) to a file.

**Parameters:**
- `filename` (str): Output file path (typically `.pkl` or `.json`)

**Example:**
```python
system.save('my_system.pkl')
```

---

#### `.load(filename)`

Load complete system from a file (class method).

**Parameters:**
- `filename` (str): Input file path

**Returns:** `MamdaniSystem` - Loaded system

**Example:**
```python
system = MamdaniSystem.load('my_system.pkl')
```

---

## SugenoSystem

Sugeno (TSK) fuzzy inference system with functional outputs.

### Constructor

```python
SugenoSystem(name="Sugeno FIS", t_norm='min', s_norm='max')
```

**Parameters:**

- `name` (str): System name (default: `"Sugeno FIS"`)
- `t_norm` (str): T-norm for AND operation (default: `'min'`)
- `s_norm` (str): S-norm for OR operation (default: `'max'`)

**Example:**
```python
from fuzzy_systems import SugenoSystem

system = SugenoSystem(name="Nonlinear Model")
```

---

### Methods

Most methods are identical to `MamdaniSystem`: `.add_input()`, `.add_term()`, `.evaluate()`, etc.

#### Key Differences

##### `.add_output(name, order=0)`

Add output variable with functional definition.

**Parameters:**
- `name` (str): Variable name
- `order` (int): Output order:
    - `0`: Constant output (zero-order Sugeno)
    - `1`: Linear function (first-order Sugeno)

**Example:**
```python
# Zero-order (constants)
system.add_output('y', order=0)

# First-order (linear functions)
system.add_output('y', order=1)
```

---

##### `.add_rule()` with functional outputs

For Sugeno systems, consequents are numbers (order 0) or coefficient lists (order 1).

**Order 0 - Constant outputs:**
```python
system.add_rules([
    ('low', 2.0),   # IF x is low THEN y = 2.0
    ('high', 8.0)   # IF x is high THEN y = 8.0
])
```

**Order 1 - Linear outputs:**
```python
# For y = a*x + b, provide (a, b)
system.add_rules([
    ('low', 2.0, 1.0),    # IF x is low THEN y = 2.0*x + 1.0
    ('high', 0.5, 3.0)    # IF x is high THEN y = 0.5*x + 3.0
])
```

**Multiple inputs (order 1):**
```python
# For y = a*x1 + b*x2 + c, provide (a, b, c)
system.add_rules([
    ('low', 'low', 1.0, 0.5, 2.0),    # y = 1.0*x1 + 0.5*x2 + 2.0
    ('high', 'high', 2.0, 1.0, 0.0)   # y = 2.0*x1 + 1.0*x2 + 0.0
])
```

---

## Complete Examples

### Example 1: Mamdani Tipping System

```python
from fuzzy_systems import MamdaniSystem

# Create system
system = MamdaniSystem(name="Tipping System")

# Add inputs
system.add_input('service', (0, 10))
system.add_input('food', (0, 10))

# Add output
system.add_output('tip', (0, 25))

# Add terms to inputs
system.add_term('service', 'poor', 'triangular', (0, 0, 5))
system.add_term('service', 'good', 'triangular', (0, 5, 10))
system.add_term('service', 'excellent', 'triangular', (5, 10, 10))

system.add_term('food', 'poor', 'triangular', (0, 0, 5))
system.add_term('food', 'good', 'triangular', (0, 5, 10))
system.add_term('food', 'delicious', 'triangular', (5, 10, 10))

# Add terms to output
system.add_term('tip', 'low', 'triangular', (0, 0, 13))
system.add_term('tip', 'medium', 'triangular', (0, 13, 25))
system.add_term('tip', 'high', 'triangular', (13, 25, 25))

# Add rules
system.add_rules([
    {'service': 'poor', 'food': 'poor', 'tip': 'low'},
    {'service': 'good', 'food': 'good', 'tip': 'medium'},
    {'service': 'excellent', 'food': 'delicious', 'tip': 'high'},
])

# Evaluate
result = system.evaluate(service=7, food=8)
print(f"Tip: {result['tip']:.1f}%")

# Visualize
system.plot_variables()
system.plot_rule_matrix()
```

---

### Example 2: Sugeno Zero-Order

```python
from fuzzy_systems import SugenoSystem

# Create system
system = SugenoSystem()

# Add input
system.add_input('x', (0, 10))
system.add_term('x', 'low', 'triangular', (0, 0, 5))
system.add_term('x', 'medium', 'triangular', (0, 5, 10))
system.add_term('x', 'high', 'triangular', (5, 10, 10))

# Add output (order 0 = constant)
system.add_output('y', order=0)

# Add rules with constant outputs
system.add_rules([
    ('low', 2.0),      # IF x is low THEN y = 2.0
    ('medium', 5.0),   # IF x is medium THEN y = 5.0
    ('high', 8.0)      # IF x is high THEN y = 8.0
])

# Evaluate
result = system.evaluate(x=6)
print(f"y = {result['y']:.2f}")
```

---

### Example 3: Sugeno First-Order

```python
from fuzzy_systems import SugenoSystem

# Create system
system = SugenoSystem()

# Add input
system.add_input('x', (0, 10))
system.add_term('x', 'low', 'triangular', (0, 0, 5))
system.add_term('x', 'high', 'triangular', (5, 10, 10))

# Add output (order 1 = linear function)
system.add_output('y', order=1)

# Add rules with linear functions: y = a*x + b
system.add_rules([
    ('low', 2.0, 1.0),    # IF x is low THEN y = 2.0*x + 1.0
    ('high', 0.5, 3.0)    # IF x is high THEN y = 0.5*x + 3.0
])

# Evaluate
result = system.evaluate(x=7)
print(f"y = {result['y']:.2f}")

# For x=7:
# - mu_low = 0.0, mu_high = 0.4
# - y_low = 2.0*7 + 1.0 = 15.0
# - y_high = 0.5*7 + 3.0 = 6.5
# - y_final = (0.0*15.0 + 0.4*6.5) / (0.0 + 0.4) = 6.5
```

---

### Example 4: Multiple Inputs & Complex Rules

```python
system = MamdaniSystem()

# Multiple inputs
system.add_input('temp', (0, 40))
system.add_input('humidity', (0, 100))
system.add_output('comfort', (0, 10))

# Add terms
for var in ['temp', 'humidity']:
    system.add_term(var, 'low', 'triangular', (0, 0, 50))
    system.add_term(var, 'high', 'triangular', (50, 100, 100))

system.add_term('comfort', 'uncomfortable', 'triangular', (0, 0, 5))
system.add_term('comfort', 'comfortable', 'triangular', (5, 10, 10))

# Rules with OR operator
system.add_rules([
    {
        'temp': 'high',
        'humidity': 'high',
        'comfort': 'uncomfortable',
        'operator': 'OR',
        'weight': 0.9
    },
    {
        'temp': 'low',
        'humidity': 'low',
        'comfort': 'comfortable',
        'operator': 'AND'
    }
])

result = system.evaluate(temp=30, humidity=70)
print(f"Comfort: {result['comfort']:.1f}/10")
```

---

## See Also

- [Core API](core.md) - Membership functions, fuzzy sets, operators
- [Learning API](learning.md) - Automatic rule generation and optimization
- [User Guide: Inference](../user_guide/inference_systems.md) - Detailed tutorials
- [Examples](../examples/gallery.md) - Interactive notebooks
