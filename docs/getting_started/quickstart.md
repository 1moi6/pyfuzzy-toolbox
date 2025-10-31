# Quickstart Guide

Get started with pyfuzzy-toolbox in 5 minutes!

## Installation

```bash
pip install pyfuzzy-toolbox
```

## Your First Fuzzy System

Let's build a simple temperature-controlled fan system:
- **Input**: Temperature (0-40°C)
- **Output**: Fan speed (0-100%)
- **Rules**: If cold → slow, If hot → fast

### Step 1: Import

```python
import fuzzy_systems as fs
```

### Step 2: Create System

```python
# Create Mamdani system
system = fs.MamdaniSystem()
```

### Step 3: Define Input Variable

```python
# Add input: temperature
system.add_input('temperature', (0, 40))

# Add linguistic terms
system.add_term('temperature', 'cold', 'triangular', (0, 0, 20))
system.add_term('temperature', 'hot', 'triangular', (20, 40, 40))
```

### Step 4: Define Output Variable

```python
# Add output: fan speed
system.add_output('fan_speed', (0, 100))

# Add linguistic terms
system.add_term('fan_speed', 'slow', 'triangular', (0, 0, 50))
system.add_term('fan_speed', 'fast', 'triangular', (50, 100, 100))
```

### Step 5: Add Rules

```python
# Define fuzzy rules
system.add_rules([
    ('cold', 'slow'),  # IF temperature is cold THEN fan_speed is slow
    ('hot', 'fast')    # IF temperature is hot THEN fan_speed is fast
])
```

### Step 6: Evaluate

```python
# Test the system
result = system.evaluate(temperature=25)
print(f"Fan speed: {result['fan_speed']:.1f}%")
# Output: Fan speed: 50.0%
```

## Complete Example

```python
import fuzzy_systems as fs

# Create and configure system
system = fs.MamdaniSystem()
system.add_input('temperature', (0, 40))
system.add_output('fan_speed', (0, 100))

# Add terms
system.add_term('temperature', 'cold', 'triangular', (0, 0, 20))
system.add_term('temperature', 'hot', 'triangular', (20, 40, 40))
system.add_term('fan_speed', 'slow', 'triangular', (0, 0, 50))
system.add_term('fan_speed', 'fast', 'triangular', (50, 100, 100))

# Add rules
system.add_rules([
    ('cold', 'slow'),
    ('hot', 'fast')
])

# Evaluate
result = system.evaluate(temperature=25)
print(f"Fan speed: {result['fan_speed']:.1f}%")
```

## Visualize Your System

```python
# Plot input variable
system.plot_variables(['temperature'])

# Plot output variable
system.plot_variables(['fan_speed'])

# Plot rule matrix
system.plot_rule_matrix()
```

## Test Multiple Values

```python
test_temps = [5, 15, 25, 35]

for temp in test_temps:
    result = system.evaluate(temperature=temp)
    print(f"Temperature: {temp}°C → Fan speed: {result['fan_speed']:.1f}%")
```

Output:
```
Temperature: 5°C → Fan speed: 12.5%
Temperature: 15°C → Fan speed: 37.5%
Temperature: 25°C → Fan speed: 62.5%
Temperature: 35°C → Fan speed: 87.5%
```

## Next Steps

Now that you have a working fuzzy system, explore more:

- **[User Guide: Fundamentals](../user_guide/fundamentals.md)** - Learn about membership functions, fuzzification, and operators
- **[User Guide: Inference](../user_guide/inference_systems.md)** - Build complex Mamdani and Sugeno systems
- **[Examples Gallery](../examples/gallery.md)** - See practical applications in Colab notebooks
- **[API Reference](../api_reference/core.md)** - Detailed documentation of all classes and methods

## Common Patterns

### Adding More Terms

```python
system.add_term('temperature', 'cold', 'triangular', (0, 0, 15))
system.add_term('temperature', 'warm', 'triangular', (10, 20, 30))
system.add_term('temperature', 'hot', 'triangular', (25, 40, 40))

system.add_term('fan_speed', 'slow', 'triangular', (0, 0, 40))
system.add_term('fan_speed', 'medium', 'triangular', (30, 50, 70))
system.add_term('fan_speed', 'fast', 'triangular', (60, 100, 100))
```

### Adding More Rules

```python
system.add_rules([
    ('cold', 'slow'),
    ('warm', 'medium'),
    ('hot', 'fast')
])
```

### Multiple Inputs

```python
system.add_input('humidity', (0, 100))
system.add_term('humidity', 'dry', 'triangular', (0, 0, 50))
system.add_term('humidity', 'humid', 'triangular', (50, 100, 100))

# Rules with multiple conditions
system.add_rules([
    {'temperature': 'hot', 'humidity': 'humid', 'fan_speed': 'fast'},
    {'temperature': 'cold', 'humidity': 'dry', 'fan_speed': 'slow'}
])
```

## Help & Support

- **Documentation**: [Full docs](https://github.com/1moi6/pyfuzzy-toolbox)
- **Issues**: [Report bugs](https://github.com/1moi6/pyfuzzy-toolbox/issues)
- **PyPI**: [Package page](https://pypi.org/project/pyfuzzy-toolbox/)
