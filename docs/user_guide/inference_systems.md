# User Guide: Inference Systems

This guide covers how to build complete fuzzy inference systems using Mamdani and Sugeno methods.

## What is a Fuzzy Inference System?

A **Fuzzy Inference System (FIS)** transforms fuzzy inputs into fuzzy (or crisp) outputs through a rule base.

**Components:**
1. **Fuzzification**: Convert crisp inputs → fuzzy degrees
2. **Rule Base**: IF-THEN rules
3. **Inference Engine**: Apply rules
4. **Aggregation**: Combine rule outputs
5. **Defuzzification**: Convert fuzzy output → crisp value

---

## Mamdani vs Sugeno

| Feature | Mamdani | Sugeno (TSK) |
|---------|---------|--------------|
| **Output** | Linguistic fuzzy sets | Mathematical functions |
| **Defuzzification** | Centroid, MOM, etc. | Weighted average |
| **Interpretability** | ⭐⭐⭐ Very high | ⭐⭐ Moderate |
| **Computation** | Slower (integration) | ⚡ Faster (direct) |
| **Best for** | Expert systems, control | Function approximation, modeling |
| **Example output** | "Fan speed is FAST" | "Fan speed = 0.8*temp + 10" |

**When to use:**
- **Mamdani**: You need interpretable rules with linguistic outputs
- **Sugeno**: You need precise numerical modeling or faster computation

---

## Mamdani Systems

### The 5 Steps of Mamdani Inference

Let's build a temperature-controlled fan system step by step.

#### Step 1: Fuzzification

```python
from fuzzy_systems import MamdaniSystem

# Create system
system = MamdaniSystem(name="Fan Controller")

# Add input
system.add_input('temperature', (0, 40))
system.add_term('temperature', 'cold', 'triangular', (0, 0, 20))
system.add_term('temperature', 'warm', 'triangular', (10, 20, 30))
system.add_term('temperature', 'hot', 'triangular', (20, 40, 40))

# Fuzzify (internal step when evaluating)
# For 25°C: cold=0.25, warm=0.5, hot=0.25
```

#### Step 2: Rule Application

```python
# Add output
system.add_output('fan_speed', (0, 100))
system.add_term('fan_speed', 'slow', 'triangular', (0, 0, 50))
system.add_term('fan_speed', 'medium', 'triangular', (25, 50, 75))
system.add_term('fan_speed', 'fast', 'triangular', (50, 100, 100))

# Add rules
system.add_rules([
    ('cold', 'slow'),    # IF temp is cold THEN speed is slow
    ('warm', 'medium'),  # IF temp is warm THEN speed is medium
    ('hot', 'fast')      # IF temp is hot THEN speed is fast
])
```

#### Step 3-5: Implication, Aggregation, Defuzzification

These happen automatically in `.evaluate()`:

```python
result = system.evaluate(temperature=25)
print(f"Fan speed: {result['fan_speed']:.1f}%")  # 50.0%
```

**What happened internally:**
1. **Implication**: Each rule "cuts" its output MF at the activation level
2. **Aggregation**: All cut MFs are combined (usually MAX)
3. **Defuzzification**: Center of gravity (COG) → crisp value

---

### Building Your First Mamdani System

#### Complete Example: Tipping System

```python
from fuzzy_systems import MamdaniSystem

# Step 1: Create system
system = MamdaniSystem(name="Tipping System")

# Step 2: Add inputs
system.add_input('service', (0, 10))
system.add_input('food', (0, 10))

# Step 3: Add input terms
for var in ['service', 'food']:
    system.add_term(var, 'poor', 'triangular', (0, 0, 5))
    system.add_term(var, 'good', 'triangular', (0, 5, 10))
    system.add_term(var, 'excellent', 'triangular', (5, 10, 10))

# Step 4: Add output
system.add_output('tip', (0, 25))

# Step 5: Add output terms
system.add_term('tip', 'low', 'triangular', (0, 0, 13))
system.add_term('tip', 'medium', 'triangular', (0, 13, 25))
system.add_term('tip', 'high', 'triangular', (13, 25, 25))

# Step 6: Add rules
system.add_rules([
    {'service': 'poor', 'food': 'poor', 'tip': 'low'},
    {'service': 'good', 'food': 'good', 'tip': 'medium'},
    {'service': 'excellent', 'food': 'excellent', 'tip': 'high'},
])

# Step 7: Evaluate
result = system.evaluate(service=7, food=8)
print(f"Tip: {result['tip']:.1f}%")
```

---

### Rule Formats

#### Format 1: Dictionary (Explicit)

Most readable for complex rules:

```python
system.add_rules([
    {
        'service': 'poor',
        'food': 'poor',
        'tip': 'low',
        'operator': 'AND',  # Optional
        'weight': 1.0       # Optional
    }
])
```

#### Format 2: Tuple (Compact)

Best for simple systems:

```python
# Order: (input1, input2, ..., output1, output2, ...)
system.add_rules([
    ('poor', 'poor', 'low'),
    ('good', 'good', 'medium'),
    ('excellent', 'excellent', 'high')
])
```

#### Format 3: Indices

Use term index instead of name:

```python
# 0 = first term, 1 = second term, etc.
system.add_rules([
    (0, 0, 0),  # poor, poor → low
    (1, 1, 1),  # good, good → medium
    (2, 2, 2)   # excellent, excellent → high
])
```

---

### Operators in Rules

#### AND (default)

Both conditions must be satisfied:

```python
system.add_rule({
    'temperature': 'hot',
    'humidity': 'high',
    'fan_speed': 'fast',
    'operator': 'AND'  # Takes MIN of activations
})
```

#### OR

At least one condition must be satisfied:

```python
system.add_rule({
    'temperature': 'hot',
    'humidity': 'high',
    'fan_speed': 'fast',
    'operator': 'OR'  # Takes MAX of activations
})
```

#### Rule Weights

Reduce a rule's influence:

```python
system.add_rule({
    'temperature': 'cold',
    'fan_speed': 'slow',
    'weight': 0.5  # Only 50% influence
})
```

---

### Defuzzification Methods

Choose how to convert the fuzzy output to a crisp value:

```python
system = MamdaniSystem(defuzz_method='centroid')  # Default
```

**Available methods:**

| Method | Description | When to use |
|--------|-------------|-------------|
| `'centroid'` (COG) | Center of gravity | **Default**, balanced |
| `'bisector'` | Divides area in half | Alternative to COG |
| `'mom'` | Mean of maximum | Emphasize peak values |
| `'som'` | Smallest of maximum | Conservative choice |
| `'lom'` | Largest of maximum | Aggressive choice |

**Example comparison:**

```python
methods = ['centroid', 'bisector', 'mom', 'som', 'lom']

for method in methods:
    system = MamdaniSystem(defuzz_method=method)
    # ... configure system ...
    result = system.evaluate(temperature=25)
    print(f"{method}: {result['fan_speed']:.2f}%")
```

---

### Visualization

#### Plot Variables

```python
# Plot all variables
system.plot_variables()

# Plot specific variables
system.plot_variables(['temperature', 'fan_speed'])
```

#### Plot Rule Matrix

For 2-input systems, shows rules as a heatmap:

```python
system.plot_rule_matrix()
```

---

### Saving and Loading

```python
# Save system
system.save('my_system.pkl')

# Load system
from fuzzy_systems import MamdaniSystem
system = MamdaniSystem.load('my_system.pkl')

# Export rules only
system.export_rules('rules.json', format='json')
system.export_rules('rules.txt', format='txt')

# Import rules
system.import_rules('rules.json', format='json')
```

---

## Sugeno Systems

### Zero-Order Sugeno

Outputs are **constants**.

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

**How it works:**
1. Fuzzify input: x=6 → low=0, medium=0.8, high=0.2
2. Apply rules: y₁=2.0 (w₁=0), y₂=5.0 (w₂=0.8), y₃=8.0 (w₃=0.2)
3. Weighted average: y = (0×2 + 0.8×5 + 0.2×8) / (0 + 0.8 + 0.2) = 5.6

---

### First-Order Sugeno

Outputs are **linear functions** of inputs.

```python
system = SugenoSystem()

# Add inputs
system.add_input('x1', (0, 10))
system.add_input('x2', (0, 10))

# Add terms
for var in ['x1', 'x2']:
    system.add_term(var, 'low', 'triangular', (0, 0, 5))
    system.add_term(var, 'high', 'triangular', (5, 10, 10))

# Add output (order 1 = linear function)
system.add_output('y', order=1)

# Rules: y = a*x1 + b*x2 + c
system.add_rules([
    # (input1_term, input2_term, a, b, c)
    ('low', 'low', 1.0, 0.5, 2.0),    # y = 1.0*x1 + 0.5*x2 + 2.0
    ('low', 'high', 2.0, 1.0, 0.0),   # y = 2.0*x1 + 1.0*x2 + 0.0
    ('high', 'low', 0.5, 2.0, 1.0),   # y = 0.5*x1 + 2.0*x2 + 1.0
    ('high', 'high', 1.0, 1.0, 3.0)   # y = 1.0*x1 + 1.0*x2 + 3.0
])

# Evaluate
result = system.evaluate(x1=7, x2=3)
print(f"y = {result['y']:.2f}")
```

**How it works:**
1. Fuzzify: x1=7 → low=0.6, high=0.4; x2=3 → low=0.4, high=0.6
2. Calculate rule activations (AND = MIN):
   - Rule 1: min(0.6, 0.4) = 0.4 → y₁ = 1.0×7 + 0.5×3 + 2.0 = 10.5
   - Rule 2: min(0.6, 0.6) = 0.6 → y₂ = 2.0×7 + 1.0×3 + 0.0 = 17.0
   - Rule 3: min(0.4, 0.4) = 0.4 → y₃ = 0.5×7 + 2.0×3 + 1.0 = 10.5
   - Rule 4: min(0.4, 0.6) = 0.4 → y₄ = 1.0×7 + 1.0×3 + 3.0 = 13.0
3. Weighted average: y = (0.4×10.5 + 0.6×17 + 0.4×10.5 + 0.4×13) / (0.4+0.6+0.4+0.4)

---

## Advanced Topics

### Custom T-norms and S-norms

```python
system = MamdaniSystem(
    t_norm='product',        # AND: a * b instead of min(a, b)
    s_norm='probabilistic',  # OR: a + b - a*b instead of max(a, b)
    implication='product'    # Larsen instead of Mamdani
)
```

**Options:**

**T-norms (AND):**
- `'min'` (default): min(a, b)
- `'product'`: a × b
- `'lukasiewicz'`: max(0, a + b - 1)

**S-norms (OR):**
- `'max'` (default): max(a, b)
- `'probabilistic'`: a + b - a×b
- `'bounded'`: min(1, a + b)

---

### Detailed Evaluation

Get intermediate results:

```python
details = system.evaluate_detailed(temperature=25)

print("Fuzzified inputs:")
print(details['inputs'])
# {'temperature': {'cold': 0.25, 'warm': 0.5, 'hot': 0.25}}

print("\nRule activations:")
for i, activation in enumerate(details['rule_activations']):
    print(f"  Rule {i+1}: {activation:.3f}")

print("\nAggregated output MF:")
print(details['aggregated'])

print("\nFinal outputs:")
print(details['outputs'])
```

---

## Design Guidelines

### 1. Number of Rules

For a system with n inputs and k terms per input:
- **Maximum rules:** k^n (combinatorial explosion!)
- **Typical rules:** 0.3 × k^n to 0.7 × k^n

**Example:** 2 inputs, 5 terms each:
- Max: 5² = 25 rules
- Typical: 8-18 rules (skip irrelevant combinations)

### 2. Term Overlap

Adjacent terms should overlap by **25-50%**:

```python
# Good overlap
system.add_term('temp', 'cold', 'triangular', (0, 0, 20))
system.add_term('temp', 'warm', 'triangular', (15, 25, 35))  # Overlaps at 15-20
system.add_term('temp', 'hot', 'triangular', (30, 40, 40))   # Overlaps at 30-35
```

### 3. Rule Completeness

Every possible input combination should activate **at least one rule**.

**Check coverage:**
```python
# Test grid
import numpy as np
temps = np.linspace(0, 40, 20)
humids = np.linspace(0, 100, 20)

for t in temps:
    for h in humids:
        try:
            result = system.evaluate(temperature=t, humidity=h)
        except:
            print(f"No coverage at temp={t}, humidity={h}")
```

### 4. Rule Consistency

Avoid contradictory rules:

**Bad:**
```python
system.add_rules([
    {'temp': 'hot', 'humidity': 'high', 'comfort': 'good'},      # ❌
    {'temp': 'hot', 'humidity': 'high', 'comfort': 'bad'}        # ❌ Conflict!
])
```

**Good:**
```python
system.add_rules([
    {'temp': 'hot', 'humidity': 'high', 'comfort': 'bad'},       # ✓
    {'temp': 'hot', 'humidity': 'low', 'comfort': 'moderate'}    # ✓ No conflict
])
```

---

## Troubleshooting

### Problem: Output is always the same

**Cause:** Rules are not being activated.

**Debug:**
```python
details = system.evaluate_detailed(temperature=25)
print(details['rule_activations'])  # All zeros?
```

**Fix:** Check term coverage with `system.plot_variables()`.

---

### Problem: Output is stuck at universe boundary

**Cause:** All active rules point to extreme values.

**Fix:** Add intermediate terms or adjust MF parameters.

---

### Problem: System is too slow

**Solutions:**
1. Use Sugeno instead of Mamdani
2. Reduce number of points in universe (default: 1000)
3. Use simpler MFs (triangular instead of gaussian)
4. Cache results for repeated inputs

---

## Next Steps

- **[Learning](learning.md)**: Automatically generate rules from data
- **[API Reference: Inference](../api_reference/inference.md)**: Complete method documentation
- **[Examples: Inference](../examples/gallery.md#inference-systems)**: Interactive notebooks

---

## Further Reading

- **Mamdani, E. H. (1974)**: "Application of fuzzy algorithms for control of simple dynamic plant". *Proceedings of the IEE*, 121(12), 1585-1588.
- **Takagi, T., & Sugeno, M. (1985)**: "Fuzzy identification of systems and its applications to modeling and control". *IEEE Transactions on Systems, Man, and Cybernetics*, (1), 116-132.
