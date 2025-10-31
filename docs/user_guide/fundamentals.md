# User Guide: Fundamentals

This guide introduces the fundamental concepts of fuzzy logic and how to use the `fuzzy_systems.core` module.

## What is Fuzzy Logic?

Classical logic uses binary values: **True** or **False** (1 or 0). Fuzzy logic extends this to handle **partial truth**: values between 0 and 1.

**Example:**
- Classical: "Is 18°C cold?" → **Yes** (1) or **No** (0)
- Fuzzy: "Is 18°C cold?" → **0.4** (somewhat cold)

This allows systems to handle **uncertainty** and **gradual transitions**, making them more human-like.

---

## Membership Functions

Membership functions (MFs) define **how much** an input belongs to a fuzzy set.

### Types of Membership Functions

#### 1. Triangular

Most common for its simplicity.

**Parameters:** `(a, b, c)` where `b` is the peak

```python
from fuzzy_systems.core import triangular
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 100)
mu = triangular(x, (2, 5, 8))

plt.plot(x, mu)
plt.title('Triangular MF')
plt.xlabel('x')
plt.ylabel('μ(x)')
plt.show()
```

**When to use:**
- Simple concepts with clear peaks
- Fast computation needed
- Educational purposes

---

#### 2. Trapezoidal

Has a **plateau** where μ = 1.

**Parameters:** `(a, b, c, d)` where `[b, c]` is the plateau

```python
from fuzzy_systems.core import trapezoidal

mu = trapezoidal(x, (1, 3, 7, 9))
```

**When to use:**
- Ranges that are "fully true" (e.g., "room temperature" = 20-24°C)
- Modeling endpoints (e.g., "very low" includes everything below 5)

---

#### 3. Gaussian

Smooth, bell-shaped curve.

**Parameters:** `(mean, sigma)` where sigma controls width

```python
from fuzzy_systems.core import gaussian

mu = gaussian(x, (5, 1.5))
```

**When to use:**
- Natural phenomena (measurements, sensors)
- Smooth transitions needed
- Mathematical modeling

---

#### 4. Sigmoid

S-shaped curve, asymmetric.

**Parameters:** `(slope, inflection_point)`

```python
from fuzzy_systems.core import sigmoid

mu = sigmoid(x, (1, 5))
```

**When to use:**
- Asymmetric concepts (e.g., "increasing", "above threshold")
- Modeling saturation effects

---

### Choosing the Right MF

| Type | Speed | Smoothness | Best For |
|------|-------|------------|----------|
| **Triangular** | ⚡⚡⚡ | ⭐ | Simple models, fast prototyping |
| **Trapezoidal** | ⚡⚡⚡ | ⭐ | Ranges with plateaus |
| **Gaussian** | ⚡⚡ | ⭐⭐⭐ | Natural phenomena, smooth transitions |
| **Sigmoid** | ⚡⚡ | ⭐⭐ | Asymmetric concepts |

**Rule of thumb:** Start with **triangular**, switch to **gaussian** if you need smoothness.

---

## Fuzzy Sets

A **FuzzySet** combines a name with a membership function.

### Creating Fuzzy Sets

```python
from fuzzy_systems.core import FuzzySet

# Create a fuzzy set for "comfortable temperature"
comfortable = FuzzySet(
    name="comfortable",
    mf_type="triangular",
    params=(18, 22, 26)
)

# Calculate membership
temp = 20
mu = comfortable.membership(temp)
print(f"20°C is {mu:.2f} comfortable")  # 0.50 comfortable
```

### Custom Membership Functions

```python
def custom_mf(x):
    """Custom bell-shaped function."""
    return np.exp(-((x - 5)**2) / 8)

custom_set = FuzzySet(
    name="custom",
    mf_type="custom",
    params=(),
    mf_func=custom_mf
)
```

---

## Linguistic Variables

A **LinguisticVariable** groups multiple fuzzy sets under one variable.

### Basic Example

```python
from fuzzy_systems.core import LinguisticVariable

# Create variable
temperature = LinguisticVariable(
    name="temperature",
    universe=(0, 40)
)

# Add fuzzy terms
temperature.add_term("cold", "trapezoidal", (0, 0, 10, 18))
temperature.add_term("warm", "triangular", (15, 22, 29))
temperature.add_term("hot", "trapezoidal", (26, 32, 40, 40))
```

**Key points:**
- Variable has a **universe** (valid range)
- Each **term** is a fuzzy set
- Terms can **overlap** (this is normal!)

---

### Fuzzification

Convert a crisp value to membership degrees in all terms.

```python
# Fuzzify a value
current_temp = 24
degrees = temperature.fuzzify(current_temp)

print(degrees)
# {'cold': 0.0, 'warm': 0.357, 'hot': 0.0}
```

**Interpretation:** 24°C is **35.7% warm** and **0% cold/hot**.

---

### Visualizing Variables

```python
temperature.plot()
```

This creates a plot showing all terms overlapping on the same axis.

---

## Fuzzy Operators

Combine fuzzy values using AND, OR, NOT.

### AND (T-norm)

**Minimum** is the standard:

```python
from fuzzy_systems.core import fuzzy_and_min

mu_warm = 0.7
mu_humid = 0.5

comfort = fuzzy_and_min(mu_warm, mu_humid)
print(comfort)  # 0.5 (takes minimum)
```

**Alternative (Product):**
```python
from fuzzy_systems.core import fuzzy_and_product

comfort = fuzzy_and_product(0.7, 0.5)
print(comfort)  # 0.35 (7 * 0.5)
```

**When to use:**
- Use **min** for standard Mamdani systems
- Use **product** for stricter combinations

---

### OR (S-norm)

**Maximum** is the standard:

```python
from fuzzy_systems.core import fuzzy_or_max

discomfort = fuzzy_or_max(0.3, 0.6)
print(discomfort)  # 0.6 (takes maximum)
```

**Alternative (Probabilistic):**
```python
from fuzzy_systems.core import fuzzy_or_probabilistic

result = fuzzy_or_probabilistic(0.3, 0.6)
print(result)  # 0.72 (= 0.3 + 0.6 - 0.3*0.6)
```

---

### NOT (Negation)

**Complement:**

```python
from fuzzy_systems.core import fuzzy_not

mu_cold = 0.2
mu_not_cold = fuzzy_not(mu_cold)
print(mu_not_cold)  # 0.8 (= 1 - 0.2)
```

---

## Practical Example: Thermal Comfort

Let's build a complete example combining everything.

### Step 1: Define Variables

```python
from fuzzy_systems.core import LinguisticVariable

# Temperature
temp_var = LinguisticVariable("temperature", (0, 40))
temp_var.add_term("cold", "trapezoidal", (0, 0, 12, 20))
temp_var.add_term("comfortable", "triangular", (18, 24, 30))
temp_var.add_term("hot", "trapezoidal", (28, 35, 40, 40))

# Humidity
humid_var = LinguisticVariable("humidity", (0, 100))
humid_var.add_term("dry", "trapezoidal", (0, 0, 30, 50))
humid_var.add_term("normal", "triangular", (40, 60, 80))
humid_var.add_term("humid", "trapezoidal", (70, 85, 100, 100))
```

### Step 2: Fuzzify Inputs

```python
current_temp = 26
current_humidity = 65

temp_degrees = temp_var.fuzzify(current_temp)
humid_degrees = humid_var.fuzzify(current_humidity)

print("Temperature:")
for term, degree in temp_degrees.items():
    print(f"  {term}: {degree:.3f}")

print("\nHumidity:")
for term, degree in humid_degrees.items():
    print(f"  {term}: {degree:.3f}")
```

Output:
```
Temperature:
  cold: 0.000
  comfortable: 0.667
  hot: 0.000

Humidity:
  dry: 0.000
  normal: 0.750
  humid: 0.000
```

### Step 3: Apply Rules

```python
from fuzzy_systems.core import fuzzy_and_min, fuzzy_or_max

# Rule 1: IF temp is comfortable AND humidity is normal THEN very comfortable
rule1 = fuzzy_and_min(temp_degrees['comfortable'], humid_degrees['normal'])
print(f"Rule 1 (very comfortable): {rule1:.3f}")  # 0.667

# Rule 2: IF temp is hot OR humidity is humid THEN uncomfortable
rule2 = fuzzy_or_max(temp_degrees['hot'], humid_degrees['humid'])
print(f"Rule 2 (uncomfortable): {rule2:.3f}")  # 0.000

# Rule 3: IF temp is cold THEN uncomfortable
rule3 = temp_degrees['cold']
print(f"Rule 3 (cold uncomfortable): {rule3:.3f}")  # 0.000
```

**Interpretation:**
- 26°C with 65% humidity is **66.7% very comfortable**
- Not uncomfortable (0%)

---

## Common Patterns

### Pattern 1: Three-Term Variable

Standard partition for most variables:

```python
var = LinguisticVariable("variable", (0, 100))
var.add_term("low", "trapezoidal", (0, 0, 20, 40))
var.add_term("medium", "triangular", (30, 50, 70))
var.add_term("high", "trapezoidal", (60, 80, 100, 100))
```

### Pattern 2: Five-Term Variable

More granular control:

```python
var = LinguisticVariable("variable", (0, 100))
var.add_term("very_low", "trapezoidal", (0, 0, 10, 25))
var.add_term("low", "triangular", (15, 25, 40))
var.add_term("medium", "triangular", (30, 50, 70))
var.add_term("high", "triangular", (60, 75, 85))
var.add_term("very_high", "trapezoidal", (75, 90, 100, 100))
```

### Pattern 3: Asymmetric Endpoints

Use trapezoidal at boundaries:

```python
var = LinguisticVariable("variable", (0, 100))
# Left endpoint: trapezoidal with flat left side
var.add_term("very_low", "trapezoidal", (0, 0, 15, 30))

# Middle: triangular
var.add_term("medium", "triangular", (25, 50, 75))

# Right endpoint: trapezoidal with flat right side
var.add_term("very_high", "trapezoidal", (70, 85, 100, 100))
```

---

## Tips and Best Practices

### 1. Overlapping is Good

Terms should **overlap** by 25-50% for smooth transitions.

**Good:**
```
     /\        /\        /\
    /  \      /  \      /  \
   /    \    /    \    /    \
  /      \  /      \  /      \
 /________\/________\/________\
  low     medium    high
```

**Bad (no overlap):**
```
  /\       /\       /\
 /  \     /  \     /  \
/____\   /____\   /____\
 low     medium    high
```

### 2. Universe Coverage

Make sure terms **cover the entire universe**:

```python
# Good: covers [0, 100]
var.add_term("low", "trapezoidal", (0, 0, 30, 50))
var.add_term("high", "trapezoidal", (50, 70, 100, 100))

# Bad: gap between 50-60
var.add_term("low", "triangular", (0, 25, 50))
var.add_term("high", "triangular", (60, 80, 100))
```

### 3. Symmetric vs Asymmetric

- **Symmetric** (triangular/gaussian): Neutral concepts (medium, normal)
- **Asymmetric** (sigmoid/trapezoidal): Directional concepts (increasing, above)

### 4. Number of Terms

- **3 terms**: Simple, fast, interpretable
- **5 terms**: Good balance
- **7-9 terms**: Complex, precise (use with learning algorithms)

**Rule:** Start with 3, add more only if needed.

---

## Troubleshooting

### Problem: "Value outside universe"

```python
temperature = LinguisticVariable("temp", (0, 40))
temp_degrees = temperature.fuzzify(50)  # ⚠️ Warning!
```

**Solution:** Extend universe or clip input:
```python
value = min(max(value, 0), 40)  # Clip to [0, 40]
```

### Problem: "All membership degrees are zero"

**Cause:** No term covers the input value.

**Solution:** Check term coverage with plots:
```python
temperature.plot()
plt.axvline(x=value, color='r', linestyle='--')  # Check if covered
plt.show()
```

### Problem: "Membership degree is always 1"

**Cause:** Terms are too wide or value is exactly at a peak.

**Solution:** Adjust term parameters to reduce overlap.

---

## Next Steps

Now that you understand fuzzy logic fundamentals:

1. **[Inference Systems](inference_systems.md)** - Build complete Mamdani and Sugeno systems
2. **[API Reference: Core](../api_reference/core.md)** - Detailed API documentation
3. **[Examples: Fundamentals](../examples/gallery.md#fundamentals)** - Interactive notebooks

---

## Further Reading

- **Zadeh, L.A. (1965)**: "Fuzzy Sets". *Information and Control*, 8(3), 338-353.
- **Ross, T.J. (2010)**: *Fuzzy Logic with Engineering Applications*. Wiley.
- **[Membership Functions](../api_reference/core.md#membership-functions)**: Complete API reference
