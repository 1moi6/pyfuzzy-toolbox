# Wang-Mendel Learning Quick Start Guide

## Overview

The Wang-Mendel algorithm (1992) is a method for **automatically generating fuzzy rules from data**. It extracts fuzzy IF-THEN rules directly from input-output pairs without requiring domain expert knowledge.

**Key Features:**
- Automatic rule extraction from data
- Conflict resolution mechanism
- Supports both regression and classification
- Fast and interpretable
- No gradient descent or iterative optimization

---

## Algorithm Steps (Background)

The Wang-Mendel method consists of 5 steps:

1. **Partition variable domains** - Define membership functions for each variable
2. **Generate candidate rules** - Create rules from each data sample
3. **Assign degree to each rule** - Calculate rule strength based on membership degrees
4. **Resolve conflicts** - Keep only the rule with highest degree when antecedents match
5. **Create final fuzzy system** - Build the rule base with conflict-free rules

---

## 1. Setup: Create and Configure Mamdani System

**Important:** Before using Wang-Mendel, you must first create a Mamdani system with **all variables and membership functions defined**.

```python
from fuzzy_systems.inference import MamdaniSystem
from fuzzy_systems.learning import WangMendelLearning
import numpy as np

# Step 1: Create Mamdani system
system = MamdaniSystem()

# Step 2: Add input variables with their ranges
system.add_input('temperature', (0, 40))
system.add_input('humidity', (0, 100))

# Step 3: Add output variable(s)
system.add_output('comfort', (0, 10))

# Step 4: Define membership functions for INPUTS
# Temperature
system.add_term('temperature', 'cold', 'trapezoidal', (0, 0, 10, 20))
system.add_term('temperature', 'warm', 'triangular', (15, 25, 35))
system.add_term('temperature', 'hot', 'trapezoidal', (30, 35, 40, 40))

# Humidity
system.add_term('humidity', 'dry', 'trapezoidal', (0, 0, 20, 40))
system.add_term('humidity', 'normal', 'triangular', (30, 50, 70))
system.add_term('humidity', 'wet', 'trapezoidal', (60, 80, 100, 100))

# Step 5: Define membership functions for OUTPUT
system.add_term('comfort', 'low', 'triangular', (0, 0, 5))
system.add_term('comfort', 'medium', 'triangular', (2, 5, 8))
system.add_term('comfort', 'high', 'triangular', (5, 10, 10))
```

### Key Points

- **All variables must be configured** before Wang-Mendel
- **Membership functions define the linguistic terms** used in rules
- The algorithm will automatically select which terms to use in rules
- Number of MFs per variable determines rule space size

---

## 2. Instantiate WangMendelLearning Class

```python
# Prepare data
X_train = ...  # shape: (n_samples, n_inputs)
y_train = ...  # shape: (n_samples, n_outputs) or (n_samples,)

# Create Wang-Mendel learner
wm = WangMendelLearning(
    system=system,              # Pre-configured Mamdani system
    X=X_train,                  # Training input data
    y=y_train,                  # Training output data
    task='auto',                # Task type: 'auto', 'regression', or 'classification'
    scale_classification=True,  # Scale classification outputs to [0,1] (structure-based)
    verbose_init=False          # Print output range info during initialization
)
```

### Key Parameters

- **`system`**: Pre-configured MamdaniSystem with all variables and terms defined
- **`X`**: Training input data (n_samples, n_features)
  - Must match number of input variables in system
- **`y`**: Training output data
  - **Regression**: (n_samples, n_outputs) or (n_samples,)
  - **Classification**: (n_samples, n_classes) one-hot encoded
- **`task`**: Task type detection
  - `'auto'`: Automatically detect (one-hot → classification, else → regression)
  - `'regression'`: Force regression mode
  - `'classification'`: Force classification mode
- **`scale_classification`**: If True, scales classification outputs based on MF structure (not data-dependent)
- **`verbose_init`**: Print achievable output ranges during initialization

### Task Detection

The algorithm automatically detects the task type when `task='auto'`:

```python
# Regression (single output or multi-output continuous)
y_regression = np.random.randn(100, 1)
wm_reg = WangMendelLearning(system, X, y_regression, task='auto')
# → Detected: regression

# Classification (one-hot encoded)
from sklearn.preprocessing import OneHotEncoder
y_classes = np.random.randint(0, 3, 100)
y_onehot = OneHotEncoder().fit_transform(y_classes.reshape(-1, 1)).toarray()
wm_clf = WangMendelLearning(system, X, y_onehot, task='auto')
# → Detected: classification
```

---

## 3. Training with `fit()`

The `fit()` method executes the complete Wang-Mendel algorithm:

```python
# Train the system
wm.fit(verbose=True)

# The trained system is now ready
trained_system = wm.system
```

### Parameters

- **`verbose`**: If True, prints progress information
  - Task type
  - Data dimensions
  - Number of candidate rules generated
  - Number of conflicts resolved
  - Final rule count

### What Happens During `fit()`

1. **Generate candidate rules** from each training sample
   - For each sample, find the most activated fuzzy term for each variable
   - Create IF-THEN rule using these terms
   - Calculate rule degree (product of all membership degrees)

2. **Resolve conflicts**
   - Group rules with same antecedents
   - Keep only the rule with highest degree
   - Discard conflicting rules

3. **Create final system**
   - Add conflict-free rules to the Mamdani system
   - System is now ready for prediction

### Training Output Example

```
🔄 Starting Wang-Mendel Algorithm...
   Task: REGRESSION
   Data: 500 samples, 2 inputs, 1 outputs

📊 Step 2: Generating candidate rules...
   Candidate rules: 127

🔍 Step 4: Resolving conflicts...
   Conflicts: 32

✅ Training completed!
   Rules generated: 95
   Conflicts resolved: 32
```

---

## 4. Making Predictions

### 4.1 Basic Prediction: `predict()`

```python
# Make predictions
X_test = ...  # shape: (n_samples, n_features)
predictions = wm.predict(X_test)
```

**Returns:**
- **Regression**: Array of continuous values (n_samples, n_outputs)
- **Classification**: Array of predicted class indices (n_samples,)

### 4.2 Classification Probabilities: `predict_proba()`

For classification tasks, get class probabilities:

```python
# Get class probabilities (only for classification)
probabilities = wm.predict_proba(X_test)
# Returns: (n_samples, n_classes) with rows summing to 1
```

**Raises ValueError** if called on regression task.

### 4.3 Membership Degrees: `predict_membership()`

Get membership degrees for each output term:

```python
# Get membership degrees for output terms
membership_dict = wm.predict_membership(X_test)

# Returns dictionary: {output_variable: {term_name: degrees_array}}
# Example:
# {
#   'comfort': {
#     'low': array([0.2, 0.7, ...]),
#     'medium': array([0.6, 0.3, ...]),
#     'high': array([0.1, 0.0, ...])
#   }
# }
```

Useful for **interpretability** - see which linguistic terms are activated.

### 4.4 Detailed Membership: `predict_membership_detailed()`

Get per-sample detailed membership information:

```python
# Get detailed membership for each sample
details = wm.predict_membership_detailed(X_test)

# Returns list of dicts (one per sample):
# [
#   {
#     'sample_idx': 0,
#     'outputs': {'comfort': 6.5},
#     'membership': {
#       'comfort': {
#         'low': 0.2,
#         'medium': 0.6,
#         'high': 0.1
#       }
#     }
#   },
#   ...
# ]
```

---

## 5. Training Statistics

Get comprehensive training statistics:

```python
stats = wm.get_training_stats()
print(stats)
```

**Returns dictionary with:**
```python
{
    'task': 'regression' or 'classification',
    'n_samples': int,
    'n_features': int,
    'n_outputs': int,
    'candidate_rules': int,  # Total rules before conflict resolution
    'final_rules': int,      # Rules after conflict resolution
    'conflicts_resolved': int,
    # For classification only:
    'n_classes': int,
    'classes': list,
    'output_ranges': dict    # Achievable ranges per output variable
}
```

---

## 6. Complete Examples

### 6.1 Regression Example

```python
import numpy as np
from fuzzy_systems.inference import MamdaniSystem
from fuzzy_systems.learning import WangMendelLearning
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Generate synthetic data
np.random.seed(42)
X = np.random.uniform(0, 10, (500, 2))
y = np.sin(X[:, 0]) + np.cos(X[:, 1]) + np.random.normal(0, 0.1, 500)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and configure system
system = MamdaniSystem()

# Configure inputs
system.add_input('x1', (0, 10))
system.add_input('x2', (0, 10))

# Add terms (3 per input)
for var in ['x1', 'x2']:
    system.add_term(var, 'low', 'triangular', (0, 0, 5))
    system.add_term(var, 'medium', 'triangular', (2.5, 5, 7.5))
    system.add_term(var, 'high', 'triangular', (5, 10, 10))

# Configure output
system.add_output('y', (-2, 2))
system.add_term('y', 'negative', 'triangular', (-2, -2, 0))
system.add_term('y', 'zero', 'triangular', (-1, 0, 1))
system.add_term('y', 'positive', 'triangular', (0, 2, 2))

# Train with Wang-Mendel
wm = WangMendelLearning(system, X_train, y_train, task='regression')
wm.fit(verbose=True)

# Evaluate
y_pred = wm.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nTest MSE: {mse:.4f}")
print(f"Test R²: {r2:.4f}")

# Get statistics
stats = wm.get_training_stats()
print(f"Rules generated: {stats['final_rules']}")
```

### 6.2 Classification Example

```python
import numpy as np
from fuzzy_systems.inference import MamdaniSystem
from fuzzy_systems.learning import WangMendelLearning
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report

# Load Iris dataset
iris = load_iris()
X = iris.data[:, :2]  # Use only 2 features for simplicity
y = iris.target

# One-hot encode targets
encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y.reshape(-1, 1))

# Split data
X_train, X_test, y_train, y_test, y_train_onehot, y_test_onehot = train_test_split(
    X, y, y_onehot, test_size=0.3, random_state=42
)

# Create system
system = MamdaniSystem()

# Configure inputs (sepal length and width)
system.add_input('sepal_length', (4, 8))
system.add_input('sepal_width', (2, 5))

# Add terms
for var in ['sepal_length', 'sepal_width']:
    system.add_term(var, 'small', 'triangular', (4, 4, 5.5))
    system.add_term(var, 'medium', 'triangular', (4.5, 6, 7))
    system.add_term(var, 'large', 'triangular', (6, 8, 8))

# Configure outputs (one per class)
for i in range(3):
    system.add_output(f'class_{i}', (0, 1))
    system.add_term(f'class_{i}', 'no', 'triangular', (0, 0, 0.5))
    system.add_term(f'class_{i}', 'yes', 'triangular', (0.5, 1, 1))

# Train with Wang-Mendel
wm = WangMendelLearning(system, X_train, y_train_onehot, task='classification')
wm.fit(verbose=True)

# Predict classes
y_pred = wm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Get probabilities
y_proba = wm.predict_proba(X_test)

print(f"\nTest Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Show sample predictions with probabilities
print("\nSample predictions:")
for i in range(5):
    print(f"Sample {i}: True={y_test[i]}, Pred={y_pred[i]}, "
          f"Proba={y_proba[i]}")
```

---

## 7. Comparing Wang-Mendel vs Other Methods

| Aspect | Wang-Mendel | ANFIS | Manual Rules |
|--------|-------------|-------|--------------|
| **Rule Generation** | Automatic from data | Optimized via learning | Manual expert knowledge |
| **Training Speed** | Very fast (single pass) | Slow (iterative) | N/A |
| **Interpretability** | High (linguistic rules) | Medium (can inspect MFs) | High |
| **Accuracy** | Good baseline | Best (optimized) | Depends on expert |
| **Requires** | Data + MF definitions | Data only | Domain expertise |
| **Overfitting Risk** | Low (conflict resolution) | Medium (can overfit) | Low |
| **Best For** | Quick baseline, interpretable systems | High accuracy requirements | Known domain rules |

---

## 8. Tips and Best Practices

### Membership Function Design

- **Use 3-5 MFs per variable** for good coverage without explosion
- **Overlap MFs** to ensure smooth transitions (typically 50% overlap)
- **Cover the entire universe** to avoid undefined regions
- **Use triangular/trapezoidal** for simplicity and interpretability

### Data Preparation

- **Normalize/scale inputs** to match MF universes
- **Remove outliers** before defining MF ranges
- **Ensure balanced classes** for classification (or use weighted sampling)

### Rule Optimization

```python
# Check rule statistics
stats = wm.get_training_stats()
print(f"Rule efficiency: {stats['final_rules']} / {stats['candidate_rules']} "
      f"({100*stats['final_rules']/stats['candidate_rules']:.1f}%)")

# Many conflicts = overlapping/redundant patterns
if stats['conflicts_resolved'] > stats['final_rules']:
    print("⚠️ High conflict rate - consider:")
    print("  - Reducing number of MFs")
    print("  - Adjusting MF overlap")
    print("  - Checking for noisy data")
```

### Classification Setup

- **Use one output per class** (one-hot encoding)
- **Define binary MFs** for each output: 'no' and 'yes'
- **Enable scaling** for better probability calibration: `scale_classification=True`
- **Check achievable ranges** with `verbose_init=True`

### Troubleshooting

| Problem | Solution |
|---------|----------|
| Too many rules | Reduce number of MFs per variable |
| Too few rules | Increase MF overlap, check data coverage |
| Poor predictions | Add more MFs, adjust MF shapes/positions |
| Rules not interpretable | Use simpler MF shapes (triangular) |
| Conflicts > 50% | Reduce MFs or adjust overlap |

---

## 9. Advantages and Limitations

### ✅ Advantages

1. **Fast**: Single-pass through data (no iterative optimization)
2. **Interpretable**: Generates readable IF-THEN rules
3. **No hyperparameters**: No learning rate, epochs, etc.
4. **Automatic**: No manual rule crafting needed
5. **Robust**: Conflict resolution handles contradictions
6. **Baseline**: Good starting point before trying complex methods

### ⚠️ Limitations

1. **Depends on MF design**: Requires good initial MF placement
2. **No optimization**: Rules not fine-tuned for accuracy
3. **Discrete selection**: Uses only most-activated terms (not fuzzy combination)
4. **Fixed structure**: Cannot adjust MF parameters during training
5. **Scalability**: Rule count grows exponentially with inputs/MFs

### When to Use Wang-Mendel

✅ **Good for:**
- Quick prototyping and baseline modeling
- Interpretable systems (medical, finance, control)
- Small-to-medium datasets
- When domain knowledge can guide MF design
- Systems requiring explainability

❌ **Not ideal for:**
- Maximum accuracy requirements (use ANFIS or deep learning)
- Very high-dimensional inputs (curse of dimensionality)
- When MF design is difficult
- Real-time learning/adaptation needed

---

## 10. Integration with Other Methods

Wang-Mendel can be combined with other techniques:

### Sequential Training

```python
# Step 1: Get initial rules with Wang-Mendel
wm = WangMendelLearning(system, X_train, y_train)
wm.fit(verbose=True)

# Step 2: Fine-tune with gradient-based learning
from fuzzy_systems.learning import MamdaniLearning
ml = MamdaniLearning(wm.system, X_train, y_train)
ml.fit(epochs=50, optimizer='adam')

# Now system has good initial rules + optimized parameters
```

### Ensemble Methods

```python
# Create multiple Wang-Mendel models with different MF configurations
models = []
for n_mfs in [3, 4, 5]:
    system = create_system(n_mfs)  # Create with different MF counts
    wm = WangMendelLearning(system, X_train, y_train)
    wm.fit()
    models.append(wm)

# Ensemble prediction (average)
predictions = [model.predict(X_test) for model in models]
y_pred = np.mean(predictions, axis=0)
```

---

## 11. Accessing Generated Rules

```python
# Access final rules
print(f"Total rules: {len(wm.final_rules)}")

# Print first 5 rules
for i, rule in enumerate(wm.final_rules[:5]):
    print(f"\nRule {i+1}:")
    print(f"  IF {rule['antecedents']}")
    print(f"  THEN {rule['consequents']}")
    print(f"  Degree: {rule['degree']:.3f}")

# Rules are also in the system
wm.system.print_rules()
```

---

## References

- Wang, L. X., & Mendel, J. M. (1992). "Generating fuzzy rules by learning from examples." IEEE Transactions on Systems, Man, and Cybernetics, 22(6), 1414-1427.
