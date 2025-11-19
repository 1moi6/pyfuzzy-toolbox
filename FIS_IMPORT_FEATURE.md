# MATLAB .fis File Import Feature

## Overview

The pyfuzzy-toolbox now supports importing fuzzy inference systems directly from MATLAB Fuzzy Logic Toolbox `.fis` files. This feature enables seamless migration from MATLAB to Python.

## Features

### 1. Direct FIS Import in Python

Import MATLAB `.fis` files using the `from_fis()` classmethod:

```python
from fuzzy_systems.inference import MamdaniSystem, SugenoSystem

# Load Mamdani system
fis = MamdaniSystem.from_fis('my_system.fis')

# Load Sugeno system
fis = SugenoSystem.from_fis('my_sugeno_system.fis')

# Use the system normally
result = fis.evaluate({'input1': 0.5, 'input2': 0.7})
```

### 2. Streamlit Interface Support

The Streamlit web interface now accepts both `.json` and `.fis` files:

1. Launch the interface: `pyfuzzy --interface`
2. Navigate to the "Mamdani" or "Sugeno (TSK)" page
3. Use the file uploader to select either:
   - `.json` files (pyfuzzy-toolbox native format)
   - `.fis` files (MATLAB Fuzzy Logic Toolbox format)
4. Click "Import FIS" to load the system

## Supported Features

The parser supports the following MATLAB FIS features:

### System Configuration
- ✅ System name and type (Mamdani/Sugeno)
- ✅ Number of inputs and outputs
- ✅ AND method (min, prod)
- ✅ OR method (max, probor)
- ✅ Defuzzification method (centroid, bisector, mom, som, lom)

### Variables
- ✅ Input and output variables
- ✅ Variable names and universe of discourse (range)
- ✅ Linguistic terms with membership functions

### Membership Functions
- ✅ Triangular (trimf)
- ✅ Trapezoidal (trapmf)
- ✅ Gaussian (gaussmf)
- ✅ Generalized Bell (gbellmf)
- ✅ Sigmoid (sigmf)

### Rules
- ✅ Antecedents and consequents
- ✅ AND/OR operators
- ✅ Rule weights
- ✅ Multiple rules (tested with 500+ rules)

## Implementation Details

### Architecture

The `from_fis()` method:
1. Parses the `.fis` file into sections (System, Input, Output, Rules)
2. Creates the appropriate FIS type (MamdaniSystem or SugenoSystem)
3. Adds input/output variables with their membership functions
4. Converts MATLAB rule format to pyfuzzy-toolbox format
5. Configures inference operators and defuzzification methods

### Rule Format Conversion

MATLAB `.fis` rule format:
```
1 2 3 4, 5 (1) : 1
```

Converts to pyfuzzy-toolbox format:
- Antecedents: `{'input1': 'term1', 'input2': 'term2', ...}`
- Consequents: `{'output1': 'term5'}`
- Operator: `'AND'` (1) or `'OR'` (2)
- Weight: `1.0`

### Membership Function Mapping

| MATLAB | pyfuzzy-toolbox |
|--------|-----------------|
| trimf | triangular |
| trapmf | trapezoidal |
| gaussmf | gaussian |
| gbellmf | generalized_bell |
| sigmf | sigmoid |

## Testing

### Test Files Included

1. **test_fis_import.py** - Comprehensive test with ASFALTO.fis
   - Tests variable loading
   - Tests rule conversion
   - Compares results with JSON version
   - Validates inference accuracy

2. **test_streamlit_fis.py** - Tests Streamlit upload simulation
   - Tests temporary file handling
   - Tests JSON conversion
   - Validates data structure

### Running Tests

```bash
# Test basic FIS import
python3 test_fis_import.py

# Test Streamlit upload simulation
python3 test_streamlit_fis.py
```

### Example: ASFALTO.fis

The included ASFALTO.fis file (asphalt adhesion coefficient):
- **4 inputs**: UMIDADE (humidity), TEXTURA (texture), SUJEIRA (dirt), PNEU (tire)
- **1 output**: CA (adhesion coefficient)
- **500 rules**: Comprehensive rule base
- **5 linguistic terms per variable**: Fine-grained control

Test results show **identical outputs** between `.fis` and `.json` versions:
```
Resultado .fis:  CA = 0.637857
Resultado .json: CA = 0.637857
Diferença: 0.0000000000
```

## Usage Examples

### Example 1: Simple Import and Evaluation

```python
from fuzzy_systems.inference import MamdaniSystem

# Load MATLAB FIS file
fis = MamdaniSystem.from_fis('temperature_control.fis')

# Evaluate
result = fis.evaluate({'temperature': 25, 'humidity': 60})
print(f"Fan speed: {result['fan_speed']}")
```

### Example 2: Convert .fis to .json

```python
from fuzzy_systems.inference import MamdaniSystem

# Load .fis
fis = MamdaniSystem.from_fis('my_system.fis')

# Save as .json (pyfuzzy-toolbox native format)
fis.to_json('my_system.json')
```

### Example 3: Batch Processing

```python
from fuzzy_systems.inference import MamdaniSystem
import pandas as pd

# Load FIS
fis = MamdaniSystem.from_fis('classifier.fis')

# Load data
data = pd.read_csv('test_data.csv')

# Process each row
results = []
for _, row in data.iterrows():
    inputs = row.to_dict()
    output = fis.evaluate(inputs)
    results.append(output)

# Save results
pd.DataFrame(results).to_csv('results.csv', index=False)
```

## Migration from MATLAB

To migrate a MATLAB Fuzzy Logic Toolbox project:

1. **Export from MATLAB**: Use `writeFIS(fis, 'filename.fis')`
2. **Import to pyfuzzy-toolbox**: Use `MamdaniSystem.from_fis('filename.fis')`
3. **Optional**: Save as JSON for faster loading: `fis.to_json('filename.json')`

## Limitations

1. **Custom Functions**: MATLAB's custom membership functions are not supported
2. **FIS Files Only**: Does not support MATLAB's binary `.mat` files
3. **Standard Operators**: Only standard t-norms and s-norms are supported

## Technical Notes

### Performance

- **Parser Performance**: ~0.1s for systems with 500+ rules
- **Memory Usage**: Same as pyfuzzy-toolbox native format
- **Inference Speed**: Identical to JSON-loaded systems

### File Format

The parser expects standard MATLAB FIS text format:
```
[System]
Name='SystemName'
Type='mamdani'
...

[Input1]
Name='InputVar'
Range=[0 100]
...

[Rules]
1 2, 3 (1) : 1
...
```

## Future Enhancements

Potential improvements for future versions:
- Support for custom membership functions via Python code generation
- Direct `.mat` file support
- Fuzzy C-Means (FCM) clustering import
- ANFIS model import from MATLAB

## References

- **MATLAB Fuzzy Logic Toolbox**: Documentation for `.fis` file format
- **Implementation**: `fuzzy_systems/inference/systems.py` (lines 2285-2570)
- **Tests**: `test_fis_import.py`, `test_streamlit_fis.py`

## Version

Feature added in version **1.1.7**
