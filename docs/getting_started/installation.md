# Installation

## Requirements

- Python 3.8 or higher
- pip (Python package manager)

## Install from PyPI

The simplest way to install pyfuzzy-toolbox is via pip:

```bash
pip install pyfuzzy-toolbox
```

### Optional Dependencies

Install with machine learning support (ANFIS, Wang-Mendel, optimization):

```bash
pip install pyfuzzy-toolbox[ml]
```

Install with development tools (testing, linting):

```bash
pip install pyfuzzy-toolbox[dev]
```

Install everything:

```bash
pip install pyfuzzy-toolbox[all]
```

## Install from Source

For development or to get the latest features:

```bash
git clone https://github.com/1moi6/pyfuzzy-toolbox.git
cd pyfuzzy-toolbox
pip install -e .
```

For editable install with development dependencies:

```bash
pip install -e .[dev]
```

## Verify Installation

```python
import fuzzy_systems as fs

print(f"pyfuzzy-toolbox version: {fs.__version__}")
```

## Import Convention

The recommended import convention is:

```python
import fuzzy_systems as fs
```

Note: The package name on PyPI is `pyfuzzy-toolbox`, but you import it as `fuzzy_systems`.

## Next Steps

- [Quickstart](quickstart.md): Create your first fuzzy system in 5 minutes
- [Key Concepts](key_concepts.md): Learn fundamental fuzzy logic concepts
