# Documentation for pyfuzzy-toolbox

This directory contains the documentation for pyfuzzy-toolbox using MkDocs with Material theme.

## 📁 Structure

```
docs/
├── index.md                      # Landing page
├── getting_started/
│   ├── installation.md           ✅ Complete
│   ├── quickstart.md             ✅ Complete
│   └── key_concepts.md           🚧 TODO
├── user_guide/
│   ├── fundamentals.md           🚧 TODO
│   ├── inference_systems.md      🚧 TODO
│   ├── learning.md               🚧 TODO
│   └── dynamics.md               🚧 TODO
├── api_reference/
│   ├── core.md                   ✅ Complete
│   ├── inference.md              🚧 TODO
│   ├── learning.md               🚧 TODO
│   └── dynamics.md               🚧 TODO
├── examples/
│   └── gallery.md                ✅ Complete
└── contributing/
    └── development.md            🚧 TODO
```

## 🚀 Quick Start

### Install MkDocs

```bash
pip install mkdocs mkdocs-material
```

### Preview Documentation Locally

```bash
cd /path/to/pyfuzzy-toolbox
mkdocs serve
```

Then open http://127.0.0.1:8000 in your browser.

### Build Documentation

```bash
mkdocs build
```

This creates a `site/` directory with the static HTML files.

## 📝 What's Been Done

### ✅ Completed

1. **Documentation Structure**
   - Created organized folder hierarchy
   - Set up MkDocs configuration with Material theme
   - Configured navigation, search, and code highlighting

2. **API Reference: Core Module**
   - Complete reference for `fuzzy_systems.core`
   - Documented all membership functions
   - Documented `FuzzySet` and `LinguisticVariable` classes
   - Documented fuzzy operators
   - Documented defuzzification methods
   - Included practical examples for each function/class

3. **Getting Started**
   - Installation guide with all options
   - Quickstart tutorial (5-minute guide)
   - Complete working example

4. **Examples Gallery**
   - Organized all 18 Colab notebooks
   - Categorized by difficulty (Beginner/Intermediate/Advanced)
   - Added descriptions and learning objectives
   - Included estimated completion times

5. **Analysis Tools**
   - Created `extract_api.py` script to analyze notebooks
   - Generated `api_analysis.json` with all classes and methods used

## 🚧 Next Steps

### Priority 1: API Reference (High Priority)

Complete API documentation for remaining modules:

#### `api_reference/inference.md`
Document:
- `MamdaniSystem`: `.add_input()`, `.add_output()`, `.add_term()`, `.add_rules()`, `.evaluate()`, `.plot_variables()`, `.plot_rule_matrix()`, `.export_rules()`, `.import_rules()`
- `SugenoSystem`: Same methods + order-specific behavior

#### `api_reference/learning.md`
Document:
- `WangMendelLearning`: `.fit()`, `.predict()`, `.get_training_stats()`
- `ANFIS`: `.fit()`, `.predict()`, `.get_training_history()`
- `MamdaniLearning`: `.from_mamdani()`, `.to_mamdani()`, `.fit()` with PSO/DE/GA

#### `api_reference/dynamics.md`
Document:
- `PFuzzyDiscrete`: `.simulate()`, `.plot_trajectory()`, `.plot_phase_space()`, `.to_csv()`
- `PFuzzyContinuous`: Same methods + continuous-specific
- `FuzzyODE`: `.solve()`, `.plot_envelope()`

### Priority 2: User Guides (Medium Priority)

Create tutorial-style guides based on notebooks:

#### `user_guide/fundamentals.md`
- When to use each membership function type
- Understanding fuzzification
- Working with linguistic variables
- Fuzzy operators in practice

#### `user_guide/inference_systems.md`
- Mamdani vs Sugeno: when to use each
- Building rule bases
- Defuzzification methods comparison
- Tuning fuzzy systems

#### `user_guide/learning.md`
- When to use Wang-Mendel vs ANFIS vs optimization
- Choosing the right learning method
- Hyperparameter tuning
- Avoiding overfitting

#### `user_guide/dynamics.md`
- p-Fuzzy systems: discrete vs continuous
- Modeling dynamic systems with fuzzy rules
- Fuzzy ODEs for uncertainty propagation
- Applications in ecology and population dynamics

### Priority 3: Additional Pages (Low Priority)

- `getting_started/key_concepts.md`: Fuzzy logic theory primer
- `contributing/development.md`: Setup dev environment, run tests
- `about/license.md`: MIT license text

## 🛠️ Tools Used

- **MkDocs**: Static site generator
- **Material for MkDocs**: Modern, responsive theme
- **Python Markdown**: Extensions for better formatting
- **PyMdown Extensions**: Code highlighting, admonitions, etc.

## 📖 Documentation Best Practices Followed

1. **✅ Organized by User Journey**
   - Getting Started → User Guide → API Reference → Examples

2. **✅ API First Approach**
   - Complete API reference for core module
   - Each class/function has: description, parameters, returns, examples

3. **✅ Practical Examples**
   - Every API entry includes working code
   - Examples gallery links to interactive notebooks

4. **✅ Search and Navigation**
   - Configured instant search
   - Tab-based navigation
   - Table of contents in each page

5. **✅ Community Standards**
   - Follows scikit-learn/pandas documentation patterns
   - Uses Material theme (modern standard)
   - Markdown-based (easy to maintain)

## 🚀 Deployment Options

### GitHub Pages (Recommended)

Add to `.github/workflows/docs.yml`:

```yaml
name: Deploy Docs
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: 3.x
      - run: pip install mkdocs mkdocs-material
      - run: mkdocs gh-deploy --force
```

### ReadTheDocs

1. Connect GitHub repository at readthedocs.org
2. ReadTheDocs auto-detects `mkdocs.yml`
3. Documentation rebuilds automatically on push

## 📊 Current Status

**Completion: ~40%**

- ✅ Infrastructure: 100%
- ✅ Core API: 100%
- ⏳ Other APIs: 0%
- ✅ Examples Gallery: 100%
- ✅ Getting Started: 66% (2/3)
- ⏳ User Guides: 0%

**Estimated time to complete:**
- API Reference: 6-8 hours
- User Guides: 8-10 hours
- Remaining pages: 2-3 hours

**Total: 16-21 hours**

## 🤝 Contributing

To add new documentation:

1. Create/edit `.md` files in `docs/`
2. Update `nav` section in `mkdocs.yml` if adding new pages
3. Preview with `mkdocs serve`
4. Commit and push

## 📞 Questions?

- Check the MkDocs documentation: https://www.mkdocs.org/
- Material theme docs: https://squidfunk.github.io/mkdocs-material/
