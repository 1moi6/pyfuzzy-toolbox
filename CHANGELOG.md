# Changelog

All notable changes to **fuzzy-systems** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Type-2 fuzzy systems support
- LSTM-Fuzzy hybrid architectures
- Advanced visualization dashboards
- Interactive web interface for system design
- Additional metaheuristic optimizers (ACO, ABC)

---
## [1.0.2] - 2025-10-27

### Added
- Unified `WangMendelLearning` class supporting both regression and classification
- Automatic task detection (one-hot → classification)
- Structure-based output scaling for classification (maps achievable fuzzy range to [0, 1])
- `predict_membership()` method to analyze linguistic term activations
- `predict_membership_detailed()` for comprehensive membership analysis
- Modern color palettes for rule visualization (`plot_rule_matrix`)
- `plot_rule_matrix_2d()` for 2-input systems
- `return_axes` parameter in `plot_variables()` for customization

### Changed
- Simplified `add_rule()` and `add_rules()` with 3 input formats (dict, list, indices)
- Optional universe of discourse for Sugeno output variables
- `WangMendelRegression` and `WangMendelClassification` now aliases of unified class
- Improved output scaling based on membership function structure (not data)

### Fixed
- Duplicate plotting issue in `plot_variables()`
- Rule matrix visualization overlapping labels

## [1.0.0] - 2024-10-25

### 🎉 Initial Public Release

First stable release of fuzzy-systems, a comprehensive Python library for fuzzy logic applications.

### Added

#### Core Fuzzy Systems
- **Membership Functions**: triangular, trapezoidal, gaussian, generalized bell, sigmoid, singleton
- **T-norms (AND)**: min, product, Łukasiewicz, drastic, Hamacher
- **S-norms (OR)**: max, probabilistic, bounded sum, drastic, Hamacher
- **Negation operators**: standard, Sugeno, Yager
- **Defuzzification methods**: centroid, bisector, mean of maximum, smallest/largest of maximum
- **Implication methods**: Mamdani (min), Larsen (product)
- **Aggregation methods**: max, sum, probabilistic

#### Inference Systems
- **Mamdani Fuzzy Inference System** - Complete implementation with linguistic variables
- **Sugeno/TSK Fuzzy Inference System** - Order 0 and Order 1 systems
- **Simplified API** - Intuitive system creation with minimal code
- **Rule management** - Add, remove, import/export rules (CSV, JSON, TXT)
- **Visualization** - Plot membership functions, surfaces, and system responses

#### Learning and Optimization
- **ANFIS** (Adaptive Neuro-Fuzzy Inference System)
  - Hybrid learning (LSE + backpropagation)
  - Lyapunov stability for adaptive learning rate
  - L1/L2 regularization
  - Mini-batch training
  - Early stopping
  - Model save/load

- **Wang-Mendel Algorithm**
  - Automatic rule extraction from data
  - Support for regression and classification
  - Partition tuning

- **MamdaniLearning**
  - Gradient-based optimization of Mamdani systems
  - Metaheuristic optimization (PSO, DE, GA)
  - Hybrid knowledge + data approach
  - Integration with MamdaniSystem (bidirectional conversion)

- **Metaheuristics**
  - Particle Swarm Optimization (PSO)
  - Differential Evolution (DE)
  - Genetic Algorithm (GA)

#### Dynamic Systems
- **Fuzzy ODE Solver**
  - Solve ODEs with fuzzy initial conditions and parameters
  - α-level method for uncertainty propagation
  - Support for multiple α-cuts
  - Grid-based discretization
  - Export results to DataFrame/CSV
  - Multiple integration methods (RK45, RK23, DOP853, etc.)

- **p-Fuzzy Systems**
  - Discrete and continuous dynamic systems
  - Evolution governed by fuzzy rules
  - Absolute and incremental modes
  - Applications: population dynamics, predator-prey models

#### Examples and Documentation
- **16 comprehensive examples** organized by complexity:
  - `01_inference/`: 5 examples (basic Mamdani, Sugeno, tipping problem, visualization, import/export)
  - `02_learning/`: 4 examples (Wang-Mendel, ANFIS with notebooks)
  - `03_dynamics/`: 3 examples (p-fuzzy progression: simple → population → predator-prey)
  - `04_complete/`: 4 end-to-end professional applications

- **Complete READMEs** with learning guides
- **Jupyter notebooks** for interactive learning
- **Detailed docstrings** throughout the codebase

### Technical Features
- **Type hints** for better IDE support
- **Modular architecture** - Easy to extend
- **NumPy vectorization** for performance
- **Comprehensive test suite**
- **Professional code quality** (black, flake8 compliant)

### Documentation
- Main README with quickstart guide
- Module-specific READMEs (learning, dynamics)
- Integration guide (MamdaniSystem ↔ MamdaniLearning)
- 16 documented examples with progression paths

### Dependencies
- Python >= 3.8
- numpy >= 1.20.0
- matplotlib >= 3.3.0
- scipy >= 1.6.0
- Optional: scikit-learn, pandas, joblib (for ML features)

---

## Version History

- **1.0.0** (2024-10-25) - Initial public release
- **0.2.0** (2024-10-24) - Internal pre-release with integration features
- **0.1.0** (2024-10-22) - Internal alpha version

---

## Notes

### Versioning Strategy
- **MAJOR**: Incompatible API changes
- **MINOR**: New features, backward-compatible
- **PATCH**: Bug fixes, backward-compatible

### Deprecation Policy
- Deprecated features will be maintained for at least 2 minor versions
- Deprecation warnings will be issued before removal
- Migration guides will be provided

### Support
- Report bugs: https://github.com/1moi6/fuzzy-systems/issues
- Feature requests: https://github.com/1moi6/fuzzy-systems/issues
- Questions: https://github.com/1moi6/fuzzy-systems/discussions

---

**Thank you for using fuzzy-systems!** 🎉