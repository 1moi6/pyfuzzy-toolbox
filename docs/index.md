# pyfuzzy-toolbox Documentation

Welcome to **pyfuzzy-toolbox**, a comprehensive Python library for Fuzzy Systems with focus on education and professional applications.

## Features

- **🧩 Core**: Membership functions, fuzzy sets, linguistic variables, operators
- **🎛️ Inference**: Mamdani and Sugeno/TSK systems
- **🧠 Learning**: ANFIS, Wang-Mendel, metaheuristic optimization (PSO, DE, GA)
- **🌊 Dynamics**: Fuzzy ODEs and p-fuzzy systems

## Quick Links

<div class="grid cards" markdown>

-   :material-clock-fast:{ .lg .middle } __Getting Started__

    ---

    Install pyfuzzy-toolbox and create your first fuzzy system in 5 minutes

    [:octicons-arrow-right-24: Installation](getting_started/installation.md)
    [:octicons-arrow-right-24: Quickstart](getting_started/quickstart.md)

-   :material-book-open-page-variant:{ .lg .middle } __User Guide__

    ---

    Learn how to use fuzzy systems to solve real-world problems

    [:octicons-arrow-right-24: Fundamentals](user_guide/fundamentals.md)
    [:octicons-arrow-right-24: Inference Systems](user_guide/inference_systems.md)

-   :material-api:{ .lg .middle } __API Reference__

    ---

    Complete reference for all classes and methods

    [:octicons-arrow-right-24: Core API](api_reference/core.md)
    [:octicons-arrow-right-24: Inference API](api_reference/inference.md)

-   :material-application-brackets:{ .lg .middle } __Examples__

    ---

    Gallery of Colab notebooks with practical examples

    [:octicons-arrow-right-24: Examples Gallery](examples/gallery.md)

</div>

## Installation

```bash
pip install pyfuzzy-toolbox
```

## Quick Example

```python
import fuzzy_systems as fs

# Create Mamdani system
system = fs.MamdaniSystem()
system.add_input('temperature', (0, 40))
system.add_output('fan_speed', (0, 100))

# Add terms
system.add_term('temperature', 'cold', 'triangular', (0, 0, 20))
system.add_term('temperature', 'hot', 'triangular', (20, 40, 40))
system.add_term('fan_speed', 'slow', 'triangular', (0, 0, 50))
system.add_term('fan_speed', 'fast', 'triangular', (50, 100, 100))

# Add rules
system.add_rules([('cold', 'slow'), ('hot', 'fast')])

# Evaluate
result = system.evaluate(temperature=25)
print(f"Fan speed: {result['fan_speed']:.1f}%")
```

## Community & Support

- **PyPI**: [pypi.org/project/pyfuzzy-toolbox](https://pypi.org/project/pyfuzzy-toolbox/)
- **GitHub**: [github.com/1moi6/pyfuzzy-toolbox](https://github.com/1moi6/pyfuzzy-toolbox)
- **Issues**: [Report bugs or request features](https://github.com/1moi6/pyfuzzy-toolbox/issues)

## Citation

```bibtex
@software{pyfuzzy_toolbox,
  title = {pyfuzzy-toolbox: A Comprehensive Python Library for Fuzzy Systems},
  author = {Cecconello, Moiseis},
  year = {2025},
  url = {https://github.com/1moi6/pyfuzzy-toolbox}
}
```

## License

MIT License - see [LICENSE](https://github.com/1moi6/pyfuzzy-toolbox/blob/main/LICENSE) for details.
