# Quick Start Guide

## Installation

### 1. Install Python dependencies

```bash
cd streamlit_app
pip install -r requirements.txt
```

### 2. Run the application

```bash
streamlit run main.py
```

### 3. Open in browser

The application will automatically open at `http://localhost:8501`

If it doesn't open automatically, click the link in the terminal.

## Navigation

The interface uses a **Single Page Application (SPA)** design:

- **Home**: Overview and quick access to modules
- **Inference Systems**: Create and test Mamdani/Sugeno systems
- **Learning & Optimization**: Train ANFIS, Wang-Mendel, and optimize systems
- **Dynamic Systems**: Solve Fuzzy ODEs and p-Fuzzy systems

## Features

### Home Page
✅ Modern design with animations
✅ Three module cards with descriptions
✅ Installation instructions
✅ Code examples
✅ Links to documentation

### Inference Systems (🚧 In Progress)
- Create Mamdani or Sugeno fuzzy systems
- Define linguistic variables
- Add fuzzy terms with membership functions
- Build rules visually
- Test and simulate
- Export to Python code

### Learning & Optimization (🚧 In Progress)
- **ANFIS**: Train adaptive neuro-fuzzy systems
- **Wang-Mendel**: Extract rules from data
- **Metaheuristics**: Optimize with PSO, DE, GA, SA
- Upload CSV datasets
- Visualize training curves

### Dynamic Systems (🚧 In Progress)
- **Fuzzy ODE**: Solve differential equations with fuzzy parameters
- **p-Fuzzy Discrete**: Discrete-time evolution
- **p-Fuzzy Continuous**: Continuous-time evolution
- Phase space visualization
- Uncertainty propagation

## Customization

### Theme

Edit `.streamlit/config.toml` to change colors:

```toml
[theme]
primaryColor = "#667eea"        # Main accent color
backgroundColor = "#ffffff"     # Background
secondaryBackgroundColor = "#f3f4f6"  # Secondary background
textColor = "#1f2937"          # Text color
```

### Animations

The home page uses Lottie animations from [LottieFiles](https://lottiefiles.com/).

To change the animation, edit `modules/home.py`:

```python
lottie_science = load_lottieurl("YOUR_LOTTIE_URL_HERE")
```

## Development

### Project Structure

```
streamlit_app/
├── main.py                 # Entry point
├── .streamlit/
│   └── config.toml        # Streamlit configuration
├── modules/
│   ├── home.py           # Home page
│   ├── inference.py      # Inference module
│   ├── learning.py       # Learning module
│   └── dynamics.py       # Dynamics module
├── components/           # (Future) Reusable components
├── utils/               # (Future) Utility functions
└── assets/              # (Future) Static files
```

### Adding New Features

1. **New Module**: Create a new file in `modules/`
2. **Import**: Add import in `main.py`
3. **Navigation**: Add page routing in `main.py`
4. **Link**: Add navigation button in other modules

Example:

```python
# modules/new_module.py
import streamlit as st

def run():
    st.title("New Module")
    # Your code here

# main.py
from modules import home, inference, learning, dynamics, new_module

if current_page == 'new_module':
    new_module.run()
```

## Troubleshooting

### Port already in use

```bash
streamlit run main.py --server.port 8502
```

### Module not found

Make sure you're in the correct directory:

```bash
cd streamlit_app
python -c "import streamlit; print(streamlit.__version__)"
```

### Lottie animations not loading

Check internet connection. Animations are loaded from CDN.

## Next Steps

1. ✅ Run the application
2. 🚧 Explore the three modules
3. 🚧 Try the code examples
4. 🚧 Read the documentation
5. 🚧 Contribute to development

## Resources

- **Documentation**: https://1moi6.github.io/pyfuzzy-toolbox/
- **GitHub**: https://github.com/1moi6/pyfuzzy-toolbox
- **PyPI**: https://pypi.org/project/pyfuzzy-toolbox/
- **Streamlit Docs**: https://docs.streamlit.io/

## Support

For issues or questions:

1. Check the [documentation](https://1moi6.github.io/pyfuzzy-toolbox/)
2. Open an issue on [GitHub](https://github.com/1moi6/pyfuzzy-toolbox/issues)
3. Read the [Streamlit community forum](https://discuss.streamlit.io/)

---

**Happy coding!** 🚀
