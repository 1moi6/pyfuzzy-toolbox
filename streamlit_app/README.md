# pyfuzzy-toolbox Interactive Interface

A modern, elegant, and academic interface for the pyfuzzy-toolbox library built with Streamlit.

## Features

- **Modern Design**: Clean, professional interface with smooth animations
- **Three Main Modules**:
  - ⚙️ **Inference Systems**: Build Mamdani and Sugeno fuzzy inference systems
  - 🧠 **Learning & Optimization**: Train ANFIS, extract rules with Wang-Mendel, optimize with metaheuristics
  - 📊 **Dynamic Systems**: Solve fuzzy ODEs and simulate p-fuzzy systems
- **Interactive Visualizations**: Real-time plots and animations
- **Code Generation**: Export Python code for your systems
- **Educational**: Perfect for teaching and learning fuzzy systems

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
streamlit run main.py
```

3. Open your browser at `http://localhost:8501`

## Project Structure

```
streamlit_app/
├── main.py                 # Main application with SPA navigation
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── modules/               # Page modules
│   ├── home.py           # Home page with overview
│   ├── inference.py      # Inference systems interface
│   ├── learning.py       # Learning algorithms interface
│   └── dynamics.py       # Dynamic systems interface
├── components/           # Reusable UI components (future)
├── utils/               # Utility functions (future)
└── assets/              # Static resources (future)
```

## Usage

### Home Page
- Overview of pyfuzzy-toolbox
- Quick access to three main modules
- Installation instructions and examples
- Links to documentation

### Inference Systems
- Create Mamdani or Sugeno systems
- Define input/output variables
- Build fuzzy rules visually
- Test and simulate systems

### Learning & Optimization
- Train ANFIS from data
- Extract rules with Wang-Mendel
- Optimize systems with PSO, DE, GA, SA
- Visualize training progress

### Dynamic Systems
- Solve fuzzy ODEs with uncertainty
- Simulate p-fuzzy discrete systems
- Simulate p-fuzzy continuous systems
- Phase space visualization

## Development Status

**Current Version**: 1.0.0 (MVP)

✅ Completed:
- Modern home page with animations
- Navigation system (SPA)
- Module structure
- Basic UI for all three modules

🚧 In Progress:
- Interactive inference system builder
- ANFIS training interface
- Fuzzy ODE solver interface
- Data import/export features

## Requirements

- Python 3.8+
- Streamlit 1.28+
- pyfuzzy-toolbox 1.0.7+
- streamlit-lottie (for animations)

## Contributing

This interface is part of the pyfuzzy-toolbox project. For contributions:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - see main repository for details

## Author

Moiseis Cecconello
- GitHub: [1moi6/pyfuzzy-toolbox](https://github.com/1moi6/pyfuzzy-toolbox)
- Documentation: [https://1moi6.github.io/pyfuzzy-toolbox/](https://1moi6.github.io/pyfuzzy-toolbox/)

## Acknowledgments

Built with:
- [Streamlit](https://streamlit.io/) - Web framework
- [Lottie](https://lottiefiles.com/) - Animations
- [pyfuzzy-toolbox](https://pypi.org/project/pyfuzzy-toolbox/) - Fuzzy systems library
