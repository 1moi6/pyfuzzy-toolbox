# Command Line Interface (CLI)

pyfuzzy-toolbox includes a modern command-line interface for quick access to the web interface and other utilities.

## Installation

To use the CLI, install pyfuzzy-toolbox with the `[ui]` extras:

=== "macOS/Linux"
    ```bash
    pip install 'pyfuzzy-toolbox[ui]'
    ```

=== "Windows"
    ```bash
    pip install pyfuzzy-toolbox[ui]
    ```

!!! tip "Why the quotes?"
    On macOS/Linux (Zsh/Bash), the square brackets `[]` need to be quoted to prevent shell expansion.

## Quick Start

Launch the web interface with a single command:

```bash
pyfuzzy interface
```

This will:

1. ✅ Start the Streamlit server
2. ✅ Automatically open your browser
3. ✅ Display the interactive ANFIS interface

Press `Ctrl+C` to stop the server.

## Available Commands

### `interface` - Launch Web Interface

Open the interactive web application.

```bash
pyfuzzy interface [OPTIONS]
```

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--port` | int | 8501 | Port number for the web server |
| `--host` | string | localhost | Host address to bind |
| `--dark-theme` | flag | False | Use dark theme |
| `--browser` / `--no-browser` | flag | True | Auto-open browser |
| `--help` | flag | - | Show help message |

**Examples:**

```bash
# Default: localhost:8501, auto-open browser
pyfuzzy interface

# Custom port
pyfuzzy interface --port 8080

# Dark theme without browser
pyfuzzy interface --dark-theme --no-browser

# Custom host (for remote access)
pyfuzzy interface --host 0.0.0.0 --port 8080
```

!!! info "Port Auto-Detection"
    If the specified port is already in use, pyfuzzy will automatically find the next available port.

### `version` - Show Version Info

Display package version and system information.

```bash
pyfuzzy version
```

**Output:**
```
╭── Package Info ──╮
│ pyfuzzy-toolbox  │
│ Version: 1.1.3   │
│ Python: 3.11.9   │
│ Platform: darwin │
╰──────────────────╯
```

### `demo` - Run Demos

Run demonstration examples (coming soon).

```bash
pyfuzzy demo [NAME]
```

### Global Options

```bash
pyfuzzy --help              # Show help for all commands
pyfuzzy --install-completion  # Install shell completion
pyfuzzy --show-completion     # Show completion script
```

## Programmatic API

You can also launch the interface from Python code:

```python
from fuzzy_systems import launch_interface

# Simple launch
launch_interface()

# Custom configuration
launch_interface(
    port=8080,
    host='localhost',
    open_browser=True,
    theme='dark'
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `port` | int | None | Port number (auto-detects if None) |
| `host` | str | 'localhost' | Host address |
| `open_browser` | bool | True | Auto-open browser |
| `theme` | str | None | 'light' or 'dark' |
| `auto_find_port` | bool | True | Find free port if occupied |

**Use Cases:**

### Jupyter Notebooks

Launch the interface alongside your notebook:

```python
import fuzzy_systems as fs

# Train your model
model = fs.learning.ANFIS(n_inputs=2, n_mfs=3)
model.fit(X_train, y_train)

# Open interface for interactive analysis
fs.launch_interface()
```

### Scripts and Automation

```python
import fuzzy_systems as fs

# Run in headless mode for server deployment
fs.launch_interface(
    host='0.0.0.0',
    port=8501,
    open_browser=False
)
```

## Alternative: Python Module

If the `pyfuzzy` command is not available, use the Python module directly:

```bash
python -m fuzzy_systems.cli interface
python -m fuzzy_systems.cli version
python -m fuzzy_systems.cli --help
```

## Troubleshooting

### Command not found

**Problem:**
```bash
pyfuzzy interface
zsh: command not found: pyfuzzy
```

**Solutions:**

1. **Verify installation:**
   ```bash
   pip show pyfuzzy-toolbox
   ```

2. **Use Python module:**
   ```bash
   python -m fuzzy_systems.cli interface
   ```

3. **Check PATH:**
   ```bash
   which python
   echo $PATH
   ```

4. **Reinstall:**
   ```bash
   pip uninstall pyfuzzy-toolbox -y
   pip install 'pyfuzzy-toolbox[ui]'
   ```

### Port already in use

**Problem:**
```
⚠️ Port 8501 is occupied. Using port 8502 instead.
```

**This is normal!** pyfuzzy automatically finds a free port. If you want a specific port:

```bash
pyfuzzy interface --port 9000
```

### Browser doesn't open

**Problem:** Server starts but browser doesn't open automatically.

**Solutions:**

1. **Manual access:** Open your browser and go to `http://localhost:8501`

2. **Check headless mode:**
   ```bash
   pyfuzzy interface --browser  # Ensure browser mode is on
   ```

3. **Try different browser:** Set as default and retry

### Streamlit not found

**Problem:**
```
ModuleNotFoundError: No module named 'streamlit'
```

**Solution:** Install with UI extras:
```bash
pip install 'pyfuzzy-toolbox[ui]'
```

## Web Interface Features

Once the interface is running, you'll have access to:

### 🎯 ANFIS Module

Complete workflow for Adaptive Neuro-Fuzzy Inference Systems:

1. **Dataset Tab**
   - Load CSV files
   - Use classic datasets (Iris, Wine, Breast Cancer, Diabetes)
   - Generate synthetic data
   - Train/Validation/Test split
   - Data normalization

2. **Training Tab**
   - Configure architecture (MFs, type)
   - Hybrid Learning (backpropagation + LSE)
   - Metaheuristic optimization (PSO, DE, GA)
   - Regularization (L1/L2)
   - Real-time training curves

3. **Metrics Tab**
   - Regression: RMSE, MAE, R², MAPE
   - Classification: Accuracy, Precision, Recall, F1
   - Scatter plots (Predicted vs Actual)
   - Residual analysis
   - Confusion matrix

4. **Prediction Tab**
   - Manual input with custom values
   - Batch prediction (CSV upload)
   - Export predictions (Train/Val/Test)

5. **Model Analysis Tab**
   - Architecture overview
   - Membership functions visualization
   - Fuzzy rules (3 views: Table, Visual Matrix, Activation)
   - 3D decision surface (2-input models)
   - Feature importance (3 methods)

6. **What is ANFIS? Tab**
   - Theory and explanation
   - Architecture details
   - Use cases

### 📊 Other Modules (Coming Soon)

- Mamdani Systems
- Sugeno Systems
- Wang-Mendel Learning
- Fuzzy ODEs
- p-Fuzzy Systems

## Next Steps

- [Installation Guide](installation.md) - Detailed installation instructions
- [Quickstart Tutorial](quickstart.md) - Learn the basics
- [ANFIS User Guide](../user_guide/learning.md#anfis) - Deep dive into ANFIS
- [API Reference](../api_reference/learning.md) - Complete API documentation

## Feedback

Having issues or suggestions? Please [open an issue](https://github.com/1moi6/pyfuzzy-toolbox/issues) on GitHub!
