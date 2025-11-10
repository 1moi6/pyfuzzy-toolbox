"""
pyfuzzy-toolbox Interface Launcher
===================================
Programmatic API for launching the Streamlit web interface

Author: Moiseis Cecconello
"""

import subprocess
import sys
import socket
from pathlib import Path
from typing import Optional, Literal


def find_free_port(start_port: int = 8501, max_attempts: int = 10) -> int:
    """
    Find an available port starting from start_port

    Parameters
    ----------
    start_port : int
        Port to start searching from (default: 8501)
    max_attempts : int
        Maximum number of ports to try (default: 10)

    Returns
    -------
    int
        Available port number

    Raises
    ------
    RuntimeError
        If no free port is found within max_attempts
    """
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"No free ports found in range {start_port}-{start_port + max_attempts}")


def launch_interface(
    port: Optional[int] = None,
    host: str = 'localhost',
    open_browser: bool = True,
    theme: Optional[Literal['light', 'dark']] = None,
    auto_find_port: bool = True
) -> None:
    """
    Launch pyfuzzy-toolbox web interface

    This function starts the Streamlit-based web interface for interactive
    fuzzy logic and ANFIS modeling. The interface provides:

    - Dataset management (CSV, classic datasets, synthetic data)
    - ANFIS training (Hybrid Learning & Metaheuristics)
    - Model evaluation and metrics
    - Interactive predictions
    - Model analysis and visualization

    Parameters
    ----------
    port : int, optional
        Port number to run the interface. If None and auto_find_port=True,
        will automatically find a free port starting from 8501.
        Default: None (auto-find)
    host : str
        Host address to bind. Default: 'localhost'
    open_browser : bool
        Automatically open web browser. Default: True
    theme : {'light', 'dark'}, optional
        Interface theme. Default: None (Streamlit default)
    auto_find_port : bool
        If True, automatically find a free port if the specified port
        is occupied. Default: True

    Returns
    -------
    None
        The function blocks until the server is stopped (Ctrl+C)

    Raises
    ------
    FileNotFoundError
        If the Streamlit app file cannot be found
    RuntimeError
        If no free port is available (when auto_find_port=True)
    subprocess.SubprocessError
        If Streamlit fails to start

    Examples
    --------
    >>> from fuzzy_systems import launch_interface

    # Simple launch with defaults
    >>> launch_interface()

    # Custom port and dark theme
    >>> launch_interface(port=8080, theme='dark')

    # Launch without opening browser (headless)
    >>> launch_interface(open_browser=False)

    # Use in Jupyter notebooks
    >>> launch_interface(port=8888)  # Different port to avoid conflicts

    Notes
    -----
    - Press Ctrl+C to stop the interface
    - The interface will be available at http://{host}:{port}
    - All Streamlit features are available (session state, caching, etc.)
    """
    # Determine port
    if port is None:
        port = 8501

    if auto_find_port:
        try:
            # Try to bind to the requested port
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((host, port))
        except OSError:
            # Port is occupied, find a free one
            original_port = port
            port = find_free_port(start_port=port)
            print(f"⚠️  Port {original_port} is occupied. Using port {port} instead.")

    # Locate the Streamlit app
    app_path = Path(__file__).parent / 'streamlit_app' / 'main.py'

    if not app_path.exists():
        raise FileNotFoundError(
            f"Streamlit app not found at {app_path}\n"
            f"Make sure pyfuzzy-toolbox is properly installed."
        )

    # Build Streamlit command
    cmd = [
        sys.executable, '-m', 'streamlit', 'run',
        str(app_path),
        f'--server.port={port}',
        f'--server.address={host}'
    ]

    # Add theme if specified
    if theme:
        cmd.append(f'--theme.base={theme}')

    # Add headless mode if browser should not open
    if not open_browser:
        cmd.append('--server.headless=true')

    # Print startup message
    print(f"🚀 Starting pyfuzzy interface on http://{host}:{port}")
    if not open_browser:
        print(f"   Open the URL above in your browser to access the interface")
    print(f"   Press Ctrl+C to stop")
    print()

    # Launch Streamlit
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n👋 Interface stopped")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to start Streamlit interface: {e}")


# Convenience function for quick testing
def start(port: int = 8501, **kwargs):
    """
    Convenience function to quickly start the interface

    Equivalent to launch_interface() with shorter name.

    Parameters
    ----------
    port : int
        Port number (default: 8501)
    **kwargs
        Additional arguments passed to launch_interface()

    Examples
    --------
    >>> from fuzzy_systems.interface import start
    >>> start(8080, theme='dark')
    """
    launch_interface(port=port, **kwargs)


if __name__ == '__main__':
    # Allow running as: python -m fuzzy_systems.interface
    launch_interface()
