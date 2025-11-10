"""
pyfuzzy-toolbox Command Line Interface
=======================================
Modern CLI for pyfuzzy-toolbox with Typer and Rich

Author: Moiseis Cecconello
"""

import typer
from typing_extensions import Annotated
from rich.console import Console
from rich.panel import Panel
import sys

# Initialize Typer app
app = typer.Typer(
    name="pyfuzzy",
    help="🔮 pyfuzzy-toolbox - Fuzzy Logic & ANFIS Toolkit",
    add_completion=True,
    rich_markup_mode="rich"
)

console = Console()


@app.command()
def interface(
    port: Annotated[int, typer.Option(help="Port number for the web interface")] = 8501,
    host: Annotated[str, typer.Option(help="Host address to bind")] = "localhost",
    dark_theme: Annotated[bool, typer.Option("--dark-theme", help="Use dark theme")] = False,
    browser: Annotated[bool, typer.Option("--browser/--no-browser", help="Auto-open browser")] = True,
):
    """
    🚀 Launch the interactive web interface

    The interface provides a complete graphical environment for:
    - ANFIS training and analysis
    - Fuzzy inference systems
    - Interactive visualizations
    - Model evaluation and predictions

    [bold cyan]Examples:[/]

        [dim]# Start with default settings[/]
        pyfuzzy interface

        [dim]# Custom port[/]
        pyfuzzy interface --port 8080

        [dim]# Dark theme without auto-opening browser[/]
        pyfuzzy interface --dark-theme --no-browser
    """
    from .interface import launch_interface

    # Show startup panel
    console.print(Panel.fit(
        f"[bold cyan]🔮 pyfuzzy-toolbox Interface[/]\n"
        f"[green]→[/] Starting on [blue underline]http://{host}:{port}[/]\n"
        f"[yellow]→[/] Theme: [magenta]{'Dark' if dark_theme else 'Light'}[/]\n"
        f"[yellow]→[/] Auto-open: [magenta]{'Yes' if browser else 'No'}[/]\n\n"
        f"[dim]Press Ctrl+C to stop[/]",
        border_style="cyan",
        title="[bold]pyfuzzy Interface[/]",
        subtitle="[dim]Adaptive Neuro-Fuzzy Inference System[/]"
    ))

    try:
        launch_interface(
            port=port,
            host=host,
            open_browser=browser,
            theme='dark' if dark_theme else 'light'
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]👋 Interface stopped by user[/]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/]")
        sys.exit(1)


@app.command()
def demo(
    example: Annotated[str, typer.Argument(help="Demo example to run")] = "anfis",
):
    """
    📚 Run demonstration examples

    Available demos:
    - [cyan]anfis[/]: ANFIS training and prediction demo
    - [cyan]fuzzy[/]: Fuzzy inference system demo
    - [cyan]timeseries[/]: Time series forecasting demo

    [bold cyan]Example:[/]

        pyfuzzy demo anfis
    """
    console.print(f"[cyan]📚 Running {example} demo...[/]")

    if example == "anfis":
        console.print("[yellow]ℹ️  ANFIS demo - Coming soon![/]")
        console.print("[dim]For now, use: pyfuzzy interface[/]")
    else:
        console.print(f"[red]❌ Unknown demo: {example}[/]")
        console.print("[dim]Available: anfis, fuzzy, timeseries[/]")
        sys.exit(1)


@app.command()
def version():
    """
    📦 Show package version and system info
    """
    try:
        from . import __version__
        console.print(Panel.fit(
            f"[bold cyan]pyfuzzy-toolbox[/]\n"
            f"[green]Version:[/] [yellow]{__version__}[/]\n"
            f"[green]Python:[/] [yellow]{sys.version.split()[0]}[/]\n"
            f"[green]Platform:[/] [yellow]{sys.platform}[/]",
            border_style="cyan",
            title="[bold]Package Info[/]"
        ))
    except ImportError:
        console.print("[red]❌ Could not determine package version[/]")
        sys.exit(1)


@app.callback()
def main():
    """
    pyfuzzy-toolbox - A comprehensive toolkit for Fuzzy Logic and ANFIS

    Run [cyan]pyfuzzy --help[/] to see available commands.
    """
    pass


if __name__ == "__main__":
    app()
