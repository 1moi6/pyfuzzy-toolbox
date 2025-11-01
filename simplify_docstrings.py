"""Script to simplify verbose docstrings in pfuzzy.py"""

filepath = 'fuzzy_systems/dynamics/pfuzzy.py'

# Mapeamento de substituições: (old_docstring_start, new_concise_docstring)
replacements = {
    # plot_trajectory
    '''    def plot_trajectory(self, variables=None, **kwargs):
        """
        Plot the temporal trajectory of state variables.

        Creates a visual representation of how state variables evolve over time
        during simulation. Each variable is plotted as a separate line with
        optional markers. Useful for analyzing system behavior and dynamics.

        Parameters:
        -----------
        variables : List[str], str, or None, default=None
            Variables to plot:
            - None: Plot all state variables (default behavior)
            - str: Plot single variable (e.g., 'x')
            - List[str]: Plot multiple specific variables (e.g., ['x', 'y'])

            Invalid variable names are skipped with a warning.

            Example:
                >>> system.plot_trajectory()  # All variables
                >>> system.plot_trajectory('x')  # Single variable
                >>> system.plot_trajectory(['x', 'y'])  # Multiple variables

        **kwargs : dict
            Optional plotting customization parameters:

            - figsize : tuple, default=(10, 6)
            Figure dimensions in inches (width, height)
            Example: figsize=(12, 8)

            - title : str, default='p-Fuzzy System Trajectory'
            Plot title
            Example: title='System Response Over Time'

            - xlabel : str, default='Time'
            X-axis label
            Example: xlabel='Time (seconds)'

            - ylabel : str, default='State'
            Y-axis label
            Example: ylabel='State Value'

            - linestyle : str, default='-'
            Line style: '-', '--', '-.', ':'

            - linewidth : float, default=2
            Line thickness

            - marker : str, default='o'
            Marker style: 'o', 's', '^', '*', '+', etc.

            - markersize : int, default=3
            Marker size in points

            - grid : bool, default=True
            Show/hide grid

            - legend : bool, default=True
            Show/hide legend

        Returns:
        --------
        tuple
            (fig, ax) : Matplotlib Figure and Axes objects
            Allows further customization after plotting

            Example:
                >>> fig, ax = system.plot_trajectory()
                >>> ax.set_ylim(-10, 10)  # Further customize
                >>> fig.savefig('trajectory.png')

        Raises:
        -------
        RuntimeError
            If simulate() has not been called yet (no trajectory data available)
        ImportError
            If Matplotlib is not installed

        Notes:
        ------
        - Requires simulate() to be called first
        - self.time and self.trajectory must be populated
        - Time vector and trajectory must have matching lengths
        - Variables are plotted with different colors (automatic)
        - Markers appear at each time step (useful for discrete systems)

        See Also:
        ---------
        - simulate() : Generate trajectory data
        - plot_phase_portrait() : 2D state-space visualization

        Example:
        --------
        >>> import fuzzy_systems as fs
        >>>
        >>> # Create and simulate system
        >>> pfuzzy = fs.dynamic.PFuzzyContinuous(fis, mode='absolute')
        >>> t, traj = pfuzzy.simulate(x0={'x': 1.0}, t_span=(0, 10), dt=0.01)
        >>>
        >>> # Plot all variables
        >>> fig, ax = pfuzzy.plot_trajectory()
        >>>
        >>> # Plot specific variables with customization
        >>> fig, ax = pfuzzy.plot_trajectory(
        ...     variables=['x', 'y'],
        ...     figsize=(14, 6),
        ...     title='State Evolution',
        ...     xlabel='Time (s)',
        ...     ylabel='Position (m)',
        ...     linewidth=2.5,
        ...     markersize=4
        ... )
        >>> fig.savefig('my_trajectory.png', dpi=300)
        """''':
    '''    def plot_trajectory(self, variables=None, **kwargs):
        """
        Plot state variables over time.

        Args:
            variables: Variables to plot (None=all, str=single, list=multiple)
            **kwargs: figsize, title, xlabel, ylabel, linestyle, linewidth, marker, markersize, grid, legend

        Returns:
            (fig, ax): Matplotlib Figure and Axes
        """''',
}

print("Starting docstring simplification...")

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

lines_before = content.count('\n') + 1

# Apply replacements
for old, new in replacements.items():
    if old in content:
        content = content.replace(old, new)
        print(f"✓ Replaced docstring ({old[:50]}...)")
    else:
        print(f"✗ Pattern not found ({old[:50]}...)")

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

lines_after = content.count('\n') + 1
lines_saved = lines_before - lines_after

print(f"\nDone!")
print(f"Lines before: {lines_before}")
print(f"Lines after: {lines_after}")
print(f"Lines saved: {lines_saved} ({lines_saved/lines_before*100:.1f}%)")
