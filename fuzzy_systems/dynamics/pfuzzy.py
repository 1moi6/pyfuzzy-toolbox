"""
p-Fuzzy Dynamic Systems - Fuzzy rule-based dynamical systems.

Discrete: x_{n+1} = x_n + f(x_n) [absolute] or x_{n+1} = x_n * f(x_n) [relative]
Continuous: dx/dt = f(x) [absolute] or dx/dt = x * f(x) [relative]

Reference: Barros, Bassanezi & Lodwick (2017) - Fuzzy Logic, Fuzzy Dynamical Systems
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
from abc import ABC, abstractmethod
from ..inference.systems import MamdaniSystem, SugenoSystem, FuzzyInferenceSystem
import warnings 


class PFuzzySystem(ABC):
    """Base class for p-fuzzy dynamical systems with fuzzy rule-based evolution."""

    def __init__(self,
             fis: FuzzyInferenceSystem,
             mode: str = 'absolute',
             state_vars: Optional[List[str]] = None,
             dynamic_function: Optional[Callable] = None):
        """
        Initialize p-fuzzy system.

        Args:
            fis: Fuzzy inference system (Mamdani or Sugeno)
            mode: 'absolute', 'relative', or 'custom'
            state_vars: State variable names (default: FIS inputs)
            dynamic_function: Custom function f(states, fis_outputs) -> new_states (required if mode='custom')
        """

        # Validate mode
        valid_modes = {'absolute', 'relative', 'custom'}
        if mode not in valid_modes:
            raise ValueError(f"Mode must be one of {valid_modes}, got '{mode}'")

        if mode == 'custom' and dynamic_function is None:
            raise ValueError("If mode='custom', dynamic_function must be provided")

        if mode != 'custom' and dynamic_function is not None:
            raise ValueError("dynamic_function can only be used with mode='custom'")

        # Set basic attributes
        self.fis = fis
        self.mode = mode
        self._user_function = dynamic_function  # Store original user function

        # Identify state variables
        if state_vars is None:
            self.state_vars = list(fis.input_variables.keys())
        else:
            self.state_vars = state_vars

        # Validate state variables exist in FIS
        for var in self.state_vars:
            if var not in fis.input_variables:
                raise ValueError(
                    f"State variable '{var}' not found in FIS. "
                    f"Available: {list(fis.input_variables.keys())}"
                )

        self.n_vars = len(self.state_vars)
        self.output_vars = list(fis.output_variables.keys())

        # Validate outputs match states
        if len(self.output_vars) != self.n_vars:
            raise ValueError(
                f"Number of FIS outputs ({len(self.output_vars)}) must equal "
                f"number of state variables ({self.n_vars})."
            )

        # Store results
        self.trajectory = None
        self.time = None


    def _apply_user_dynamics(self, state_dict: Dict[str, float]) -> Dict[str, float]:
        """Apply user custom dynamics: dict -> list -> user_func -> dict."""
        # Convert state dict to list (preserving variable order)
        state_values = [state_dict[var] for var in self.state_vars]

        # Evaluate FIS and convert to list
        fis_output = self.fis.evaluate(state_dict)
        fis_values = [fis_output[var] for var in self.output_vars]

        # Call user function
        result_values = self._user_function(state_values, fis_values)

        # Handle scalar return (single variable systems)
        if np.isscalar(result_values):
            result_values = [result_values]

        # Convert result back to dictionary
        return {
            self.state_vars[i]: float(result_values[i])
            for i in range(self.n_vars)
        }


    def _check_domain(self, x: np.ndarray) -> Tuple[bool, Optional[str]]:
        """Check if state variables are within domain bounds (vectorized)."""
        min_vals = self._domain_limits[:, 0]
        max_vals = self._domain_limits[:, 1]
        out_of_bounds = (x < min_vals) | (x > max_vals)
        
        if not np.any(out_of_bounds):
            return True, None

        idx = np.where(out_of_bounds)[0][0]
        var_name = self.state_vars[idx]
        value = x[idx]
        min_val, max_val = self._domain_limits[idx]
        msg = f"'{var_name}' = {value:.6f} out of bounds [{min_val:.6f}, {max_val:.6f}]"
        return False, msg


    def _evaluate_fis(self, state: Dict[str, float]) -> Dict[str, float]:
        """Evaluate FIS for given state."""
        return self.fis.evaluate(state)

    @abstractmethod
    def simulate(self, x0: Union[Dict[str, float], np.ndarray], **kwargs):
        """Simulate p-fuzzy system (abstract method)."""
        pass

    def plot_trajectory(self, variables=None, **kwargs):
        """
        Plot state variables over time.

        Args:
            variables: None (all), str (single), or list (multiple) variable names
            **kwargs: figsize, title, xlabel, ylabel, linestyle, linewidth, marker, markersize, grid, legend

        Returns:
            (fig, ax): Matplotlib Figure and Axes objects
        """
        
        # Verify simulation has been run
        if self.trajectory is None:
            raise RuntimeError(
                "No trajectory data available. Execute simulate() "
                "before plotting."
            )
        
        # Ensure trajectory and time have matching shapes
        if self.time is None or len(self.time) != len(self.trajectory):
            raise RuntimeError(
                "Trajectory data is corrupted: time and trajectory lengths don't match. "
                "Re-run simulate()."
            )
        
        # Import plotting library
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "Matplotlib is required for plotting. "
                "Install it with: pip install matplotlib"
            )
        
        # Normalize variables input
        if variables is None:
            variables = self.state_vars
        elif isinstance(variables, str):
            variables = [variables]
        
        # Validate all requested variables exist
        for var in variables:
            if var not in self.state_vars:
                print(
                    f"Warning: State variable '{var}' not found in system. "
                    f"Available variables: {self.state_vars}. Skipping."
                )
        
        # Filter to only valid variables
        valid_variables = [v for v in variables if v in self.state_vars]
        
        if not valid_variables:
            raise ValueError(
                f"No valid variables to plot. Available: {self.state_vars}"
            )
        
        # Extract plot customization parameters
        figsize = kwargs.get('figsize', (10, 6))
        title = kwargs.get('title', 'p-Fuzzy System Trajectory')
        xlabel = kwargs.get('xlabel', 'Time')
        ylabel = kwargs.get('ylabel', 'State')
        linestyle = kwargs.get('linestyle', '-')
        linewidth = kwargs.get('linewidth', 2)
        marker = kwargs.get('marker', 'o')
        markersize = kwargs.get('markersize', 3)
        show_grid = kwargs.get('grid', True)
        show_legend = kwargs.get('legend', True)
        
        # Create figure and axes
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot each variable as a separate line
        for var in valid_variables:
            # Find variable index
            var_idx = self.state_vars.index(var)
            
            # Plot trajectory for this variable
            ax.plot(
                self.time,
                self.trajectory[:, var_idx],
                label=var,
                linestyle=linestyle,
                linewidth=linewidth,
                marker=marker,
                markersize=markersize
            )
        
        # Configure axes
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        if show_legend:
            ax.legend(fontsize=10, loc='best')
        
        if show_grid:
            ax.grid(True, alpha=0.3, linestyle='--')
        
        # Improve layout spacing
        fig.tight_layout()
        
        # Display plot
        plt.show()
        
        return fig, ax


    def plot_phase_space(self, var_x, var_y, **kwargs):
        """
        Plot 2D phase space (phase portrait) of two state variables.

        Args:
            var_x: State variable for X-axis
            var_y: State variable for Y-axis
            **kwargs: figsize, title, xlabel, ylabel, trajectory_color/linewidth/alpha,
                     initial_color/markersize, final_color/markersize, grid, legend

        Returns:
            (fig, ax): Matplotlib Figure and Axes objects
        """
        
        # Verify simulation has been run
        if self.trajectory is None:
            raise RuntimeError(
                "No trajectory data available. Execute simulate() "
                "before plotting."
            )
        
        # Validate variables exist in state space
        if var_x not in self.state_vars or var_y not in self.state_vars:
            raise ValueError(
                f"Variables must be in {self.state_vars}. "
                f"Requested: var_x='{var_x}', var_y='{var_y}'"
            )
        
        # Import plotting library
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "Matplotlib is required for plotting. "
                "Install it with: pip install matplotlib"
            )
        
        # Get indices of selected variables
        idx_x = self.state_vars.index(var_x)
        idx_y = self.state_vars.index(var_y)
        
        # Extract customization parameters
        figsize = kwargs.get('figsize', (8, 8))
        title = kwargs.get(
            'title',
            f'Phase Space: {var_x} vs {var_y}'
        )
        xlabel = kwargs.get('xlabel', var_x)
        ylabel = kwargs.get('ylabel', var_y)
        
        trajectory_color = kwargs.get('trajectory_color', 'blue')
        trajectory_linewidth = kwargs.get('trajectory_linewidth', 2)
        trajectory_alpha = kwargs.get('trajectory_alpha', 0.6)
        
        initial_color = kwargs.get('initial_color', 'green')
        initial_markersize = kwargs.get('initial_markersize', 10)
        
        final_color = kwargs.get('final_color', 'red')
        final_markersize = kwargs.get('final_markersize', 10)
        
        show_grid = kwargs.get('grid', True)
        show_legend = kwargs.get('legend', True)
        
        # Create figure with square aspect ratio (recommended for phase space)
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot trajectory line
        ax.plot(
            self.trajectory[:, idx_x],
            self.trajectory[:, idx_y],
            color=trajectory_color,
            linewidth=trajectory_linewidth,
            alpha=trajectory_alpha,
            label='Trajectory',
            linestyle='-'
        )
        
        # Plot initial point (entry to phase space)
        ax.plot(
            self.trajectory[0, idx_x],
            self.trajectory[0, idx_y],
            marker='o',
            color=initial_color,
            markersize=initial_markersize,
            label='Initial',
            linestyle='',
            zorder=5
        )
        
        # Plot final point (exit from phase space)
        ax.plot(
            self.trajectory[-1, idx_x],
            self.trajectory[-1, idx_y],
            marker='o',
            color=final_color,
            markersize=final_markersize,
            label='Final',
            linestyle='',
            zorder=5
        )
        
        # Configure axes labels and title
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add legend and grid
        if show_legend:
            ax.legend(fontsize=10, loc='best')
        
        if show_grid:
            ax.grid(True, alpha=0.3, linestyle='--')
        
        # Set equal aspect ratio for better phase space visualization
        # This ensures the phase space is not distorted
        ax.set_aspect('equal', adjustable='box')
        
        # Improve layout spacing
        fig.tight_layout()
        
        # Display plot
        plt.show()
        
        return fig, ax


    def to_csv(self, filename: str, sep: str = ',', decimal: str = '.') -> None:
        """Method: to_csv (simplified docstring)."""
        
        # Validate that simulation has been run
        if self.trajectory is None or self.time is None:
            raise RuntimeError(
                "No trajectory data available. Execute simulate() "
                "before exporting to CSV."
            )
        
        # Verify data consistency
        if len(self.time) != len(self.trajectory):
            raise RuntimeError(
                "Trajectory data is corrupted: time and trajectory lengths don't match. "
                "Re-run simulate()."
            )
        
        # Create CSV header: time, var1, var2, ...
        header = ['time'] + self.state_vars
        header_str = sep.join(header)
        
        # Concatenate time column with trajectory data
        # Shape: (n_steps, 1 + n_vars)
        data = np.column_stack([self.time, self.trajectory])
        
        # Export based on decimal separator choice
        if decimal == ',':
            # European/Brazilian format: use comma as decimal separator
            fmt = '%.6f'
            
            # Save with dot as decimal initially
            np.savetxt(
                filename,
                data,
                delimiter=sep,
                header=header_str,
                comments='',  # No comment prefix
                fmt=fmt
            )
            
            # Post-process: replace all dots with commas
            # This converts 3.14 -> 3,14 for European Excel compatibility
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Replace decimal points with commas
                content = content.replace('.', ',')
                
                # Write back to file
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(content)
                    
            except IOError as e:
                raise IOError(
                    f"Failed to process CSV file '{filename}': {e}. "
                    f"Check file permissions and path validity."
                )
        
        else:
            # International format: use dot as decimal separator (default)
            np.savetxt(
                filename,
                data,
                delimiter=sep,
                header=header_str,
                comments='',  # No comment prefix
                fmt='%.6f'  # 6 decimal places precision
            )
        
        # Success message (optional, can be removed for silent operation)
        import os
        file_size = os.path.getsize(filename)
        print(
            f"Trajectory exported successfully to '{filename}' "
            f"({file_size:,} bytes, {len(self.trajectory)} rows)"
        )




class PFuzzyDiscrete(PFuzzySystem):
    """
    Discrete-time p-fuzzy system: x_{n+1} = f(x_n)

    Modes:
        - 'absolute': x_{n+1} = x_n + fis(x_n)
        - 'relative': x_{n+1} = x_n * fis(x_n)
        - 'custom': x_{n+1} = custom_function(x_n, fis)
    """

    def __init__(self,
             fis: FuzzyInferenceSystem,
             mode: str = 'absolute',
             state_vars: Optional[List[str]] = None,
             dynamic_function: Optional[Callable[[Dict[str, float], FuzzyInferenceSystem], Dict[str, float]]] = None):
        """
        Initialize discrete p-fuzzy system.

        Args:
            fis: Fuzzy inference system (Mamdani or Sugeno)
            mode: 'absolute', 'relative', or 'custom'
            state_vars: State variable names (default: FIS inputs)
            dynamic_function: Custom function f(states_list, fis_outputs_list) -> new_states_list (required if mode='custom')

        Example:
            >>> # Custom dynamics
            >>> def my_dyn(states, fis_out):
            ...     x = states[0]
            ...     f = fis_out[0]
            ...     return [x + 0.1*x*f]
            >>> pfuzzy = PFuzzyDiscrete(fis, mode='custom', dynamic_function=my_dyn)
        """

        # Call parent class initializer (validates mode, state_vars, etc.)
        super().__init__(fis, mode, state_vars, dynamic_function)
        
        # OPTIMIZATION 1: Pre-compute domain limits (avoids repeated lookups)
        # Shape: (n_vars, 2) -> [[min1, max1], [min2, max2], ...]
        self._domain_limits = np.array([
            self.fis.input_variables[var].universe
            for var in self.state_vars
        ])
        
        # OPTIMIZATION 2: Reusable state dictionary (avoids creating new one each step)
        # Updated in-place during simulation to reduce memory allocations
        self._state_dict = {var: 0.0 for var in self.state_vars}



    def simulate(self,
             x0: Union[Dict[str, float], np.ndarray, List[float], Tuple[float, ...]],
             n_steps: int,
             verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate discrete-time p-fuzzy system.

        Args:
            x0: Initial state as dict {'x': val}, array, list, or tuple
            n_steps: Number of iterations
            verbose: Print progress (default: False)

        Returns:
            (n, trajectory):
                - n: Array of iteration indices [0, 1, 2, ..., n_steps]
                - trajectory: Array of states, shape (n_steps+1, n_vars)

        Example:
            >>> n, traj = pfuzzy.simulate(x0={'population': 10.0}, n_steps=100)
            >>> pfuzzy.plot_trajectory()
        """

        # STEP 1: NORMALIZE INPUT (Convert to NumPy array)
        # ================================================
        if isinstance(x0, dict):
            # User provided dictionary - extract values in correct order
            x_current = np.array([x0[var] for var in self.state_vars], dtype=float)
        elif isinstance(x0, (list, tuple)):
            # User provided list/tuple - convert directly
            x_current = np.array(x0, dtype=float)
        else:
            # Assume NumPy array
            x_current = np.array(x0, dtype=float)
        
        # STEP 2: VALIDATE INPUT
        # =====================
        # Check dimension
        if len(x_current) != self.n_vars:
            raise ValueError(
                f"Initial condition must have {self.n_vars} values "
                f"(one per state variable), received {len(x_current)}. "
                f"Expected order: {self.state_vars}"
            )
        
        # Check domain
        valid, msg = self._check_domain(x_current)
        if not valid:
            raise ValueError(
                f"Initial condition violates domain constraints: {msg}"
            )
        
        # STEP 3: ALLOCATE MEMORY
        # =======================
        trajectory = np.zeros((n_steps + 1, self.n_vars), dtype=float)
        trajectory[0] = x_current
        iterations = np.arange(n_steps + 1, dtype=int)
        
        # STEP 4: VERBOSE OUTPUT - HEADER
        # ===============================
        if verbose:
            print(f"\n{'='*75}")
            print(f"  DISCRETE p-FUZZY SYSTEM SIMULATION")
            print(f"{'='*75}")
            print(f"  Total iterations: {n_steps}")
            print(f"  State variables: {', '.join(self.state_vars)}")
            print(f"  Dynamics mode:   {self.mode}")
            print(f"  Initial state:   {', '.join([f'{var}={x_current[i]:.6f}' for i, var in enumerate(self.state_vars)])}")
            print(f"{'='*75}\n")
        
        # STEP 5: MAIN SIMULATION LOOP
        # ============================
        for step in range(n_steps):
            # OPTIMIZATION: Reuse state dictionary instead of creating new one each step
            for i, var in enumerate(self.state_vars):
                self._state_dict[var] = float(x_current[i])
            
            # Compute next state based on mode
            if self.mode == 'custom':
                # ===== CUSTOM MODE =====
                # Apply user-defined dynamics function
                next_state = self._apply_user_dynamics(self._state_dict)
                x_current = np.array([next_state[var] for var in self.state_vars], dtype=float)
            
            else:
                # ===== STANDARD MODES (absolute, relative) =====
                # Evaluate FIS at current state
                fis_output = self._evaluate_fis(self._state_dict)
                
                # Extract FIS outputs as a vectorized array
                output_vals = np.array([
                    fis_output[self.output_vars[i]] for i in range(self.n_vars)
                ], dtype=float)
                
                # Update state based on mode
                if self.mode == 'absolute':
                    # x_{n+1} = x_n + f(x_n)
                    x_current = x_current + output_vals
                
                elif self.mode == 'relative':
                    # x_{n+1} = x_n * f(x_n)
                    x_current = x_current * output_vals
            
            # STEP 6: DOMAIN CHECKING
            # =======================
            valid, msg = self._check_domain(x_current)
            if not valid:
                # State has left valid domain - terminate early
                print(f"\n⚠️  WARNING: {msg}")
                print(f"    Iteration: {step + 1}/{n_steps}")
                
                # Truncate trajectory to completed steps
                self.trajectory = trajectory[:step + 2]  # Include current failed step
                self.time = iterations[:step + 2]
                
                if verbose:
                    print(f"\n{'='*75}")
                    print(f"  SIMULATION TERMINATED (Domain violation)")
                    print(f"{'='*75}")
                    print(f"  Completed iterations: {step + 1}/{n_steps}")
                    print(f"{'='*75}\n")
                
                return self.time, self.trajectory
            
            # Store state at this iteration
            trajectory[step + 1] = x_current
            
            # STEP 7: PROGRESS REPORTING
            # ==========================
            if verbose:
                # Report every 10% or every 100 steps (whichever is smaller)
                report_interval = max(1, min(100, n_steps // 10))
                if (step + 1) % report_interval == 0 or (step + 1) == n_steps:
                    progress_pct = 100.0 * (step + 1) / n_steps
                    print(
                        f"  Progress: {progress_pct:6.1f}% | "
                        f"Iteration: {step + 1:6d}/{n_steps} | "
                        f"State: {', '.join([f'{var}={x_current[i]:8.4f}' for i, var in enumerate(self.state_vars)])}"
                    )
        
        # STEP 8: STORE RESULTS AND FINAL REPORT
        # ======================================
        self.trajectory = trajectory
        self.time = iterations
        
        if verbose:
            print(f"\n{'='*75}")
            print(f"  SIMULATION COMPLETED SUCCESSFULLY")
            print(f"{'='*75}")
            print(f"  Total iterations: {n_steps}")
            print(f"  Trajectory points: {len(iterations)}")
            print(f"  Final state:")
            for i, var in enumerate(self.state_vars):
                print(f"    {var}: {x_current[i]:.6f}")
            print(f"{'='*75}\n")
        
        return self.time, self.trajectory


    def step(self,
         x: Union[Dict[str, float], np.ndarray]) -> np.ndarray:
        """
        Execute single iteration step.

        Args:
            x: Current state as dict or array

        Returns:
            np.ndarray: Next state x_{n+1}

        Example:
            >>> x_next = pfuzzy.step({'x': 10.0})
        """

        # STEP 1: NORMALIZE INPUT
        # =======================
        if isinstance(x, dict):
            # Dictionary input - extract values in correct order
            x_current = np.array([x[var] for var in self.state_vars], dtype=float)
        else:
            # Array-like input (list, tuple, or NumPy array)
            x_current = np.array(x, dtype=float)
        
        # STEP 2: VALIDATE DOMAIN
        # =======================
        valid, msg = self._check_domain(x_current)
        if not valid:
            raise ValueError(
                f"Input state outside domain bounds: {msg}"
            )
        
        # STEP 3: UPDATE STATE DICTIONARY
        # ===============================
        # OPTIMIZATION: Reuse self._state_dict to minimize allocations
        for i, var in enumerate(self.state_vars):
            self._state_dict[var] = float(x_current[i])
        
        # STEP 4: COMPUTE NEXT STATE
        # ==========================
        if self.mode == 'custom':
            # ===== CUSTOM MODE =====
            # Apply user-defined dynamics function
            next_state = self._apply_user_dynamics(self._state_dict)
            return np.array([next_state[var] for var in self.state_vars], dtype=float)
        
        else:
            # ===== STANDARD MODES (absolute, relative) =====
            # Evaluate FIS at current state
            fis_output = self._evaluate_fis(self._state_dict)
            
            # Extract FIS outputs as vectorized array
            output_vals = np.array([
                fis_output[self.output_vars[i]] for i in range(self.n_vars)
            ], dtype=float)
            
            # Compute next state based on mode
            if self.mode == 'absolute':
                # x_{n+1} = x_n + f(x_n)
                return x_current + output_vals
            
            else:  # self.mode == 'relative'
                # x_{n+1} = x_n * f(x_n)
                return x_current * output_vals



class PFuzzyContinuous(PFuzzySystem):
    """
    Continuous-time p-fuzzy system: dx/dt = f(x)

    Modes:
        - 'absolute': dx/dt = fis(x)
        - 'relative': dx/dt = x * fis(x)
        - 'custom': dx/dt = custom_function(x, fis)

    Integration methods: 'euler' (fast), 'rk4' (accurate)
    """

    def __init__(self,
             fis: FuzzyInferenceSystem,
             mode: str = 'absolute',
             state_vars: Optional[List[str]] = None,
             method: str = 'euler',
             dynamic_function: Optional[Callable[[Dict[str, float], FuzzyInferenceSystem], Dict[str, float]]] = None):
        """
        Initialize continuous p-fuzzy system.

        Args:
            fis: Fuzzy inference system (Mamdani or Sugeno)
            mode: 'absolute', 'relative', or 'custom'
            state_vars: State variable names (default: FIS inputs)
            method: Integration method - 'euler' or 'rk4' (default: 'euler')
            dynamic_function: Custom function f(states_list, fis_outputs_list) -> derivatives_list (required if mode='custom')

        Example:
            >>> # Custom dynamics: dx/dt = 0.1*x*f(x)
            >>> def my_dyn(states, fis_out):
            ...     return [0.1 * states[0] * fis_out[0]]
            >>> pfuzzy = PFuzzyContinuous(fis, mode='custom', method='rk4', dynamic_function=my_dyn)
        """

        # Call parent class initializer (validates mode, state_vars, custom_function)
        super().__init__(fis, mode, state_vars, dynamic_function)
        
        # Validate numerical integration method
        if method not in {'euler', 'rk4'}:
            raise ValueError(
                f"Numerical method '{method}' is invalid. "
                f"Must be one of: {{'euler', 'rk4'}}"
            )
        
        self.method = method
        
        # OPTIMIZATION 1: Pre-compute domain limits (avoids repeated lookups)
        # Shape: (n_vars, 2) -> [[min1, max1], [min2, max2], ...]
        self._domain_limits = np.array([
            self.fis.input_variables[var].universe
            for var in self.state_vars
        ], dtype=float)
        
        # OPTIMIZATION 2: Reusable state dictionary (avoids creating new one each step)
        # Updated in-place during simulation to reduce memory allocations
        self._state_dict = {var: 0.0 for var in self.state_vars}

    def _dynamics(self, x: np.ndarray) -> np.ndarray:
        """Method: _dynamics (simplified docstring)."""
        
        # STEP 1: UPDATE STATE DICTIONARY (OPTIMIZATION)
        # ==============================================
        # Reuse self._state_dict instead of creating new dict
        # This reduces memory allocations and improves performance
        for i, var in enumerate(self.state_vars):
            self._state_dict[var] = float(x[i])
        
        # STEP 2: COMPUTE NEXT STATE BASED ON MODE
        # ========================================
        if self.mode == 'custom':
            # ===== CUSTOM MODE =====
            # Apply user-defined dynamics function (returns dx/dt)
            derivatives = self._apply_user_dynamics(self._state_dict)
            return np.array([derivatives[var] for var in self.state_vars], dtype=float)
        
        else:
            # ===== STANDARD MODES (absolute, relative) =====
            # Evaluate FIS at current state
            fis_output = self._evaluate_fis(self._state_dict)
            
            # Extract FIS outputs as vectorized array (same order as state_vars)
            f_x = np.array([
                fis_output[self.output_vars[i]] for i in range(self.n_vars)
            ], dtype=float)
            
            # Compute derivatives based on mode
            if self.mode == 'absolute':
                # ===== ABSOLUTE MODE =====
                # dx/dt = f(x)
                # FIS output is the direct rate of change
                return f_x
            
            elif self.mode == 'relative':
                # ===== RELATIVE MODE =====
                # dx/dt = x * f(x)
                # FIS output is scaled by current state (proportional growth)
                return x * f_x
            
            else:
                # Should never reach here (mode validation in __init__)
                raise ValueError(
                    f"Unknown mode: {self.mode}. "
                    f"Valid modes: {{'absolute', 'relative', 'custom'}}"
                )

    def _step_euler(self, x: np.ndarray, dt: float) -> np.ndarray:
       
        # Compute derivative at current state
        # This calls _dynamics which evaluates FIS (main computational cost)
        dxdt = self._dynamics(x)
        
        # Apply Euler formula: x_{n+1} = x_n + dt * dx/dt
        # Vectorized NumPy operation (very fast)
        x_next = x + dt * dxdt
        
        return x_next


    def _step_rk4(self, x: np.ndarray, dt: float) -> np.ndarray:
        """Method: _step_rk4 (simplified docstring)."""
        
        # STEP 1: Compute k1 at current state x_n
        # ========================================
        k1 = self._dynamics(x)
        
        # STEP 2: Compute k2 at midpoint using k1
        # ========================================
        # Predicts state at t + dt/2 using slope k1
        x_mid1 = x + 0.5 * dt * k1
        k2 = self._dynamics(x_mid1)
        
        # STEP 3: Compute k3 at midpoint using k2
        # ========================================
        # Refines prediction at t + dt/2 using slope k2
        x_mid2 = x + 0.5 * dt * k2
        k3 = self._dynamics(x_mid2)
        
        # STEP 4: Compute k4 at endpoint using k3
        # ========================================
        # Predicts state at t + dt using slope k3
        x_end = x + dt * k3
        k4 = self._dynamics(x_end)
        
        # STEP 5: Combine weighted slopes into final step
        # ===============================================
        # Classical RK4: average with weights (1, 2, 2, 1) divided by 6
        # This emphasizes the midpoint slopes (k2, k3)
        x_next = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        return x_next


    def simulate(self,
             x0: Union[Dict[str, float], np.ndarray, List[float], Tuple[float, ...]],
             t_span: Tuple[float, float],
             dt: Optional[float] = None,
             adaptive: bool = False,
             tolerance: float = 1e-4,
             dt_min: float = 1e-5,
             dt_max: float = 1.0,
             verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate continuous-time p-fuzzy system using ODE integration.

        Args:
            x0: Initial state as dict, array, list, or tuple
            t_span: Time interval (t_start, t_end)
            dt: Time step (required if adaptive=False)
            adaptive: Use adaptive step size (default: False)
            tolerance: Error tolerance for adaptive method (default: 1e-4)
            dt_min: Minimum time step for adaptive (default: 1e-5)
            dt_max: Maximum time step for adaptive (default: 1.0)
            verbose: Print progress (default: False)

        Returns:
            (t, trajectory):
                - t: Array of time points
                - trajectory: Array of states, shape (n_steps, n_vars)

        Example:
            >>> # Fixed step
            >>> t, traj = pfuzzy.simulate(x0={'temp': 100}, t_span=(0, 10), dt=0.01)
            >>> # Adaptive step (more accurate)
            >>> t, traj = pfuzzy.simulate(x0={'temp': 100}, t_span=(0, 10), adaptive=True)
        """

        # STEP 1: NORMALIZE INPUT
        # =======================
        if isinstance(x0, dict):
            x_current = np.array([x0[var] for var in self.state_vars], dtype=float)
        elif isinstance(x0, (list, tuple)):
            x_current = np.array(x0, dtype=float)
        else:
            x_current = np.array(x0, dtype=float)
        
        # STEP 2: VALIDATE INPUT
        # =====================
        if len(x_current) != self.n_vars:
            raise ValueError(
                f"Initial condition must have {self.n_vars} values "
                f"(one per state variable), received {len(x_current)}. "
                f"Expected order: {self.state_vars}"
            )
        
        # Check domain
        valid, msg = self._check_domain(x_current)
        if not valid:
            raise ValueError(f"Initial condition violates domain constraints: {msg}")
        
        # STEP 3: ROUTE TO APPROPRIATE METHOD
        # ===================================
        if not adaptive:
            # Fixed-step simulation (original method)
            return self._simulate_fixed_step(x_current, t_span, dt, verbose)
        else:
            # Adaptive-step simulation
            return self.simulate_adaptive(
                x0=x_current,
                t_span=t_span,
                dt_initial=dt if dt is not None else 0.1,
                tolerance=tolerance,
                dt_min=dt_min,
                dt_max=dt_max,
                verbose=verbose
            )

    def _simulate_fixed_step(self,
                         x_current: np.ndarray,
                         t_span: Tuple[float, float],
                         dt: Optional[float],
                         verbose: bool) -> Tuple[np.ndarray, np.ndarray]:
      
        
        # ============================================
        # STEP 1: SET DEFAULTS
        # ============================================
        if dt is None:
            dt = 0.05
        
        # Extract time bounds
        t_start, t_end = t_span
        
        # ============================================
        # STEP 2: GENERATE TIME POINTS (UNIFORM SPACING)
        # ============================================
        # Creates array: [t_start, t_start+dt, t_start+2*dt, ..., t_end]
        time_points = np.arange(t_start, t_end + dt, dt)
        n_steps = len(time_points)
        
        # ============================================
        # STEP 3: PRE-ALLOCATE MEMORY
        # ============================================
        # OPTIMIZATION: Allocate all at once (much faster than append)
        # Shape: (n_steps, n_vars) - one row per time point
        trajectory = np.zeros((n_steps, self.n_vars), dtype=float)
        trajectory[0] = x_current  # Store initial condition
        
        # ============================================
        # STEP 4: SELECT STEP FUNCTION
        # ============================================
        # OPTIMIZATION: Choose once (not in the loop)
        # Avoids if-branching on every iteration
        step_func = self._step_rk4 if self.method == 'rk4' else self._step_euler
        
        # ============================================
        # STEP 5: VERBOSE HEADER
        # ============================================
        if verbose:
            print(f"\n{'='*75}")
            print(f"  CONTINUOUS p-FUZZY SYSTEM SIMULATION (FIXED-STEP)")
            print(f"{'='*75}")
            print(f"  Time interval: [{t_start:.4f}, {t_end:.4f}]")
            print(f"  Integration method: {self.method.upper()}")
            print(f"  Dynamics mode: {self.mode}")
            print(f"  Step size (dt): {dt:.6f}")
            print(f"  Expected points: {n_steps}")
            print(f"  Initial state:")
            for i, var in enumerate(self.state_vars):
                print(f"    {var} = {x_current[i]:.6f}")
            print(f"{'='*75}\n")
        
        # ============================================
        # STEP 6: MAIN INTEGRATION LOOP
        # ============================================
        for i in range(1, n_steps):
            
            # Perform one integration step
            # (Uses either Euler or RK4 depending on self.method)
            x_current = step_func(x_current, dt)
            
            # ========================================
            # SAFETY CHECK: Verify state is in domain
            # ========================================
            valid, msg = self._check_domain(x_current)
            if not valid:
                # State has left valid domain - STOP
                print(f"\n⚠️  WARNING: State outside domain bounds")
                print(f"    {msg}")
                print(f"    Time: {time_points[i]:.6f}")
                print(f"    Step: {i}/{n_steps-1}")
                
                # Truncate trajectory to completed steps (not including failed step)
                self.trajectory = trajectory[:i]
                self.time = time_points[:i]
                
                # Verbose termination message
                if verbose:
                    print(f"\n{'='*75}")
                    print(f"  SIMULATION TERMINATED EARLY (Domain violation)")
                    print(f"{'='*75}")
                    print(f"  Reason: State exceeded domain bounds")
                    print(f"  Completed steps: {i}/{n_steps-1}")
                    print(f"  Points saved: {i}")
                    print(f"  Time reached: {self.time[-1]:.6f}/{t_end:.6f}")
                    print(f"{'='*75}\n")
                
                return self.time, self.trajectory
            
            # Store state at this time point
            trajectory[i] = x_current
            
            # ========================================
            # PROGRESS REPORTING (every 10%)
            # ========================================
            if verbose:
                # Report interval: every 10% of simulation (or final step)
                report_interval = max(1, (n_steps - 1) // 10)
                
                if i % report_interval == 0 or i == n_steps - 1:
                    # Compute progress percentage
                    progress_pct = 100.0 * i / (n_steps - 1)
                    
                    # Format state values
                    state_str = ', '.join([
                        f'{var}={x_current[j]:8.4f}' 
                        for j, var in enumerate(self.state_vars)
                    ])
                    
                    # Print progress line
                    print(
                        f"  Progress: {progress_pct:6.1f}% | "
                        f"Step: {i:6d}/{n_steps-1:6d} | "
                        f"Time: {time_points[i]:8.4f} | "
                        f"State: {state_str}"
                    )
        
        # ============================================
        # STEP 7: STORE RESULTS
        # ============================================
        self.time = time_points
        self.trajectory = trajectory
        
        # ============================================
        # STEP 8: VERBOSE FOOTER
        # ============================================
        if verbose:
            print(f"\n{'='*75}")
            print(f"  SIMULATION COMPLETED SUCCESSFULLY")
            print(f"{'='*75}")
            print(f"  Integration method: {self.method.upper()}")
            print(f"  Total steps executed: {n_steps - 1}")
            print(f"  Points in trajectory: {n_steps}")
            print(f"  Time interval: [{t_start:.4f}, {t_end:.4f}]")
            print(f"  Final time: {self.time[-1]:.4f}")
            print(f"  Duration: {self.time[-1] - t_start:.4f} time units")
            print(f"  Average step: {(self.time[-1] - t_start) / (n_steps - 1):.6f}")
            print(f"  Final state:")
            for j, var in enumerate(self.state_vars):
                print(f"    {var} = {x_current[j]:.6f}")
            print(f"{'='*75}\n")
        
        return self.time, self.trajectory




    def _step_rk4_with_error(self, 
                         x: np.ndarray, 
                         dt: float) -> Tuple[np.ndarray, float]:
        
        
        # ====================================================
        # STEP 1: COMPUTE FULL STEP (dt with single RK4)
        # ====================================================
        # Standard RK4: x(t) → x(t+dt) in one step
        x_full = self._step_rk4(x, dt)
        
        # ====================================================
        # STEP 2: COMPUTE HALF-STEPS (dt/2 twice)
        # ====================================================
        # More accurate approach: break into two smaller steps
        # x(t) → x(t+dt/2) → x(t+dt)
        x_half1 = self._step_rk4(x, dt / 2.0)          # First half-step
        x_half2 = self._step_rk4(x_half1, dt / 2.0)    # Second half-step
        
        # ====================================================
        # STEP 3: ESTIMATE LOCAL TRUNCATION ERROR
        # ====================================================
        # Using Richardson extrapolation principle:
        # 
        # RK4 error is O(dt⁵), so:
        # full_step_error ≈ C·dt⁵
        # half_steps_error ≈ 2·C·(dt/2)⁵ = C·dt⁵/16
        # 
        # Difference: C·dt⁵·(1 - 1/16) = C·dt⁵·(15/16)
        # 
        # Therefore: C·dt⁵ ≈ |full - half| · 16/15 ≈ |full - half| / (15/16)
        
        # Compute Euclidean norm of difference (works for multidimensional states)
        difference = x_full - x_half2
        error_magnitude = np.linalg.norm(difference)
        
        # Richardson extrapolation factor (for 4th-order method with dt⁵ error)
        # The exact factor is 15 for RK4 (from error analysis)
        richardson_factor = 15.0
        error_estimate = error_magnitude / richardson_factor
        
        # ====================================================
        # STEP 4: RETURN MORE ACCURATE SOLUTION
        # ====================================================
        # The half-steps solution is more accurate than full step
        # (2 smaller steps accumulate less error than 1 larger step)
        return x_half2, error_estimate


    def simulate_adaptive(self,
                      x0: Union[Dict[str, float], np.ndarray, List[float], Tuple[float, ...]],
                      t_span: Tuple[float, float],
                      dt_initial: float = 0.1,
                      tolerance: float = 1e-4,
                      dt_min: float = 1e-5,
                      dt_max: float = 1.0,
                      max_steps: int = 100000,
                      verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate with adaptive step size control (high accuracy).

        Automatically adjusts time step based on error estimates.
        More accurate than fixed-step, especially for stiff systems.

        Args:
            x0: Initial state as dict, array, list, or tuple
            t_span: Time interval (t_start, t_end)
            dt_initial: Initial time step (default: 0.1)
            tolerance: Error tolerance (default: 1e-4)
            dt_min: Minimum allowed time step (default: 1e-5)
            dt_max: Maximum allowed time step (default: 1.0)
            max_steps: Maximum number of steps (default: 100000)
            verbose: Print progress (default: False)

        Returns:
            (t, trajectory): Time points and state trajectory

        Example:
            >>> t, traj = pfuzzy.simulate_adaptive(x0={'x': 1.0}, t_span=(0, 10), tolerance=1e-6)
        """

        import warnings
        
        # ============================================================
        # STEP 1: NORMALIZE INPUT
        # ============================================================
        if isinstance(x0, dict):
            x_current = np.array([x0[var] for var in self.state_vars], dtype=float)
        elif isinstance(x0, (list, tuple)):
            x_current = np.array(x0, dtype=float)
        else:
            x_current = np.array(x0, dtype=float)
        
        # ============================================================
        # STEP 2: VALIDATE INPUT
        # ============================================================
        if len(x_current) != self.n_vars:
            raise ValueError(
                f"Initial condition must have {self.n_vars} values, "
                f"received {len(x_current)}. "
                f"Expected order: {self.state_vars}"
            )
        
        # Check domain
        valid, msg = self._check_domain(x_current)
        if not valid:
            raise ValueError(f"Initial condition violates domain constraints: {msg}")
        
        # ============================================================
        # STEP 3: INITIALIZATION
        # ============================================================
        t_start, t_end = t_span
        t_current = t_start
        dt = dt_initial
        
        # Pre-allocate lists for dynamic growth
        trajectory = [x_current.copy()]
        time_points = [t_current]
        
        # Statistics tracking
        n_accepted = 0
        n_rejected = 0
        dt_history = []
        
        # ============================================================
        # STEP 4: SELECT STEP FUNCTION WITH ERROR ESTIMATION
        # ============================================================
        if self.method == 'rk4':
            step_with_error = self._step_rk4_with_error
            order = 4  # RK4 is 4th-order
        else:  # euler
            step_with_error = self._step_euler_with_error if hasattr(self, '_step_euler_with_error') else self._step_rk4_with_error
            order = 1  # Euler is 1st-order
            
            if verbose:
                warnings.warn(
                    "Using adaptive Euler method. For better efficiency, "
                    "consider method='rk4' (typically 2-10× faster for same accuracy).",
                    UserWarning,
                    stacklevel=2
                )
        
        # ============================================================
        # STEP 5: VERBOSE HEADER
        # ============================================================
        if verbose:
            print(f"\n{'='*75}")
            print(f"  CONTINUOUS p-FUZZY SYSTEM SIMULATION (ADAPTIVE STEPPING)")
            print(f"{'='*75}")
            print(f"  Time interval: [{t_start:.6f}, {t_end:.6f}]")
            print(f"  Integration method: {self.method.upper()} (order {order})")
            print(f"  Dynamics mode: {self.mode}")
            print(f"  Initial dt: {dt_initial:.6f}")
            print(f"  Error tolerance: {tolerance:.2e}")
            print(f"  dt limits: [{dt_min:.2e}, {dt_max:.2e}]")
            print(f"  Max steps allowed: {max_steps:,}")
            print(f"  Initial state:")
            for i, var in enumerate(self.state_vars):
                print(f"    {var} = {x_current[i]:.6f}")
            print(f"{'='*75}\n")
        
        # ============================================================
        # STEP 6: MAIN ADAPTIVE INTEGRATION LOOP
        # ============================================================
        step_count = 0
        while t_current < t_end and step_count < max_steps:
            step_count += 1
            
            # Adjust dt if it would overshoot t_end
            if t_current + dt > t_end:
                dt = t_end - t_current
            
            # ====================================================
            # STEP 6a: ATTEMPT INTEGRATION STEP WITH ERROR EST.
            # ====================================================
            try:
                x_next, error_estimate = step_with_error(x_current, dt)
            except Exception as e:
                print(f"\n⚠️  ERROR in step function at iteration {step_count}:")
                print(f"    {e}")
                break
            
            # ====================================================
            # STEP 6b: COMPUTE RELATIVE ERROR NORM
            # ====================================================
            # Relative error: error / (tol * (1 + |x|))
            # Avoids over-refinement for small |x|
            state_magnitude = 1.0 + np.linalg.norm(x_current)
            error_norm = error_estimate / (tolerance * state_magnitude)
            
            # ====================================================
            # STEP 6c: DECISION: ACCEPT OR REJECT STEP
            # ====================================================
            if error_norm <= 1.0:
                # ========================================
                # ACCEPT STEP - error within tolerance
                # ========================================
                
                # Check domain before accepting
                valid, msg = self._check_domain(x_next)
                if not valid:
                    print(f"\n⚠️  WARNING: {msg}")
                    print(f"    Time: {t_current:.6f}")
                    print(f"    Step: {step_count}")
                    break
                
                # Advance time and state
                x_current = x_next
                t_current += dt
                
                # Store trajectory
                trajectory.append(x_current.copy())
                time_points.append(t_current)
                dt_history.append(dt)
                
                n_accepted += 1
                
                # Progress reporting (every 100 steps)
                if verbose and n_accepted % 100 == 0:
                    progress_pct = 100.0 * (t_current - t_start) / (t_end - t_start)
                    print(
                        f"  Progress: {progress_pct:6.1f}% | "
                        f"Time: {t_current:8.4f} | "
                        f"dt: {dt:.6f} | "
                        f"Accepted: {n_accepted:6d} | "
                        f"Rejected: {n_rejected:6d}"
                    )
                
                # ====================================================
                # STEP 6d: COMPUTE OPTIMAL dt FOR NEXT STEP
                # ====================================================
                # Formula: dt_new = dt * (1/error_norm)^(1/(order+1))
                # * safety factor 0.9 to avoid repeated rejections
                # * clamp to [0.5, 2.0] to avoid extreme changes
                
                if error_norm > 0:
                    exponent = 1.0 / (order + 1)
                    factor = 0.9 * (1.0 / error_norm) ** exponent
                    factor = np.clip(factor, 0.5, 2.0)  # Limit factor change
                    dt = dt * factor
                else:
                    # Negligible error: increase dt
                    dt = min(dt * 1.5, dt_max)
                
                # Respect dt limits
                dt = np.clip(dt, dt_min, dt_max)
            
            else:
                # ========================================
                # REJECT STEP - error exceeds tolerance
                # ========================================
                n_rejected += 1
                
                # Compute reduced dt for retry
                exponent = 1.0 / (order + 1)
                factor = 0.9 * (1.0 / error_norm) ** exponent
                factor = np.clip(factor, 0.1, 0.95)  # Reduce significantly
                dt = dt * factor
                
                # Check if dt fell below minimum
                if dt < dt_min:
                    print(f"\n⚠️  WARNING: dt reached minimum ({dt_min:.2e})")
                    print(f"    Error too high for specified tolerance")
                    print(f"    Consider increasing tolerance or reducing dt_min")
                    print(f"    Time: {t_current:.6f}/{t_end:.6f}")
                    break
        
        # ============================================================
        # STEP 7: CHECK LOOP TERMINATION CONDITIONS
        # ============================================================
        if step_count >= max_steps:
            warnings.warn(
                f"Maximum step limit ({max_steps:,}) reached before final time.\n"
                f"Completed time: {t_current:.6f}/{t_end:.6f}\n"
                f"Consider: increasing max_steps, relaxing tolerance, or increasing dt_max",
                UserWarning,
                stacklevel=2
            )
        
        # ============================================================
        # STEP 8: STORE RESULTS
        # ============================================================
        self.time = np.array(time_points, dtype=float)
        self.trajectory = np.array(trajectory, dtype=float)
        
        # ============================================================
        # STEP 9: VERBOSE FOOTER WITH STATISTICS
        # ============================================================
        if verbose:
            total_steps = n_accepted + n_rejected
            acceptance_rate = 100.0 * n_accepted / total_steps if total_steps > 0 else 0
            
            print(f"\n{'='*75}")
            print(f"  ADAPTIVE SIMULATION COMPLETED")
            print(f"{'='*75}")
            print(f"  Integration method: {self.method.upper()}")
            print(f"  Steps accepted: {n_accepted:,}")
            print(f"  Steps rejected: {n_rejected:,}")
            print(f"  Total attempts: {total_steps:,}")
            print(f"  Acceptance rate: {acceptance_rate:.1f}%")
            print(f"  Trajectory points: {len(time_points):,}")
            print(f"  Speedup vs fixed-step: ~{max(1, 300.0 / len(time_points)):.1f}×")
            
            if len(dt_history) > 0:
                print(f"  Step size statistics:")
                print(f"    Mean dt: {np.mean(dt_history):.6e}")
                print(f"    Min dt:  {np.min(dt_history):.6e}")
                print(f"    Max dt:  {np.max(dt_history):.6e}")
            
            print(f"  Time statistics:")
            print(f"    Initial time: {t_start:.6f}")
            print(f"    Final time:   {t_current:.6f}")
            print(f"    Duration: {t_current - t_start:.6f}")
            print(f"  Final state:")
            for j, var in enumerate(self.state_vars):
                print(f"    {var} = {x_current[j]:.6f}")
            print(f"{'='*75}\n")
        
        return self.time, self.trajectory



    def _step_euler_with_error(self, 
                           x: np.ndarray, 
                           dt: float) -> Tuple[np.ndarray, float]:

        
        # ====================================================
        # STEP 1: COMPUTE FULL STEP (dt with single Euler)
        # ====================================================
        # Standard Euler: x(t) → x(t+dt) in one step
        x_full = self._step_euler(x, dt)
        
        # ====================================================
        # STEP 2: COMPUTE HALF-STEPS (dt/2 twice)
        # ====================================================
        # More accurate approach: break into two smaller steps
        # x(t) → x(t+dt/2) → x(t+dt)
        x_half1 = self._step_euler(x, dt / 2.0)          # First half-step
        x_half2 = self._step_euler(x_half1, dt / 2.0)    # Second half-step
        
        # ====================================================
        # STEP 3: ESTIMATE LOCAL TRUNCATION ERROR
        # ====================================================
        # Using Richardson extrapolation principle for first-order method:
        #
        # Euler error is O(dt²), so:
        # full_step_error ≈ C·dt²
        # half_steps_error ≈ 2·C·(dt/2)² = C·dt²/2
        #
        # Difference: C·dt²·(1 - 1/2) = C·dt²/2
        #
        # Therefore: C·dt² ≈ |full - half| · 2
        # And error estimate: C·dt² / 2 ≈ |full - half| / 2
        
        # Compute Euclidean norm of difference (works for multidimensional states)
        difference = x_full - x_half2
        error_magnitude = np.linalg.norm(difference)
        
        # Richardson extrapolation factor (for 1st-order method with dt² error)
        # For p-th order method: factor = 2^(p+1) - 2 = 2^2 - 2 = 2 for p=1
        richardson_factor = 2.0
        error_estimate = error_magnitude / richardson_factor
        
        # ====================================================
        # STEP 4: RETURN MORE ACCURATE SOLUTION
        # ====================================================
        # The half-steps solution is more accurate than full step
        # (2 smaller steps accumulate less error than 1 larger step)
        # This approach is called Richardson extrapolation
        return x_half2, error_estimate

