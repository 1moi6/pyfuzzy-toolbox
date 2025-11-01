import numpy as np
from typing import Callable, List, Tuple, Union, Optional, Dict
from dataclasses import dataclass
from scipy.integrate import solve_ivp
import warnings
import itertools

# Optional joblib import for parallelization
try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False
    warnings.warn(
        "joblib not found. Parallel processing disabled. "
        "Install with: pip install fuzzy-systems[ml]",
        ImportWarning
    )

# Core integration
from ..core import FuzzySet, triangular, trapezoidal, gaussian


@dataclass
class FuzzyNumber:
    """
    Fuzzy number based on core FuzzySet.

    Represents a fuzzy number through its membership function.
    Fully integrated with fuzzy_systems.core.

    Attributes:
        fuzzy_set: Core FuzzySet (triangular, gaussian, etc)
        support: Fuzzy number support [min, max]
        name: Descriptive name (optional)

    Examples:
        >>> # Triangular number ~5
        >>> num1 = FuzzyNumber.triangular(center=5, spread=1)

        >>> # Gaussian number ~10
        >>> num2 = FuzzyNumber.gaussian(mean=10, sigma=2)

        >>> # Trapezoidal number
        >>> num3 = FuzzyNumber.trapezoidal(a=1, b=2, c=3, d=4)
    """
    fuzzy_set: FuzzySet
    support: Tuple[float, float]
    name: str = "fuzzy_number"

    @classmethod
    def triangular(cls, center: float, spread: float,
                   name: str = "triangular") -> 'FuzzyNumber':
        """
        Creates triangular fuzzy number.

        Args:
            center: Center (peak, μ=1)
            spread: Spread (distance from center to extremes)
            name: Number name

        Returns:
            Triangular FuzzyNumber
        """
        a = center - spread
        b = center
        c = center + spread

        fuzzy_set = FuzzySet(
            name=name,
            mf_type='triangular',
            params=(a, b, c)
        )

        return cls(
            fuzzy_set=fuzzy_set,
            support=(a, c),
            name=name
        )

    @classmethod
    def trapezoidal(cls, a: float, b: float, c: float, d: float,
                    name: str = "trapezoidal") -> 'FuzzyNumber':
        """
        Creates trapezoidal fuzzy number.

        Args:
            a: Lower bound
            b: Plateau start
            c: Plateau end
            d: Upper bound
            name: Number name

        Returns:
            Trapezoidal FuzzyNumber
        """
        fuzzy_set = FuzzySet(
            name=name,
            mf_type='trapezoidal',
            params=(a, b, c, d)
        )

        return cls(
            fuzzy_set=fuzzy_set,
            support=(a, d),
            name=name
        )

    @classmethod
    def gaussian(cls, mean: float, sigma: float,
                 n_sigmas: float = 3.0,
                 name: str = "gaussian") -> 'FuzzyNumber':
        """
        Creates gaussian fuzzy number.

        Args:
            mean: Mean (center)
            sigma: Standard deviation
            n_sigmas: How many sigmas to define support (default: 3)
            name: Number name

        Returns:
            Gaussian FuzzyNumber
        """
        fuzzy_set = FuzzySet(
            name=name,
            mf_type='gaussian',
            params=(mean, sigma)
        )

        support = (mean - n_sigmas * sigma, mean + n_sigmas * sigma)

        return cls(
            fuzzy_set=fuzzy_set,
            support=support,
            name=name
        )

    @classmethod
    def from_fuzzy_set(cls, fuzzy_set: FuzzySet, support: Tuple[float, float]) -> 'FuzzyNumber':
        """
        Creates FuzzyNumber from a core FuzzySet.

        Args:
            fuzzy_set: Core FuzzySet
            support: Support [min, max]

        Returns:
            FuzzyNumber
        """
        return cls(
            fuzzy_set=fuzzy_set,
            support=support,
            name=fuzzy_set.name
        )

    def alpha_cut(self, alpha: float, n_points: int = 100) -> Tuple[float, float]:
        """
        Extracts α-cut from fuzzy number.

        Args:
            alpha: α level (0 to 1)
            n_points: Points for numerical search

        Returns:
            (min, max) of α-cut
        """
        if not (0 <= alpha <= 1):
            raise ValueError(f"Alpha must be in [0, 1], received: {alpha}")

        # Special case: α = 0 returns full support
        if alpha == 0:
            return self.support

        # Numerical search for points where μ(x) >= alpha
        x = np.linspace(self.support[0], self.support[1], n_points)
        mu = self.fuzzy_set.membership(x)

        # Points that satisfy μ(x) >= alpha
        valid_indices = np.where(mu >= alpha - 1e-10)[0]

        if len(valid_indices) == 0:
            # Alpha too high, return center
            center = (self.support[0] + self.support[1]) / 2
            return (center, center)

        x_min = x[valid_indices[0]]
        x_max = x[valid_indices[-1]]

        return (x_min, x_max)

    def membership(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Computes membership degree."""
        return self.fuzzy_set.membership(x)

    def __repr__(self) -> str:
        return (f"FuzzyNumber(name='{self.name}', "
                f"type='{self.fuzzy_set.mf_type}', "
                f"support={self.support})")


@dataclass
class FuzzySolution:
    """
    Solution of a fuzzy ODE.

    Contains envelopes (min/max) for each α-level at each time instant.

    Attributes:
        t: Time array
        y_min: Array [n_alpha, n_vars, n_time] with lower envelope
        y_max: Array [n_alpha, n_vars, n_time] with upper envelope
        alphas: α levels used
        var_names: Variable names
    """
    t: np.ndarray
    y_min: np.ndarray  # shape: (n_alpha, n_vars, n_time)
    y_max: np.ndarray  # shape: (n_alpha, n_vars, n_time)
    alphas: np.ndarray
    var_names: List[str] = None

    def __post_init__(self):
        if self.var_names is None:
            n_vars = self.y_min.shape[1]
            self.var_names = [f"y{i}" for i in range(n_vars)]

    def get_alpha_level(self, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns envelopes for a specific α-level.

        Args:
            alpha: Desired α level

        Returns:
            (y_min, y_max) for the nearest α
        """
        idx = np.argmin(np.abs(self.alphas - alpha))
        return self.y_min[idx], self.y_max[idx]

    def plot(self, var_idx: int = None, ax=None, alpha_levels=None,
             show=True, figsize=None, **kwargs):
        """
        Plots fuzzy solution with α-levels.

        Args:
            var_idx: Variable index to plot (None = plot all variables)
            ax: Matplotlib axis or array of axes (None = create new)
            alpha_levels: List of αs to plot (None = all)
            show: If True, calls plt.show()
            figsize: Figure size (width, height). If None, auto-sized based on n_vars
            **kwargs: Arguments for fill_between
        """
        import matplotlib.pyplot as plt

        if alpha_levels is None:
            alpha_levels = self.alphas

        # Determine which variables to plot
        n_vars = len(self.var_names)

        if var_idx is None:
            # Plot all variables
            vars_to_plot = list(range(n_vars))
        else:
            # Plot single variable
            vars_to_plot = [var_idx]

        n_plots = len(vars_to_plot)

        # Create figure and axes if not provided
        if ax is None:
            if n_plots == 1:
                if figsize is None:
                    figsize = (10, 6)
                fig, ax = plt.subplots(figsize=figsize)
                axes = [ax]
            else:
                # Multiple subplots
                if figsize is None:
                    # Auto-size: 10 width, 4 height per subplot
                    figsize = (10, 4 * n_plots)
                fig, axes = plt.subplots(n_plots, 1, figsize=figsize)
                # Ensure axes is always a list
                if n_plots == 1:
                    axes = [axes]
        else:
            # Use provided axes
            if isinstance(ax, (list, tuple)):
                axes = ax
            else:
                axes = [ax]
            fig = axes[0].get_figure()

        # Colormap for α-levels
        cmap = plt.cm.Blues

        # Plot each variable
        for plot_idx, v_idx in enumerate(vars_to_plot):
            current_ax = axes[plot_idx]

            for i, alpha in enumerate(self.alphas):
                if alpha not in alpha_levels:
                    continue

                y_min_alpha, y_max_alpha = self.get_alpha_level(alpha)

                # Color intensity proportional to α
                color = cmap(0.3 + 0.7 * alpha)

                # Plot envelope
                current_ax.fill_between(
                    self.t,
                    y_min_alpha[v_idx],
                    y_max_alpha[v_idx],
                    alpha=0.3,
                    color=color,
                    label=f'α={alpha:.2f}' if i % max(
                        1, len(self.alphas) // 5) == 0 else None,
                    **kwargs
                )

            current_ax.set_xlabel('Time', fontsize=12)
            current_ax.set_ylabel(self.var_names[v_idx], fontsize=12)
            current_ax.set_title(f'Fuzzy Solution: {self.var_names[v_idx]}',
                         fontsize=14, fontweight='bold')
            current_ax.legend(loc='best')
            current_ax.grid(True, alpha=0.3)

        if show:
            plt.tight_layout()
            plt.show()

        return fig, axes

    def to_dataframe(self, alpha: Optional[float] = None):
        """
        Converts fuzzy solution to pandas DataFrame.

        Args:
            alpha: Specific α level (None = uses α=1.0, fuzzy core)

        Returns:
            pandas.DataFrame with columns:
                - time: Time
                - {var}_min: Lower envelope for each variable
                - {var}_max: Upper envelope for each variable

        Raises:
            ImportError: If pandas is not installed

        Example:
            >>> sol = solver.solve()
            >>> df = sol.to_dataframe(alpha=0.5)
            >>> df.head()
            >>>
            >>> # Export to CSV
            >>> df.to_csv('fuzzy_solution.csv', index=False)
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for to_dataframe(). "
                "Install with: pip install pandas"
            )

        # If alpha not provided, use α=1.0 (core)
        if alpha is None:
            alpha = 1.0

        # Find nearest α-level
        idx = np.argmin(np.abs(self.alphas - alpha))
        alpha_real = self.alphas[idx]

        y_min, y_max = self.get_alpha_level(alpha_real)

        # Build data dictionary
        data = {'time': self.t}

        for i, var_name in enumerate(self.var_names):
            data[f'{var_name}_min'] = y_min[i]
            data[f'{var_name}_max'] = y_max[i]

        df = pd.DataFrame(data)

        # Add metadata as attributes
        df.attrs['alpha_level'] = float(alpha_real)
        df.attrs['n_alpha_levels'] = len(self.alphas)
        df.attrs['var_names'] = self.var_names

        return df

    def to_csv(self, filename: str, alpha: Optional[float] = None,
               sep: str = ',', decimal: str = '.', **kwargs):
        """
        Exports fuzzy solution to CSV file.

        Args:
            filename: CSV file path
            alpha: α level (None = α=1.0)
            sep: Column separator (default: ',')
            decimal: Decimal separator (default: '.' for international,
                    use ',' for Brazilian/European format)
            **kwargs: Additional arguments for pd.DataFrame.to_csv()

        Example:
            >>> sol.to_csv('solution.csv')
            >>>
            >>> # Brazilian format (Excel)
            >>> sol.to_csv('solution.csv', sep=';', decimal=',')
            >>>
            >>> # Specific α-level
            >>> sol.to_csv('solution_alpha05.csv', alpha=0.5)
        """
        df = self.to_dataframe(alpha=alpha)

        # Default CSV settings
        csv_kwargs = {
            'index': False,
            'sep': sep,
            'decimal': decimal,
        }
        csv_kwargs.update(kwargs)

        df.to_csv(filename, **csv_kwargs)

    def __repr__(self) -> str:
        return (f"FuzzySolution(n_vars={len(self.var_names)}, "
                f"n_alpha={len(self.alphas)}, "
                f"t_span=({self.t[0]:.2f}, {self.t[-1]:.2f}), "
                f"n_time={len(self.t)})")


class FuzzyODESolver:

    def __init__(
        self,
        ode_func: Callable,
        t_span: Tuple[float, float],
        initial_condition: List[Union[FuzzyNumber, float]],
        params: Optional[Dict[str, Union[FuzzyNumber, float]]] = None,
        n_alpha_cuts: int = 11,
        method: str = 'RK45',
        t_eval: Optional[np.ndarray] = None,
        n_jobs: int = -1,
        rtol: float = 1e-6,
        atol: float = 1e-9,
        var_names: Optional[List[str]] = None
    ):

        self.ode_func = ode_func
        self.t_span = t_span
        self.initial_condition = initial_condition
        self.params = params or {}
        self.n_alpha_cuts = n_alpha_cuts

        self.method = method
        self.t_eval = t_eval
        self.n_jobs = n_jobs
        self.rtol = rtol
        self.atol = atol
        self.var_names = var_names

        # Dimensions
        self.n_vars = len(initial_condition)
        self.n_params = len(self.params)

        # Validate
        self._validate_inputs()

    def _validate_inputs(self):
        """Validates inputs."""
        if self.n_vars == 0:
            raise ValueError("y0_fuzzy cannot be empty")

        if self.n_alpha_cuts < 2:
            raise ValueError("n_alpha_cuts must be >= 2")

    def _generate_alpha_levels(self) -> np.ndarray:
        """Generates uniformly spaced α levels."""
        return np.linspace(0, 1, self.n_alpha_cuts)

    def _extract_alpha_cuts(
        self,
        alpha: float
    ) -> Tuple[List[Tuple[float, float]], Dict[str, Tuple[float, float]]]:
        """
        Extracts α-cuts from all fuzzy variables and parameters.

        Args:
            alpha: α level

        Returns:
            (y0_intervals, params_intervals)
        """
        # α-cuts of initial conditions
        y0_intervals = []
        for y0 in self.initial_condition:
            if isinstance(y0, FuzzyNumber):
                interval = y0.alpha_cut(alpha)
            else:
                # Crisp value
                interval = (float(y0), float(y0))
            y0_intervals.append(interval)

        # α-cuts of parameters
        params_intervals = {}
        for param_name, param_value in self.params.items():
            if isinstance(param_value, FuzzyNumber):
                interval = param_value.alpha_cut(alpha)
            else:
                interval = (float(param_value), float(param_value))
            params_intervals[param_name] = interval

        return y0_intervals, params_intervals

    def _create_grid(
        self,
        y0_intervals: List[Tuple[float, float]],
        params_intervals: Dict[str, Tuple[float, float]]
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Creates grid of initial points and parameters (vectorized).

        Args:
            y0_intervals: [min, max] intervals for each y0
            params_intervals: Intervals for each parameter

        Returns:
            (y0_grid, params_grid)
            y0_grid: array (n_points, n_vars)
            params_grid: list of parameter dicts
        """
        # Create points for each dimension
        y0_points = []
        for y_min, y_max in y0_intervals:
            if y_min == y_max:
                # Crisp value
                points = np.array([y_min])
            else:
                # Uniformly spaced points in interval
                points = np.linspace(y_min, y_max, self.n_grid_points)
            y0_points.append(points)

        params_points = {}
        for param_name, (p_min, p_max) in params_intervals.items():
            if p_min == p_max:
                points = np.array([p_min])
            else:
                points = np.linspace(p_min, p_max, self.n_grid_points)
            params_points[param_name] = points

        # Cartesian product (full grid)
        # For y0
        y0_meshgrid = np.meshgrid(*y0_points, indexing='ij')
        y0_grid = np.stack([grid.flatten() for grid in y0_meshgrid], axis=1)

        # For params
        if params_points:
            param_names = list(params_points.keys())
            param_values = [params_points[name] for name in param_names]
            param_meshgrid = np.meshgrid(*param_values, indexing='ij')

            # Repeat for each y0 combination
            n_y0_combinations = y0_grid.shape[0]
            n_param_combinations = param_meshgrid[0].size

            # Expand y0_grid to include all parameter combinations
            y0_grid_expanded = np.repeat(y0_grid, n_param_combinations, axis=0)

            # Create list of parameter dicts
            params_grid = []
            for _ in range(n_y0_combinations):
                for idx in range(n_param_combinations):
                    param_dict = {
                        name: param_meshgrid[i].flatten()[idx]
                        for i, name in enumerate(param_names)
                    }
                    params_grid.append(param_dict)

            y0_grid = y0_grid_expanded
        else:
            # No fuzzy parameters
            params_grid = [{} for _ in range(y0_grid.shape[0])]

        return y0_grid, params_grid

    def _solve_single_ode_with_t_eval(
        self,
        y0: np.ndarray,
        params: Dict,
        t_eval: np.ndarray
    ) -> np.ndarray:
        """
        Solves a single ODE with specific parameters and times.

        Args:
            y0: Initial condition
            params: Parameters
            t_eval: Times for evaluation

        Returns:
            Array (n_vars, len(t_eval)) with solution
        """

        # Wrapper to include parameters
        def ode_wrapper(t, y):
            return self.ode_func(t, y, **params)

        # Solve
        sol = solve_ivp(
            ode_wrapper,
            self.t_span,
            y0,
            method=self.method,
            t_eval=t_eval,  # *** ALWAYS specify t_eval ***
            rtol=self.rtol,
            atol=self.atol,
            dense_output=False
        )

        if not sol.success:
            warnings.warn(
                f"ODE solver failed for y0={y0}, params={params}: {sol.message}",
                RuntimeWarning
            )
            # Return NaNs with correct shape
            return np.full((self.n_vars, len(t_eval)), np.nan)

        # Ensure output has shape (n_vars, len(t_eval))
        if sol.y.shape[0] != self.n_vars or sol.y.shape[1] != len(t_eval):
            # If something went wrong, return NaNs
            return np.full((self.n_vars, len(t_eval)), np.nan)

        return sol.y

    def _solve_alpha_level(
        self,
        alpha: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solves ODE for a specific α-level.

        Args:
            alpha: α level

        Returns:
            (t, y_min, y_max)
        """

        # 1. Extract α-cuts
        y0_intervals, params_intervals = self._extract_alpha_cuts(alpha)

        # 2. Create grid
        y0_grid, params_grid = self._create_grid(
            y0_intervals, params_intervals)

        # *** FIX: Define t_eval EXPLICITLY if not provided ***
        if self.t_eval is None:
            # Create consistent time mesh
            t_eval_internal = np.linspace(
                self.t_span[0],
                self.t_span[1],
                100  # 100 uniform points
            )
        else:
            t_eval_internal = self.t_eval

        # 3. Solve ODEs (parallel if joblib available, otherwise serial)
        if HAS_JOBLIB and self.n_jobs != 1:
            solutions = Parallel(n_jobs=self.n_jobs, backend='loky')(
                delayed(self._solve_single_ode_with_t_eval)(
                    y0, params, t_eval_internal
                )
                for y0, params in zip(y0_grid, params_grid)
            )
        else:
            # Fallback: serial processing
            solutions = [
                self._solve_single_ode_with_t_eval(y0, params, t_eval_internal)
                for y0, params in zip(y0_grid, params_grid)
            ]

        # 4. Filter valid solutions (without NaNs)
        valid_solutions = [
            sol for sol in solutions
            if not np.any(np.isnan(sol))
        ]

        if len(valid_solutions) == 0:
            raise RuntimeError(f"No valid solution for α={alpha}")

        # *** NOW ALL SOLUTIONS HAVE THE SAME SHAPE ***
        # (n_solutions, n_vars, n_time)
        solutions_array = np.stack(valid_solutions, axis=0)

        # 5. Extract envelopes (min/max)
        y_min = np.min(solutions_array, axis=0)  # (n_vars, n_time)
        y_max = np.max(solutions_array, axis=0)  # (n_vars, n_time)

        return t_eval_internal, y_min, y_max

    def _solve_standard(self, n_grid_points=5, verbose: bool = False) -> 'FuzzySolution':
        """
        Original standard method - solves each complete α.
        (This is the original solve() code)
        """
        if n_grid_points < 2:
            raise ValueError("n_grid_points must be >= 2")

        self.n_grid_points = n_grid_points
        if verbose:
            print("=" * 70)
            print("FUZZY ODE SOLVER - STANDARD METHOD")
            print("=" * 70)
            print(f"Variables: {self.n_vars}")
            print(f"Fuzzy parameters: {self.n_params}")
            print(f"α-levels: {self.n_alpha_cuts}")
            print(f"Points per dimension: {self.n_grid_points}")
            print(f"Method: {self.method}")
            print("=" * 70 + "\n")

        alphas = self._generate_alpha_levels()

        if verbose:
            print("Solving for each α-level...")

        results = []

        for i, alpha in enumerate(alphas):
            if verbose:
                print(f"  α = {alpha:.3f} ({i+1}/{self.n_alpha_cuts})")

            t, y_min, y_max = self._solve_alpha_level(alpha)
            results.append((t, y_min, y_max))

        t_final = results[0][0]
        y_min_all = np.stack([res[1] for res in results], axis=0)
        y_max_all = np.stack([res[2] for res in results], axis=0)

        if verbose:
            print("\n✅ Solution computed!")
            print("=" * 70)

        return FuzzySolution(
            t=t_final,
            y_min=y_min_all,
            y_max=y_max_all,
            alphas=alphas,
            var_names=self.var_names
        )

    def _solve_single_ode(self, y0: np.ndarray, params: Dict,
                          t_eval: np.ndarray = None) -> np.ndarray:
        """Wrapper for compatibility."""
        return self._solve_single_ode_with_t_eval(y0, params, t_eval)

    def solve_with_method_option(self,
                                 method: str = 'standard',
                                 verbose: bool = False,
                                 **method_kwargs) -> 'FuzzySolution':
        """
        Solves fuzzy ODE with multiple available methods.

        Args:
            method: Method to use
                - 'standard': Solve each α-level completely (default)
                - 'hierarchical': Hierarchical reuse of α-levels (3-5x faster)
                - 'monte_carlo': Monte Carlo (scalable, 10-400x in high dimension)

            verbose: If True, print progress

            **method_kwargs: Method-specific arguments
                For 'hierarchical': (none)
                For 'monte_carlo': n_samples=500, random_seed=None

        Returns:
            FuzzySolution

        """

        # Normalize method
        method_lower = method.lower().strip()

        if method_lower == 'standard':
            n_grid_points = method_kwargs.get('n_grid_points', 20)
            return self._solve_standard(n_grid_points=n_grid_points, verbose=verbose)

        elif method_lower == 'hierarchical':
            return self._solve_hierarchical(verbose=verbose)

        elif method_lower == 'monte_carlo':
            n_samples = method_kwargs.get('n_samples', 1000)
            random_seed = method_kwargs.get('random_seed', None)
            return self._solve_monte_carlo(
                n_samples=n_samples,
                random_seed=random_seed,
                verbose=verbose
            )

        else:
            raise ValueError(
                f"Unknown method: '{method}'. "
                f"Valid options: 'standard', 'hierarchical', 'monte_carlo'"
            )

    def _solve_hierarchical(self, verbose: bool = False) -> 'FuzzySolution':
        """
        Hierarchical method - reuses larger α-levels.
        3-5x faster than standard.
        """
        # Import optimization class

        optimizer = HierarchicalFuzzyODESolver(self)
        return optimizer.solve_optimized(verbose=verbose)

    def _solve_monte_carlo(self,
                           n_samples: int = 1000,
                           random_seed: int = None,
                           verbose: bool = False) -> 'FuzzySolution':
        """
        Monte Carlo method with inherited membership.
        10-400x faster in high dimensionality.
        """

        mc_solver = MonteCarloFuzzyODESolver(
            self,
            n_samples=n_samples,
            random_seed=random_seed
        )
        return mc_solver.solve_monte_carlo(verbose=verbose)
    def solve(self, method: str = 'standard',
              verbose: bool = False, **method_kwargs) -> FuzzySolution:
        """Solves fuzzy ODE with multiple methods."""
        return self.solve_with_method_option(method, verbose, **method_kwargs)


class MonteCarloFuzzyODESolver:
    """
    Monte Carlo solver with INHERITED membership from IC.

    Key idea:
        μ(y(t)) = μ(y0)
    """

    def __init__(self, base_solver, n_samples: int = 1000, random_seed: int = None):
        """
        Args:
            base_solver: Original FuzzyODESolver
            n_samples: Number of random samples
            random_seed: For reproducibility
        """
        self.solver = base_solver
        self.n_samples = n_samples
        if random_seed is not None:
            np.random.seed(random_seed)

        self.sampled_points = []
        self.solutions = []
        # IC membership (inherited to all solutions)
        self.pertinences_CI = []

    def _compute_ci_pertinence(self,
                               y0: np.ndarray,
                               params: Dict[str, float]) -> float:
        """
        Computes membership of an initial condition in fuzzy space.

        Membership is calculated as:
        μ(IC) = min(μ_y0_1, μ_y0_2, ..., μ_param_1, μ_param_2, ...)

        Where:
        - μ_y0_i: membership of i-th initial condition
        - μ_param_j: membership of j-th fuzzy parameter

        Args:
            y0 (np.ndarray): Initial conditions vector
                            Shape: (n_vars,)
                            Values evaluated in fuzzy membership functions

            params (Dict[str, float]): Parameter dictionary
                                    Keys: parameter names
                                    Values: numerical values

        Returns:
            float: Total membership degree ∈ [0, 1]
                1.0 = Maximum membership (fuzzy core)
                0.0 = Minimum membership (outside support)

        Raises:
            IndexError: If y0 has different size than expected n_vars
            KeyError: If params doesn't contain expected parameter
        """

        # Initialize with maximum membership
        pertinence = 1.0

        # ========================================================================
        # PHASE 1: Compute fuzzy initial conditions membership
        # ========================================================================

        for i, y0_var in enumerate(self.solver.initial_condition):

            # Get i-th IC value
            y0_value = y0[i]

            # Check if it's FuzzyNumber or crisp
            if hasattr(y0_var, 'fuzzy_set'):
                # It's FuzzyNumber: compute membership
                mu_y0 = y0_var.fuzzy_set.membership(y0_value)
            else:
                # It's crisp (float/int): membership is 1.0
                # (crisp doesn't restrict fuzzy space)
                mu_y0 = 1.0

            # Apply t-norm (minimum) to combine memberships
            pertinence = min(pertinence, mu_y0)

        # ========================================================================
        # PHASE 2: Compute fuzzy parameters membership
        # ========================================================================

        for param_name, param_var in self.solver.params.items():

            # Check if it's FuzzyNumber or crisp
            if hasattr(param_var, 'fuzzy_set'):
                # It's FuzzyNumber: get parameter value
                param_value = params.get(param_name)

                if param_value is None:
                    # Parameter not provided, assume default value
                    # This shouldn't happen, but it's a protection
                    continue

                # Compute parameter membership
                mu_param = param_var.fuzzy_set.membership(param_value)
            else:
                # It's crisp: membership is 1.0
                mu_param = 1.0

            # Apply t-norm (minimum)
            pertinence = min(pertinence, mu_param)

        # ========================================================================
        # PHASE 3: Normalize result to [0, 1]
        # ========================================================================

        # Ensure result is in [0, 1] interval
        # (protection against numerical errors)
        pertinence = max(0.0, min(1.0, pertinence))

        return pertinence

    def _sample_hypercube_with_pertinence(
    self,
    n_samples: int = 1000,
    verbose: bool = False
) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
        """
        Samples points in fuzzy hypercube in a simple and direct way.

        STRATEGY:
        1. Generates 1000 INDEPENDENT random samples in each dimension
        2. Uses zip() to combine into 1000 points (hypercube)
        3. Adds combinations of EXTREMES from α=0 (itertools.product)
        4. Adds combinations of EXTREMES from α=1.0
        5. Computes membership of ALL

        Args:
            n_samples: Number of samples per dimension (default: 1000)
            verbose: If True, print statistics

        Returns:
            (y0_samples, param_samples, pertinences_CI)
        """

        # ========================================================================
        # PHASE 1: EXTRACT INTERVALS
        # ========================================================================

        # α=0 (full support)
        y0_intervals_alpha_0, params_intervals_alpha_0 = (
            self.solver._extract_alpha_cuts(0.0)
        )

        # α=1.0 (core)
        y0_intervals_alpha_1, params_intervals_alpha_1 = (
            self.solver._extract_alpha_cuts(1.0)
        )

        if verbose:
            print(f"\n  PHASE 1: Generating {n_samples} samples per dimension...")

        # ========================================================================
        # PHASE 2: INDEPENDENT RANDOM SAMPLES IN EACH DIMENSION
        # ========================================================================

        # Samples for each IC
        y0_samples_per_dim = []
        for i, (y_min, y_max) in enumerate(y0_intervals_alpha_0):
            if y_min == y_max:
                samples = np.full(n_samples, y_min)
            else:
                samples = np.random.uniform(y_min, y_max, n_samples)
            y0_samples_per_dim.append(samples)
            if verbose:
                print(f"    - IC {i}: [{y_min:.3f}, {y_max:.3f}]")

        # Samples for each fuzzy parameter
        param_names = sorted([k for k, v in self.solver.params.items()
                            if hasattr(v, 'fuzzy_set')])

        param_samples_per_dim = {}
        for param_name in param_names:
            p_min, p_max = params_intervals_alpha_0[param_name]
            if p_min == p_max:
                samples = np.full(n_samples, p_min)
            else:
                samples = np.random.uniform(p_min, p_max, n_samples)
            param_samples_per_dim[param_name] = samples
            if verbose:
                print(f"    - Param '{param_name}': [{p_min:.3f}, {p_max:.3f}]")

        # ========================================================================
        # PHASE 3: COMBINE INTO 1000 POINTS WITH zip()
        # ========================================================================

        if verbose:
            print(f"\n  PHASE 2: Combining samples with zip()...")

        # Combine samples: [(y0_1[0], y0_2[0], ...), ...]
        y0_combined = list(zip(*y0_samples_per_dim))
        y0_combined = np.array(y0_combined)

        # Combine parameters: [{'r': r[0], 'K': K[0]}, ...]
        param_combined = []
        for i in range(n_samples):
            param_dict = {}
            for param_name in param_names:
                param_dict[param_name] = param_samples_per_dim[param_name][i]
            param_combined.append(param_dict)

        n_sampled_points = len(y0_combined)
        if verbose:
            print(f"    ✓ {n_sampled_points} hypercube points")

        # ========================================================================
        # PHASE 4: COMBINATIONS OF EXTREMES FROM α=0
        # ========================================================================

        if verbose:
            print(f"\n  PHASE 3: Adding extremes from α=0...")

        # Extremes of each IC
        y0_extremes_alpha_0 = []
        for y_min, y_max in y0_intervals_alpha_0:
            y0_extremes_alpha_0.append([y_min, y_max])

        # Extremes of each parameter
        param_extremes_alpha_0 = {}
        for param_name in param_names:
            p_min, p_max = params_intervals_alpha_0[param_name]
            param_extremes_alpha_0[param_name] = [p_min, p_max]

        # CARTESIAN PRODUCT of extremes
        # Example: [[y0_min, y0_max]] × [[r_min, r_max]] × [[K_min, K_max]]
        y0_extremes_product = list(itertools.product(*y0_extremes_alpha_0))
        param_extremes_product = list(itertools.product(*[
            param_extremes_alpha_0[pname] for pname in param_names
        ]))

        n_vertices_alpha_0 = len(y0_extremes_product) * len(param_extremes_product)

        # Combine y0 extremes with param extremes
        for y0_vertex in y0_extremes_product:
            for param_vertex in param_extremes_product:
                y0_combined = np.vstack([y0_combined, [y0_vertex]])

                param_dict = {}
                for param_idx, param_name in enumerate(param_names):
                    param_dict[param_name] = param_vertex[param_idx]
                param_combined.append(param_dict)

        if verbose:
            print(f"    ✓ {n_vertices_alpha_0} extreme combinations")
            print(f"      = 2^{len(y0_extremes_alpha_0)} × 2^{len(param_names)} = {len(y0_extremes_product)} × {len(param_extremes_product)}")

        # ========================================================================
        # PHASE 5: COMBINATIONS OF EXTREMES FROM α=1.0
        # ========================================================================

        if verbose:
            print(f"\n  PHASE 4: Adding extremes from α=1.0 (core)...")

        # Extremes of α=1.0 (may be just one point if triangular)
        y0_extremes_alpha_1 = []
        for y_min, y_max in y0_intervals_alpha_1:
            y0_extremes_alpha_1.append([y_min, y_max])

        param_extremes_alpha_1 = {}
        for param_name in param_names:
            p_min, p_max = params_intervals_alpha_1[param_name]
            param_extremes_alpha_1[param_name] = [p_min, p_max]

        # CARTESIAN PRODUCT of extremes α=1.0
        y0_extremes_product_alpha_1 = list(itertools.product(*y0_extremes_alpha_1))
        param_extremes_product_alpha_1 = list(itertools.product(*[
            param_extremes_alpha_1[pname] for pname in param_names
        ]))

        n_vertices_alpha_1 = len(y0_extremes_product_alpha_1) * len(param_extremes_product_alpha_1)

        # Add to points
        for y0_vertex in y0_extremes_product_alpha_1:
            for param_vertex in param_extremes_product_alpha_1:
                y0_combined = np.vstack([y0_combined, [y0_vertex]])

                param_dict = {}
                for param_idx, param_name in enumerate(param_names):
                    param_dict[param_name] = param_vertex[param_idx]
                param_combined.append(param_dict)

        if verbose:
            print(f"    ✓ {n_vertices_alpha_1} extreme combinations α=1")
            print(f"      = 2^{len(y0_extremes_alpha_1)} × 2^{len(param_names)} = {len(y0_extremes_product_alpha_1)} × {len(param_extremes_product_alpha_1)}")

        # ========================================================================
        # PHASE 6: COMPUTE MEMBERSHIPS
        # ========================================================================

        if verbose:
            print(f"\n  PHASE 5: Computing memberships...")

        n_total = len(y0_combined)
        pertinences_CI = []

        for i in range(n_total):
            mu_i = self._compute_ci_pertinence(y0_combined[i], param_combined[i])
            pertinences_CI.append(mu_i)

        pertinences_CI = np.array(pertinences_CI)

        # ========================================================================
        # PHASE 7: RETURN
        # ========================================================================

        if verbose:
            print(f"\n  SUMMARY:")
            print(f"    Total points: {n_total}")
            print(f"      = {n_sampled_points} (zip) + {n_vertices_alpha_0} (extremes α=0) + {n_vertices_alpha_1} (extremes α=1)")
            print(f"\n    Memberships: min={np.min(pertinences_CI):.3f}, max={np.max(pertinences_CI):.3f}, mean={np.mean(pertinences_CI):.3f}")

        # Reorganize param_combined into dicts per parameter
        param_samples_final = {pname: [] for pname in param_names}
        for param_dict in param_combined:
            for pname in param_names:
                param_samples_final[pname].append(param_dict[pname])

        for pname in param_names:
            param_samples_final[pname] = np.array(param_samples_final[pname])

        return y0_combined, param_samples_final, pertinences_CI

    def _solve_all_samples(
        self,
        y0_samples: np.ndarray,
        param_samples: Dict[str, np.ndarray]
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Solves ODEs for all sampled points.
        """

        t_eval = self.solver.t_eval
        if t_eval is None:
            t_eval = np.linspace(
                self.solver.t_span[0],
                self.solver.t_span[1],
                100
            )

        # Prepara lista de tarefas
        solve_tasks = []
        param_names = sorted(param_samples.keys())

        for i in range(len(y0_samples)):
            y0 = y0_samples[i]

            params = {}
            for pname in param_names:
                pvals = param_samples[pname]
                if len(pvals) > i:
                    params[pname] = pvals[i]
                else:
                    params[pname] = pvals[0]

            # Add crisp parameters
            for param_name, param_val in self.solver.params.items():
                if param_name not in params:
                    params[param_name] = param_val

            solve_tasks.append((y0, params))

        # Solve
        try:
            from joblib import Parallel, delayed
            HAS_JOBLIB = True
        except ImportError:
            HAS_JOBLIB = False

        if HAS_JOBLIB and self.solver.n_jobs != 1:
            solutions = Parallel(n_jobs=self.solver.n_jobs, backend='loky')(
                delayed(self.solver._solve_single_ode)(y0, params, t_eval)
                for y0, params in solve_tasks
            )
        else:
            solutions = [
                self.solver._solve_single_ode(y0, params, t_eval)
                for y0, params in solve_tasks
            ]

        return solutions, t_eval

    def _compute_alpha_levels_from_ci_pertinence(
        self,
        solutions: List[np.ndarray],
        pertinences_CI: np.ndarray,
        alphas: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes α-levels using ONLY IC membership.

        ALGORITHM (very simple!):

        1. Sort solutions by membership: descending
        2. For each α:
           - Select solutions with μ(IC) ≥ α
           - Compute min/max
        """

        # Validate solutions
        valid_mask = np.array([
            not np.any(np.isnan(sol)) for sol in solutions
        ])

        valid_solutions = [sol for sol, valid in zip(solutions, valid_mask)
                           if valid]
        valid_pertinences = pertinences_CI[valid_mask]

        if len(valid_solutions) == 0:
            raise RuntimeError("No valid solutions")

        solutions_array = np.stack(valid_solutions, axis=0)
        n_points, n_vars, n_time = solutions_array.shape

        y_min_all = []
        y_max_all = []
        alphas_valid = []

        for alpha in alphas:
            # Select ONLY solutions with μ(IC) ≥ α
            alpha_mask = valid_pertinences >= (alpha - 1e-10)

            if np.sum(alpha_mask) == 0:
                # If no solution reaches this α, skip
                continue

            solutions_alpha = solutions_array[alpha_mask]

            y_min_alpha = np.min(solutions_alpha, axis=0)
            y_max_alpha = np.max(solutions_alpha, axis=0)

            y_min_all.append(y_min_alpha)
            y_max_all.append(y_max_alpha)
            alphas_valid.append(alpha)

        return np.stack(y_min_all), np.stack(y_max_all), np.array(alphas_valid)

    def solve_monte_carlo(self, verbose: bool = False):
        """
        Solves with the CORRECT Monte Carlo method.
        """

        if verbose:
            print("=" * 80)
            print("MONTE CARLO SOLVER + MEMBERSHIP (CORRECTED)")
            print("=" * 80)

        # 1. Sampling with memberships
        if verbose:
            print(f"\n⏳ Sampling {self.n_samples} points...")

        y0_samples, param_samples, pertinences_CI = (
            self._sample_hypercube_with_pertinence(
                self.n_samples,
                verbose=verbose  # ← PASS
            )
        )

        n_total = len(y0_samples)+len(param_samples)

        if verbose:
            print(f"✓ {n_total} sampled points (+ vertices)")
            print(f"  Memberships: min={np.min(pertinences_CI):.3f}, "
                  f"max={np.max(pertinences_CI):.3f}, "
                  f"mean={np.mean(pertinences_CI):.3f}")

        # 2. Solve ALL ODEs once
        if verbose:
            print(f"\n⏳ Solving {n_total} ODEs...")

        solutions, t_eval = self._solve_all_samples(y0_samples, param_samples)

        if verbose:
            print(f"✓ {n_total} ODEs solved")

        # 3. Compute α-levels (VERY fast!)
        if verbose:
            print(f"\n⏳ Computing α-levels...")

        alphas = self.solver._generate_alpha_levels()

        y_min_all, y_max_all, alphas_valid = (
            self._compute_alpha_levels_from_ci_pertinence(
                solutions, pertinences_CI, alphas
            )
        )

        if verbose:
            print(f"✓ {len(alphas_valid)} α-levels computed")
            print("\nSTATISTICS:")
            print(f"  Total ODEs solved: {n_total}")
            print(
                f"  Vs. standard method: 1/{self.solver.n_alpha_cuts} of cost")
            print("=" * 80)

        return FuzzySolution(
            t=t_eval,
            y_min=y_min_all,
            y_max=y_max_all,
            alphas=alphas_valid,
            var_names=self.solver.var_names
        )


class HierarchicalFuzzyODESolver:

    def __init__(self, base_solver):
        """
        Args:
            base_solver: FuzzyODESolver instance
        """
        self.solver = base_solver
        self.alpha_levels = self._generate_alpha_levels()

        # New attributes to store data
        self.solutions_per_alpha = {}   # alpha -> [solutions]
        self.pertinences_per_alpha = {}  # alpha -> [memberships]
        self.y0_grids_per_alpha = {}    # alpha -> IC grid
        self.t_eval = None              # Evaluation times

    def _generate_alpha_levels(self) -> np.ndarray:
        """Generates α-levels in DESCENDING order."""
        alphas = np.linspace(0, 1, self.solver.n_alpha_cuts)
        return alphas[::-1]  # [1.0, 0.9, ..., 0.1, 0.0]

    def _compute_ci_pertinence(
        self,
        y0: np.ndarray,
        params: Dict[str, float]
    ) -> float:
        """
        Computes membership of an initial condition.

        μ(IC) = min(μ_y0_1, μ_y0_2, ..., μ_r, μ_K, ...)
        (t-norm: minimum)
        """
        pertinence = 1.0

        # Membership in fuzzy initial conditions
        for i, y0_fuzzy_var in enumerate(self.solver.initial_condition):
            mu_y0 = y0_fuzzy_var.fuzzy_set.membership(y0[i])
            pertinence = min(pertinence, mu_y0)

        # Membership in fuzzy parameters
        for param_name, param_fuzzy in self.solver.params.items():
            if hasattr(param_fuzzy, 'fuzzy_set'):  # It's FuzzyNumber
                param_val = params.get(param_name, param_fuzzy)
                mu_param = param_fuzzy.fuzzy_set.membership(param_val)
                pertinence = min(pertinence, mu_param)

        return max(0.0, min(1.0, pertinence))

    def _solve_alpha_level_with_storage(
        self,
        alpha: float,
        verbose: bool = False
    ) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
        """
        Solves α-level storing:
        - Individual solutions for each point
        - Membership of each IC

        Returns:
            (t_eval, solutions, pertinences)
        """

        # Extract α-cuts
        y0_intervals, params_intervals = (
            self.solver._extract_alpha_cuts(alpha)
        )

        # Create grid
        y0_grid, params_grid = self.solver._create_grid(
            y0_intervals, params_intervals
        )

        # Define evaluation times (using first resolution)
        if self.t_eval is None:
            self.t_eval = np.linspace(
                self.solver.t_span[0],
                self.solver.t_span[1],
                100
            )

        # Solve ODEs with parallelization
        try:
            from joblib import Parallel, delayed
            HAS_JOBLIB = True
        except ImportError:
            HAS_JOBLIB = False

        if HAS_JOBLIB and self.solver.n_jobs != 1:
            solutions = Parallel(n_jobs=self.solver.n_jobs, backend='loky')(
                delayed(self.solver._solve_single_ode_with_t_eval)(
                    y0, params, self.t_eval
                )
                for y0, params in zip(y0_grid, params_grid)
            )
        else:
            solutions = [
                self.solver._solve_single_ode_with_t_eval(
                    y0, params, self.t_eval)
                for y0, params in zip(y0_grid, params_grid)
            ]

        # Compute IC memberships
        pertinences = []
        for y0, params in zip(y0_grid, params_grid):
            mu = self._compute_ci_pertinence(y0, params)
            pertinences.append(mu)

        pertinences = np.array(pertinences)

        if verbose:
            print(f"  α = {alpha:.3f}: {len(y0_grid)} ODEs, "
                  f"memberships ∈ [{np.min(pertinences):.3f}, "
                  f"{np.max(pertinences):.3f}]")

        return self.t_eval, solutions, pertinences

    def _compute_alpha_levels_with_filtering(
        self,
        alphas: np.ndarray,
        verbose: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Recomputes α-levels filtering by membership.

        CORRECT ALGORITHM:
        For each α:
            Select solutions with μ(IC) ≥ α
            Compute min/max of THESE

        Returns:
            (y_min_all, y_max_all) with shape (n_alpha, n_vars, n_time)
        """

        y_min_all = []
        y_max_all = []
        alphas_valid = []

        for alpha in alphas:
            # Get stored data for this α
            if alpha not in self.solutions_per_alpha:
                continue

            solutions = self.solutions_per_alpha[alpha]
            pertinences = self.pertinences_per_alpha[alpha]

            # CRITICAL FILTER: select by membership
            valid_mask = pertinences >= (alpha - 1e-10)

            if np.sum(valid_mask) == 0:
                # No solution with membership ≥ α
                # Use the one with highest membership
                max_pert = np.max(pertinences)
                valid_mask = pertinences >= (max_pert - 1e-10)

            # Select only valid solutions
            valid_solutions = [
                sol for sol, valid in zip(solutions, valid_mask)
                if valid and not np.any(np.isnan(sol))
            ]

            if len(valid_solutions) == 0:
                if verbose:
                    print(f"  ⚠ α = {alpha:.3f}: no valid solutions")
                continue

            # Compute envelopes ONLY from selected solutions
            solutions_array = np.stack(valid_solutions, axis=0)
            y_min = np.min(solutions_array, axis=0)
            y_max = np.max(solutions_array, axis=0)

            y_min_all.append(y_min)
            y_max_all.append(y_max)
            alphas_valid.append(alpha)

            if verbose:
                n_used = np.sum(valid_mask)
                n_total = len(valid_mask)
                print(f"  ✓ α = {alpha:.3f}: {n_used}/{n_total} solutions "
                      f"({100*n_used/n_total:.0f}%)")

        return np.stack(y_min_all), np.stack(y_max_all), np.array(alphas_valid)

    def solve_hierarchical(self, verbose: bool = True) -> 'FuzzySolution':
        """
        Solves with CORRECTED hierarchical method.

        Flow:
        1. Solve each α-level (in descending order)
        2. Store ALL individual solutions
        3. Store IC memberships
        4. Filter and recompute α-levels
        """

        if verbose:
            print("=" * 80)
            print("HIERARCHICAL SOLVER (CORRECTED)")
            print("=" * 80)
            print(f"Variables: {self.solver.n_vars}")
            print(f"α-levels: {len(self.alpha_levels)}")
            print(f"Method: Store individual solutions + membership filter")
            print("=" * 80 + "\n")

        # PHASE 1: Solve and store for each α
        if verbose:
            print("PHASE 1: Solving ODEs for each α-level...")

        for idx, alpha in enumerate(self.alpha_levels):
            if verbose and idx > 0:
                print()  # Blank line

            t, solutions, pertinences = self._solve_alpha_level_with_storage(
                alpha, verbose=verbose
            )

            # Store data
            self.solutions_per_alpha[alpha] = solutions
            self.pertinences_per_alpha[alpha] = pertinences

        # PHASE 2: Recompute α-levels with correct filter
        if verbose:
            print("\n" + "-" * 80)
            print("PHASE 2: Computing α-levels with membership filter...")
            print("-" * 80)

        y_min_all, y_max_all, alphas_valid = (
            self._compute_alpha_levels_with_filtering(
                self.alpha_levels, verbose=verbose
            )
        )

        if verbose:
            print("\n" + "=" * 80)
            print("✅ CORRECTED hierarchical solution computed!")
            print("=" * 80 + "\n")

        from fuzzy_ode import FuzzySolution

        return FuzzySolution(
            t=self.t_eval,
            y_min=y_min_all,
            y_max=y_max_all,
            alphas=alphas_valid,
            var_names=self.solver.var_names
        )


@dataclass
class AlphaGridPoint:
    """Point in grid associated with a fuzzy interval."""
    point: np.ndarray  # Coordinates (y0_1, y0_2, ..., y0_n)
    alpha_min: float   # Minimum α for which this point belongs to interval
    alpha_max: float   # Maximum α
    grid_index: int    # Index in Cartesian grid


# Success message
if HAS_JOBLIB:
    print("Automatic parallelization (joblib)")
else:
    print("Serial processing (install joblib for parallelization)")
