"""
Mamdani Fuzzy Rule Learning Module

This module implements automatic learning of fuzzy rules for Mamdani systems
using metaheuristics (SA, GA, PSO, DE).
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
import itertools


class EfficientMamdaniFIS:
    """
    Efficient Mamdani Fuzzy Inference System.
    Uses all functions and operators directly from the original FIS.
    """

    def __init__(self,
                 input_mf_functions: List[List[Callable]],
                 output_mf_functions: List[Callable],
                 input_ranges: List[Tuple[float, float]],
                 output_range: Tuple[float, float],
                 and_operator: Callable,
                 implication_function: Callable,
                 aggregation_function: Callable,
                 defuzzify_function: Callable,
                 n_output_points: int = 1000):
        """
        Parameters:
        - input_mf_functions: list of lists with input MF functions from FIS
        - output_mf_functions: list with output MF functions from FIS
        - input_ranges: list of tuples (min, max) for each input
        - output_range: tuple (min, max) for output
        - and_operator: AND operator function from FIS
        - implication_function: Implication method from FIS
        - aggregation_function: Aggregation method from FIS
        - defuzzify_function: Defuzzification function from FIS
        - n_output_points: number of points in output domain
        """
        self.input_mf_functions = input_mf_functions
        self.output_mf_functions = output_mf_functions
        self.input_ranges = input_ranges
        self.output_range = output_range
        self.and_op = and_operator
        self.implication = implication_function
        self.aggregation = aggregation_function
        self.defuzzify_function = defuzzify_function
        self.n_inputs = len(input_mf_functions)
        self.n_output_mfs = len(output_mf_functions)

        # PRE-COMPUTE output domain
        self.output_domain = np.linspace(output_range[0], output_range[1], n_output_points)
        self.n_output_points = n_output_points

        # PRE-COMPUTE output MFs on domain
        self.output_mfs_values = np.zeros((self.n_output_mfs, n_output_points))
        for i, mf_func in enumerate(self.output_mf_functions):
            self.output_mfs_values[i] = mf_func(self.output_domain)

    def _compute_activations(self, input_data):
        """Compute rule activations using AND operator from FIS."""
        n_samples = input_data.shape[0]
        membership_degrees = []

        for i in range(self.n_inputs):
            var_data = input_data[:, i]
            n_mfs = len(self.input_mf_functions[i])
            degrees = np.zeros((n_samples, n_mfs))

            for j, mf_func in enumerate(self.input_mf_functions[i]):
                degrees[:, j] = mf_func(var_data)

            membership_degrees.append(degrees)

        shapes = [md.shape[1] for md in membership_degrees]
        n_rules = np.prod(shapes)
        activations = np.zeros((n_samples, n_rules))

        rule_idx = 0
        for combo in np.ndindex(tuple(shapes)):
            activation = membership_degrees[0][:, combo[0]].copy()
            for i in range(1, self.n_inputs):
                activation = self.and_op(activation, membership_degrees[i][:, combo[i]])
            activations[:, rule_idx] = activation
            rule_idx += 1

        return activations

    def _aggregate_consequents(self, activations, consequent_indices):
        """Aggregate consequents using implication and aggregation from FIS."""
        n_samples = activations.shape[0]
        aggregated = np.zeros((n_samples, self.n_output_points))

        for rule_idx, consequent_idx in enumerate(consequent_indices):
            rule_activation = activations[:, rule_idx:rule_idx+1]
            output_mf_values = self.output_mfs_values[consequent_idx]
            implied = self.implication(rule_activation, output_mf_values)
            aggregated = self.aggregation(aggregated, implied)

        return aggregated

    def evaluate(self, input_data, consequent_indices):
        """Evaluate FIS for input data."""
        activations = self._compute_activations(input_data)
        aggregated = self._aggregate_consequents(activations, consequent_indices)
        crisp_outputs = np.array([
            self.defuzzify_function(self.output_domain, aggregated[i])
            for i in range(len(aggregated))
        ])
        return crisp_outputs

    def evaluate_with_precomputed(self, activations, consequent_indices):
        """Evaluate FIS using PRE-COMPUTED activations."""
        aggregated = self._aggregate_consequents(activations, consequent_indices)
        crisp_outputs = np.array([
            self.defuzzify_function(self.output_domain, aggregated[i])
            for i in range(len(aggregated))
        ])
        return crisp_outputs

    def get_rule_activations(self, input_data):
        """Return activations of all rules for analysis."""
        return self._compute_activations(input_data)


class MamdaniLearning:
    """
    Class for learning fuzzy rules in Mamdani systems.
    Takes a MamdaniSystem and learns optimal rules using metaheuristics.
    """

    def __init__(self,
                 fis,
                 num_points: int = 1000,
                 verbose: bool = True):
        """
        Initialize the Mamdani rule learning system.

        Parameters
        ----------
        fis : MamdaniSystem
            The fuzzy inference system to optimize
        num_points : int, default=1000
            Number of discretization points for output universe
        verbose : bool, default=True
            Print progress information
        """
        self.fis = fis
        self.num_points = num_points
        self.verbose = verbose

        # Internal state
        self.X_train = None
        self.y_train = None
        self.precomputed_activations = None

        # Extract variable information
        self.input_var_names = list(fis.input_variables.keys())
        self.output_var_names = list(fis.output_variables.keys())

        # Extract output terms
        self.output_terms = {}
        for out_var_name, out_var in fis.output_variables.items():
            self.output_terms[out_var_name] = list(out_var.terms.keys())

        self.n_rules = None
        self.n_output_mfs = len(self.output_terms[self.output_var_names[0]])

        # Optimization results
        self.efficient_fis = None
        self.best_rules = None
        self.best_cost = None
        self.optimization_history = None

    def fit_rules(self,
                  X_train: np.ndarray,
                  y_train: np.ndarray,
                  optimizer: str = 'sa',
                  optimizer_params: Optional[Dict] = None,
                  initial_solution_method: str = 'random') -> 'MamdaniLearning':
        """
        Learn fuzzy system rules from training data using metaheuristics.

        Parameters
        ----------
        X_train : np.ndarray, shape (n_samples, n_features)
            Training input data
        y_train : np.ndarray, shape (n_samples,)
            Training target values
        optimizer : str, default='sa'
            Optimization algorithm to use:
            - 'sa': Simulated Annealing
            - 'ga': Genetic Algorithm (BinaryGA)
            - 'pso': Particle Swarm Optimization (discrete)
            - 'de': Differential Evolution (discrete)
        optimizer_params : dict, optional
            Parameters for the optimizer. If None, uses defaults.

            For 'sa' (Simulated Annealing):
                - temperature_init: float = 100.0
                - temperature_min: float = 0.01
                - cooling_rate: float = 0.95
                - max_iterations: int = 5000
                - plateau_iterations: int = 1000
                - cooling_schedule: str = 'exponential'

            For 'ga' (Binary Genetic Algorithm):
                - pop_size: int = 100
                - max_gen: int = 500
                - elite_ratio: float = 0.15
                - crossover_rate: float = 0.8
                - crossover_type: str = 'uniform'
                - mutation_rate: float = 0.05
                - tournament_size: int = 5
                - adaptive_mutation: bool = True

            For 'pso' (Particle Swarm Optimization):
                - n_particles: int = 30
                - n_iterations: int = 100
                - w_max: float = 0.9
                - w_min: float = 0.4
                - c1: float = 1.49618
                - c2: float = 1.49618

            For 'de' (Differential Evolution):
                - pop_size: int = 50
                - max_iter: int = 100
                - F: float = 0.8
                - CR: float = 0.9

        initial_solution_method : str, default='random'
            Method for generating initial solution:
            - 'random': Random consequent indices
            - 'uniform': All rules start with middle MF
            - 'gradient': Data-driven initialization

        Returns
        -------
        self : MamdaniLearning
            Returns self for method chaining

        Examples
        --------
        >>> # Using Simulated Annealing (default)
        >>> learner = MamdaniLearning(fis)
        >>> learner.fit_rules(X_train, y_train)

        >>> # Using Genetic Algorithm with custom parameters
        >>> learner.fit_rules(
        ...     X_train, y_train,
        ...     optimizer='ga',
        ...     optimizer_params={
        ...         'pop_size': 150,
        ...         'max_gen': 300,
        ...         'mutation_rate': 0.08
        ...     }
        ... )

        >>> # Using PSO
        >>> learner.fit_rules(
        ...     X_train, y_train,
        ...     optimizer='pso',
        ...     optimizer_params={'n_particles': 50, 'n_iterations': 200}
        ... )
        """
        self.X_train = X_train
        self.y_train = y_train

        if self.verbose:
            print("\n" + "=" * 70)
            print("📚 MAMDANI FUZZY RULE LEARNING")
            print("=" * 70)
            print(f"Optimizer: {optimizer.upper()}")

        # Setup efficient FIS
        self._create_efficient_fis()
        self._create_rule_base_if_empty()

        if self.verbose:
            print(f"\nDataset size: {len(X_train)} samples")
            print("\nPre-computing rule activations...")

        # Pre-compute activations once
        self.precomputed_activations = self.efficient_fis._compute_activations(X_train)

        if self.verbose:
            print(f"Pre-computed activations: shape {self.precomputed_activations.shape}")
            print(f" - {self.precomputed_activations.shape[0]} samples")
            print(f" - {self.precomputed_activations.shape[1]} rules")
            print(f"\nInitialization method: {initial_solution_method}")

        # Set default parameters if not provided
        if optimizer_params is None:
            optimizer_params = {}

        # Optimize based on selected optimizer
        optimizer_lower = optimizer.lower()

        if optimizer_lower == 'sa':
            self._optimize_with_sa(optimizer_params, initial_solution_method)

        elif optimizer_lower == 'ga':
            self._optimize_with_ga(optimizer_params, initial_solution_method)

        elif optimizer_lower == 'pso':
            self._optimize_with_pso(optimizer_params, initial_solution_method)

        elif optimizer_lower == 'de':
            self._optimize_with_de(optimizer_params, initial_solution_method)

        else:
            raise ValueError(
                f"Unknown optimizer: '{optimizer}'. "
                f"Available: 'sa', 'ga', 'pso', 'de'"
            )

        # Update FIS with learned rules
        self._update_fis_rules(self.best_rules)

        if self.verbose:
            print("\n" + "=" * 70)
            print("✅ LEARNING COMPLETED")
            print("=" * 70)
            print(f"Optimized rule vector: {self.best_rules}")
            print(f"Final cost (RMSE): {self.best_cost:.6f}")
            print("=" * 70)

        return self

    # ==================== Optimizer-specific methods ====================

    def _optimize_with_sa(self, params: Dict, init_method: str):
        """Optimize using Simulated Annealing."""
        from .metaheuristics import SimulatedAnnealing

        # Default SA parameters
        sa_params = {
            'temperature_init': 100.0,
            'temperature_min': 0.01,
            'cooling_rate': 0.95,
            'max_iterations': 5000,
            'plateau_iterations': 1000,
            'cooling_schedule': 'exponential',
            'verbose': self.verbose
        }
        sa_params.update(params)

        optimizer = SimulatedAnnealing(
            cost_function=self._cost_function,
            neighbor_function=self._get_neighbor,
            initial_solution_function=lambda: self._generate_initial_solution(init_method),
            **sa_params
        )

        self.best_rules, self.best_cost, self.optimization_history = optimizer.optimize()

    def _optimize_with_ga(self, params: Dict, init_method: str):
        """Optimize using Binary Genetic Algorithm."""
        from .metaheuristics import BinaryGA

        # Default GA parameters
        ga_params = {
            'pop_size': 100,
            'max_gen': 500,
            'elite_ratio': 0.15,
            'crossover_rate': 0.8,
            'crossover_type': 'uniform',
            'mutation_rate': 0.05,
            'tournament_size': 5,
            'adaptive_mutation': True,
            'plateau_generations': 50,
            'mutation_boost_factor': 2.0
        }
        ga_params.update(params)

        # Extract verbose separately
        verbose = ga_params.pop('verbose', self.verbose)

        # Create optimizer
        optimizer = BinaryGA(**ga_params)

        # Define bounds for discrete problem
        bounds = np.array([[0, self.n_output_mfs - 1]] * self.n_rules)

        # Optimize
        self.best_rules, self.best_cost, history = optimizer.optimize(
            objective_func=self._cost_function,
            bounds=bounds,
            minimize=True,
            verbose=verbose
        )

        # Store history
        self.optimization_history = history

    def _optimize_with_pso(self, params: Dict, init_method: str):
        """Optimize using Particle Swarm Optimization (adapted for discrete)."""
        from .metaheuristics import PSO

        # Default PSO parameters
        pso_params = {
            'n_particles': 30,
            'n_iterations': 100,
            'w_max': 0.9,
            'w_min': 0.4,
            'c1': 1.49618,
            'c2': 1.49618
        }
        pso_params.update(params)

        optimizer = PSO(**pso_params)

        # Define bounds
        bounds = np.array([[0, self.n_output_mfs - 1]] * self.n_rules)

        # Wrapper to handle continuous -> discrete conversion
        def discrete_cost_function(x_continuous):
            x_discrete = np.round(x_continuous).astype(int)
            x_discrete = np.clip(x_discrete, 0, self.n_output_mfs - 1)
            return self._cost_function(x_discrete)

        # Optimize
        best_continuous, self.best_cost, history = optimizer.optimize(
            objective_func=discrete_cost_function,
            bounds=bounds,
            minimize=True,
            verbose=self.verbose
        )

        # Convert back to discrete
        self.best_rules = np.round(best_continuous).astype(int)
        self.best_rules = np.clip(self.best_rules, 0, self.n_output_mfs - 1)

        # Store history
        self.optimization_history = history

    def _optimize_with_de(self, params: Dict, init_method: str):
        """Optimize using Differential Evolution (adapted for discrete)."""
        from .metaheuristics import DE

        # Default DE parameters
        de_params = {
            'pop_size': 50,
            'max_iter': 100,
            'F': 0.8,
            'CR': 0.9
        }
        de_params.update(params)

        optimizer = DE(**de_params)

        # Define bounds
        bounds = np.array([[0, self.n_output_mfs - 1]] * self.n_rules)

        # Wrapper to handle continuous -> discrete conversion
        def discrete_cost_function(x_continuous):
            x_discrete = np.round(x_continuous).astype(int)
            x_discrete = np.clip(x_discrete, 0, self.n_output_mfs - 1)
            return self._cost_function(x_discrete)

        # Optimize
        best_continuous, self.best_cost, history = optimizer.optimize(
            objective_func=discrete_cost_function,
            bounds=bounds,
            minimize=True,
            verbose=self.verbose
        )

        # Convert back to discrete
        self.best_rules = np.round(best_continuous).astype(int)
        self.best_rules = np.clip(self.best_rules, 0, self.n_output_mfs - 1)

        # Store history
        self.optimization_history = history

    # ==================== Helper methods ====================

    def _create_efficient_fis(self):
        """Create EfficientMamdaniFIS using all functions and operators from FIS."""
        from ..core.defuzzification import defuzzify

        # Extract input MF functions
        input_mf_functions = []
        input_ranges = []

        for var_name in self.input_var_names:
            var = self.fis.input_variables[var_name]
            var_mf_functions = []
            for term_name in var.terms.keys():
                fuzzy_set = var.terms[term_name]
                var_mf_functions.append(fuzzy_set.membership)
            input_mf_functions.append(var_mf_functions)
            input_ranges.append((var.universe[0], var.universe[-1]))

        # Extract output MF functions
        out_var_name = self.output_var_names[0]
        out_var = self.fis.output_variables[out_var_name]
        output_mf_functions = []
        for term_name in self.output_terms[out_var_name]:
            fuzzy_set = out_var.terms[term_name]
            output_mf_functions.append(fuzzy_set.membership)

        output_range = (out_var.universe[0], out_var.universe[-1])

        # Get AND operator from inference_engine's fuzzy_op
        inference_engine = self.fis.inference_engine
        and_operator = inference_engine.fuzzy_op.AND

        # Implication function
        if inference_engine.implication_method == 'min':
            implication_func = np.minimum
        elif inference_engine.implication_method == 'product':
            implication_func = np.multiply
        else:
            implication_func = np.minimum

        # Aggregation function
        if inference_engine.aggregation_method == 'max':
            aggregation_func = np.maximum
        elif inference_engine.aggregation_method == 'sum':
            aggregation_func = lambda a, b: np.minimum(1.0, a + b)
        elif inference_engine.aggregation_method == 'probabilistic':
            aggregation_func = lambda a, b: a + b - a * b
        else:
            aggregation_func = np.maximum

        # Defuzzification
        defuzz_method = self.fis.defuzzification_method
        defuzzify_func = lambda x, mf: defuzzify(x, mf, defuzz_method)

        if self.verbose:
            print(f"\n🔧 Creating efficient FIS representation:")
            print(f" - Input variables: {len(self.input_var_names)}")
            print(f" - MFs per input: {[len(mfs) for mfs in input_mf_functions]}")
            print(f" - Output MFs: {len(output_mf_functions)}")
            print(f" - AND operator: {inference_engine.fuzzy_op.and_method}")
            print(f" - Implication: {inference_engine.implication_method}")
            print(f" - Aggregation: {inference_engine.aggregation_method}")
            print(f" - Defuzzification: {defuzz_method}")

        # Create EfficientMamdaniFIS
        self.efficient_fis = EfficientMamdaniFIS(
            input_mf_functions=input_mf_functions,
            output_mf_functions=output_mf_functions,
            input_ranges=input_ranges,
            output_range=output_range,
            and_operator=and_operator,
            implication_function=implication_func,
            aggregation_function=aggregation_func,
            defuzzify_function=defuzzify_func,
            n_output_points=self.num_points
        )

        self.n_rules = np.prod([len(mfs) for mfs in input_mf_functions])

        if self.verbose:
            print(f" - Total rules: {self.n_rules}")

    def _create_rule_base_if_empty(self):
        """Create complete rule base if empty."""
        if len(self.fis.rule_base.rules) > 0:
            return

        input_terms_lists = []
        for var_name in self.input_var_names:
            terms = list(self.fis.input_variables[var_name].terms.keys())
            input_terms_lists.append(terms)

        all_combinations = list(itertools.product(*input_terms_lists))
        out_var_name = self.output_var_names[0]
        default_consequent = self.output_terms[out_var_name][0]

        if self.verbose:
            print(f"\n⚠️  Rule base is empty - creating from input combinations...")
            print(f" - Terms per variable: {[len(terms) for terms in input_terms_lists]}")
            print(f" - Total rules to create: {len(all_combinations)}")

        from ..inference.rules import FuzzyRule

        for combination in all_combinations:
            antecedent = {self.input_var_names[i]: combination[i]
                         for i in range(len(self.input_var_names))}
            consequent = {out_var_name: default_consequent}
            rule = FuzzyRule(antecedent, consequent)
            self.fis.rule_base.add_rule(rule)

        if self.verbose:
            print(f"✅ Created {len(all_combinations)} rules")

    def _cost_function(self, consequent_indices: np.ndarray) -> float:
        """Cost function: RMSE."""
        y_pred = self.efficient_fis.evaluate_with_precomputed(
            self.precomputed_activations,
            consequent_indices
        )
        rmse = np.sqrt(np.mean((y_pred - self.y_train) ** 2))
        return rmse

    def _generate_initial_solution(self, method: str = 'random') -> np.ndarray:
        """Generate initial solution."""
        if method == 'random':
            return np.random.randint(0, self.n_output_mfs, size=self.n_rules)

        elif method == 'uniform':
            mid_mf = self.n_output_mfs // 2
            return np.full(self.n_rules, mid_mf)

        elif method == 'gradient':
            return self._gradient_init()

        else:
            raise ValueError(f"Unknown method: {method}")

    def _gradient_init(self) -> np.ndarray:
        """Gradient-based initialization."""
        consequents = np.zeros(self.n_rules, dtype=int)
        output_min, output_max = self.efficient_fis.output_range

        for rule_idx in range(self.n_rules):
            mask = self.precomputed_activations[:, rule_idx] > 0.1

            if mask.sum() > 0:
                avg_output = self.y_train[mask].mean()
                normalized = (avg_output - output_min) / (output_max - output_min)
                mf_idx = int(normalized * (self.n_output_mfs - 1))
                consequents[rule_idx] = np.clip(mf_idx, 0, self.n_output_mfs - 1)
            else:
                consequents[rule_idx] = np.random.randint(0, self.n_output_mfs)

        return consequents

    def _get_neighbor(self, solution: np.ndarray) -> np.ndarray:
        """Generate neighbor (for SA)."""
        neighbor = solution.copy()
        n_modifications = np.random.randint(1, min(4, self.n_rules + 1))

        for _ in range(n_modifications):
            rule_idx = np.random.randint(0, self.n_rules)
            current_mf = neighbor[rule_idx]
            available_mfs = [i for i in range(self.n_output_mfs) if i != current_mf]

            if available_mfs:
                neighbor[rule_idx] = np.random.choice(available_mfs)

        return neighbor

    def _update_fis_rules(self, consequent_indices: np.ndarray):
        """Update FIS rules with learned consequents."""
        out_var_name = self.output_var_names[0]
        terms = self.output_terms[out_var_name]

        for rule_idx, rule in enumerate(self.fis.rule_base.rules):
            term_idx = consequent_indices[rule_idx]
            term_name = terms[term_idx]
            rule.consequent[out_var_name] = term_name

    # ==================== Public API ====================

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.best_rules is None:
            raise ValueError("Rules have not been learned yet. Run fit_rules() first.")
        return self.efficient_fis.evaluate(X, self.best_rules)

    def get_rules(self) -> np.ndarray:
        """Return best learned rules."""
        if self.best_rules is None:
            raise ValueError("Rules have not been learned yet. Run fit_rules() first.")
        return self.best_rules

    def get_cost(self) -> float:
        """Return best cost achieved."""
        if self.best_cost is None:
            raise ValueError("Rules have not been learned yet. Run fit_rules() first.")
        return self.best_cost

    def get_history(self) -> Dict:
        """Return optimization history."""
        if self.optimization_history is None:
            raise ValueError("Rules have not been learned yet. Run fit_rules() first.")
        return self.optimization_history

    def score(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """Calculate RMSE on test set."""
        y_pred = self.predict(X_test)
        rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
        return rmse
