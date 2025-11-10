"""
ANFIS - Adaptive Neuro-Fuzzy Inference System
==============================================

Implementstion Python complete do ANFIS com:
- Learning hybrid (LSE + Gradient Descent)
- Multiple membership functions (gaussian, generalized bell, sigmoid)
- Regularization L1/L2 (Lasso, Ridge, Elastic Net)
- Minibatch training for large datasets
- Early stopping e validation
- Detailed metrics de training
- Constraints de domain adaptive

References:
    Jang, J. S. (1993). "ANFIS: adaptive-network-based fuzzy inference system."
    IEEE transactions on systems, man, and cybernetics, 23(3), 665-685.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Union, Callable
import warnings
import time
import itertools

from ..core.membership import gaussian, generalized_bell, sigmoid


class ANFIS:
    """
    ANFIS - Adaptive Neuro-Fuzzy Inference System

    Sistema hybrid que combines neural networks e fuzzy logic for
    supervised learning. Implements:

    - Architecture de 5 layers (fuzzification, rules, normalization,
      consequents, aggregation)
    - Learning hybrid: LSE for formeters consequents + gradiente
      descendente for formeters premises
    - Regularization L1/L2 for avoid overfitting
    - Minibatch training for computational efficiency
    - Early stopping based on validation

    Example:
        >>> import numpy as np
        >>> from fuzzy_systems.learning.anfis_moderno import ANFIS
        >>>
        >>> # Training data
        >>> X_train = np.random.uniform(-3, 3, (100, 2))
        >>> y_train = np.sin(X_train[:, 0]) + np.cos(X_train[:, 1])
        >>>
        >>> # Create and train ANFIS
        >>> anfis = ANFIS(n_inputs=2, n_mfs=[3, 3], mf_type='gaussmf',
        ...               learning_rate=0.01, lambda_l2=0.01, batch_size=32)
        >>> anfis.fit(X_train, y_train, epochs=100, verbose=True)
        >>>
        >>> # Prediction
        >>> y_pred = anfis.predict(X_train)
        >>>
        >>> # Visualize MFs e convergessnce
        >>> anfis.visualizar_mfs()
        >>> anfis.metricas.plotar_convergesncia()
    """

    def __init__(self,
             n_inputs: int,
             n_mfs: Union[int, List[int]],
             mf_type: str = 'gaussmf',
             learning_rate: float = 0.01,
             input_ranges: Optional[List[Tuple[float, float]]] = None,
             lambda_l1: float = 0.0,
             lambda_l2: float = 0.0,
             batch_size: Optional[int] = None,
             use_adaptive_lr: bool = False,
             classification: bool = False):
        """
        Initializes the ANFIS with regularization and minibatch training.

        Parameters:
            n_inputs: Number of input variables
            n_mfs: Number of membership functions per input.
                Can be int (same number for all) or list of ints
            mf_type: Type of membership function:
                    'gaussmf' - Gaussian (default)
                    'gbellmf' - Generalized bell
                    'sigmf' - Sigmoid
            learning_rate: Learning rate for gradient descent
            input_ranges: List with ranges (min, max) of each input.
                        If None, uses (-8, 8) for all
            lambda_l1: L1 regularization coefficient (Lasso) on MF widths
            lambda_l2: L2 regularization coefficient (Ridge) on MF widths
            batch_size: Batch size for minibatch training
                    - None: Batch gradient descent (uses all data)
                    - 1: Stochastic gradient descent
                    - 16-128: Minibatch gradient descent (recommended)
            use_adaptive_lr: If True, uses adaptive learning rate based
                            on Lyapunov. Guarantees theoretical convergence but may
                            be slower. If False, uses fixed learning_rate.
            classification: If True, configures ANFIS for classification tasks

        Note on regularization:
            Regularization is applied ONLY on the widths (sigmas) of membership
            functions, not on centers. This makes sense because:
            - Centers should adapt freely to data position
            - Widths should be regularized to avoid overfitting

        Note on Lyapunov:
            When use_adaptive_lr=True, the learning rate is calculated
            dynamically to guarantee stability: η = 1.99/||∇E||²
            This guarantees theoretical convergence according to Lyapunov theory.
        """
        self.n_inputs = n_inputs
        
        # Allow n_mfs as int or list
        if isinstance(n_mfs, int):
            self.n_mfs = [n_mfs] * n_inputs
        else:
            if len(n_mfs) != n_inputs:
                raise ValueError(f"n_mfs must have {n_inputs} elements")
            self.n_mfs = list(n_mfs)
        
        self.mf_type = mf_type
        self.lr = learning_rate
        self.n_rules = int(np.prod(self.n_mfs))
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.batch_size = batch_size
        self.use_adaptive_lr = use_adaptive_lr
        self.classification = classification
        self._reg = (self.lambda_l1 > 0) or (self.lambda_l2 > 0)
        # Validate MF type
        valid_types = ['gaussmf', 'gbellmf', 'sigmf']
        if mf_type not in valid_types:
            raise ValueError(f"mf_type must be one of {valid_types}")
        
        # Define ranges
        if input_ranges is None:
            self.input_ranges = [(-8.0, 8.0)] * n_inputs
        else:
            if len(input_ranges) != n_inputs:
                raise ValueError(f"input_ranges must have {n_inputs} elements")
            self.input_ranges = input_ranges
        
        # Initialize parameters (will be done in fit with real data)
        self.mf_params = None
        self.consequent_params = None
        self.input_bounds = None
        self._rule_indices_cache = None
        
        # Classification attributes
        self.classes_ = None
        self.n_classes_ = None
        
        # Training history with regression and classification metrics
        self.history = {
            'train': {
                'loss': [],
                'rmse': [],
                'mae': [],
                'max_error': [],
                'r2': [],
                'mape': [],
                'accuracy': [],
                'precision': [],
                'recall': [],
                'f1_score': []
            },
            'val': {
                'loss': [],
                'rmse': [],
                'mae': [],
                'max_error': [],
                'r2': [],
                'mape': [],
                'accuracy': [],
                'precision': [],
                'recall': [],
                'f1_score': []
            },
            'epoch_times': [],
            'gradient_norms': [],
            'learning_rates': []
        }
        
        # Cache of rule indices
        self._rule_indices_cache = None
        
        # Penalty history
        self.l1_history = []
        self.l2_history = []
        self.total_cost_history = []

    def _initialize_premise_params(self, X: np.ndarray):
        """
        Initializes parameters of the membership functions based on data.
        
        Parameters:
            X: Input data (n_samples, n_inputs)
        """
        # Calculate actual bounds from data
        self.input_bounds = np.array([(X[:, i].min(), X[:, i].max()) 
                                    for i in range(self.n_inputs)])
        
        self.mf_params = []
        
        for i in range(self.n_inputs):
            x_min, x_max = self.input_bounds[i]
            x_range = x_max - x_min
            
            # Guarantee minimum range
            if x_range < 1e-6:
                x_range = 1.0
                x_min = x_min - 0.5
                x_max = x_max + 0.5
            
            mf_params = []
            n_mf = self.n_mfs[i]
            centers = np.linspace(x_min, x_max, n_mf)
            
            for j in range(n_mf):
                if self.mf_type == 'gaussmf':
                    # Parameters: mean, sigma
                    sigma = x_range / (2 * n_mf)
                    center = centers[j] + np.random.uniform(-0.1, 0.1)
                    params = np.array([center, sigma])
                    
                elif self.mf_type == 'gbellmf':
                    # Parameters: a (width), b (slope), c (center)
                    width = x_range / (2 * n_mf)
                    center = centers[j] + np.random.uniform(-0.1, 0.1)
                    params = np.array([width, 2.0, center])
                    
                elif self.mf_type == 'sigmf':
                    # Parameters: a (slope), c (center)
                    slope = 4.0 / (x_range / n_mf)
                    center = centers[j] + np.random.uniform(-0.1, 0.1) * (x_range / n_mf)
                    params = np.array([slope, center])
                
                # Add small random perturbation
                mf_params.append(params)
            
            self.mf_params.append(np.array(mf_params))
            if self.consequent_params is None:
                # Initialize with small random values
                self.consequent_params = np.random.randn(self.n_rules, self.n_inputs + 1) * 0.01
            self._rule_indices_cache = list(itertools.product(*[range(n) for n in self.n_mfs]))
            self.n_rules = len(self._rule_indices_cache)


    def _apply_domain_constraints(self):
        """
        Applies domain constraints to premise parameters to guarantee
        they remain within valid ranges.
        """
        for i in range(self.n_inputs):
            x_min, x_max = self.input_bounds[i]
            
            for j in range(self.n_mfs[i]):
                if self.mf_type == 'gaussmf':
                    # mean, sigma
                    # Clip center to input range
                    self.mf_params[i][j, 0] = np.clip(self.mf_params[i][j, 0], x_min, x_max)
                    # Guarantee positive sigma
                    self.mf_params[i][j, 1] = np.maximum(self.mf_params[i][j, 1], 1e-6)
                    
                elif self.mf_type == 'gbellmf':
                    # a, b, c
                    # Clip center to input range
                    self.mf_params[i][j, 2] = np.clip(self.mf_params[i][j, 2], x_min, x_max)
                    # Guarantee positive width (a)
                    self.mf_params[i][j, 0] = np.maximum(self.mf_params[i][j, 0], 1e-6)
                    # Guarantee positive slope (b)
                    self.mf_params[i][j, 1] = np.maximum(self.mf_params[i][j, 1], 0.1)
                    
                elif self.mf_type == 'sigmf':
                    # a, c
                    # Clip center to input range
                    self.mf_params[i][j, 1] = np.clip(self.mf_params[i][j, 1], x_min, x_max)


    def _generate_rule_indices(self) -> List[Tuple[int, ...]]:
        """
        Generates rule indices as cartesian product of membership functions.
        
        Returns:
            List of tuples with MF indices for each rule
        """
        # Special case: single input
        if self.n_inputs == 1:
            return [(i,) for i in range(self.n_mfs[0])]
        
        # Generate all combinations (cartesian product)
        indexes = list(itertools.product(*[range(n) for n in self.n_mfs]))
        
        return indexes


    def _create_batches(self, X: np.ndarray, y: np.ndarray, 
                   batch_size: int, shuffle: bool = True) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Splits data into batches for minibatch training.
        
        Parameters:
            X: Input data
            y: Output data
            batch_size: Batch size
            shuffle: If True, shuffles the data before splitting
            
        Returns:
            List of tuples (X_batch, y_batch)
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        if shuffle:
            np.random.shuffle(indices)
        
        batches = []
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            batches.append((X[batch_indices], y[batch_indices]))
        
        return batches


    def _eval_mf(self, x: float, params: np.ndarray) -> float:
        """
        Evaluates membership function for a given input value.
        
        Parameters:
            x: Input value
            params: Parameters of the MF
            
        Returns:
            Membership degree
        """
        if self.mf_type == 'gaussmf':
            return gaussian(x, tuple(params))
        elif self.mf_type == 'gbellmf':
            return generalized_bell(x, tuple(params))
        elif self.mf_type == 'sigmf':
            return sigmoid(x, tuple(params))


    def layer1_fuzzification(self, X: np.ndarray) -> List[np.ndarray]:
        """
        Layer 1: Fuzzification - calculates membership degrees.
        
        Parameters:
            X: Input vector (n_inputs,)
            
        Returns:
            List with membership degrees for each input
        """
        mu = []
        for i in range(self.n_inputs):
            mu_i = np.array([self._eval_mf(X[i], params) 
                            for params in self.mf_params[i]])
            mu.append(mu_i)
        
        return mu


    def layer2_rules(self, mu: List[np.ndarray]) -> np.ndarray:
        """
        Layer 2: Firing strength of rules - product of MFs.
        
        Parameters:
            mu: List with membership degrees
            
        Returns:
            Array with firing strength of each rule
        """
        w = np.zeros(self.n_rules)
        
        for rule_idx, mf_indices in enumerate(self._rule_indices_cache):
            w[rule_idx] = np.prod([mu[i][mf_idx] for i, mf_idx in enumerate(mf_indices)])
        
        return w


    def layer3_normalization(self, w: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Layer 3: Normalization of firing strengths.
        
        Parameters:
            w: Firing strengths
            
        Returns:
            Tuple (w_norm, sum_w)
        """
        sum_w = np.sum(w) + 1e-10
        w_norm = w / sum_w
        
        return w_norm, sum_w


    def layer4_consequents(self, X: np.ndarray, w_norm: np.ndarray) -> np.ndarray:
        """
        Layer 4: Calculates outputs of consequents (Takagi-Sugeno).
        
        Parameters:
            X: Input vector
            w_norm: Normalized firing strengths
            
        Returns:
            Outputs of each rule
        """
        outputs = np.zeros(self.n_rules)
        
        for i in range(self.n_rules):
            params = self.consequent_params[i]
            # f_i = p0 + p1*x1 + p2*x2 + ...
            f_i = params[0] + np.dot(params[1:], X)
            outputs[i] = w_norm[i] * f_i
        
        return outputs


    def layer5_aggregation(self, outputs: np.ndarray) -> float:
        """
        Layer 5: Final aggregation - sum of outputs.
        
        Parameters:
            outputs: Outputs of each rule
            
        Returns:
            Final output of ANFIS
        """
        return np.sum(outputs)


    def forward(self, X: np.ndarray) -> Tuple:
        """
        Complete forward propagation through the 5 layers.
        
        Parameters:
            X: Input vector (n_inputs,)
            
        Returns:
            Tuple (y_pred, cache) where cache contains intermediate values
        """
        # Layer 1: Fuzzification
        mu = self.layer1_fuzzification(X)
        
        # Layer 2: Rules
        w = self.layer2_rules(mu)
        
        # Layer 3: Normalization
        w_norm, sum_w = self.layer3_normalization(w)
        
        # Layer 4: Consequents
        outputs = self.layer4_consequents(X, w_norm)
        
        # Layer 5: Aggregation
        y_pred = self.layer5_aggregation(outputs)
        
        return y_pred, mu, w, w_norm, sum_w


    def forward_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Vectorized forward propagation for multiple samples.
        
        Parameters:
            X: Input data (n_samples, n_inputs)
            
        Returns:
            Array with predictions (n_samples,)
        """
        n_samples = X.shape[0]
        
        # Layer 1: Fuzzification (vectorized)
        # mu[i][j,k] = membership of sample k in MF j of input i
        mu_batch = []
        for i in range(self.n_inputs):
            # Apply all MFs of input i to all samples
            mu_i = np.array([self._eval_mf(X[:, i], params) 
                            for params in self.mf_params[i]])  # (n_mfs[i], n_samples)
            mu_batch.append(mu_i.T)  # (n_samples, n_mfs[i])
        
        # Layer 2: Rules (vectorized)
        # w[k,j] = firing strength of rule j for sample k
        w_batch = np.ones((n_samples, self.n_rules))
        for rule_idx, mf_indices in enumerate(self._rule_indices_cache):
            for input_idx, mf_idx in enumerate(mf_indices):
                w_batch[:, rule_idx] *= mu_batch[input_idx][:, mf_idx]
        
        # Layer 3: Normalization (vectorized)
        sum_w_batch = np.sum(w_batch, axis=1, keepdims=True) + 1e-10  # (n_samples, 1)
        w_norm_batch = w_batch / sum_w_batch  # (n_samples, n_rules)
        
        # Layer 4: Consequents (vectorized)
        # f_i = p0 + p1*x1 + p2*x2 + ... for each rule
        X_extended = np.hstack([np.ones((n_samples, 1)), X])  # (n_samples, n_inputs+1)
        f_batch = X_extended @ self.consequent_params.T  # (n_samples, n_rules)
        outputs_batch = w_norm_batch * f_batch  # (n_samples, n_rules)
        
        # Layer 5: Aggregation (vectorized)
        y_pred_batch = np.sum(outputs_batch, axis=1)  # (n_samples,)
        
        return y_pred_batch


    def _calculate_l1_penalty(self) -> float:
        """
        Calculates L1 penalty (Lasso) on MF widths.
        
        Applies regularization ONLY on the widths (sigmas), not on centers.
        
        Returns:
            L1 penalty value
        """
        penalty = 0.0
        
        for input_idx in range(self.n_inputs):
            for mf_idx in range(self.n_mfs[input_idx]):
                params = self.mf_params[input_idx][mf_idx]
                
                if self.mf_type == 'gaussmf':
                    # params = [center, sigma] → regularize only sigma
                    sigma = params[1]
                    penalty += np.abs(sigma)
                    
                elif self.mf_type == 'gbellmf':
                    # params = [a, b, c] → regularize a (width) and b (slope)
                    a, b = params[0], params[1]
                    penalty += np.abs(a) + np.abs(b)
                    
                elif self.mf_type == 'sigmf':
                    # params = [a, c] → regularize a (slope)
                    a = params[0]
                    penalty += np.abs(a)
        
        return penalty


    def _calculate_l2_penalty(self) -> float:
        """
        Calculates L2 penalty (Ridge) on MF widths.
        
        Applies regularization ONLY on the widths (sigmas), not on centers.
        
        Returns:
            L2 penalty value
        """
        penalty = 0.0
        
        for input_idx in range(self.n_inputs):
            for mf_idx in range(self.n_mfs[input_idx]):
                params = self.mf_params[input_idx][mf_idx]
                
                if self.mf_type == 'gaussmf':
                    # params = [center, sigma] → regularize only sigma
                    sigma = params[1]
                    penalty += sigma ** 2
                    
                elif self.mf_type == 'gbellmf':
                    # params = [a, b, c] → regularize a (width) and b (slope)
                    a, b = params[0], params[1]
                    penalty += a ** 2 + b ** 2
                    
                elif self.mf_type == 'sigmf':
                    # params = [a, c] → regularize a (slope)
                    a = params[0]
                    penalty += a ** 2
        
        return penalty


    def _calculate_total_cost(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculates total cost: MSE + regularization penalties.
        
        Parameters:
            X: Input data
            y: Output data
            
        Returns:
            Tuple (mse, l1_penalty, l2_penalty)
        """
        y_pred = self.predict(X, score=0)
        mse = np.mean((y - y_pred) ** 2)
        
        l1_penalty = self._calculate_l1_penalty()
        l2_penalty = self._calculate_l2_penalty()
        
        return mse, l1_penalty, l2_penalty


    def _adjust_consequents_least_squares(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Adjusts consequent parameters using Least Squares (hybrid method).
        
        This is the most efficient method to adjust the consequents,
        calculating the optimal analytical solution.
        
        Parameters:
            X: Input data (n_samples, n_inputs)
            y: Output data (n_samples,)
            
        Returns:
            RMSE after adjustment
        """
        n_samples = len(X)
        n_params = self.n_rules * (self.n_inputs + 1)
        
        # Build matrix A of the linear system
        A = np.zeros((n_samples, n_params))
        y_target = y.copy()
        
        for i in range(n_samples):
            _, mu, w, w_norm, sum_w = self.forward(X[i])
            
            for j in range(self.n_rules):
                start_idx = j * (self.n_inputs + 1)
                end_idx = start_idx + (self.n_inputs + 1)
                
                # [w_norm_j, w_norm_j*x1, w_norm_j*x2, ...]
                A[i, start_idx:end_idx] = w_norm[j] * np.concatenate([[1], X[i]])
        
        # Solve with Tikhonov regularization for numerical stability
        try:
            lambda_reg = 1e-6
            ATA = A.T @ A + lambda_reg * np.eye(A.shape[1])
            ATy = A.T @ y_target
            p_flat = np.linalg.solve(ATA, ATy)
            
            self.consequent_params = p_flat.reshape(self.n_rules, self.n_inputs + 1)
            
        except np.linalg.LinAlgError:
            warnings.warn("Error solving linear system for consequents")
        
        # Calculate RMSE
        y_pred = self.predict(X, score=0)
        rmse = np.sqrt(np.mean((y_target - y_pred) ** 2))
        
        return rmse


    def _gradient_l1(self, parametro: float) -> float:
        """Subgradient penalty L1."""
        if parametro > 0:
            return 1.0
        elif parametro < 0:
            return -1.0
        else:
            return 0.0

    def _gradient_l2(self, parametro: float) -> float:
        """Gradient penalty L2."""
        return 2.0 * parametro

    def _compute_adaptive_learning_rate(self, gradient: np.ndarray, max_lr: float = 0.01) -> float:
        """
        Calculates rate de learning adaptive based em stability de Lyapunov.

        A theory de stability de Lyapunov guarantees that o algorithm converges
        se a rate de learning satisfies: 0 < η < 2/||∇E||²

        For guarantee stability, usesmos: η = 1.99/||∇E||²
        limited por um value maximum for avoidsr passos muito grandes.

        Formeters:
            gradient: Vector de gradients
            max_lr: Rate de learning máxima allowed

        Returns:
            Rate de learning adaptive que garante stability

        Referência:
            Jang, J. S. (1993). "ANFIS: adaptive-network-based fuzzy inference system."
            IEEE transactions on systems, man, and cybernetics, 23(3), 665-685.
        """
        grad_norm_squared = np.sum(gradient ** 2)

        if grad_norm_squared < 1e-10:
            # Gradient very small → usesr lr maximum
            return max_lr

        # Criterion de stability de Lyapunov: η < 2/||∇E||²
        # Usamos 1.99 for margin de safety
        stable_lr = 1.99 / grad_norm_squared

        # Limit by maximum specified
        return min(stable_lr, max_lr)


    def _adjust_premises_gradient(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """
        Adjusts premise parameters using gradient descent with regularization.
        
        Calculates gradients analytically and applies L1/L2 regularization.
        
        Parameters:
            X: Input data (n_samples, n_inputs)
            y: Output data (n_samples,)
            
        Returns:
            Tuple (total gradient norm, effective learning rate)
        """
        n_samples = len(X)
        
        # Gradient accumulators
        grad_accumulated = []
        for input_idx in range(self.n_inputs):
            grad_accumulated.append([])
            for mf_idx in range(self.n_mfs[input_idx]):
                n_params = len(self.mf_params[input_idx][mf_idx])
                grad_accumulated[input_idx].append(np.zeros(n_params))
        
        # Accumulate gradients from all samples in the batch
        for sample_idx in range(n_samples):
            X_sample = X[sample_idx]
            target = y[sample_idx]
            
            # Forward pass
            y_pred, mu, w, w_norm, sum_w = self.forward(X_sample)
            error = target - y_pred
            
            # Consequent contributions
            contributions = np.zeros(self.n_rules)
            for j in range(self.n_rules):
                params = self.consequent_params[j]
                contributions[j] = params[0] + np.dot(params[1:], X_sample)
            
            # Calculate gradients for each MF
            for input_idx in range(self.n_inputs):
                x_val = X_sample[input_idx]
                
                for rule_idx, mf_indices in enumerate(self._rule_indices_cache):
                    mf_idx = mf_indices[input_idx]
                    params = self.mf_params[input_idx][mf_idx]
                    mu_val = mu[input_idx][mf_idx]
                    
                    # Common chain rule components
                    dw_dmu = w[rule_idx] / (mu_val + 1e-10)
                    dy_dwn = contributions[rule_idx]
                    dwn_dw = (1 - w_norm[rule_idx]) / sum_w if sum_w > 1e-10 else 0
                    
                    chain_common = -error * dy_dwn * dwn_dw * dw_dmu
                    
                    # Gradients depend on MF type
                    if self.mf_type == 'gaussmf':
                        # Parameters: [center, sigma]
                        center, sigma = params
                        dmu_dc = mu_val * (x_val - center) / (sigma ** 2)
                        dmu_ds = mu_val * ((x_val - center) ** 2) / (sigma ** 3)
                        
                        grad_accumulated[input_idx][mf_idx][0] += chain_common * dmu_dc
                        grad_accumulated[input_idx][mf_idx][1] += chain_common * dmu_ds
                        
                    elif self.mf_type == 'gbellmf':
                        # Parameters: [a (width), b (slope), c (center)]
                        a, b, c = params
                        diff = x_val - c
                        abs_ratio = np.abs(diff / (a + 1e-10))
                        denominator = 1 + abs_ratio ** (2 * b)
                        
                        if abs_ratio > 1e-10:  # Avoid division by zero
                            # Gradient w.r.t. a
                            dmu_da = 2 * b * (mu_val ** 2) * (abs_ratio ** (2 * b)) / (a + 1e-10)
                            
                            # Gradient w.r.t. b
                            log_ratio = np.log(abs_ratio + 1e-10)
                            dmu_db = -2 * (mu_val ** 2) * (abs_ratio ** (2 * b)) * log_ratio
                            
                            # Gradient w.r.t. c
                            sign_diff = np.sign(diff)
                            dmu_dc = -2 * b * (mu_val ** 2) * (abs_ratio ** (2 * b)) * sign_diff / (a + 1e-10)
                            
                            grad_accumulated[input_idx][mf_idx][0] += chain_common * dmu_da
                            grad_accumulated[input_idx][mf_idx][1] += chain_common * dmu_db
                            grad_accumulated[input_idx][mf_idx][2] += chain_common * dmu_dc
                        
                    elif self.mf_type == 'sigmf':
                        # Parameters: [a (slope), c (center)]
                        a, c = params
                        
                        # Gradient w.r.t. a
                        dmu_da = mu_val * (1 - mu_val) * (x_val - c)
                        
                        # Gradient w.r.t. c
                        dmu_dc = -a * mu_val * (1 - mu_val)
                        
                        grad_accumulated[input_idx][mf_idx][0] += chain_common * dmu_da
                        grad_accumulated[input_idx][mf_idx][1] += chain_common * dmu_dc
        
        # Update parameters using average gradients from batch
        grad_norm_total = 0
        all_gradients = []
        
        for input_idx in range(self.n_inputs):
            for mf_idx in range(self.n_mfs[input_idx]):
                params = self.mf_params[input_idx][mf_idx]
                n_params = len(params)
                
                for param_idx in range(n_params):
                    param_val = params[param_idx]
                    
                    # Average gradient from MSE
                    grad_mse = grad_accumulated[input_idx][mf_idx][param_idx] / n_samples
                    
                    # Determine if it's a width parameter (should be regularized)
                    is_width = False
                    if self.mf_type == 'gaussmf':
                        # params = [center, sigma] → regularize only sigma
                        is_width = (param_idx == 1)
                    elif self.mf_type == 'gbellmf':
                        # params = [a, b, c] → regularize a and b
                        is_width = (param_idx in [0, 1])
                    elif self.mf_type == 'sigmf':
                        # params = [a, c] → regularize a
                        is_width = (param_idx == 0)
                    
                    # Apply regularization ONLY on widths
                    if is_width and self._reg:
                        grad_l1 = self.lambda_l1 * self._gradient_l1(param_val)
                        grad_l2 = self.lambda_l2 * self._gradient_l2(param_val)
                        grad_total = grad_mse + grad_l1 + grad_l2
                    else:
                        # Centers without regularization
                        grad_total = grad_mse
                    
                    all_gradients.append(grad_total)
                    grad_norm_total += grad_total ** 2
        
        # Calculate adaptive learning rate based on Lyapunov or use fixed
        if self.use_adaptive_lr:
            # Adaptive rate based on Lyapunov
            lr_effective = self._compute_adaptive_learning_rate(
                np.array(all_gradients), 
                max_lr=self.lr
            )
        else:
            # Fixed learning rate
            lr_effective = self.lr
        
        # Apply update with calculated learning rate
        grad_idx = 0
        for input_idx in range(self.n_inputs):
            for mf_idx in range(self.n_mfs[input_idx]):
                n_params = len(self.mf_params[input_idx][mf_idx])
                for param_idx in range(n_params):
                    grad_total = all_gradients[grad_idx]
                    self.mf_params[input_idx][mf_idx][param_idx] -= lr_effective * grad_total
                    grad_idx += 1
        
        # Apply domain constraints
        self._apply_domain_constraints()
        
        return np.sqrt(grad_norm_total), lr_effective


    def _validate_input(self, X: np.ndarray, y: Optional[np.ndarray] = None, 
                   name_X: str = 'X', name_y: str = 'y') -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Validates input data in a robust way.
        
        Parameters:
            X: Input data
            y: Output data (optional)
            name_X: Variable name X for error messages
            name_y: Variable name y for error messages
            
        Returns:
            Tuple (X_validated, y_validated)
            
        Raises:
            TypeError: If types are not numpy arrays
            ValueError: If there are invalid values or incorrect dimensions
        """
        # Validate type of X
        if not isinstance(X, np.ndarray):
            raise TypeError(f"{name_X} must be numpy.ndarray, received {type(X)}")
        
        # Validate dimensions of X
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim != 2:
            raise ValueError(f"{name_X} must have 1 or 2 dimensions, received {X.ndim}")
        
        # Validate number of features
        if hasattr(self, 'n_inputs'):
            if X.shape[1] != self.n_inputs:
                raise ValueError(
                    f"{name_X} must have {self.n_inputs} columns, received {X.shape[1]}"
                )
        
        # Validate NaN/Inf values
        if np.any(np.isnan(X)):
            raise ValueError(f"{name_X} contains NaN values")
        if np.any(np.isinf(X)):
            raise ValueError(f"{name_X} contains Inf values")
        
        # Validate y if provided
        if y is not None:
            if not isinstance(y, np.ndarray):
                raise TypeError(f"{name_y} must be numpy.ndarray, received {type(y)}")
            
            # Accept 1D or 2D
            if y.ndim == 2:
                if y.shape[1] == 1:
                    y = y.ravel()
                else:
                    raise ValueError(f"{name_y} must have 1 column, received {y.shape[1]}")
            elif y.ndim != 1:
                raise ValueError(f"{name_y} must have 1 or 2 dimensions, received {y.ndim}")
            
            # Validate compatible length
            if y.shape[0] != X.shape[0]:
                raise ValueError(
                    f"{name_X} and {name_y} must have same number of samples. "
                    f"{name_X}: {X.shape[0]}, {name_y}: {y.shape[0]}"
                )
            
            # Validate NaN/Inf values
            if np.any(np.isnan(y)):
                raise ValueError(f"{name_y} contains NaN values")
            if np.any(np.isinf(y)):
                raise ValueError(f"{name_y} contains Inf values")
        
        return X, y


    def _calculate_metrics(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Calculates performance metrics.
        
        Parameters:
            X: Input data
            y: Output data
            
        Returns:
            Dictionary with metrics appropriate based on task type
        """
        y_pred = self.predict(X, score=0)
        
        if self.classification:
            metrics = self._calculate_classification_metrics(y, y_pred)
        else:
            # Regression metrics
            errors = y - y_pred
            mse = np.mean(errors ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(errors))
            max_error = np.max(np.abs(errors))
            
            ss_res = np.sum(errors ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-10))
            
            mape = np.mean(np.abs(errors / (y + 1e-10))) * 100
            
            metrics = {
                'loss': mse,
                'rmse': rmse,
                'mae': mae,
                'max_error': max_error,
                'r2': r2,
                'mape': mape,
                'accuracy': np.nan,
                'precision': np.nan,
                'recall': np.nan,
                'f1_score': np.nan
            }
        
        return metrics


    def fit(self, X, y, epochs=100, verbose=True, train_premises=True,
        X_val=None, y_val=None, early_stopping_patience=20,
        restore_best_weights=True):
        """
        Trains the ANFIS model using hybrid learning.
        
        Combines Least Squares Estimation (LSE) for consequent parameters
        and Gradient Descent for premise parameters (membership functions).
        
        Parameters:
            X: Training input data (n_samples, n_inputs)
            y: Training target values (n_samples,)
            epochs: Number of training epochs
            verbose: Print training progress
            train_premises: If True, trains membership function parameters via gradient descent
                        If False, only trains consequent parameters via LSE
            X_val: Validation input data (optional)
            y_val: Validation target values (optional)
            early_stopping_patience: Number of epochs to wait for improvement before stopping
            restore_best_weights: If True, restores best model weights when early stopping or
                                at the end of training (requires validation data)
            
        Returns:
            self (for method chaining)
            
        Examples:
            # Basic training
            anfis.fit(X_train, y_train, epochs=100)
            
            # With validation and early stopping
            anfis.fit(X_train, y_train, X_val=X_val, y_val=y_val, 
                    early_stopping_patience=20)
            
            # Only train consequents (faster, simpler)
            anfis.fit(X_train, y_train, train_premises=False)
        """
        # Validate input
        X, y = self._validate_input(X, y, 'X', 'y')
        if self.classification:
            # Store unique classes for classification
            self.classes_ = np.unique(y)
            self.n_classes = len(self.classes_)
        
        if X_val is not None and y_val is not None:
            X_val, y_val = self._validate_input(X_val, y_val, 'X_val', 'y_val')
        
        # Initialize parameters if not already done
        if self.mf_params is None:
            self._initialize_premise_params(X)
        
        # Timing
        start_time = time.time()
        
        # Early stopping variables
        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        early_stopped = False
        
        # Store best parameters for restoration
        best_mf_params = None
        best_consequent_params = None
        
        # Determine batch size
        batch_size_effective = self.batch_size if self.batch_size is not None else X.shape[0]
        
        # Print training info
        if verbose:
            print("="*70)
            print("ANFIS TRAINING")
            print("="*70)
            print(f"  Inputs:           {self.n_inputs}")
            print(f"  MFs per input:    {self.n_mfs}")
            print(f"  Total rules:      {self.n_rules}")
            print(f"  Training samples: {X.shape[0]}")
            if X_val is not None:
                print(f"  Validation samples: {X_val.shape[0]}")
            print(f"  Epochs:           {epochs}")
            print(f"  Batch size:       {batch_size_effective}")
            print(f"  Learning rate:    {self.lr}")
            if train_premises:
                print(f"  Adaptive LR:      {self.use_adaptive_lr}")
            print(f"  Regularization:   {self._get_reg_type()}")
            if self.lambda_l1 > 0 or self.lambda_l2 > 0:
                print(f"    L1 (λ₁):        {self.lambda_l1}")
                print(f"    L2 (λ₂):        {self.lambda_l2}")
            print(f"  Train premises:   {train_premises}")
            if X_val is not None:
                print(f"  Early stopping:   {early_stopping_patience} epochs")
                print(f"  Restore best:     {restore_best_weights}")
            print("="*70 + "\n")
        
        # Training loop
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Create mini-batches
            batches = self._create_batches(X, y, batch_size_effective, shuffle=True)
            
            # Train on each batch
            for X_batch, y_batch in batches:
                # Step 1: Adjust consequent parameters using Least Squares
                self._adjust_consequents_least_squares(X_batch, y_batch)
                
                # Step 2: Adjust premise parameters using Gradient Descent (if enabled)
                if train_premises:
                    grad_norm, lr_effective = self._adjust_premises_gradient(X_batch, y_batch)
                    self.history['gradient_norms'].append(grad_norm)
                    self.history['learning_rates'].append(lr_effective)
            
            # Calculate training metrics for full dataset
            train_metrics = self._calculate_metrics(X, y)
            
            # Append to history
            for key, value in train_metrics.items():
                self.history['train'][key].append(value)

            if self._reg:
                # Calculate L1 penalty
                l1_penalty = 0.0
                for mf_params in self.mf_params:
                    l1_penalty += np.sum(np.abs(mf_params))
                l1_penalty += np.sum(np.abs(self.consequent_params))
                
                # Calculate L2 penalty
                l2_penalty = 0.0
                for mf_params in self.mf_params:
                    l2_penalty += np.sum(mf_params ** 2)
                l2_penalty += np.sum(self.consequent_params ** 2)
                
                # Total cost
                total_cost = train_metrics['loss'] + \
                            self.lambda_l1 * l1_penalty + \
                            self.lambda_l2 * l2_penalty
                
                # Initialize if first epoch
                if not hasattr(self, 'total_cost_history'):
                    self.total_cost_history = []
                    self.l1_history = []
                    self.l2_history = []
                
                # Append
                self.total_cost_history.append(total_cost)
                self.l1_history.append(l1_penalty)
                self.l2_history.append(l2_penalty)
            
            # Validation metrics
            if X_val is not None and y_val is not None:
                val_metrics = self._calculate_metrics(X_val, y_val)
                val_loss = val_metrics['loss']
                
                # Append validation metrics to history
                for key, value in val_metrics.items():
                    self.history['val'][key].append(value)
                
                # Check for improvement
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    patience_counter = 0
                    
                    # Save best parameters (deep copy)
                    best_mf_params = [p.copy() for p in self.mf_params]
                    best_consequent_params = self.consequent_params.copy()
                    
                    if verbose and epoch > 0:  # Don't print on first epoch
                        print(f"    ✅ New best validation loss: {best_val_loss:.6f}")
                else:
                    patience_counter += 1
                
                # Check for early stopping
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"\n⚠️  Early stopping triggered at epoch {epoch + 1}")
                        print(f"   No improvement for {early_stopping_patience} epochs")
                    early_stopped = True
                    break
            
            # Print epoch progress
            if verbose:
                if self.classification:
                    # Classification metrics
                    metrics_str = (
                        f"Epoch {epoch+1:3d}/{epochs} | "
                        f"Train - Loss: {train_metrics['loss']:.6f}, "
                        f"Acc: {train_metrics['accuracy']:.4f}"
                    )
                    if self._reg:
                        metrics_str += f", Cost: {total_cost:.6f}"
                    
                    metrics_str += f", Acc: {train_metrics['accuracy']:.4f}"
                    
                    if X_val is not None and y_val is not None:
                        metrics_str += (
                            f" | Val - Loss: {val_metrics['loss']:.6f}, "
                            f"Acc: {val_metrics['accuracy']:.4f}"
                        )
                else:
                    # Regression metrics
                    metrics_str = (
                        f"Epoch {epoch+1:3d}/{epochs} | "
                        f"Train - RMSE: {train_metrics['rmse']:.6f}, "
                        f"R²: {train_metrics['r2']:.4f}"
                    )

                    if self._reg:
                        metrics_str += f", Cost: {total_cost:.6f}"
                    
                    if X_val is not None and y_val is not None:
                        metrics_str += (
                            f" | Val - RMSE: {val_metrics['rmse']:.6f}, "
                            f"R²: {val_metrics['r2']:.4f}"
                        )
                
                # Add epoch time
                epoch_time = time.time() - epoch_start_time
                metrics_str += f" | Time: {epoch_time:.2f}s"
                
                print(metrics_str)
            
            # Save epoch timing
            epoch_time = time.time() - epoch_start_time
            self.history['epoch_times'].append(epoch_time)
            
            # Save regularization history
            if hasattr(self, 'total_cost_history'):
                # Already saved during training
                pass
        
        # Restore best weights if applicable
        if restore_best_weights and best_mf_params is not None:
            if X_val is not None and y_val is not None:
                if verbose:
                    print(f"\n🔄 Restoring best model from epoch {best_epoch + 1}")
                    print(f"   Best validation loss: {best_val_loss:.6f}")
                
                # Restore parameters
                self.mf_params = best_mf_params
                self.consequent_params = best_consequent_params
        
        # Print final summary
        if verbose:
            self._print_training_summary(
                X=X,
                y=y,
                X_val=X_val,
                y_val=y_val,
                epoch=epoch,
                epochs=epochs,
                batch_size_effective=batch_size_effective,
                start_time=start_time,
                early_stopped=early_stopped,
                best_val_loss=best_val_loss if early_stopped else None,
                best_epoch=best_epoch if early_stopped else None
            )
        
        return self



    def fit_metaheuristic(self, X, y, optimizer='pso', n_particles=30, 
                      n_iterations=100, verbose=True,
                      X_val=None, y_val=None,
                      early_stopping_patience=20,
                      restore_best_weights=True,
                      **optimizer_kwargs):
        """
        Trains ANFIS using metaheuristic optimization.
        
        Optimizes all parameters (membership functions and consequents) simultaneously
        using evolutionary algorithms. Supports early stopping based on validation loss
        and stores comprehensive convergence metrics.
        
        Parameters:
            X: Training input data (n_samples, n_inputs)
            y: Training target values (n_samples,)
            optimizer: Optimizer name ('pso', 'de', 'ga')
            n_particles: Number of particles/population size
            n_iterations: Number of optimization iterations
            verbose: Print progress information
            X_val: Validation input data (optional, for early stopping)
            y_val: Validation target values (optional, for early stopping)
            early_stopping_patience: Iterations without improvement before stopping
            restore_best_weights: If True, restores best parameters at the end
            **optimizer_kwargs: Additional optimizer-specific parameters
                PSO: w (inertia), c1, c2 (cognitive/social parameters)
                DE: F (differential weight), CR (crossover probability)
                GA: crossover_rate, mutation_rate
            
        Returns:
            self (for method chaining)
            
        Stores:
            - self.metaheuristic_history: Dict with convergence metrics and best fitness
            
        Examples:
            # Basic PSO optimization
            anfis.fit_metaheuristic(X_train, y_train, optimizer='pso', 
                                n_particles=30, n_iterations=100)
            
            # With validation and early stopping
            anfis.fit_metaheuristic(X_train, y_train, X_val=X_val, y_val=y_val,
                                optimizer='pso', early_stopping_patience=20)
            
            # Using Differential Evolution with custom parameters
            anfis.fit_metaheuristic(X_train, y_train, optimizer='de',
                                n_particles=50, F=0.8, CR=0.9)
            
            # Method chaining
            model = ANFIS(n_inputs=2, n_mfs=[3,3]).fit_metaheuristic(X, y)
        """
        # Validate input
        X, y = self._validate_input(X, y, 'X', 'y')
        from .metaheuristics import get_optimizer
        
        if X_val is not None and y_val is not None:
            X_val, y_val = self._validate_input(X_val, y_val, 'X_val', 'y_val')
        
        # For classification, store unique classes
        if self.classification:
            self.classes_ = np.unique(y)
            self.n_classes = len(self.classes_)
            
            if verbose:
                print(f"\n📊 Classification Mode: {self.n_classes} classes detected")
                print(f"   Classes: {self.classes_}\n")
        
        # Initialize parameters if needed
        if self.mf_params is None:
            self._initialize_premise_params(X)
        
        # Timing
        start_time = time.time()
        
        # Early stopping variables
        best_val_loss = float('inf')
        best_train_loss = float('inf')
        best_iteration = 0
        patience_counter = 0
        early_stopped = False
        
        # Store best parameters
        best_params_vector = None
        best_mf_params = None
        best_consequent_params = None
        
        # Convergence tracking
        convergence_history = []
        
        # Print training info
        if verbose:
            print("="*70)
            print("ANFIS METAHEURISTIC TRAINING")
            print("="*70)
            print(f"  Inputs:           {self.n_inputs}")
            print(f"  MFs per input:    {self.n_mfs}")
            print(f"  Total rules:      {self.n_rules}")
            print(f"  Training samples: {X.shape[0]}")
            if X_val is not None:
                print(f"  Validation samples: {X_val.shape[0]}")
            print(f"  Optimizer:        {optimizer.upper()}")
            print(f"  Population:       {n_particles}")
            print(f"  Iterations:       {n_iterations}")
            if X_val is not None and early_stopping_patience > 0:
                print(f"  Early stopping:   {early_stopping_patience} iterations")
                print(f"  Restore best:     {restore_best_weights}")
            print("="*70 + "\n")
        
        # Calculate parameter bounds
        bounds = self._create_optimization_bounds(X)

        
        # Objective function with tracking
        def objective_with_tracking(params_vec):
            """
            Objective function that evaluates fitness and tracks metrics.
            """
            try:
                # Set parameters from vector
                self._vector_to_params(params_vec)
                
                # Forward pass for training data
                y_pred_train = self.forward_batch(X)
                train_loss = np.mean((y - y_pred_train) ** 2)
                
                # Calculate validation loss if available
                if X_val is not None and y_val is not None:
                    y_pred_val = self.forward_batch(X_val)
                    val_loss = np.mean((y_val - y_pred_val) ** 2)
                    fitness = val_loss  # Optimize on validation
                else:
                    val_loss = None
                    fitness = train_loss  # Optimize on training
                
                # Apply regularization to fitness
                if self.lambda_l1 > 0 or self.lambda_l2 > 0:
                    l1_penalty = 0
                    l2_penalty = 0
                    
                    # Regularize MF parameters
                    for mf_params in self.mf_params:
                        l1_penalty += np.sum(np.abs(mf_params))
                        l2_penalty += np.sum(mf_params ** 2)
                    
                    # Regularize consequent parameters
                    l1_penalty += np.sum(np.abs(self.consequent_params))
                    l2_penalty += np.sum(self.consequent_params ** 2)
                    
                    fitness += self.lambda_l1 * l1_penalty + self.lambda_l2 * l2_penalty
                
                return fitness
                
            except Exception as e:
                if verbose:
                    print(f"⚠️  Warning in objective function: {e}")
                return 1e10  # Return large penalty for invalid parameters
        
       # Create optimizer
        if optimizer.lower() == 'pso':
            opt_params = {'n_particles': n_particles, 'n_iterations': n_iterations}
        elif optimizer.lower() in ['de', 'ga']:
            opt_params = {'popsize': n_particles, 'maxiter': n_iterations}
            if optimizer.lower() == 'ga':
                opt_params['maxgen'] = opt_params.pop('maxiter')
        else:
            raise ValueError(f"Optimizer {optimizer} unknown. Use 'pso', 'de' or 'ga'")
        
        # Update with user-provided parameters
        opt_params.update(optimizer_kwargs)
        
        # Create optimizer instance (unpack with **)
        opt = get_optimizer(optimizer, **opt_params)
        
        if verbose:
            print("Starting optimization...\n")
        
        # Custom callback for tracking and early stopping
        iteration_count = [0]  # Use list to allow modification in nested function
        
        def callback_wrapper(params_vec, fitness):
            """Callback to track convergence and check early stopping."""
            nonlocal best_val_loss, best_train_loss, best_iteration, patience_counter
            nonlocal early_stopped, best_params_vector, best_mf_params, best_consequent_params
            
            # Set parameters to evaluate
            self._vector_to_params(params_vec)
            
            # Calculate metrics
            y_pred_train = self.forward_batch(X)
            train_loss = np.mean((y - y_pred_train) ** 2)
            
            if X_val is not None and y_val is not None:
                y_pred_val = self.forward_batch(X_val)
                val_loss = np.mean((y_val - y_pred_val) ** 2)
            else:
                val_loss = None
            
            # Store convergence data
            convergence_history.append({
                'iteration': iteration_count[0],
                'train_loss': train_loss,
                'val_loss': val_loss,
                'fitness': fitness,
                'timestamp': time.time() - start_time
            })
            
            # Check for improvement
            current_loss = val_loss if val_loss is not None else train_loss
            
            if current_loss < best_val_loss:
                best_val_loss = current_loss
                best_train_loss = train_loss
                best_iteration = iteration_count[0]
                patience_counter = 0
                
                # Save best parameters
                best_params_vector = params_vec.copy()
                best_mf_params = [p.copy() for p in self.mf_params]
                best_consequent_params = self.consequent_params.copy()
                
                if verbose:
                    loss_str = f"Val: {val_loss:.6f}" if val_loss is not None else f"Train: {train_loss:.6f}"
                    print(f"  ✅ Iteration {iteration_count[0]:3d} | New best {loss_str}")
            else:
                patience_counter += 1
            
            # Check early stopping
            if X_val is not None and early_stopping_patience > 0:
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"\n⚠️  Early stopping at iteration {iteration_count[0]}")
                        print(f"   No improvement for {early_stopping_patience} iterations")
                    early_stopped = True
                    return True  # Signal to stop optimization
            
            # Print progress
            if verbose and (iteration_count[0] + 1) % max(1, n_iterations // 10) == 0:
                if val_loss is not None:
                    print(f"  Iteration {iteration_count[0]+1:3d}/{n_iterations} | "
                        f"Train: {train_loss:.6f} | Val: {val_loss:.6f}")
                else:
                    print(f"  Iteration {iteration_count[0]+1:3d}/{n_iterations} | "
                        f"Train: {train_loss:.6f}")
            
            iteration_count[0] += 1
            return False  # Continue optimization
        
        # Run optimization with callback (if optimizer supports it)
        try:
            best_params, best_fitness, opt_history = opt.optimize(
                objective_with_tracking, bounds, minimize=True, 
                callback=callback_wrapper, verbose=False
            )
        except TypeError:
            # Optimizer doesn't support callback, run normally
            best_params, best_fitness, opt_history = opt.optimize(
                objective_with_tracking, bounds, minimize=True, verbose=verbose
            )
            
            # Manual tracking for optimizers without callback
            if not convergence_history:
                for i in range(len(opt_history)):
                    self._vector_to_params(best_params)
                    y_pred_train = self.forward_batch(X)
                    train_loss = np.mean((y - y_pred_train) ** 2)
                    
                    val_loss = None
                    if X_val is not None and y_val is not None:
                        y_pred_val = self.forward_batch(X_val)
                        val_loss = np.mean((y_val - y_pred_val) ** 2)
                    
                    convergence_history.append({
                        'iteration': i,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'fitness': opt_history[i],
                        'timestamp': time.time() - start_time
                    })
        
        # Set final parameters
        if best_params_vector is not None:
            final_params = best_params_vector
        else:
            final_params = best_params
        
        self._vector_to_params(final_params)
        
        # Restore best weights if applicable
        if restore_best_weights and best_mf_params is not None:
            if verbose:
                print(f"\n🔄 Restoring best model from iteration {best_iteration}")
                loss_str = f"Val: {best_val_loss:.6f}" if X_val is not None else f"Train: {best_train_loss:.6f}"
                print(f"   Best loss: {loss_str}")
            
            self.mf_params = best_mf_params
            self.consequent_params = best_consequent_params
        
        # Store metaheuristic history
        self.metaheuristic_history = {
            'optimizer': optimizer,
            'n_particles': n_particles,
            'n_iterations': n_iterations,
            'convergence': convergence_history,
            'best_fitness': best_val_loss if X_val is not None else best_train_loss,
            'best_iteration': best_iteration,
            'early_stopped': early_stopped,
            'total_time': time.time() - start_time,
            'optimizer_params': opt_params
        }
        
        # Print final summary
        if verbose:
            self._print_training_summary(
                X=X,
                y=y,
                X_val=X_val,
                y_val=y_val,
                epoch=len(convergence_history) - 1,
                epochs=n_iterations,
                batch_size_effective=None,
                start_time=start_time,
                early_stopped=early_stopped,
                best_val_loss=best_val_loss if X_val is not None else None,
                best_epoch=best_iteration
            )
        
        return self



    def _params_to_vector(self) -> np.ndarray:
        """
        Converts ANFIS parameters to 1D vector for optimization.
        
        Returns:
            1D vector with all parameters
        """
        vector = []
        
        # Premise parameters
        for i in range(self.n_inputs):
            for j in range(self.n_mfs[i]):
                params = self.mf_params[i][j]
                vector.extend(params)
        
        # Consequent parameters
        vector.extend(self.consequent_params.flatten())
        
        return np.array(vector)


    def _vector_to_params(self, vector: np.ndarray):
        """
        Converts 1D vector to ANFIS parameters.
        
        Parameters:
            vector: 1D vector with all parameters
        """
        idx = 0
        
        # Premise parameters
        for i in range(self.n_inputs):
            for j in range(self.n_mfs[i]):
                n_params = len(self.mf_params[i][j])
                self.mf_params[i][j] = vector[idx:idx+n_params].copy()
                idx += n_params
        
        # Consequent parameters
        n_conseq = self.n_rules * (self.n_inputs + 1)
        self.consequent_params = vector[idx:idx+n_conseq].reshape(self.n_rules, self.n_inputs + 1)


    def _create_optimization_bounds(self, X: np.ndarray) -> np.ndarray:
        """
        Creates bounds for metaheuristic optimization.
        
        Parameters:
            X: Input data
            
        Returns:
            Array of bounds (n_params, 2)
        """
        bounds = []
        
        # Bounds for premise parameters
        for i in range(self.n_inputs):
            x_min, x_max = self.input_bounds[i]
            x_range = x_max - x_min
            
            for j in range(self.n_mfs[i]):
                if self.mf_type == 'gaussmf':
                    # center, sigma
                    bounds.append([x_min, x_max])  # center
                    bounds.append([x_range * 0.05, x_range * 2.0])  # sigma
                    
                elif self.mf_type == 'gbellmf':
                    # a, b, c
                    bounds.append([x_range * 0.05, x_range * 2.0])  # a (width)
                    bounds.append([0.5, 5.0])  # b (slope)
                    bounds.append([x_min, x_max])  # c (center)
                    
                elif self.mf_type == 'sigmf':
                    # a, c
                    bounds.append([-10.0, 10.0])  # a (slope)
                    bounds.append([x_min, x_max])  # c (center)
        
        # Bounds for consequent parameters
        for _ in range(self.n_rules * (self.n_inputs + 1)):
            bounds.append([-10.0, 10.0])
        
        return np.array(bounds)

    def predict(self, X: np.ndarray, score: float = 0.5) -> np.ndarray:
        """
        Performs predictions for new data in vectorized form.
        
        Parameters:
            X: Input data (n_samples, n_inputs) or (n_inputs,)
            score: Threshold for classification (default: 0.5)
                Only used if classification=True
            
        Returns:
            Array with predictions (n_samples,) or scalar if X is 1D
        """
        # Validate input
        X, _ = self._validate_input(X, None, 'X', 'y')
        
        # Check if input was 1D
        input_1d = (X.shape[0] == 1)
        
        # Vectorized prediction
        predictions = self.forward_batch(X)
        
        if self.classification and score:
            # Binarize predictions
            y_pred_bin = predictions.copy()
            y_pred_bin = (y_pred_bin > score).astype(int)
            predictions = y_pred_bin
            
            # Map to original classes
            predictions = self.classes_[predictions]
        
        # Return scalar if input was 1D
        return predictions[0] if input_1d else predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts class probabilities (for classification).

        For ANFIS trained with MSE, the raw outputs are already in approximate
        range [0, 1] representing class membership. We clip to ensure valid
        probabilities without distorting the learned values with sigmoid.

        Parameters:
            X: Input data (n_samples, n_inputs)

        Returns:
            Array with probabilities (n_samples, n_classes)
        """
        if not self.classification:
            raise ValueError("predict_proba only available for classification tasks")

        # Validate input
        X, _ = self._validate_input(X, None, 'X', 'y')

        # Get raw predictions (already ~[0,1] from MSE training)
        predictions = self.forward_batch(X)

        # Clip to valid probability range (no sigmoid needed for MSE-trained models)
        proba_class1 = np.clip(predictions, 0, 1)
        proba_class0 = 1 - proba_class1

        return np.column_stack([proba_class0, proba_class1])

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """
        Sigmoid function for converting outputs to probabilities.
        
        Parameters:
            x: Input values
            
        Returns:
            Values between 0 and 1
        """
        return 1 / (1 + np.exp(-x))

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Returns the coefficient of determination R² of the prediction.
        
        Compatible method with scikit-learn API.
        
        Parameters:
            X: Input data (n_samples, n_inputs)
            y: True values (n_samples,)
            
        Returns:
            R² score (best value is 1.0, can be negative if model is worse than baseline)
        """
        # Validate inputs
        X, y = self._validate_input(X, y, 'X', 'y')
        
        # Calculate metrics
        metrics = self._calculate_metrics(X, y)
        
        return metrics['r2']

    def save(self, filepath: str):
        """
        Saves the trained model to file.
        
        Parameters:
            filepath: Path to file (extension .npz will be added if not present)
        """
        import os
        
        # Add extension if not present
        if not filepath.endswith('.npz'):
            filepath = filepath + '.npz'
        
        # Prepare data to save
        save_dict = {
            # Architecture
            'n_inputs': self.n_inputs,
            'n_mfs': np.array(self.n_mfs),
            'n_rules': self.n_rules,
            'mf_type': self.mf_type,
            
            # Parameters
            'consequent_params': self.consequent_params,
            'input_bounds': self.input_bounds,
            
            # Regularization
            'lambda_l1': self.lambda_l1,
            'lambda_l2': self.lambda_l2,
            
            # Training config
            'batch_size': self.batch_size,
            'use_adaptive_lr': self.use_adaptive_lr,
            'classification': self.classification,
            
            # Rule indices
            'rule_indices': np.array([list(idx) for idx in self._rule_indices_cache])
        }
        
        # Save MF params (list of arrays with different sizes)
        for i in range(self.n_inputs):
            save_dict[f'mf_params_{i}'] = self.mf_params[i]
        
        # Classification attributes
        if self.classification:
            save_dict['classes_'] = self.classes_
            save_dict['n_classes_'] = self.n_classes_
        
        # Save
        np.savez_compressed(filepath, **save_dict)
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'ANFIS':
        """
        Loads trained model from file.
        
        Parameters:
            filepath: Path to file
            
        Returns:
            Loaded ANFIS instance
        """
        import os
        
        # Add extension if not present
        if not filepath.endswith('.npz'):
            filepath = filepath + '.npz'
        
        # Check if file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Load data
        data = np.load(filepath, allow_pickle=True)
        
        # Reconstruct n_mfs
        n_mfs = data['n_mfs'].tolist()
        if isinstance(n_mfs, int):
            n_mfs = [n_mfs]
        
        # Create model
        model = cls(
            n_inputs=int(data['n_inputs']),
            n_mfs=n_mfs,
            mf_type=str(data['mf_type']),
            lambda_l1=float(data['lambda_l1']),
            lambda_l2=float(data['lambda_l2']),
            batch_size=int(data['batch_size']) if data['batch_size'] is not None else None,
            use_adaptive_lr=bool(data['use_adaptive_lr']),
            classification=bool(data['classification'])
        )
        
        # Restore parameters
        model.consequent_params = data['consequent_params']
        model.input_bounds = data['input_bounds']
        model._rule_indices_cache = [tuple(idx) for idx in data['rule_indices']]
        
        # Restore MF params
        model.mf_params = []
        for i in range(model.n_inputs):
            model.mf_params.append(data[f'mf_params_{i}'])
        
        # Restore classification attributes
        if model.classification:
            model.classes_ = data['classes_']
            model.n_classes_ = int(data['n_classes_'])
        
        print(f"Model loaded from {filepath}")
        
        return model

    def plot_membership_functions(self, figsize_per_input=(6, 4)):
        """
        Visualizes the learned membership functions.
        
        Parameters:
            figsize_per_input: Size of each subplot
            
        Returns:
            Matplotlib figure
        """
        ncols = min(3, self.n_inputs)
        nrows = int(np.ceil(self.n_inputs / ncols))
        
        fig, axes = plt.subplots(nrows, ncols, 
                                figsize=(figsize_per_input[0]*ncols, 
                                        figsize_per_input[1]*nrows))
        
        # Handle single input case
        if self.n_inputs == 1:
            axes = np.array([axes])
        
        axes = axes.flatten()
        
        for input_idx in range(self.n_inputs):
            ax = axes[input_idx]
            
            x_min, x_max = self.input_bounds[input_idx]
            x_range = np.linspace(x_min, x_max, 200)
            
            for mf_idx, params in enumerate(self.mf_params[input_idx]):
                # CORREÇÃO: Processar o array inteiro de uma vez
                if self.mf_type == 'gaussmf':
                    from ..core.membership import gaussian
                    mu = gaussian(x_range, tuple(params))
                    label = f"MF{mf_idx+1} (μ={params[0]:.2f}, σ={params[1]:.2f})"
                    
                elif self.mf_type == 'gbellmf':
                    from ..core.membership import generalized_bell
                    mu = generalized_bell(x_range, tuple(params))
                    label = f"MF{mf_idx+1} (a={params[0]:.2f}, b={params[1]:.2f}, c={params[2]:.2f})"
                    
                elif self.mf_type == 'sigmf':
                    from ..core.membership import sigmoid
                    mu = sigmoid(x_range, tuple(params))
                    label = f"MF{mf_idx+1} (a={params[0]:.2f}, c={params[1]:.2f})"
                
                ax.plot(x_range, mu, linewidth=2, label=label)
            
            ax.set_xlabel(f'Input {input_idx+1}', fontsize=12)
            ax.set_ylabel('Membership Degree', fontsize=12)
            ax.set_title(f'MFs - Input {input_idx+1} ({self.mf_type})', 
                        fontsize=13, weight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.05, 1.1)
        
        # Hide unused subplots
        for idx in range(self.n_inputs, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        return fig

    def plot_regularization(self, figsize=(16, 5)):
        """
        Plots evolution of regularization penalties.
        
        Parameters:
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Check if there is data to plot
        if len(self.total_cost_history) == 0:
            print("No regularization data available. Train the model first.")
            return None
        
        # Create new figure (avoid overlap)
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        epochs = np.arange(1, len(self.total_cost_history) + 1)
        
        # Total cost
        ax = axes[0]
        ax.plot(epochs, self.total_cost_history, 'b-', linewidth=2, label='Total Cost')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Total Cost', fontsize=12)
        ax.set_title(f'J = MSE + λ₁L1 + λ₂L2', fontsize=13, weight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # L1 penalty
        ax = axes[1]
        ax.plot(epochs, self.l1_history, 'r-', linewidth=2, label='L1')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('L1 Penalty', fontsize=12)
        ax.set_title(f'L1 Penalty (λ₁={self.lambda_l1})', fontsize=13, weight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # L2 penalty
        ax = axes[2]
        ax.plot(epochs, self.l2_history, 'g-', linewidth=2, label='L2')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('L2 Penalty', fontsize=12)
        ax.set_title(f'L2 Penalty (λ₂={self.lambda_l2})', fontsize=13, weight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        return fig

    def _get_reg_type(self) -> str:
        """
        Returns description of the regularization type used.
        
        Returns:
            String describing the regularization type
        """
        if self.lambda_l1 > 0 and self.lambda_l2 > 0:
            return "Elastic Net (L1 + L2)"
        elif self.lambda_l1 > 0:
            return "Lasso (L1)"
        elif self.lambda_l2 > 0:
            return "Ridge (L2)"
        else:
            return "No regularization"

    def summary(self):
        """
        Shows summary of the model architecture and configuration.
        """
        n_params = self.consequent_params.size + sum(p.size for p in self.mf_params)
        batch_size_str = str(self.batch_size) if self.batch_size is not None else "Full batch"
        
        print("=" * 70)
        print("ANFIS - Model Summary")
        print("=" * 70)
        print(" Architecture")
        print(f"   Inputs: {self.n_inputs}")
        print(f"   MFs per input: {self.n_mfs}")
        print(f"   Total rules: {self.n_rules}")
        print(f"   MF type: {self.mf_type}")
        print(f"   Total parameters: {n_params}")
        print("")
        print(" Training Configuration")
        print(f"   Batch size: {batch_size_str}")
        print(f"   Learning rate: {self.lr}")
        print("")
        print(" Regularization (applied only to widths)")
        print(f"   Type: {self._get_reg_type()}")
        print(f"   L1: {self.lambda_l1}")
        print(f"   L2: {self.lambda_l2}")
        print(f"   Centers: Free (not regularized)")
        print(f"   Widths: Regularized")
        print("=" * 70)

    def _calculate_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Calculates classification metrics.
        
        Parameters:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary with accuracy, precision, recall, f1_score and RMSE
        """
        try:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            # Binarize predictions
            y_pred_bin = y_pred.copy()
            y_pred_bin = (y_pred_bin > 0.5).astype(int)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_true, y_pred_bin)
            
            # Handle binary and multiclass
            average = 'binary' if self.n_classes_ == 2 else 'weighted'
            precision = precision_score(y_true, y_pred_bin, average=average, zero_division=0)
            recall = recall_score(y_true, y_pred_bin, average=average, zero_division=0)
            f1 = f1_score(y_true, y_pred_bin, average=average, zero_division=0)
            
            # Calculate RMSE based on errors
            errors = y_true - y_pred
            mse = np.mean(errors ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(errors))
            max_error = np.max(np.abs(errors))

            ss_res = np.sum(errors ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-10))
            
            return {
                'loss': mse,
                'rmse': rmse,
                'mae': mae,
                'max_error': max_error,
                'r2': r2,
                'mape': np.nan,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            
        except Exception as e:
            print(f"Error calculating classification metrics: {e}")
            return {
                'loss': np.nan,
                'rmse': np.nan,
                'mae': np.nan,
                'max_error': np.nan,
                'r2': np.nan,
                'mape': np.nan,
                'accuracy': np.nan,
                'precision': np.nan,
                'recall': np.nan,
                'f1_score': np.nan
            }

    def plot_metrics(self, metrics: Union[str, List[str]] = 'all', 
                 figsize=(15, 10), metaheuristic=False):
        """
        Plots training and validation metrics evolution.
        
        Parameters:
            metrics: Metrics to plot. Options:
                    - 'all': plots all available metrics
                    - List of metric names: ['rmse', 'mae', 'r2', etc.]
                    Available metrics:
                    - Regression: 'loss', 'rmse', 'mae', 'max_error', 'r2', 'mape'
                    - Classification: 'accuracy', 'precision', 'recall', 'f1_score'
                    - Training: 'gradient_norms', 'learning_rates'
            figsize: Figure size
            metaheuristic: If True, plots metaheuristic convergence
                        If False, plots hybrid training metrics (default)
            
        Returns:
            Matplotlib figure
            
        Examples:
            # Plot hybrid training metrics
            anfis.plot_metrics(['rmse', 'r2'])
            
            # Plot metaheuristic convergence
            anfis.plot_metrics(metaheuristic=True)
        """
        import matplotlib.pyplot as plt
        
        # ============================================================================
        # Check available data and print helpful message
        # ============================================================================
        has_hybrid = hasattr(self, 'history') and self.history is not None and \
                    len(self.history.get('train', {}).get('loss', [])) > 0
        has_metaheuristic = hasattr(self, 'metaheuristic_history') and \
                            self.metaheuristic_history is not None and \
                            len(self.metaheuristic_history.get('convergence', [])) > 0
        
        if not has_hybrid and not has_metaheuristic:
            print("❌ No training history available.")
            print("   Train the model first using fit() or fit_metaheuristic()")
            return None
        
        # Print info about available data
        if has_hybrid and has_metaheuristic:
            if not metaheuristic:
                print("ℹ️  Showing hybrid training metrics (fit).")
                print("   To see metaheuristic convergence, use: plot_metrics(metaheuristic=True)")
            else:
                print("ℹ️  Showing metaheuristic convergence.")
                print("   To see hybrid training metrics, use: plot_metrics(metaheuristic=False)")
        elif has_hybrid and not has_metaheuristic:
            if metaheuristic:
                print("⚠️  No metaheuristic data available. Showing hybrid metrics instead.")
                print("   Train with fit_metaheuristic() to generate metaheuristic data.")
                metaheuristic = False
        elif has_metaheuristic and not has_hybrid:
            if not metaheuristic:
                print("⚠️  No hybrid training data available. Showing metaheuristic instead.")
                print("   Train with fit() to generate hybrid training data.")
                metaheuristic = True
        
        # ============================================================================
        # Plot metaheuristic convergence
        # ============================================================================
        if metaheuristic:
            conv = self.metaheuristic_history['convergence']
            
            # Extract data
            iterations = [c['iteration'] for c in conv]
            train_loss = [c['train_loss'] for c in conv]
            val_loss = [c['val_loss'] for c in conv if c['val_loss'] is not None]
            has_validation = len(val_loss) > 0
            
            # Create figure with 2 subplots
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Plot 1: Linear scale
            ax = axes[0]
            ax.plot(iterations, train_loss, 'b-', linewidth=2.5, label='Train Loss', alpha=0.8)
            
            if has_validation:
                ax.plot(iterations[:len(val_loss)], val_loss, 'r--', linewidth=2.5, 
                        label='Validation Loss', alpha=0.8)
            
            # Mark best iteration
            if self.metaheuristic_history.get('early_stopped', False):
                best_iter = self.metaheuristic_history['best_iteration']
                ax.axvline(best_iter, color='green', linestyle=':', linewidth=2.5, 
                        label=f'Best (iter {best_iter})', alpha=0.7)
            
            ax.set_xlabel('Iteration', fontsize=12, weight='bold')
            ax.set_ylabel('Loss (MSE)', fontsize=12, weight='bold')
            ax.set_title(f'{self.metaheuristic_history["optimizer"].upper()} Convergence', 
                        fontsize=14, weight='bold')
            ax.legend(loc='best', fontsize=11)
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Plot 2: Log scale
            ax = axes[1]
            ax.semilogy(iterations, train_loss, 'b-', linewidth=2.5, label='Train Loss', alpha=0.8)
            
            if has_validation:
                ax.semilogy(iterations[:len(val_loss)], val_loss, 'r--', linewidth=2.5, 
                        label='Validation Loss', alpha=0.8)
            
            if self.metaheuristic_history.get('early_stopped', False):
                best_iter = self.metaheuristic_history['best_iteration']
                ax.axvline(best_iter, color='green', linestyle=':', linewidth=2.5, 
                        label=f'Best (iter {best_iter})', alpha=0.7)
            
            ax.set_xlabel('Iteration', fontsize=12, weight='bold')
            ax.set_ylabel('Loss (MSE) - Log Scale', fontsize=12, weight='bold')
            ax.set_title('Convergence (Log Scale)', fontsize=14, weight='bold')
            ax.legend(loc='best', fontsize=11)
            ax.grid(True, alpha=0.3, linestyle='--', which='both')
            
            # Add info box
            info_text = f"Optimizer: {self.metaheuristic_history['optimizer'].upper()}\n"
            info_text += f"Population: {self.metaheuristic_history['n_particles']}\n"
            info_text += f"Iterations: {len(iterations)}/{self.metaheuristic_history['n_iterations']}\n"
            info_text += f"Best Loss: {self.metaheuristic_history['best_fitness']:.6f}\n"
            info_text += f"Time: {self.metaheuristic_history['total_time']:.2f}s"
            
            if self.metaheuristic_history.get('early_stopped', False):
                info_text += f"\n⚠️ Early Stopped"
            
            fig.text(0.98, 0.02, info_text, fontsize=9, ha='right', va='bottom',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
            
            plt.tight_layout()
            return fig
        
        # ============================================================================
        # Plot hybrid training metrics (original code)
        # ============================================================================
        # Define available metrics
        regression_metrics = ['loss', 'rmse', 'mae', 'max_error', 'r2', 'mape']
        classification_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        training_metrics = ['gradient_norms', 'learning_rates']
        
        # Determine which metrics to plot
        if metrics == 'all':
            if self.classification:
                plot_metrics_list = regression_metrics + classification_metrics
            else:
                plot_metrics_list = regression_metrics
        else:
            if isinstance(metrics, str):
                plot_metrics_list = [metrics]
            else:
                plot_metrics_list = metrics
        
        # Filter only metrics that have data
        available_metrics = []
        for metric in plot_metrics_list:
            if metric in training_metrics:
                if len(self.history[metric]) > 0:
                    available_metrics.append(metric)
            elif metric in self.history['train'] and len(self.history['train'][metric]) > 0:
                # Check if metric has valid data (not all NaN)
                if not all(np.isnan(self.history['train'][metric])):
                    available_metrics.append(metric)
        
        if len(available_metrics) == 0:
            print("❌ No metrics available to plot. Train the model first.")
            return None
        
        # Calculate grid layout
        n_metrics = len(available_metrics)
        ncols = min(3, n_metrics)
        nrows = int(np.ceil(n_metrics / ncols))
        
        # Create new figure
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        
        # Handle single metric case
        if n_metrics == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        epochs = np.arange(1, len(self.history['train']['loss']) + 1)
        
        # Plot each metric
        for idx, metric in enumerate(available_metrics):
            ax = axes[idx]
            
            if metric in training_metrics:
                # Training-specific metrics (gradient norms, learning rates)
                data = self.history[metric]
                epochs_metric = np.arange(1, len(data) + 1)
                ax.plot(epochs_metric, data, 'b-', linewidth=2, label=metric.replace('_', ' ').title())
                ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
                
            else:
                # Regular metrics (train and validation)
                train_data = self.history['train'][metric]
                ax.plot(epochs, train_data, 'b-', linewidth=2, label='Train')
                
                # Plot validation if available
                if len(self.history['val'][metric]) > 0 and not all(np.isnan(self.history['val'][metric])):
                    val_data = self.history['val'][metric]
                    ax.plot(epochs, val_data, 'r--', linewidth=2, label='Validation')
                
                ax.set_ylabel(metric.upper() if len(metric) <= 4 else metric.replace('_', ' ').title(), 
                            fontsize=12)
            
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_title(f'{metric.upper() if len(metric) <= 4 else metric.replace("_", " ").title()} Evolution', 
                        fontsize=13, weight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10, loc='best')
        
        # Hide unused subplots
        for idx in range(n_metrics, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        return fig


    def rules_to_dataframe(self, input_names=None, output_name='Output', linguistic_terms=None):
        """
        Generates a DataFrame with fuzzy rules in linguistic terms.
        
        Each row represents a rule, columns represent input variables and output.
        Cells contain the linguistic terms (e.g., "Low", "Medium", "High").
        
        Parameters:
            input_names: List of input variable names. 
                        If None, uses ['Input_1', 'Input_2', ...]
            output_name: Name of output variable (default: 'Output')
            linguistic_terms: Dictionary mapping input indices to lists of term names.
                            Example: {0: ['Low', 'High'], 1: ['Small', 'Medium', 'Large']}
                            If None, uses ['MF1', 'MF2', ...] for each input
            
        Returns:
            pandas DataFrame with fuzzy rules
            
        Examples:
            # Basic usage (auto-generated names)
            df = anfis.rules_to_dataframe()
            
            # With custom input names
            df = anfis.rules_to_dataframe(
                input_names=['Temperature', 'Humidity']
            )
            
            # With linguistic terms
            df = anfis.rules_to_dataframe(
                input_names=['Temperature', 'Humidity'],
                output_name='Comfort',
                linguistic_terms={
                    0: ['Cold', 'Warm', 'Hot'],
                    1: ['Dry', 'Normal', 'Humid']
                }
            )
        """
        import pandas as pd
        
        # Generate input names if not provided
        if input_names is None:
            input_names = [f'Input_{i+1}' for i in range(self.n_inputs)]
        elif len(input_names) != self.n_inputs:
            raise ValueError(f"input_names must have {self.n_inputs} elements, got {len(input_names)}")
        
        # Generate linguistic terms if not provided
        if linguistic_terms is None:
            linguistic_terms = {}
            for i in range(self.n_inputs):
                linguistic_terms[i] = [f'MF{j+1}' for j in range(self.n_mfs[i])]
        
        # Validate linguistic_terms
        for i in range(self.n_inputs):
            if i not in linguistic_terms:
                linguistic_terms[i] = [f'MF{j+1}' for j in range(self.n_mfs[i])]
            elif len(linguistic_terms[i]) != self.n_mfs[i]:
                raise ValueError(
                    f"linguistic_terms[{i}] must have {self.n_mfs[i]} terms, got {len(linguistic_terms[i])}"
                )
        
        # Build rules table
        rules_data = []
        
        for rule_idx, mf_indices in enumerate(self._rule_indices_cache):
            rule_row = {}
            
            # Add input terms
            for input_idx in range(self.n_inputs):
                mf_idx = mf_indices[input_idx]
                term = linguistic_terms[input_idx][mf_idx]
                rule_row[input_names[input_idx]] = term
            
            # Add consequent parameters
            consequent = self.consequent_params[rule_idx]
            
            # Format consequent as equation: y = p0 + p1*x1 + p2*x2 + ...
            equation_parts = [f"{consequent[0]:.3f}"]
            for i in range(self.n_inputs):
                coef = consequent[i+1]
                sign = '+' if coef >= 0 else ''
                equation_parts.append(f"{sign}{coef:.3f}*{input_names[i]}")
            
            equation = ' '.join(equation_parts)
            rule_row[output_name] = equation
            
            rules_data.append(rule_row)
        
        # Create DataFrame
        df = pd.DataFrame(rules_data)
        
        # Add rule index
        df.insert(0, 'Rule', [f'R{i+1}' for i in range(self.n_rules)])
        
        return df

    def show_rules_table(self, input_names=None, output_name='Output', 
                     linguistic_terms=None, cmap=None, figsize=None,
                     value_format='.2f'):
        """
        Displays fuzzy rules as a colored table (DataFrame visualization).
        
        Each row represents a rule, columns show input linguistic terms.
        Cells are colored according to the linguistic term (blue gradient).
        
        Parameters:
            input_names: List of input variable names
                        If None, uses ['Input_1', 'Input_2', ...]
            output_name: Name of output variable (default: 'Output')
            linguistic_terms: Dictionary mapping input indices to term names
                            Example: {0: ['Low', 'High'], 1: ['Small', 'Med', 'Large']}
                            If None, uses ['MF1', 'MF2', ...]
            cmap: Colormap for the cells (default: plt.cm.Blues)
            figsize: Figure size. If None, auto-calculated
            value_format: Format string for coefficient values (default: '.2f')
            
        Returns:
            Matplotlib figure
            
        Examples:
            # Basic usage
            anfis.show_rules_table()
            
            # With custom names
            anfis.show_rules_table(
                input_names=['Temp', 'Humidity'],
                linguistic_terms={
                    0: ['Cold', 'Warm', 'Hot'],
                    1: ['Dry', 'Normal', 'Humid']
                }
            )
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        import pandas as pd
        
        # Default colormap (blue tones)
        if cmap is None:
            cmap = plt.cm.Blues
        
        # Generate input names if not provided
        if input_names is None:
            input_names = [f'Input_{i+1}' for i in range(self.n_inputs)]
        elif len(input_names) != self.n_inputs:
            raise ValueError(f"input_names must have {self.n_inputs} elements")
        
        # Generate linguistic terms if not provided
        if linguistic_terms is None:
            linguistic_terms = {}
            for i in range(self.n_inputs):
                linguistic_terms[i] = [f'MF{j+1}' for j in range(self.n_mfs[i])]
        
        # Validate linguistic terms
        for i in range(self.n_inputs):
            if i not in linguistic_terms:
                linguistic_terms[i] = [f'MF{j+1}' for j in range(self.n_mfs[i])]
            elif len(linguistic_terms[i]) != self.n_mfs[i]:
                raise ValueError(
                    f"linguistic_terms[{i}] must have {self.n_mfs[i]} terms"
                )
        
        # Build rules data
        rules_data = []
        for rule_idx, mf_indices in enumerate(self._rule_indices_cache):
            rule_row = {}
            
            # Add input terms and their MF indices
            for input_idx in range(self.n_inputs):
                mf_idx = mf_indices[input_idx]
                term = linguistic_terms[input_idx][mf_idx]
                rule_row[input_names[input_idx]] = term
                rule_row[f'_{input_names[input_idx]}_idx'] = mf_idx  # Hidden column for coloring
            
            # Build output formula
            consequent = self.consequent_params[rule_idx]
            formula_parts = []
            
            # Constant term
            const = consequent[0]
            if abs(const) > 1e-6:  # Only show if significant
                formula_parts.append(f"{const:{value_format}}")
            
            # Linear terms
            for i in range(self.n_inputs):
                coef = consequent[i+1]
                if abs(coef) > 1e-6:  # Only show if significant
                    sign = '+' if coef >= 0 and len(formula_parts) > 0 else ''
                    # Simplify coefficient display
                    if abs(coef - 1.0) < 1e-6:
                        formula_parts.append(f"{sign}{input_names[i]}")
                    elif abs(coef + 1.0) < 1e-6:
                        formula_parts.append(f"-{input_names[i]}")
                    else:
                        formula_parts.append(f"{sign}{coef:{value_format}}{input_names[i]}")
            
            # Join formula or show 0 if empty
            formula = ''.join(formula_parts) if formula_parts else '0'
            rule_row[output_name] = formula
            rule_row['_rule_idx'] = rule_idx
            
            rules_data.append(rule_row)
        
        df = pd.DataFrame(rules_data)
        
        # Calculate figure size
        n_rows = len(df)
        n_cols = self.n_inputs + 1  # inputs + output (sem coluna de Rule)
        
        if figsize is None:
            cell_width = 3.0
            cell_height = 0.7
            figsize = (n_cols * cell_width, min(n_rows * cell_height + 1.5, 20))
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis('off')
        
        # Table dimensions
        col_width = 1.0 / n_cols
        row_height = 1.0 / (n_rows + 1)  # +1 for header
        
        # Draw header
        header_y = 1 - row_height
        columns_to_show = input_names + [output_name]
        
        for col_idx, col_name in enumerate(columns_to_show):
            x = col_idx * col_width
            
            # Header rectangle
            rect = Rectangle((x, header_y), col_width, row_height,
                            facecolor='#2c3e50', edgecolor='black', linewidth=1.5)
            ax.add_patch(rect)
            
            # Header text
            ax.text(x + col_width/2, header_y + row_height/2, col_name,
                ha='center', va='center', fontsize=12, weight='bold', color='white')
        
        # Draw data rows
        for row_idx in range(n_rows):
            y = 1 - (row_idx + 2) * row_height
            
            # Rule ID (outside table, on the left)
            rule_id = df.iloc[row_idx]['_rule_idx'] + 1
            ax.text(-0.02, y + row_height/2, f'R{rule_id}',
                ha='right', va='center', fontsize=11, weight='bold',
                color='black', transform=ax.transData)
            
            # Input columns (with blue gradient colors)
            for input_idx, input_name in enumerate(input_names):
                col_idx = input_idx
                x = col_idx * col_width
                
                # Get MF index for coloring
                mf_idx = df.iloc[row_idx][f'_{input_name}_idx']
                n_mfs = self.n_mfs[input_idx]
                
                # Color based on MF position (normalized 0.3-0.9 for better blue gradient)
                color_val = 0.3 + (mf_idx / (n_mfs - 1) * 0.6) if n_mfs > 1 else 0.6
                cell_color = cmap(color_val)
                
                # Draw cell
                rect = Rectangle((x, y), col_width, row_height,
                            facecolor=cell_color, edgecolor='black', linewidth=1)
                ax.add_patch(rect)
                
                # Get term text
                term = df.iloc[row_idx][input_name]
                
                # Determine text color based on background brightness
                brightness = 0.299*cell_color[0] + 0.587*cell_color[1] + 0.114*cell_color[2]
                text_color = 'white' if brightness < 0.5 else 'black'
                
                # Draw text
                ax.text(x + col_width/2, y + row_height/2, term,
                    ha='center', va='center', fontsize=11, weight='bold',
                    color=text_color)
            
            # Output column (formula)
            col_idx = self.n_inputs
            x = col_idx * col_width
            
            formula = df.iloc[row_idx][output_name]
            
            rect = Rectangle((x, y), col_width, row_height,
                        facecolor='#ecf0f1', edgecolor='black', linewidth=1)
            ax.add_patch(rect)
            
            ax.text(x + col_width/2, y + row_height/2, formula,
                ha='center', va='center', fontsize=10, family='monospace',
                color='#2c3e50')
        
        # Set axis limits (com espaço para IDs à esquerda)
        ax.set_xlim(-0.05, 1)
        ax.set_ylim(0, 1)
        
        # Title
        fig.suptitle('Fuzzy Rules', fontsize=16, weight='bold', y=0.98)
        
        plt.tight_layout()
        return fig

    def _print_training_summary(self, X, y, X_val=None, y_val=None, 
                           epoch=None, epochs=None, batch_size_effective=None,
                           start_time=None, early_stopped=False, 
                           best_val_loss=None, best_epoch=None):
        """
        Prints comprehensive training summary with final metrics.
        
        Parameters:
            X: Training data
            y: Training targets
            X_val: Validation data (optional)
            y_val: Validation targets (optional)
            epoch: Final epoch number
            epochs: Total epochs configured
            batch_size_effective: Batch size used
            start_time: Training start time
            early_stopped: Whether early stopping was triggered
            best_val_loss: Best validation loss (if early stopping)
            best_epoch: Epoch with best validation loss
        """
        print("\n" + "="*70)
        print("TRAINING COMPLETED")
        print("="*70)
        
        # Calculate final metrics on training set
        train_metrics = self._calculate_metrics(X, y)
        
        print("\n📊 Final Training Metrics:")
        print("-" * 70)
        
        if self.classification:
            # Classification metrics
            print(f"  Loss (MSE):       {train_metrics['loss']:.6f}")
            print(f"  RMSE:             {train_metrics['rmse']:.6f}")
            print(f"  Accuracy:         {train_metrics['accuracy']:.4f} ({train_metrics['accuracy']*100:.2f}%)")
            print(f"  Precision:        {train_metrics['precision']:.4f}")
            print(f"  Recall:           {train_metrics['recall']:.4f}")
            print(f"  F1-Score:         {train_metrics['f1_score']:.4f}")
        else:
            # Regression metrics
            print(f"  Loss (MSE):       {train_metrics['loss']:.6f}")
            print(f"  RMSE:             {train_metrics['rmse']:.6f}")
            print(f"  MAE:              {train_metrics['mae']:.6f}")
            print(f"  R² Score:         {train_metrics['r2']:.4f}")
            print(f"  Max Error:        {train_metrics['max_error']:.6f}")
            if not np.isnan(train_metrics['mape']):
                print(f"  MAPE:             {train_metrics['mape']:.2f}%")
        
        # Validation metrics (if available)
        if X_val is not None and y_val is not None:
            val_metrics = self._calculate_metrics(X_val, y_val)
            
            print("\n📈 Final Validation Metrics:")
            print("-" * 70)
            
            if self.classification:
                print(f"  Loss (MSE):       {val_metrics['loss']:.6f}")
                print(f"  RMSE:             {val_metrics['rmse']:.6f}")
                print(f"  Accuracy:         {val_metrics['accuracy']:.4f} ({val_metrics['accuracy']*100:.2f}%)")
                print(f"  Precision:        {val_metrics['precision']:.4f}")
                print(f"  Recall:           {val_metrics['recall']:.4f}")
                print(f"  F1-Score:         {val_metrics['f1_score']:.4f}")
            else:
                print(f"  Loss (MSE):       {val_metrics['loss']:.6f}")
                print(f"  RMSE:             {val_metrics['rmse']:.6f}")
                print(f"  MAE:              {val_metrics['mae']:.6f}")
                print(f"  R² Score:         {val_metrics['r2']:.4f}")
                print(f"  Max Error:        {val_metrics['max_error']:.6f}")
                if not np.isnan(val_metrics['mape']):
                    print(f"  MAPE:             {val_metrics['mape']:.2f}%")
        
        # Training configuration
        print("\n⚙️  Training Configuration:")
        print("-" * 70)
        if epoch is not None and epochs is not None:
            print(f"  Total Epochs:     {epoch + 1}/{epochs}")
        if batch_size_effective is not None:
            print(f"  Batch Size:       {batch_size_effective if batch_size_effective else 'Full batch'}")
        print(f"  Learning Rate:    {self.lr}")
        print(f"  Regularization:   {self._get_reg_type()}")
        if self.lambda_l1 > 0 or self.lambda_l2 > 0:
            print(f"    L1 (λ₁):        {self.lambda_l1}")
            print(f"    L2 (λ₂):        {self.lambda_l2}")
        
        # Total time
        if start_time is not None:
            total_time = time.time() - start_time
            print(f"\n⏱️  Total Training Time: {total_time:.2f}s ({total_time/60:.2f}m)")
        
        # Early stopping info
        if early_stopped:
            print(f"\n⚠️  Early stopping triggered at epoch {epoch + 1}")
            if best_val_loss is not None and best_epoch is not None:
                print(f"   Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")
        
        print("=" * 70 + "\n")

