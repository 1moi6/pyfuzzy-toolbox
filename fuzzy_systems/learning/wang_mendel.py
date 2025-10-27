"""
Wang-Mendel Method for Automatic Fuzzy Rule Generation
=======================================================

This module implements the Wang-Mendel algorithm (1992) for automatically
generating fuzzy rules from data.

Reference:
Wang, L. X., & Mendel, J. M. (1992). "Generating fuzzy rules by
learning from examples." IEEE Transactions on Systems, Man, and
Cybernetics, 22(6), 1414-1427.

The algorithm consists of 5 steps:
1. Partition variable domains (fuzzification)
2. Generate candidate rules from data
3. Assign degree to each rule
4. Resolve conflicts (keep rule with highest degree)
5. Create final fuzzy system

UPDATED: Unified class with structure-based output scaling (2025)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Literal
from ..core.fuzzification import LinguisticVariable, FuzzySet
from ..inference.systems import MamdaniSystem


class WangMendelLearning:
    """
    Unified Wang-Mendel Learning Algorithm.
    
    Supports both regression and classification tasks with a single interface.
    The task type can be specified explicitly or auto-detected from data.
    
    For classification, outputs can be automatically scaled based on the
    achievable range of the membership functions (not data-dependent).
    
    Parameters:
        system: Mamdani system with variables and terms already configured
        X: Input data (n_samples, n_features)
        y: Output data (n_samples, n_outputs) or (n_samples,)
        task: 'auto', 'regression', or 'classification'
              If 'auto', detects based on y (one-hot → classification)
        scale_classification: If True, scales classification outputs based on
                            achievable range determined by membership function structure
        verbose_init: If True, prints output range information during initialization
    
    Attributes:
        task: Detected or specified task type ('regression' or 'classification')
        is_classification: Boolean indicating if task is classification
        classes_: Unique classes (only for classification)
        n_classes: Number of classes (only for classification)
        output_ranges: Achievable output ranges per variable (only for classification)
    
    Example (Regression):
        >>> system = MamdaniSystem()
        >>> # ... configure system ...
        >>> wm = WangMendelLearning(system, X_train, y_train, task='regression')
        >>> wm.fit(verbose=True)
        >>> y_pred = wm.predict(X_test)
    
    Example (Classification with one-hot):
        >>> system = MamdaniSystem()
        >>> # ... configure system with binary outputs per class ...
        >>> y_onehot = OneHotEncoder().fit_transform(y_train)
        >>> wm = WangMendelLearning(system, X_train, y_onehot, task='auto')
        >>> wm.fit(verbose=True)
        >>> y_pred_classes = wm.predict(X_test)  # Returns class indices
        >>> y_proba = wm.predict_proba(X_test)   # Returns probabilities
    """
    
    def __init__(self, 
                 system: MamdaniSystem,
                 X: np.ndarray,
                 y: np.ndarray,
                 task: Literal['auto', 'regression', 'classification'] = 'auto',
                 scale_classification: bool = True,
                 verbose_init: bool = False):
        """
        Initialize Wang-Mendel learning algorithm.
        
        Parameters:
            system: Pre-configured Mamdani system
            X: Input array (n_samples, n_features)
            y: Output array (n_samples, n_outputs) or (n_samples,)
            task: 'auto' (detect), 'regression', or 'classification'
            scale_classification: If True, scales classification outputs to [0, 1]
                                based on membership function structure
            verbose_init: If True, prints achievable output ranges
        """
        self.system = system
        self.X = np.atleast_2d(X)
        self.y = np.atleast_2d(y) if y.ndim == 1 else y
        self.scale_classification = scale_classification
        self._verbose_init = verbose_init
        
        # Variable names
        self.input_names = list(system.input_variables.keys())
        self.output_names = list(system.output_variables.keys())
        
        # Validation
        if self.X.shape[1] != len(self.input_names):
            raise ValueError(
                f"Number of features ({self.X.shape[1]}) does not match "
                f"number of system inputs ({len(self.input_names)})"
            )
        
        if self.y.shape[1] != len(self.output_names):
            raise ValueError(
                f"Number of outputs in data ({self.y.shape[1]}) does not match "
                f"number of system outputs ({len(self.output_names)})"
            )
        
        # Detect or set task type
        self._setup_task(task)
        
        # Training statistics
        self.candidate_rules: Dict[Tuple, Dict] = {}
        self.conflicts_count = 0
        self.final_rules: List[Dict] = []
    
    def _setup_task(self, task: str):
        """Detect or validate task type."""
        if task == 'auto':
            self.task = self._detect_task()
        elif task in ['regression', 'classification']:
            self.task = task
        else:
            raise ValueError(f"task must be 'auto', 'regression', or 'classification'. Got: {task}")
        
        self.is_classification = (self.task == 'classification')
        
        # Setup classification-specific attributes
        if self.is_classification:
            self._setup_classification()
        else:
            self.classes_ = None
            self.n_classes = None
            self.output_ranges = None
    
    def _detect_task(self) -> str:
        """Auto-detect if task is regression or classification."""
        if self.y.shape[1] == 1:
            return 'regression'
        
        # Check if y looks like one-hot encoding
        is_binary = np.all((self.y == 0) | (self.y == 1))
        row_sums = self.y.sum(axis=1)
        sums_to_one = np.allclose(row_sums, 1.0)
        
        if is_binary and sums_to_one:
            return 'classification'
        else:
            return 'regression'
    
    def _compute_output_range(self, var_name: str, n_samples: int = 1000) -> Tuple[float, float]:
        """
        Compute the achievable output range for a fuzzy variable.
        
        This accounts for the fact that defuzzification may not reach
        the extremes of the universe due to membership function shapes.
        
        Parameters:
            var_name: Name of output variable
            n_samples: Number of samples to test across universe
        
        Returns:
            (min_achievable, max_achievable) tuple
        """
        var = self.system.output_variables[var_name]
        x_min, x_max = var.universe
        
        # Sample the universe
        x = np.linspace(x_min, x_max, n_samples)
        
        # Extreme 1: Only first term fully activated
        first_term = list(var.terms.values())[0]
        y_first = np.array([first_term.membership(xi) for xi in x])
        if y_first.sum() > 0:
            min_output = np.sum(x * y_first) / np.sum(y_first)
        else:
            min_output = x_min
        
        # Extreme 2: Only last term fully activated
        last_term = list(var.terms.values())[-1]
        y_last = np.array([last_term.membership(xi) for xi in x])
        if y_last.sum() > 0:
            max_output = np.sum(x * y_last) / np.sum(y_last)
        else:
            max_output = x_max
        
        return min_output, max_output
    
    def _setup_classification(self):
        """Setup classification-specific attributes including output scaling."""
        # Detect classes from one-hot encoding
        self.n_classes = self.y.shape[1]
        self.classes_ = np.arange(self.n_classes)
        
        # Store original class labels
        self.y_labels = np.argmax(self.y, axis=1)
        
        # Compute achievable output ranges for scaling
        self.output_ranges = {}
        for var_name in self.output_names:
            min_val, max_val = self._compute_output_range(var_name)
            self.output_ranges[var_name] = (min_val, max_val)
        
        if self._verbose_init:
            print("\n📏 Computed achievable output ranges:")
            for var_name, (min_val, max_val) in self.output_ranges.items():
                universe = self.system.output_variables[var_name].universe
                print(f"   {var_name}: universe={universe}, achievable=[{min_val:.4f}, {max_val:.4f}]")
    
    def _scale_classification_outputs(self, outputs: np.ndarray) -> np.ndarray:
        """
        Scale fuzzy outputs based on achievable range of membership functions.
        
        This maps the natural output range of the fuzzy system (determined by
        the structure of membership functions) to [0, 1] for each output variable.
        
        Parameters:
            outputs: Raw fuzzy outputs (n_samples, n_classes)
        
        Returns:
            Scaled outputs in [0, 1] (n_samples, n_classes)
        """
        scaled = np.zeros_like(outputs)
        
        for j, var_name in enumerate(self.output_names):
            min_val, max_val = self.output_ranges[var_name]
            
            # Linear scaling: [min_val, max_val] → [0, 1]
            range_val = max_val - min_val
            if range_val < 1e-10:
                # If range is zero, keep original values
                scaled[:, j] = outputs[:, j]
            else:
                scaled[:, j] = (outputs[:, j] - min_val) / range_val
                # Clip to [0, 1] to handle numerical errors
                scaled[:, j] = np.clip(scaled[:, j], 0.0, 1.0)
        
        return scaled
    
    def fit(self, verbose: bool = False) -> MamdaniSystem:
        """
        Execute Wang-Mendel algorithm and train the system.
        
        Parameters:
            verbose: If True, displays progress information
        
        Returns:
            Trained Mamdani system
        """
        if verbose:
            print("🔄 Starting Wang-Mendel Algorithm...")
            print(f"   Task: {self.task.upper()}")
            print(f"   Data: {self.X.shape[0]} samples, "
                  f"{self.X.shape[1]} inputs, {self.y.shape[1]} outputs")
            if self.is_classification:
                print(f"   Classes: {self.n_classes}")
                if self.scale_classification:
                    print(f"   Output scaling: ENABLED (structure-based)")
        
        # Step 2: Generate candidate rules
        self._generate_candidate_rules(verbose)
        
        # Step 4: Resolve conflicts
        self._resolve_conflicts(verbose)
        
        # Step 5: Create final system
        self._create_final_system(verbose)
        
        if verbose:
            print(f"\n✅ Training completed!")
            print(f"   Rules generated: {len(self.final_rules)}")
            print(f"   Conflicts resolved: {self.conflicts_count}")
        
        return self.system
    
    def _generate_candidate_rules(self, verbose: bool = False):
        """Step 2: Generate candidate rules from data."""
        if verbose:
            print("\n📊 Step 2: Generating candidate rules...")
        
        for sample_idx in range(self.X.shape[0]):
            x_sample = self.X[sample_idx]
            y_sample = self.y[sample_idx]
            
            # Find most activated fuzzy terms for inputs
            antecedents = []
            antecedent_degrees = []
            
            for i, var_name in enumerate(self.input_names):
                var = self.system.input_variables[var_name]
                max_degree = -1
                best_term = None
                
                for term_name, fuzzy_set in var.terms.items():
                    degree = fuzzy_set.membership(x_sample[i])
                    if degree > max_degree:
                        max_degree = degree
                        best_term = term_name
                
                antecedents.append(best_term)
                antecedent_degrees.append(max_degree)
            
            # Find most activated fuzzy terms for outputs
            consequents = []
            consequent_degrees = []
            
            for i, var_name in enumerate(self.output_names):
                var = self.system.output_variables[var_name]
                max_degree = -1
                best_term = None
                
                for term_name, fuzzy_set in var.terms.items():
                    degree = fuzzy_set.membership(y_sample[i])
                    if degree > max_degree:
                        max_degree = degree
                        best_term = term_name
                
                consequents.append(best_term)
                consequent_degrees.append(max_degree)
            
            # Step 3: Calculate rule degree
            rule_degree = np.prod(antecedent_degrees) * np.prod(consequent_degrees)
            
            # Store candidate rule
            antecedent_tuple = tuple(antecedents)
            consequent_tuple = tuple(consequents)
            rule_key = (antecedent_tuple, consequent_tuple)
            
            if rule_key not in self.candidate_rules:
                self.candidate_rules[rule_key] = {
                    'antecedents': antecedent_tuple,
                    'consequents': consequent_tuple,
                    'degree': rule_degree
                }
            else:
                if rule_degree > self.candidate_rules[rule_key]['degree']:
                    self.candidate_rules[rule_key]['degree'] = rule_degree
        
        if verbose:
            print(f"   Candidate rules generated: {len(self.candidate_rules)}")
    
    def _resolve_conflicts(self, verbose: bool = False):
        """Step 4: Resolve conflicts between rules."""
        if verbose:
            print("\n🔍 Step 4: Resolving conflicts...")
        
        rules_by_antecedent = {}
        
        for rule_key, rule_data in self.candidate_rules.items():
            antecedent_tuple = rule_data['antecedents']
            
            if antecedent_tuple not in rules_by_antecedent:
                rules_by_antecedent[antecedent_tuple] = []
            
            rules_by_antecedent[antecedent_tuple].append(rule_data)
        
        for antecedent, rules in rules_by_antecedent.items():
            if len(rules) > 1:
                self.conflicts_count += len(rules) - 1
                rules.sort(key=lambda r: r['degree'], reverse=True)
                self.final_rules.append(rules[0])
            else:
                self.final_rules.append(rules[0])
        
        if verbose:
            print(f"   Conflicts found: {self.conflicts_count}")
            print(f"   Final rules: {len(self.final_rules)}")
    
    def _create_final_system(self, verbose: bool = False):
        """Step 5: Add rules to system."""
        if verbose:
            print("\n🔧 Step 5: Creating final system...")
        
        rules_to_add = []
        
        for rule in self.final_rules:
            rule_dict = {}
            
            for i, term in enumerate(rule['antecedents']):
                rule_dict[self.input_names[i]] = term
            
            for i, term in enumerate(rule['consequents']):
                rule_dict[self.output_names[i]] = term
            
            rules_to_add.append(rule_dict)
        
        self.system.add_rules(rules_to_add)
        
        if verbose:
            print(f"   ✓ {len(rules_to_add)} rules added to system")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using trained system.
        
        Parameters:
            X: Input array (n_samples, n_features)
        
        Returns:
            For regression: predictions (n_samples, n_outputs)
            For classification: predicted classes (n_samples,)
        """
        X = np.atleast_2d(X)
        predictions = np.zeros((X.shape[0], len(self.output_names)))
        
        for i in range(X.shape[0]):
            input_dict = {name: X[i, j] for j, name in enumerate(self.input_names)}
            output_dict = self.system.evaluate(input_dict)
            
            for j, name in enumerate(self.output_names):
                predictions[i, j] = output_dict[name]
        
        # Apply scaling for classification based on achievable range
        if self.is_classification and self.scale_classification:
            predictions = self._scale_classification_outputs(predictions)
        
        # For classification, return class indices
        if self.is_classification:
            return np.argmax(predictions, axis=1)
        else:
            return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return class probabilities (normalized scores).
        Only available for classification tasks.
        
        Parameters:
            X: Input array (n_samples, n_features)
        
        Returns:
            Probability matrix (n_samples, n_classes)
        
        Raises:
            ValueError: If task is regression
        """
        if not self.is_classification:
            raise ValueError("predict_proba() only available for classification tasks")
        
        X = np.atleast_2d(X)
        predictions = np.zeros((X.shape[0], len(self.output_names)))
        
        for i in range(X.shape[0]):
            input_dict = {name: X[i, j] for j, name in enumerate(self.input_names)}
            output_dict = self.system.evaluate(input_dict)
            
            for j, name in enumerate(self.output_names):
                predictions[i, j] = output_dict[name]
        
        # Apply scaling
        if self.scale_classification:
            predictions = self._scale_classification_outputs(predictions)
        
        # Normalize to sum to 1
        row_sums = predictions.sum(axis=1, keepdims=True)
        return predictions / (row_sums + 1e-10)
    
    def get_training_stats(self) -> Dict:
        """Return training statistics."""
        stats = {
            'task': self.task,
            'n_samples': self.X.shape[0],
            'n_features': self.X.shape[1],
            'n_outputs': self.y.shape[1],
            'candidate_rules': len(self.candidate_rules),
            'final_rules': len(self.final_rules),
            'conflicts_resolved': self.conflicts_count
        }
        
        if self.is_classification:
            stats['n_classes'] = self.n_classes
            stats['classes'] = self.classes_.tolist()
            if self.output_ranges:
                stats['output_ranges'] = {
                    var: {'min': float(min_val), 'max': float(max_val)}
                    for var, (min_val, max_val) in self.output_ranges.items()
                }
        
        return stats

    def predict_membership(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Return membership degrees for each output in its linguistic terms.
        
        This is useful for understanding which linguistic terms are activated
        for each prediction and with what degree.
        
        Parameters:
            X: Input array (n_samples, n_features)
        
        Returns:
            Dictionary mapping output variable names to membership arrays.
            Each array has shape (n_samples, n_terms) where n_terms is the
            number of linguistic terms for that output variable.
        
        Example:
            >>> memberships = wm.predict_membership(X_test)
            >>> # For classification with outputs 'setosa', 'versicolor', 'virginica'
            >>> # each having terms 'no' and 'yes':
            >>> memberships['setosa']  # shape: (n_samples, 2)
            >>> # [[0.05, 0.95],   # Sample 1: 5% 'no', 95% 'yes'
            >>> #  [0.82, 0.18],   # Sample 2: 82% 'no', 18% 'yes'
            >>> #  ...]
        
        Raises:
            ValueError: If called before training (no rules exist)
        """
        if len(self.final_rules) == 0:
            raise ValueError("Model must be trained before calling predict_membership()")
        
        X = np.atleast_2d(X)
        n_samples = X.shape[0]
        
        # Initialize membership dictionary
        memberships = {}
        
        # Get raw fuzzy outputs first
        raw_outputs = np.zeros((n_samples, len(self.output_names)))
        
        for i in range(n_samples):
            input_dict = {name: X[i, j] for j, name in enumerate(self.input_names)}
            output_dict = self.system.evaluate(input_dict)
            
            for j, name in enumerate(self.output_names):
                raw_outputs[i, j] = output_dict[name]
        
        # For each output variable, compute membership in each term
        for j, var_name in enumerate(self.output_names):
            var = self.system.output_variables[var_name]
            term_names = list(var.terms.keys())
            n_terms = len(term_names)
            
            # Initialize membership array for this variable
            var_memberships = np.zeros((n_samples, n_terms))
            
            # Compute membership for each sample
            for i in range(n_samples):
                output_value = raw_outputs[i, j]
                
                for k, term_name in enumerate(term_names):
                    fuzzy_set = var.terms[term_name]
                    var_memberships[i, k] = fuzzy_set.membership(output_value)
            
            memberships[var_name] = var_memberships
        
        return memberships


    def predict_membership_detailed(self, X: np.ndarray) -> List[Dict]:
        """
        Return detailed membership information for each sample.
        
        Returns a list of dictionaries, one per sample, with complete
        membership information for all output variables and their terms.
        
        Parameters:
            X: Input array (n_samples, n_features)
        
        Returns:
            List of dictionaries, each containing:
            - 'sample_id': Sample index
            - 'raw_outputs': Raw fuzzy output values
            - 'memberships': Dict of {var_name: {term_name: membership_degree}}
            - 'dominant_terms': Dict of {var_name: (term_name, membership_degree)}
        
        Example:
            >>> details = wm.predict_membership_detailed(X_test[:2])
            >>> print(details[0])
            {
                'sample_id': 0,
                'raw_outputs': {'setosa': 0.75, 'versicolor': 0.25, 'virginica': 0.18},
                'memberships': {
                    'setosa': {'no': 0.05, 'yes': 0.95},
                    'versicolor': {'no': 0.75, 'yes': 0.25},
                    'virginica': {'no': 0.82, 'yes': 0.18}
                },
                'dominant_terms': {
                    'setosa': ('yes', 0.95),
                    'versicolor': ('no', 0.75),
                    'virginica': ('no', 0.82)
                }
            }
        """
        if len(self.final_rules) == 0:
            raise ValueError("Model must be trained before calling predict_membership_detailed()")
        
        X = np.atleast_2d(X)
        n_samples = X.shape[0]
        
        detailed_results = []
        
        for i in range(n_samples):
            # Get raw outputs
            input_dict = {name: X[i, j] for j, name in enumerate(self.input_names)}
            output_dict = self.system.evaluate(input_dict)
            
            # Initialize sample result
            sample_result = {
                'sample_id': i,
                'raw_outputs': {},
                'memberships': {},
                'dominant_terms': {}
            }
            
            # For each output variable
            for var_name in self.output_names:
                output_value = output_dict[var_name]
                sample_result['raw_outputs'][var_name] = output_value
                
                var = self.system.output_variables[var_name]
                term_memberships = {}
                max_membership = -1
                dominant_term = None
                
                # Compute membership in each term
                for term_name, fuzzy_set in var.terms.items():
                    membership = fuzzy_set.membership(output_value)
                    term_memberships[term_name] = membership
                    
                    if membership > max_membership:
                        max_membership = membership
                        dominant_term = term_name
                
                sample_result['memberships'][var_name] = term_memberships
                sample_result['dominant_terms'][var_name] = (dominant_term, max_membership)
            
            detailed_results.append(sample_result)
        
        return detailed_results

# Aliases for backwards compatibility and convenience
WangMendelRegression = WangMendelLearning
WangMendelClassification = WangMendelLearning
WM = WangMendelLearning
