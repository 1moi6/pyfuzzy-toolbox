"""
Inference Engine Module
Converts Streamlit session state to pyfuzzy-toolbox FIS and performs inference
"""

import numpy as np
from typing import Dict, Any, Optional
import fuzzy_systems as fs


class InferenceEngine:
    """
    Engine to convert Streamlit FIS format to pyfuzzy-toolbox and perform inference
    """

    def __init__(self, fis_data: Dict[str, Any]):
        """
        Initialize the inference engine with FIS data from Streamlit

        Parameters
        ----------
        fis_data : dict
            FIS data from st.session_state containing:
            - name: str
            - type: str ('Mamdani' or 'Sugeno (TSK)')
            - input_variables: list
            - output_variables: list
            - fuzzy_rules: list
        """
        self.fis_data = fis_data
        self.system = None
        self._build_system()

    def _build_system(self):
        """Build the pyfuzzy-toolbox FIS from Streamlit data"""
        try:
            # Determine system type
            if 'Sugeno' in self.fis_data['type'] or 'TSK' in self.fis_data['type']:
                self.system = fs.SugenoSystem()
            else:
                self.system = fs.MamdaniSystem()

            # Add input variables
            for var in self.fis_data['input_variables']:
                self.system.add_input(var['name'], (var['min'], var['max']))

                # Add terms for this variable
                for term in var['terms']:
                    self.system.add_term(
                        var['name'],
                        term['name'],
                        term['mf_type'],
                        tuple(term['params'])
                    )

            # Add output variables
            for var in self.fis_data['output_variables']:
                self.system.add_output(var['name'], (var['min'], var['max']))

                # Add terms for this variable
                for term in var['terms']:
                    self.system.add_term(
                        var['name'],
                        term['name'],
                        term['mf_type'],
                        tuple(term['params'])
                    )

            # Add rules
            rules = []
            for rule in self.fis_data['fuzzy_rules']:
                # Create rule tuple: (antecedent_terms..., consequent_terms...)
                ant_terms = tuple(rule['antecedents'].values())
                cons_terms = tuple(rule['consequents'].values())
                rules.append(ant_terms + cons_terms)

            if rules:
                self.system.add_rules(rules)

        except Exception as e:
            raise ValueError(f"Error building FIS: {str(e)}")

    def evaluate(self, inputs: Dict[str, float]) -> Dict[str, float]:
        """
        Evaluate the FIS with given inputs

        Parameters
        ----------
        inputs : dict
            Dictionary mapping input variable names to values

        Returns
        -------
        dict
            Dictionary mapping output variable names to computed values
        """
        if self.system is None:
            raise ValueError("System not built. Check FIS configuration.")

        try:
            # Convert inputs to kwargs for evaluate
            result = self.system.evaluate(**inputs)
            return result
        except Exception as e:
            raise ValueError(f"Error during evaluation: {str(e)}")

    def get_fuzzification(self, var_name: str, value: float) -> Dict[str, float]:
        """
        Get fuzzification degrees for a variable value

        Parameters
        ----------
        var_name : str
            Variable name
        value : float
            Input value

        Returns
        -------
        dict
            Dictionary mapping term names to membership degrees
        """
        if self.system is None:
            raise ValueError("System not built.")

        try:
            # Get variable
            if var_name in self.system.input_variables:
                var = self.system.input_variables[var_name]
            elif var_name in self.system.output_variables:
                var = self.system.output_variables[var_name]
            else:
                raise ValueError(f"Variable '{var_name}' not found")

            # Compute membership for each term
            memberships = {}
            for term_name, term in var.terms.items():
                memberships[term_name] = term.membership(value)

            return memberships
        except Exception as e:
            raise ValueError(f"Error computing fuzzification: {str(e)}")

    def get_rule_activations(self, inputs: Dict[str, float]) -> list:
        """
        Get activation degree for each rule

        Parameters
        ----------
        inputs : dict
            Dictionary mapping input variable names to values

        Returns
        -------
        list
            List of tuples (rule_index, activation_degree, rule_dict)
        """
        if self.system is None:
            raise ValueError("System not built.")

        try:
            activations = []

            for idx, rule in enumerate(self.fis_data['fuzzy_rules']):
                # Compute activation for this rule
                degrees = []

                for var_name, term_name in rule['antecedents'].items():
                    if var_name in inputs:
                        var = self.system.input_variables[var_name]
                        term = var.terms[term_name]
                        degree = term.membership(inputs[var_name])
                        degrees.append(degree)

                # Use min for AND (default)
                activation = min(degrees) if degrees else 0.0

                activations.append((idx, activation, rule))

            return activations
        except Exception as e:
            raise ValueError(f"Error computing rule activations: {str(e)}")

    def get_universe(self, var_name: str, n_points: int = 200) -> np.ndarray:
        """
        Get universe of discourse for a variable

        Parameters
        ----------
        var_name : str
            Variable name
        n_points : int
            Number of points

        Returns
        -------
        np.ndarray
            Array of values spanning the universe
        """
        if self.system is None:
            raise ValueError("System not built.")

        # Find variable
        if var_name in self.system.input_variables:
            var = self.system.input_variables[var_name]
        elif var_name in self.system.output_variables:
            var = self.system.output_variables[var_name]
        else:
            raise ValueError(f"Variable '{var_name}' not found")

        return np.linspace(var.universe[0], var.universe[1], n_points)

    def get_term_membership_curve(self, var_name: str, term_name: str,
                                   n_points: int = 200) -> tuple:
        """
        Get membership function curve for a term

        Parameters
        ----------
        var_name : str
            Variable name
        term_name : str
            Term name
        n_points : int
            Number of points

        Returns
        -------
        tuple
            (x_values, y_values) for plotting
        """
        if self.system is None:
            raise ValueError("System not built.")

        try:
            # Get universe
            x = self.get_universe(var_name, n_points)

            # Get variable and term
            if var_name in self.system.input_variables:
                var = self.system.input_variables[var_name]
            else:
                var = self.system.output_variables[var_name]

            term = var.terms[term_name]

            # Compute membership values
            y = np.array([term.membership(xi) for xi in x])

            return x, y
        except Exception as e:
            raise ValueError(f"Error computing membership curve: {str(e)}")

    def validate_fis(self) -> tuple:
        """
        Validate if FIS is ready for inference

        Returns
        -------
        tuple
            (is_valid: bool, message: str)
        """
        # Check input variables
        if not self.fis_data['input_variables']:
            return False, "No input variables defined"

        # Check output variables
        if not self.fis_data['output_variables']:
            return False, "No output variables defined"

        # Check if all variables have terms
        for var in self.fis_data['input_variables']:
            if not var['terms']:
                return False, f"Input variable '{var['name']}' has no terms"

        for var in self.fis_data['output_variables']:
            if not var['terms']:
                return False, f"Output variable '{var['name']}' has no terms"

        # Check rules
        if not self.fis_data['fuzzy_rules']:
            return False, "No rules defined"

        # Check if system was built successfully
        if self.system is None:
            return False, "Failed to build FIS system"

        return True, "FIS is ready for inference"
