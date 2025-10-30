"""
Módulo de Sistemas de Inferência Fuzzy

Este módulo implementa sistemas FIS completos (Mamdani e Sugeno)
que integram todos os componentes: fuzzificação, inferência e defuzzificação.
"""

import numpy as np
from typing import Dict, List, Union, Tuple, Optional, Callable
from ..core.fuzzification import LinguisticVariable, Fuzzifier, FuzzySet
from .rules import RuleBase, FuzzyRule, MamdaniInference, SugenoInference
from ..core.defuzzification import DefuzzMethod, defuzzify, mamdani_defuzzify
from ..core.operators import TNorm, SNorm


class FuzzyInferenceSystem:
    """
    Classe base abstrata para sistemas de inferência fuzzy.
    """

    def __init__(self, name: str = "FIS"):
        """
        Inicializa o sistema FIS.

        Parâmetros:
            name: Nome do sistema
        """
        self.name = name
        self.input_variables: Dict[str, LinguisticVariable] = {}
        self.output_variables: Dict[str, LinguisticVariable] = {}
        self.rule_base = RuleBase()

    def add_input(self,
                  name_or_variable: Union[str, LinguisticVariable],
                  universe: Optional[Tuple[float, float]] = None) -> LinguisticVariable:
        """
        Adiciona uma variável de entrada ao sistema.

        Aceita duas formas:

        Forma 1 (Direta - Recomendada):
            >>> system.add_input('temperatura', (0, 40))

        Forma 2 (Com LinguisticVariable):
            >>> system.add_input(fis.LinguisticVariable('temperatura', (0, 40)))

        Parâmetros:
            name_or_variable: Nome da variável (str) ou LinguisticVariable completa
            universe: Universo de discurso [min, max] (apenas se name_or_variable for str)

        Retorna:
            A variável linguística adicionada

        Raises:
            ValueError: Se parâmetros inválidos
        """
        # Forma 2: Objeto LinguisticVariable completo
        if isinstance(name_or_variable, LinguisticVariable):
            variable = name_or_variable
            self.input_variables[variable.name] = variable
            return variable

        # Forma 1: Parâmetros diretos
        elif isinstance(name_or_variable, str):
            if universe is None:
                raise ValueError(
                    "Quando passar nome como string, deve fornecer universe. "
                    "Exemplo: add_input('temperatura', (0, 40))"
                )

            variable = LinguisticVariable(name_or_variable, universe)
            self.input_variables[variable.name] = variable
            return variable

        else:
            raise TypeError(
                f"Primeiro parâmetro deve ser str ou LinguisticVariable, "
                f"recebido: {type(name_or_variable)}"
            )

    def add_output(self,
                   name_or_variable: Union[str, LinguisticVariable],
                   universe: Optional[Tuple[float, float]] = None) -> LinguisticVariable:
        """
        Adiciona uma variável de saída ao sistema.

        Aceita duas formas:

        Forma 1 (Direta - Recomendada):
            >>> system.add_output('ventilador', (0, 100))

        Forma 2 (Com LinguisticVariable):
            >>> system.add_output(fis.LinguisticVariable('ventilador', (0, 100)))

        Parâmetros:
            name_or_variable: Nome da variável (str) ou LinguisticVariable completa
            universe: Universo de discurso [min, max] (apenas se name_or_variable for str)

        Retorna:
            A variável linguística adicionada

        Raises:
            ValueError: Se parâmetros inválidos
        """
        # Forma 2: Objeto LinguisticVariable completo
        if isinstance(name_or_variable, LinguisticVariable):
            variable = name_or_variable
            self.output_variables[variable.name] = variable
            return variable

        # Forma 1: Parâmetros diretos
        elif isinstance(name_or_variable, str):
            if universe is None:
                raise ValueError(
                    "Quando passar nome como string, deve fornecer universe. "
                    "Exemplo: add_output('ventilador', (0, 100))"
                )

            variable = LinguisticVariable(name_or_variable, universe)
            self.output_variables[variable.name] = variable
            return variable

        else:
            raise TypeError(
                f"Primeiro parâmetro deve ser str ou LinguisticVariable, "
                f"recebido: {type(name_or_variable)}"
            )

    def add_rule(self, 
             rule_input: Union[Dict[str, Union[str, float]], List[Union[str, int]], Tuple[Union[str, int], ...]],
             operator: str = 'AND',
             weight: float = 1.0) -> None:
        """
        Adds a rule to the system in a simplified way.
        
        Parameters:
            rule_input: Can be:
                - Dict: {'var1': 'term1', ..., 'out1': 'term_out1', 'operator': 'OR', 'weight': 0.8}
                - List/Tuple of strings: ['term_in1', ..., 'term_out1', ...]
                - List/Tuple of integers: [idx_in1, ..., idx_out1, ...]
            operator: 'AND' or 'OR' (default: 'AND')
            weight: Rule weight between 0 and 1 (default: 1.0)
        """
        input_vars = list(self.input_variables.keys())
        output_vars = list(self.output_variables.keys())
        n_inputs = len(input_vars)
        n_outputs = len(output_vars)
        total_expected = n_inputs + n_outputs
        
        # Detect if Sugeno system
        is_sugeno = isinstance(self, SugenoSystem)
        
        if isinstance(rule_input, dict):
            antecedents = {}
            consequents = {}
            
            # Extract operator and weight if present
            rule_operator = rule_input.get('operator', operator)
            rule_weight = rule_input.get('weight', weight)
            
            # Validate operator
            if not isinstance(rule_operator, str) or rule_operator not in ['AND', 'OR']:
                raise ValueError(f"'operator' must be 'AND' or 'OR'. Received: {rule_operator}")
            
            # Validate weight
            if not isinstance(rule_weight, (int, float)):
                raise ValueError(f"'weight' must be numeric. Received: {rule_weight}")
            rule_weight = float(rule_weight)
            
            # Process variables
            for var, term in rule_input.items():
                if var in ['operator', 'weight']:
                    continue
                    
                if var in self.input_variables:
                    antecedents[var] = term
                elif var in self.output_variables:
                    if is_sugeno:
                        consequents[var] = term
                    else:
                        if not isinstance(term, str):
                            raise ValueError(
                                f"In Mamdani, consequent must be linguistic term (string). "
                                f"Received: {term}"
                            )
                        consequents[var] = term
                else:
                    raise ValueError(f"Variable '{var}' not found in system")
            
            # Validate number of variables
            if len(antecedents) != n_inputs:
                raise ValueError(f"Expected {n_inputs} input variables, received {len(antecedents)}")
            if len(consequents) != n_outputs:
                raise ValueError(f"Expected {n_outputs} output variables, received {len(consequents)}")
        
        elif isinstance(rule_input, (list, tuple)):
            if len(rule_input) != total_expected:
                raise ValueError(
                    f"List must have {total_expected} elements ({n_inputs} inputs + {n_outputs} outputs). "
                    f"Received {len(rule_input)}"
                )
            
            input_items = rule_input[:n_inputs]
            output_items = rule_input[n_inputs:]
            
            # For inputs: detect if indices or names
            use_index_inputs = all(isinstance(item, int) for item in input_items)
            
            if use_index_inputs:
                antecedents = {}
                for i, idx in enumerate(input_items):
                    var_name = input_vars[i]
                    term_name = self._index_to_term(var_name, idx, is_input=True)
                    antecedents[var_name] = term_name
            else:
                antecedents = {input_vars[i]: input_items[i] for i in range(n_inputs)}
            
            # For outputs: behavior depends on system type
            if is_sugeno:
                # Sugeno: accepts direct values
                consequents = {output_vars[i]: output_items[i] for i in range(n_outputs)}
            else:
                # Mamdani: convert indices to terms if needed
                use_index_outputs = all(isinstance(item, int) for item in output_items)
                if use_index_outputs:
                    consequents = {}
                    for i, idx in enumerate(output_items):
                        var_name = output_vars[i]
                        term_name = self._index_to_term(var_name, idx, is_input=False)
                        consequents[var_name] = term_name
                else:
                    consequents = {output_vars[i]: output_items[i] for i in range(n_outputs)}
            
            rule_operator = operator
            rule_weight = weight
        
        else:
            raise TypeError(f"rule_input must be dict, list or tuple. Received {type(rule_input).__name__}")
        
        # Create and add rule
        rule = FuzzyRule(antecedents, consequents, rule_operator, rule_weight)
        self.rule_base.add_rule(rule)
        self._remove_duplicate_rules()

    def _index_to_term(self, var_name: str, idx: int, is_input: bool) -> str:
        """
        Converts index to term name in a linguistic variable.
        
        Parameters:
            var_name: Variable name
            idx: Term index (0-based)
            is_input: True if input variable, False if output
        
        Returns:
            Term name corresponding to the index
        """
        var_dict = self.input_variables if is_input else self.output_variables
        
        if var_name not in var_dict:
            raise ValueError(f"Variable '{var_name}' not found")
        
        variable = var_dict[var_name]
        term_names = list(variable.terms.keys())
        
        if not isinstance(idx, int):
            raise TypeError(f"With indices, expected integer, received {type(idx).__name__}: {idx}")
        
        if idx < 0 or idx >= len(term_names):
            raise IndexError(
                f"Index {idx} out of range for variable '{var_name}'. "
                f"Available terms: {len(term_names)} (indices 0-{len(term_names)-1})"
            )
        
        return term_names[idx]


    def add_rules(self, 
              rules: List[Union[Dict[str, Union[str, float]], List[Union[str, int]], Tuple[Union[str, int], ...]]],
              operator: str = 'AND',
              weight: float = 1.0) -> None:
        """
        Adds multiple rules to the system in a simplified way.
        
        Parameters:
            rules: List of rules
            operator: Default operator for all rules
            weight: Default weight for all rules
        """
        input_vars = list(self.input_variables.keys())
        output_vars = list(self.output_variables.keys())
        n_inputs = len(input_vars)
        n_outputs = len(output_vars)
        total_vars = n_inputs + n_outputs
        
        for rule_input in rules:
            if isinstance(rule_input, dict):
                self.add_rule(rule_input, operator=operator, weight=weight)
                continue
            
            if isinstance(rule_input, (list, tuple)):
                rule_length = len(rule_input)
                
                if rule_length == total_vars:
                    rule_operator = operator
                    rule_weight = weight
                    actual_rule = rule_input
                    
                elif rule_length == total_vars + 1:
                    extra = rule_input[-1]
                    
                    if isinstance(extra, str) and extra in ['AND', 'OR']:
                        rule_operator = extra
                        rule_weight = weight
                        actual_rule = rule_input[:-1]
                    elif isinstance(extra, (int, float)):
                        rule_operator = operator
                        rule_weight = float(extra)
                        actual_rule = rule_input[:-1]
                    else:
                        raise ValueError(f"Extra element must be operator or weight. Received: {extra}")
                
                elif rule_length == total_vars + 2:
                    penultimate = rule_input[-2]
                    ultimate = rule_input[-1]
                    
                    if isinstance(penultimate, str) and penultimate in ['AND', 'OR']:
                        if isinstance(ultimate, (int, float)):
                            rule_operator = penultimate
                            rule_weight = float(ultimate)
                            actual_rule = rule_input[:-2]
                        else:
                            raise ValueError(f"Last element must be numeric weight. Received: {ultimate}")
                    else:
                        raise ValueError(f"Penultimate must be operator. Received: {penultimate}")
                
                else:
                    raise ValueError(
                        f"Rule must have {total_vars} variables (or +1/+2 for operator/weight). "
                        f"Received {rule_length} elements."
                    )
                
                self.add_rule(actual_rule, operator=rule_operator, weight=rule_weight)
            
            else:
                raise TypeError(f"Each rule must be dict, list or tuple")


    def add_term(self,
                 variable_name: str,
                 term_name: str,
                 mf_type: str,
                 params: Tuple,
                 mf_func: Optional[Callable] = None) -> None:
        """
        Adiciona um termo fuzzy a uma variável do sistema.

        Busca automaticamente a variável (entrada ou saída) pelo nome
        e adiciona o termo a ela.

        Parâmetros:
            variable_name: Nome da variável (entrada ou saída)
            term_name: Nome do termo fuzzy
            mf_type: Tipo da função de pertinência
            params: Parâmetros da função
            mf_func: Função customizada opcional

        Raises:
            ValueError: Se a variável não existir

        Exemplo:
            >>> system = fis.MamdaniSystem()
            >>> system.add_input(fis.LinguisticVariable('temperatura', (0, 40)))
            >>> system.add_term('temperatura', 'baixa', 'triangular', (0, 0, 20))
            >>> system.add_term('temperatura', 'alta', 'triangular', (20, 40, 40))
        """
        # Busca primeiro nas entradas
        if variable_name in self.input_variables:
            self.input_variables[variable_name].add_term(
                term_name, mf_type, params, mf_func
            )
            return

        # Busca nas saídas
        if variable_name in self.output_variables:
            self.output_variables[variable_name].add_term(
                term_name, mf_type, params, mf_func
            )
            return

        # Variável não encontrada
        available_vars = list(self.input_variables.keys()) + list(self.output_variables.keys())
        raise ValueError(
            f"Variável '{variable_name}' não encontrada no sistema. "
            f"Variáveis disponíveis: {available_vars}"
        )

    def _normalize_inputs(self, *args, **kwargs) -> Dict[str, float]:
        """
        Normaliza diferentes formatos de entrada para dicionário.

        Aceita:
        1. Dicionário: {'var1': val1, 'var2': val2}
        2. Lista/Tupla: [val1, val2] (ordem de adição das variáveis)
        3. Args diretos: val1, val2

        Retorna:
            Dicionário {variável: valor}
        """
        # Se tem kwargs, usa como dicionário
        if kwargs:
            return kwargs

        # Se tem apenas um argumento
        if len(args) == 1:
            arg = args[0]

            # Se já é dicionário, retorna
            if isinstance(arg, dict):
                return arg

            # Se é lista/tupla, converte para dicionário usando ordem das variáveis
            if isinstance(arg, (list, tuple, np.ndarray)):
                if len(arg) != len(self.input_variables):
                    raise ValueError(
                        f"Número de valores ({len(arg)}) não corresponde ao "
                        f"número de variáveis de entrada ({len(self.input_variables)})"
                    )

                # Usa a ordem de inserção das variáveis (Python 3.7+ garante ordem em dicts)
                var_names = list(self.input_variables.keys())
                return {var_names[i]: float(arg[i]) for i in range(len(arg))}

            # Se é um único valor numérico e só há uma variável
            if len(self.input_variables) == 1:
                var_name = list(self.input_variables.keys())[0]
                return {var_name: float(arg)}

        # Se tem múltiplos args, trata como valores ordenados
        elif len(args) > 1:
            if len(args) != len(self.input_variables):
                raise ValueError(
                    f"Número de argumentos ({len(args)}) não corresponde ao "
                    f"número de variáveis de entrada ({len(self.input_variables)})"
                )

            var_names = list(self.input_variables.keys())
            return {var_names[i]: float(args[i]) for i in range(len(args))}

        raise ValueError("Formato de entrada inválido. Use dicionário, lista, tupla ou argumentos diretos.")

    def evaluate(self, *args, **kwargs) -> Dict[str, float]:
        """
        Avalia o sistema fuzzy para as entradas fornecidas.

        Aceita múltiplos formatos de entrada:

        1. Dicionário:
            >>> system.evaluate({'temperatura': 25, 'umidade': 60})
            >>> system.evaluate(temperatura=25, umidade=60)

        2. Lista/Tupla (ordem de adição das variáveis):
            >>> system.evaluate([25, 60])
            >>> system.evaluate((25, 60))

        3. Argumentos diretos:
            >>> system.evaluate(25, 60)

        Parâmetros:
            *args: Valores de entrada (vários formatos)
            **kwargs: Valores de entrada como argumentos nomeados

        Retorna:
            Dicionário {variável_saída: valor}
        """
        raise NotImplementedError("Subclasses devem implementar evaluate()")

    def compute(self, *args, **kwargs) -> Dict[str, float]:
        """
        Alias para evaluate() mantido para compatibilidade.

        DEPRECATED: Use evaluate() ao invés de compute().

        Parâmetros:
            *args: Valores de entrada
            **kwargs: Valores de entrada como argumentos nomeados

        Retorna:
            Dicionário {variável_saída: valor}
        """
        import warnings
        warnings.warn(
            "compute() está deprecated. Use evaluate() ao invés.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.evaluate(*args, **kwargs)

    def plot_variables(self, 
                   variables: Optional[List[str]] = None,
                   show_terms: bool = True,
                   figsize: Optional[Tuple[float, float]] = None,
                   return_axes: bool = False) -> Optional[Tuple]:
        """
        Plots linguistic variables and their fuzzy terms.
        
        Parameters:
            variables: List of variable names to plot. If None, plots all variables.
            show_terms: If True, displays term names on the plot
            figsize: Figure size (width, height). If None, automatically calculated.
            return_axes: If True, returns (fig, axes) without showing. If False, shows plot.
        
        Returns:
            If return_axes=True: tuple (fig, axes)
            If return_axes=False: None (displays plot)
        
        Examples:
            >>> # Plot and show
            >>> system.plot_variables()
            
            >>> # Get axes for customization
            >>> fig, axes = system.plot_variables(return_axes=True)
            >>> axes[0].set_title('My Custom Title')
            >>> plt.show()
        """
        import matplotlib.pyplot as plt
        
        # Collect variables to plot
        all_vars = {}
        all_vars.update(self.input_variables)
        all_vars.update(self.output_variables)
        
        if variables is None:
            vars_to_plot = all_vars
        else:
            vars_to_plot = {name: all_vars[name] for name in variables if name in all_vars}
        
        if not vars_to_plot:
            print("⚠️  No variables to plot!")
            return None if return_axes else None
        
        n_vars = len(vars_to_plot)
        
        # Calculate figure size
        if figsize is None:
            width = 12
            height = 3 * n_vars
            figsize = (width, height)
        
        # Create subplots
        fig, axes = plt.subplots(n_vars, 1, figsize=figsize, squeeze=False)
        axes = axes.flatten()
        
        # Plot each variable
        for idx, (var_name, var) in enumerate(vars_to_plot.items()):
            ax = axes[idx]
            
            # Get universe
            x_min, x_max = var.universe
            x = np.linspace(x_min, x_max, 1000)
            
            # Plot each term
            for term_name, fuzzy_set in var.terms.items():
                y = fuzzy_set.membership(x)
                ax.plot(x, y, linewidth=2.5, label=term_name, alpha=0.8)
                
                # Add term label if requested
                if show_terms:
                    # Find peak of membership function
                    max_idx = np.argmax(y)
                    ax.text(x[max_idx], y[max_idx], term_name,
                        ha='center', va='bottom', fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.3', 
                                    facecolor='white', alpha=0.7, edgecolor='gray'))
            
            # Styling
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(-0.05, 1.1)
            ax.set_xlabel(var_name, fontsize=12, fontweight='bold')
            ax.set_ylabel('Membership', fontsize=11)
            ax.set_title(f'Variable: {var_name}', fontsize=13, fontweight='bold', pad=10)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=10)
            
            # Add horizontal line at membership=1
            ax.axhline(y=1.0, color='black', linestyle='--', linewidth=0.8, alpha=0.3)
        
        plt.tight_layout()
        
        # Return or show
        if return_axes:
            return fig, axes
        else:
            plt.show()
            return None


    def plot_output(self, input_var, output_var, num_points=100, **kwargs):
        """
        Plota a saída do sistema em função de uma entrada (gráfico 2D).

        Parâmetros:
            input_var: Nome da variável de entrada
            output_var: Nome da variável de saída
            num_points: Número de pontos para avaliar
            **kwargs: Argumentos adicionais
                - figsize: Tamanho da figura (default: (10, 6))
                - color: Cor da linha
                - linewidth: Espessura da linha
                - grid: Se True, mostra grid

        Retorna:
            fig, ax: Figura e axes matplotlib

        Exemplo:
            >>> system.plot_output('temperatura', 'ventilador')

        Nota:
            Para sistemas com múltiplas entradas, as outras entradas
            serão fixadas no ponto médio do universo de discurso.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "Matplotlib não está instalado. "
                "Instale com: pip install matplotlib"
            )

        # Valida variáveis
        if input_var not in self.input_variables:
            raise ValueError(
                f"Variável de entrada '{input_var}' não encontrada. "
                f"Disponíveis: {list(self.input_variables.keys())}"
            )

        if output_var not in self.output_variables:
            raise ValueError(
                f"Variável de saída '{output_var}' não encontrada. "
                f"Disponíveis: {list(self.output_variables.keys())}"
            )

        # Gera valores de entrada
        input_variable = self.input_variables[input_var]
        x_values = np.linspace(
            input_variable.universe[0],
            input_variable.universe[1],
            num_points
        )

        # Para outras entradas, usa ponto médio
        fixed_inputs = {}
        for var_name, var in self.input_variables.items():
            if var_name != input_var:
                mid_point = (var.universe[0] + var.universe[1]) / 2
                fixed_inputs[var_name] = mid_point

        # Avalia sistema para cada valor
        y_values = []
        for x in x_values:
            inputs = {input_var: x, **fixed_inputs}
            output = self.evaluate(inputs)
            y_values.append(output[output_var])

        # Cria plot
        figsize = kwargs.get('figsize', (10, 6))
        fig, ax = plt.subplots(figsize=figsize)

        color = kwargs.get('color', 'blue')
        linewidth = kwargs.get('linewidth', 2)

        ax.plot(x_values, y_values, color=color, linewidth=linewidth)

        # Configurações
        ax.set_xlabel(input_var, fontsize=12)
        ax.set_ylabel(output_var, fontsize=12)
        ax.set_title(
            f'Resposta do Sistema: {output_var} vs {input_var}',
            fontsize=14,
            fontweight='bold'
        )

        if kwargs.get('grid', True):
            ax.grid(True, alpha=0.3, linestyle='--')

        # Mostra valores fixos de outras entradas (se houver)
        if fixed_inputs:
            fixed_str = ', '.join([f'{k}={v:.1f}' for k, v in fixed_inputs.items()])
            ax.text(
                0.02, 0.98, f'Fixo: {fixed_str}',
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
                fontsize=9
            )

        plt.tight_layout()
        plt.show()

        return fig, ax

    def rules_to_dataframe(self, format='standard'):
        """
        Converte as regras do sistema para um DataFrame Pandas.

        Parâmetros:
            format: Formato do DataFrame:
                   - 'standard' (default): Uma coluna por variável (apenas termos)
                   - 'compact': Colunas 'antecedents' e 'consequents' como texto

        Retorna:
            DataFrame com as regras

        Exemplo:
            >>> # Formato padrão (recomendado para CSV)
            >>> df = system.rules_to_dataframe()
            >>> print(df)
            >>> # Colunas: rule_id, var1, var2, ..., output1, output2, ..., operator, weight
            >>>
            >>> # Formato compacto
            >>> df = system.rules_to_dataframe(format='compact')
            >>> # Colunas: rule_id, antecedents, consequents, operator, weight
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "Pandas não está instalado. "
                "Instale com: pip install pandas"
            )

        if len(self.rule_base.rules) == 0:
            return pd.DataFrame()

        rules_data = []

        if format == 'standard':
            # Formato padrão: uma coluna por variável (apenas termos)
            for i, rule in enumerate(self.rule_base.rules, 1):
                row = {'rule_id': i}

                # Antecedentes (inputs) - apenas os termos
                for var_name in self.input_variables.keys():
                    term = rule.antecedents.get(var_name, '')
                    row[var_name] = term

                # Consequentes (outputs) - apenas os termos/valores
                for var_name in self.output_variables.keys():
                    value = rule.consequent.get(var_name, '')
                    if isinstance(value, (list, tuple)):
                        row[var_name] = str(value)
                    else:
                        row[var_name] = value

                row['operator'] = rule.operator
                row['weight'] = rule.weight

                if rule.label:
                    row['label'] = rule.label

                rules_data.append(row)

        elif format == 'compact':
            # Formato compacto: colunas de texto (backward compatibility)
            for i, rule in enumerate(self.rule_base.rules, 1):
                ant_str = ', '.join([f'{k}={v}' for k, v in rule.antecedents.items()])

                cons_items = []
                for k, v in rule.consequent.items():
                    if isinstance(v, (list, tuple)):
                        cons_items.append(f'{k}={v}')
                    else:
                        cons_items.append(f'{k}={v}')
                cons_str = ', '.join(cons_items)

                row = {
                    'rule_id': i,
                    'antecedents': ant_str,
                    'consequents': cons_str,
                    'operator': rule.operator,
                    'weight': rule.weight
                }

                if rule.label:
                    row['label'] = rule.label

                rules_data.append(row)
        else:
            raise ValueError(f"Formato '{format}' inválido. Use 'standard' ou 'compact'.")

        return pd.DataFrame(rules_data)

    def info(self, show_terms=True, show_rules_summary=True, show_usage_example=True):
        """
        Exibe informações completas sobre o sistema fuzzy.

        Parâmetros:
            show_terms: Se True, mostra os termos de cada variável
            show_rules_summary: Se True, mostra resumo das regras
            show_usage_example: Se True, mostra exemplo de uso do add_rule

        Exemplo:
            >>> system.info()
            >>> system.info(show_terms=False)
        """
        print(f"\n{'=' * 70}")
        print(f"📊 INFORMAÇÕES DO SISTEMA: {self.name}")
        print(f"{'=' * 70}\n")

        # Informações básicas
        print(f"🔧 Tipo: {self.__class__.__name__}")
        print(f"📝 Nome: {self.name}")
        print()

        # Variáveis de entrada
        print(f"📥 VARIÁVEIS DE ENTRADA ({len(self.input_variables)}):")
        print(f"{'─' * 70}")
        for i, (var_name, var) in enumerate(self.input_variables.items(), 1):
            print(f"  {i}. '{var_name}'")
            print(f"     Universo: {var.universe}")
            if show_terms:
                print(f"     Termos ({len(var.terms)}):", end='')
                terms_str = ', '.join([f"'{t}'" for t in var.terms.keys()])
                print(f" {terms_str}")
            else:
                print(f"     Termos: {len(var.terms)}")
            print()

        # Variáveis de saída
        print(f"📤 VARIÁVEIS DE SAÍDA ({len(self.output_variables)}):")
        print(f"{'─' * 70}")
        for i, (var_name, var) in enumerate(self.output_variables.items(), 1):
            print(f"  {i}. '{var_name}'")
            print(f"     Universo: {var.universe}")
            if show_terms:
                print(f"     Termos ({len(var.terms)}):", end='')
                terms_str = ', '.join([f"'{t}'" for t in var.terms.keys()])
                print(f" {terms_str}")
            else:
                print(f"     Termos: {len(var.terms)}")
            print()

        # Regras
        if show_rules_summary:
            print(f"📋 REGRAS:")
            print(f"{'─' * 70}")
            print(f"  Total: {len(self.rule_base.rules)} regras")
            if len(self.rule_base.rules) > 0:
                # Contar operadores
                operators = {}
                weights = []
                for rule in self.rule_base.rules:
                    op = rule.operator
                    operators[op] = operators.get(op, 0) + 1
                    weights.append(rule.weight)

                print(f"  Operadores: {dict(operators)}")
                print(f"  Peso médio: {sum(weights)/len(weights):.2f}")
                print(f"  Peso mín/máx: {min(weights):.2f} / {max(weights):.2f}")
            print()

        # Configurações
        print(f"⚙️  CONFIGURAÇÕES:")
        print(f"{'─' * 70}")
        if hasattr(self, 'defuzzification_method'):
            print(f"  Defuzzificação: {self.defuzzification_method}")
        if hasattr(self, 'aggregation_method'):
            print(f"  Agregação: {self.aggregation_method}")
        print()

        # Exemplo de uso
        if show_usage_example and len(self.input_variables) > 0 and len(self.output_variables) > 0:
            print(f"💡 EXEMPLO DE USO:")
            print(f"{'─' * 70}")

            # Pegar primeiro termo de cada variável
            input_vars = list(self.input_variables.keys())
            output_vars = list(self.output_variables.keys())

            first_input_terms = []
            for var_name in input_vars:
                terms = list(self.input_variables[var_name].terms.keys())
                first_input_terms.append(terms[0] if terms else '???')

            first_output_terms = []
            for var_name in output_vars:
                terms = list(self.output_variables[var_name].terms.keys())
                first_output_terms.append(terms[0] if terms else '???')

            # Mostrar exemplo de add_rule com tupla plana
            all_terms = first_input_terms + first_output_terms
            terms_str = ', '.join([f"'{t}'" for t in all_terms])

            print(f"  # Adicionar uma regra (sintaxe recomendada - tupla plana):")
            print(f"  system.add_rule({terms_str})")
            print()

            print(f"  # Adicionar múltiplas regras:")
            print(f"  system.add_rules([")
            print(f"      ({terms_str}),")
            print(f"      # ... mais regras ...")
            print(f"  ])")
            print()

            # Mostrar exemplo de evaluate
            input_example = {var: f"{self.input_variables[var].universe[0]}"
                           for var in input_vars}
            input_str = ', '.join([f"{k}={v}" for k, v in input_example.items()])
            print(f"  # Avaliar o sistema:")
            print(f"  result = system.evaluate({{{input_str}}})")
            print()

        print(f"{'=' * 70}\n")

    def print_rules(self, style='table', show_stats=True):
        """
        Imprime as regras do sistema de forma formatada.

        Parâmetros:
            style: Estilo de formatação ('table', 'compact', 'detailed', 'if-then')
            show_stats: Se True, mostra estatísticas no final

        Exemplo:
            >>> system.print_rules()
            >>> system.print_rules(style='compact')
        """
        if len(self.rule_base.rules) == 0:
            print("Sistema não possui regras.")
            return

        print(f"\n{'=' * 70}")
        print(f"REGRAS DO SISTEMA: {self.name}")
        print(f"{'=' * 70}\n")

        if style == 'table':
            self._print_rules_table()
        elif style == 'compact':
            self._print_rules_compact()
        elif style == 'detailed':
            self._print_rules_detailed()
        elif style == 'if-then':
            self._print_rules_if_then()
        else:
            raise ValueError(f"Estilo '{style}' inválido. Use: 'table', 'compact', 'detailed', 'if-then'")

        if show_stats:
            self._print_rules_stats()

    def _print_rules_table(self):
        """Imprime regras em formato de tabela"""
        print(f"{'ID':<5} {'IF':<35} {'THEN':<25} {'Op':<5} {'Peso':<5}")
        print("-" * 75)

        for i, rule in enumerate(self.rule_base.rules, 1):
            ant = ' AND '.join([f'{k}={v}' for k, v in rule.antecedents.items()])
            if rule.operator == 'OR':
                ant = ' OR '.join([f'{k}={v}' for k, v in rule.antecedents.items()])

            cons = ', '.join([f'{k}={v}' for k, v in rule.consequent.items()])

            # Quebra linhas longas
            if len(ant) > 33:
                ant = ant[:30] + '...'
            if len(cons) > 23:
                cons = cons[:20] + '...'

            print(f"{i:<5} {ant:<35} {cons:<25} {rule.operator:<5} {rule.weight:<5.2f}")

    def _print_rules_compact(self):
        """Imprime regras em formato compacto"""
        for i, rule in enumerate(self.rule_base.rules, 1):
            ant = f" {rule.operator} ".join([f'{k}={v}' for k, v in rule.antecedents.items()])
            cons = ', '.join([f'{k}={v}' for k, v in rule.consequent.items()])
            print(f"{i}. IF {ant} THEN {cons}")

    def _print_rules_detailed(self):
        """Imprime regras em formato detalhado"""
        for i, rule in enumerate(self.rule_base.rules, 1):
            print(f"Regra {i}:")
            print(f"  Antecedentes:")
            for var, term in rule.antecedents.items():
                print(f"    - {var} = {term}")
            print(f"  Consequentes:")
            for var, value in rule.consequent.items():
                print(f"    - {var} = {value}")
            print(f"  Operador: {rule.operator}")
            print(f"  Peso: {rule.weight}")
            if rule.label:
                print(f"  Rótulo: {rule.label}")
            print()

    def _print_rules_if_then(self):
        """Imprime regras em linguagem natural"""
        for i, rule in enumerate(self.rule_base.rules, 1):
            # Monta IF
            if_parts = [f"{var} É {term}" for var, term in rule.antecedents.items()]
            if_str = f" {rule.operator} ".join(if_parts)

            # Monta THEN
            then_parts = [f"{var} É {value}" for var, value in rule.consequent.items()]
            then_str = " E ".join(then_parts)

            print(f"Regra {i}:")
            print(f"  SE {if_str}")
            print(f"  ENTÃO {then_str}")
            if rule.weight != 1.0:
                print(f"  (Peso: {rule.weight})")
            print()

    def _print_rules_stats(self):
        """Imprime estatísticas das regras"""
        stats = self.rules_statistics()

        print(f"\n{'-' * 70}")
        print("ESTATÍSTICAS:")
        print(f"  Total de regras: {stats['total']}")
        print(f"  Operadores: {dict(stats['by_operator'])}")
        print(f"  Média de antecedentes por regra: {stats['avg_antecedents']:.1f}")
        print(f"  Média de consequentes por regra: {stats['avg_consequents']:.1f}")
        print(f"  Peso médio: {stats['avg_weight']:.2f}")
        if stats['min_weight'] != stats['max_weight']:
            print(f"  Peso mín/máx: {stats['min_weight']:.2f} / {stats['max_weight']:.2f}")

    def rules_statistics(self):
        """
        Retorna estatísticas sobre as regras do sistema.

        Retorna:
            Dicionário com estatísticas
        """
        if len(self.rule_base.rules) == 0:
            return {
                'total': 0,
                'by_operator': {},
                'avg_antecedents': 0,
                'avg_consequents': 0,
                'avg_weight': 0,
                'min_weight': 0,
                'max_weight': 0
            }

        operators = {}
        total_antecedents = 0
        total_consequents = 0
        weights = []

        for rule in self.rule_base.rules:
            # Conta operadores
            operators[rule.operator] = operators.get(rule.operator, 0) + 1

            # Conta antecedentes e consequentes
            total_antecedents += len(rule.antecedents)
            total_consequents += len(rule.consequent)

            # Coleta pesos
            weights.append(rule.weight)

        n_rules = len(self.rule_base.rules)

        return {
            'total': n_rules,
            'by_operator': operators,
            'avg_antecedents': total_antecedents / n_rules,
            'avg_consequents': total_consequents / n_rules,
            'avg_weight': sum(weights) / n_rules,
            'min_weight': min(weights),
            'max_weight': max(weights)
        }

    def export_rules(self, filename, format='auto'):
        """
        Exporta as regras para um arquivo.

        Parâmetros:
            filename: Nome do arquivo de saída
            format: Formato do arquivo ('auto', 'csv', 'json', 'txt', 'excel')
                   'auto' detecta pela extensão do arquivo

        Exemplo:
            >>> system.export_rules('regras.csv')
            >>> system.export_rules('regras.json')
            >>> system.export_rules('regras.txt', format='txt')
        """
        import os

        # Detecta formato pela extensão
        if format == 'auto':
            ext = os.path.splitext(filename)[1].lower()
            format_map = {
                '.csv': 'csv',
                '.json': 'json',
                '.txt': 'txt',
                '.xlsx': 'excel',
                '.xls': 'excel'
            }
            format = format_map.get(ext, 'csv')

        if format == 'csv':
            self._export_rules_csv(filename)
        elif format == 'json':
            self._export_rules_json(filename)
        elif format == 'txt':
            self._export_rules_txt(filename)
        elif format == 'excel':
            self._export_rules_excel(filename)
        else:
            raise ValueError(f"Formato '{format}' não suportado. Use: csv, json, txt, excel")

        print(f"✓ Regras exportadas para: {filename}")

    def _export_rules_csv(self, filename):
        """Exporta regras para CSV (formato padrão: uma coluna por variável)"""
        df = self.rules_to_dataframe(format='standard')
        df.to_csv(filename, index=False, encoding='utf-8')

    def _export_rules_json(self, filename):
        """Exporta regras para JSON"""
        import json

        data = {
            'system_name': self.name,
            'system_type': self.__class__.__name__,
            'inputs': list(self.input_variables.keys()),
            'outputs': list(self.output_variables.keys()),
            'rules': []
        }

        for i, rule in enumerate(self.rule_base.rules, 1):
            rule_data = {
                'id': i,
                'if': rule.antecedents,
                'then': rule.consequent,
                'operator': rule.operator,
                'weight': rule.weight
            }
            if rule.label:
                rule_data['label'] = rule.label
            data['rules'].append(rule_data)

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _export_rules_txt(self, filename):
        """Exporta regras para arquivo de texto"""
        import sys
        from io import StringIO

        # Captura print_rules output
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        self.print_rules(style='if-then', show_stats=True)

        content = sys.stdout.getvalue()
        sys.stdout = old_stdout

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)

    def _export_rules_excel(self, filename):
        """Exporta regras para Excel (formato padrão: uma coluna por variável)"""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("Pandas não está instalado. Instale com: pip install pandas openpyxl")

        try:
            df = self.rules_to_dataframe(format='standard')
            df.to_excel(filename, index=False, sheet_name='Regras')
        except ImportError:
            raise ImportError("openpyxl não está instalado. Instale com: pip install openpyxl")

    def import_rules(self, filename, format='auto', clear_existing=False):
        """
        Importa regras de um arquivo.

        Parâmetros:
            filename: Nome do arquivo de entrada
            format: Formato do arquivo ('auto', 'csv', 'json')
            clear_existing: Se True, limpa regras existentes antes de importar

        Exemplo:
            >>> system.import_rules('regras.csv')
            >>> system.import_rules('regras.json', clear_existing=True)
        """
        import os

        if not os.path.exists(filename):
            raise FileNotFoundError(f"Arquivo não encontrado: {filename}")

        # Detecta formato
        if format == 'auto':
            ext = os.path.splitext(filename)[1].lower()
            format_map = {
                '.csv': 'csv',
                '.json': 'json',
                '.xlsx': 'excel',
                '.xls': 'excel'
            }
            format = format_map.get(ext, 'csv')

        if clear_existing:
            self.rule_base.rules.clear()

        if format == 'csv':
            self._import_rules_csv(filename)
        elif format == 'json':
            self._import_rules_json(filename)
        elif format == 'excel':
            self._import_rules_excel(filename)
        else:
            raise ValueError(f"Formato '{format}' não suportado para importação")

        print(f"✓ {len(self.rule_base.rules)} regras importadas de: {filename}")

    def _import_rules_csv(self, filename):
        """Importa regras de CSV (suporta formato padrão e compacto)"""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("Pandas necessário para importar CSV")

        df = pd.read_csv(filename)

        # Detecta formato
        if 'antecedents' in df.columns and 'consequents' in df.columns:
            # Formato compacto (antigo): 'antecedents' e 'consequents' como texto
            for _, row in df.iterrows():
                # Parse antecedents
                ant_dict = {}
                for item in row['antecedents'].split(','):
                    k, v = item.strip().split('=')
                    ant_dict[k.strip()] = v.strip()

                # Parse consequents
                cons_dict = {}
                for item in row['consequents'].split(','):
                    k, v = item.strip().split('=')
                    # Tenta converter para número
                    try:
                        v = float(v)
                    except:
                        pass
                    cons_dict[k.strip()] = v

                operator = row.get('operator', 'AND')
                weight = row.get('weight', 1.0)

                self.add_rule(ant_dict, cons_dict, operator, weight)
        else:
            # Formato padrão (novo): uma coluna por variável (apenas termos)
            # Colunas: rule_id, var1, var2, ..., output1, output2, ..., operator, weight

            # Identificar quais colunas são variáveis (não são metadata)
            meta_cols = {'rule_id', 'operator', 'weight', 'label'}
            var_cols = [col for col in df.columns if col not in meta_cols]

            # Separar inputs e outputs baseado nas variáveis do sistema
            input_vars = set(self.input_variables.keys())
            output_vars = set(self.output_variables.keys())

            for _, row in df.iterrows():
                ant_dict = {}
                cons_dict = {}

                for col in var_cols:
                    value = row[col]

                    # Ignorar valores vazios (NaN ou string vazia)
                    if pd.isna(value) or value == '':
                        continue

                    # Classificar como input ou output
                    if col in input_vars:
                        ant_dict[col] = str(value).strip()
                    elif col in output_vars:
                        # Tentar converter para número (Sugeno)
                        try:
                            cons_dict[col] = float(value)
                        except (ValueError, TypeError):
                            cons_dict[col] = str(value).strip()
                    else:
                        # Coluna desconhecida - tentar adivinhar
                        # Se o sistema ainda não tem variáveis definidas, adicionar como input
                        if len(input_vars) == 0 and len(output_vars) == 0:
                            # Sistema vazio - assumir primeiras são inputs
                            ant_dict[col] = str(value).strip()
                        else:
                            # Assumir que é output
                            try:
                                cons_dict[col] = float(value)
                            except (ValueError, TypeError):
                                cons_dict[col] = str(value).strip()

                operator = row.get('operator', 'AND')
                weight = row.get('weight', 1.0)

                if ant_dict and cons_dict:
                    self.add_rule(ant_dict, cons_dict, operator, weight)

    def _import_rules_json(self, filename):
        """Importa regras de JSON"""
        import json

        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for rule_data in data['rules']:
            self.add_rule(
                rule_data['if'],
                rule_data['then'],
                rule_data.get('operator', 'AND'),
                rule_data.get('weight', 1.0)
            )

    def _import_rules_excel(self, filename):
        """Importa regras de Excel"""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("Pandas necessário para importar Excel")

        df = pd.read_excel(filename)
        # Usa mesma lógica do CSV
        # (Reutiliza _import_rules_csv convertendo df)
        temp_csv = filename + '.temp.csv'
        df.to_csv(temp_csv, index=False)
        self._import_rules_csv(temp_csv)
        import os
        os.remove(temp_csv)

    def _remove_duplicate_rules(self) -> None:
        """
        Removes duplicate rules from the rule base.
        Two rules are considered duplicates if they have the same antecedents and consequents.
        """
        seen = []
        unique_rules = []
        
        for rule in self.rule_base.rules:
            # Create a hashable representation of the rule
            rule_signature = (
                tuple(sorted(rule.antecedents.items())),
                tuple(sorted(rule.consequent.items())),
                rule.operator
            )
            
            if rule_signature not in seen:
                seen.append(rule_signature)
                unique_rules.append(rule)
        
        # Update rule base with unique rules
        self.rule_base.rules = unique_rules

    def plot_rule_matrix(self, figsize=(14, 8), cmap='RdYlGn', 
                         title='Fuzzy Rule Base Visualization'):
        """
        Visualizes the fuzzy rule base as a colored matrix.
        
        Parameters:
            figsize: Figure size
            cmap: Color map ('RdYlGn', 'viridis', 'coolwarm', etc.)
            title: Plot title
        
        Returns:
            fig, ax: Matplotlib figure and axes objects
        
        Example:
            >>> sistema.plot_rule_matrix(figsize=(14, 10), cmap='coolwarm')
            >>> plt.show()
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import LinearSegmentedColormap
    
        # Define modern color palettes
        color_palettes = {
            'custom': ['#E8F4F8', '#B8E0F0', '#6BB6D6', '#3E8FB0', '#1E5A7A'],  # Modern blue
            'ocean': ['#F0F9FF', '#BAE6FD', '#7DD3FC', '#38BDF8', '#0EA5E9'],   # Sky blue
            'sunset': ['#FFF7ED', '#FFEDD5', '#FED7AA', '#FB923C', '#F97316'],  # Warm orange
            'forest': ['#F0FDF4', '#BBF7D0', '#86EFAC', '#4ADE80', '#22C55E'],  # Fresh green
            'purple': ['#FAF5FF', '#E9D5FF', '#D8B4FE', '#C084FC', '#A855F7'],  # Soft purple
            'minimal': ['#F9FAFB', '#E5E7EB', '#D1D5DB', '#9CA3AF', '#6B7280']  # Modern gray
        }
        
        # Create custom colormap
        if cmap in color_palettes:
            colors = color_palettes[cmap]
            custom_cmap = LinearSegmentedColormap.from_list('modern', colors, N=256)
        else:
            custom_cmap = cmap
        
        rules = self.rule_base.rules
        
        if len(rules) == 0:
            print("⚠️  No rules to visualize!")
            return None, None
        
        # Collect rule information
        input_vars = list(self.input_variables.keys())
        output_vars = list(self.output_variables.keys())
        all_vars = input_vars + output_vars
        n_vars = len(all_vars)
        n_rules = len(rules)
        
        # Collect all unique terms per variable
        terms_by_var = {}
        for var in all_vars:
            if var in self.input_variables:
                terms_by_var[var] = list(self.input_variables[var].terms.keys())
            else:
                terms_by_var[var] = list(self.output_variables[var].terms.keys())
        
        # Create data matrix (rules x variables)
        data_matrix = np.zeros((n_rules, n_vars))
        text_matrix = [['' for _ in range(n_vars)] for _ in range(n_rules)]
        
        # Fill matrix
        for i, rule in enumerate(rules):
            for j, var in enumerate(all_vars):
                if var in input_vars:
                    # Input variable
                    if var in rule.antecedents:
                        term = rule.antecedents[var]
                        text_matrix[i][j] = term
                        if term in terms_by_var[var]:
                            data_matrix[i][j] = terms_by_var[var].index(term)
                else:
                    # Output variable
                    if var in rule.consequent:
                        term = rule.consequent[var]
                        # For Sugeno, can be number or list
                        if isinstance(term, (int, float)):
                            text_matrix[i][j] = f"{term:.1f}"
                            data_matrix[i][j] = term
                        elif isinstance(term, list):
                            text_matrix[i][j] = f"{term[0]:.1f}..."
                            data_matrix[i][j] = term[0]
                        else:
                            text_matrix[i][j] = str(term)
                            if term in terms_by_var[var]:
                                data_matrix[i][j] = terms_by_var[var].index(term)
        
        # Normalize data for colormap
        data_normalized = np.zeros_like(data_matrix)
        for j in range(n_vars):
            col_data = data_matrix[:, j]
            if col_data.max() > col_data.min():
                data_normalized[:, j] = (col_data - col_data.min()) / (col_data.max() - col_data.min())
            else:
                data_normalized[:, j] = 0.5
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot matrix with colors
        im = ax.imshow(data_normalized, cmap=custom_cmap, aspect='auto', vmin=0, vmax=1)
        
        # Configure axes
        ax.set_xticks(np.arange(n_vars))
        ax.set_yticks(np.arange(n_rules))
        ax.set_xticklabels(all_vars, fontsize=11, fontweight='bold')
        ax.set_yticklabels([f'R{i+1}' for i in range(n_rules)], fontsize=10)
        
        # Rotate x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text to cells
        for i in range(n_rules):
            for j in range(n_vars):
                ax.text(j, i, text_matrix[i][j],
                       ha="center", va="center", color="black",
                       fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                alpha=0.7, edgecolor='none'))
        
        # Add separation lines
        for i in range(n_rules + 1):
            ax.axhline(i - 0.5, color='white', linewidth=2)
        for j in range(n_vars + 1):
            ax.axvline(j - 0.5, color='white', linewidth=2)
        
        # Highlight separation between inputs and outputs
        sep_line = len(input_vars) - 0.5
        ax.axvline(sep_line, color='black', linewidth=3, linestyle='--', alpha=0.6)
        
        # Add section labels
        ax.text(len(input_vars)/2 - 0.5, -0.7, 'ANTECEDENTS (IF)', 
               ha='center', fontsize=12, fontweight='bold', color='navy')
        ax.text(len(input_vars) + len(output_vars)/2 - 0.5, -0.7, 'CONSEQUENTS (THEN)', 
               ha='center', fontsize=12, fontweight='bold', color='darkred')
        
        # Title and adjustments
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('')
        ax.set_ylabel('Rules', fontsize=12, fontweight='bold')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Term Intensity', rotation=270, labelpad=20, fontsize=10)
        
        plt.tight_layout()
        return fig, ax
    
    def plot_rule_matrix_2d(self, figsize=(10, 8), cmap='RdYlGn'):
        """
        Visualizes fuzzy rules as 2D matrix (for 2 inputs).
        Rows = first input, Columns = second input.
        
        Parameters:
            figsize: Figure size
            cmap: Color map
        
        Returns:
            fig, ax: Matplotlib figure and axes objects
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import LinearSegmentedColormap
    
        # Define modern color palettes
        color_palettes = {
            'custom': ['#E8F4F8', '#B8E0F0', '#6BB6D6', '#3E8FB0', '#1E5A7A'],  # Modern blue
            'ocean': ['#F0F9FF', '#BAE6FD', '#7DD3FC', '#38BDF8', '#0EA5E9'],   # Sky blue
            'sunset': ['#FFF7ED', '#FFEDD5', '#FED7AA', '#FB923C', '#F97316'],  # Warm orange
            'forest': ['#F0FDF4', '#BBF7D0', '#86EFAC', '#4ADE80', '#22C55E'],  # Fresh green
            'purple': ['#FAF5FF', '#E9D5FF', '#D8B4FE', '#C084FC', '#A855F7'],  # Soft purple
            'minimal': ['#F9FAFB', '#E5E7EB', '#D1D5DB', '#9CA3AF', '#6B7280']  # Modern gray
        }
        
        # Create custom colormap
        if cmap in color_palettes:
            colors = color_palettes[cmap]
            custom_cmap = LinearSegmentedColormap.from_list('modern', colors, N=256)
        else:
            custom_cmap = cmap
        
        rules = self.rule_base.rules
        input_vars = list(self.input_variables.keys())
        output_vars = list(self.output_variables.keys())
        
        if len(input_vars) != 2:
            print("⚠️  Function requires exactly 2 input variables!")
            return None, None
        
        var1_name, var2_name = input_vars
        output_name = output_vars[0]
        
        # Get terms
        terms1 = list(self.input_variables[var1_name].terms.keys())
        terms2 = list(self.input_variables[var2_name].terms.keys())
        output_terms = list(self.output_variables[output_name].terms.keys())
        
        # Create matrix
        n1, n2 = len(terms1), len(terms2)
        matrix = np.full((n1, n2), np.nan)
        text_matrix = [['' for _ in range(n2)] for _ in range(n1)]
        
        # Fill with rules
        for rule in rules:
            if var1_name in rule.antecedents and var2_name in rule.antecedents:
                term1 = rule.antecedents[var1_name]
                term2 = rule.antecedents[var2_name]
                output_term = rule.consequent[output_name]
                
                i = terms1.index(term1)
                j = terms2.index(term2)
                
                if output_term in output_terms:
                    matrix[i][j] = output_terms.index(output_term)
                    text_matrix[i][j] = output_term
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Normalize for colormap
        matrix_norm = (matrix - np.nanmin(matrix)) / (np.nanmax(matrix) - np.nanmin(matrix))
        
        im = ax.imshow(matrix_norm, cmap=custom_cmap, aspect='auto', vmin=0, vmax=1)
        
        # Labels
        ax.set_xticks(np.arange(n2))
        ax.set_yticks(np.arange(n1))
        ax.set_xticklabels(terms2, fontsize=11)
        ax.set_yticklabels(terms1, fontsize=11)
        
        ax.set_xlabel(var2_name.upper(), fontsize=12, fontweight='bold')
        ax.set_ylabel(var1_name.upper(), fontsize=12, fontweight='bold')
        ax.set_title(f'Rule Matrix: {output_name.upper()}', fontsize=14, fontweight='bold')
        
        # Text in cells
        for i in range(n1):
            for j in range(n2):
                if text_matrix[i][j]:
                    ax.text(j, i, text_matrix[i][j], ha="center", va="center",
                           color="black", fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Grid
        for i in range(n1 + 1):
            ax.axhline(i - 0.5, color='white', linewidth=2)
        for j in range(n2 + 1):
            ax.axvline(j - 0.5, color='white', linewidth=2)
        
        plt.tight_layout()
        return fig, ax


    def __repr__(self) -> str:
        n_inputs = len(self.input_variables)
        n_outputs = len(self.output_variables)
        n_rules = len(self.rule_base)
        return f"{self.__class__.__name__}(name='{self.name}', inputs={n_inputs}, outputs={n_outputs}, rules={n_rules})"


class MamdaniSystem(FuzzyInferenceSystem):
    """
    Sistema de Inferência Fuzzy tipo Mamdani.

    Características:
    - Fuzzificação das entradas
    - Inferência usando min/max (ou variantes)
    - Agregação de regras
    - Defuzzificação
    """

    def __init__(self,
                 name: str = "Mamdani FIS",
                 and_method: TNorm = TNorm.MIN,
                 or_method: SNorm = SNorm.MAX,
                 implication_method: str = 'min',
                 aggregation_method: str = 'max',
                 defuzzification_method: Union[str, DefuzzMethod] = DefuzzMethod.CENTROID):
        """
        Inicializa o sistema Mamdani.

        Parâmetros:
            name: Nome do sistema
            and_method: T-norma para AND
            or_method: S-norma para OR
            implication_method: Método de implicação ('min' ou 'product')
            aggregation_method: Método de agregação ('max', 'sum', 'probabilistic')
            defuzzification_method: Método de defuzzificação
        """
        super().__init__(name)
        self.inference_engine = MamdaniInference(
            and_method=and_method,
            or_method=or_method,
            implication_method=implication_method,
            aggregation_method=aggregation_method
        )
        self.defuzzification_method = defuzzification_method

    def evaluate(self, *args, num_points: int = 1000, **kwargs) -> Dict[str, float]:
        """
        Avalia as saídas do sistema Mamdani.

        Aceita múltiplos formatos de entrada:
        - Dicionário: evaluate({'temperatura': 25})
        - Lista/Tupla: evaluate([25, 60])
        - Args diretos: evaluate(25, 60)
        - Kwargs: evaluate(temperatura=25, umidade=60)

        Parâmetros:
            *args: Valores de entrada (vários formatos)
            num_points: Número de pontos para discretização
            **kwargs: Valores de entrada como argumentos nomeados

        Retorna:
            Dicionário {variável_saída: valor_defuzzificado}
        """
        # Normaliza entradas para dicionário
        inputs = self._normalize_inputs(*args, **kwargs)

        # Valida entradas
        for var_name in inputs:
            if var_name not in self.input_variables:
                raise ValueError(f"Variável de entrada '{var_name}' não definida no sistema")

        # 1. Fuzzificação
        fuzzified = {}
        for var_name, value in inputs.items():
            fuzzified[var_name] = self.input_variables[var_name].fuzzify(value)

        # 2. Inferência e Defuzzificação para cada variável de saída
        outputs = {}

        for out_var_name, out_variable in self.output_variables.items():
            # Inferência
            x, aggregated_mf = self.inference_engine.infer(
                fuzzified,
                self.rule_base.rules,
                out_variable,
                num_points
            )

            # Defuzzificação
            crisp_output = defuzzify(x, aggregated_mf, self.defuzzification_method)
            outputs[out_var_name] = crisp_output

        return outputs

    def evaluate_detailed(self, *args, num_points: int = 1000, **kwargs) -> Dict:
        """
        Avalia as saídas com informações detalhadas do processo.

        Aceita os mesmos formatos de entrada que evaluate().

        Parâmetros:
            *args: Valores de entrada (vários formatos)
            num_points: Número de pontos para discretização
            **kwargs: Valores de entrada como argumentos nomeados

        Retorna:
            Dicionário com informações detalhadas incluindo:
            - outputs: saídas finais
            - fuzzified_inputs: valores fuzzificados
            - activated_rules: regras ativadas e seus graus
            - aggregated_mf: funções de pertinência agregadas
        """
        # Normaliza entradas
        inputs = self._normalize_inputs(*args, **kwargs)

        # Fuzzificação
        fuzzified = {}
        for var_name, value in inputs.items():
            fuzzified[var_name] = self.input_variables[var_name].fuzzify(value)

        # Informações sobre regras ativadas
        activated_rules = []
        for i, rule in enumerate(self.rule_base.rules):
            firing_strength = rule.evaluate_antecedent(
                fuzzified,
                self.inference_engine.fuzzy_op
            )
            if firing_strength > 0:
                activated_rules.append({
                    'rule_index': i,
                    'rule': str(rule),
                    'firing_strength': firing_strength
                })

        # Inferência e defuzzificação
        outputs = {}
        aggregated_mfs = {}

        for out_var_name, out_variable in self.output_variables.items():
            x, aggregated_mf = self.inference_engine.infer(
                fuzzified,
                self.rule_base.rules,
                out_variable,
                num_points
            )

            crisp_output = defuzzify(x, aggregated_mf, self.defuzzification_method)

            outputs[out_var_name] = crisp_output
            aggregated_mfs[out_var_name] = (x, aggregated_mf)

        return {
            'outputs': outputs,
            'fuzzified_inputs': fuzzified,
            'activated_rules': activated_rules,
            'aggregated_mf': aggregated_mfs
        }


    @classmethod
    def create_automatic(cls,
                        n_inputs: int,
                        n_outputs: int = 1,
                        n_mfs: Union[int, List[int]] = 3,
                        mf_type: Union[str, List[str]] = 'triangular',
                        input_universes: Optional[Union[Tuple[float, float], List[Tuple[float, float]]]] = None,
                        output_universes: Optional[Union[Tuple[float, float], List[Tuple[float, float]]]] = None,
                        input_names: Optional[List[str]] = None,
                        output_names: Optional[List[str]] = None,
                        name: str = "Auto Mamdani FIS",
                        **kwargs) -> 'MamdaniSystem':
        """
        Cria automaticamente um sistema Mamdani com MFs igualmente espaçadas.

        Gera funções de pertinência distribuídas uniformemente no domínio,
        considerando centros também nos extremos dos universos.

        Parameters
        ----------
        n_inputs : int
            Número de variáveis de entrada
        n_outputs : int, default=1
            Número de variáveis de saída
        n_mfs : int or list of int, default=3
            Número de MFs por variável.
            - Se int: mesmo número para todas as variáveis
            - Se list: número específico para cada variável (entrada + saída)
        mf_type : str or list of str, default='triangular'
            Tipo de MF: 'triangular', 'gaussian', 'trapezoidal', 'bell'
            - Se str: mesmo tipo para todas as MFs
            - Se list: tipo específico para cada variável (entrada + saída)
        input_universes : tuple or list of tuples, optional
            Universos de discurso das entradas (min, max)
            - Se None: usa (0, 1) para todas
            - Se tuple: mesmo universo para todas as entradas
            - Se list of tuples: universo específico para cada entrada
        output_universes : tuple or list of tuples, optional
            Universos de discurso das saídas (min, max)
            - Se None: usa (0, 1) para todas
            - Se tuple: mesmo universo para todas as saídas
            - Se list of tuples: universo específico para cada saída
        input_names : list of str, optional
            Nomes das variáveis de entrada
            Se None: usa ["input_1", "input_2", ...]
        output_names : list of str, optional
            Nomes das variáveis de saída
            Se None: usa ["output_1", "output_2", ...]
        name : str, default="Auto Mamdani FIS"
            Nome do sistema
        **kwargs
            Argumentos adicionais para MamdaniSystem

        Returns
        -------
        MamdaniSystem
            Sistema Mamdani configurado automaticamente

        Examples
        --------
        >>> # Sistema simples: 2 entradas, 1 saída, 3 MFs triangulares cada
        >>> fis = MamdaniSystem.create_automatic(n_inputs=2)

        >>> # Sistema com MFs diferentes por variável
        >>> fis = MamdaniSystem.create_automatic(
        ...     n_inputs=2,
        ...     n_outputs=1,
        ...     n_mfs=[3, 5, 3],  # input1=3, input2=5, output=3
        ...     mf_type=['triangular', 'gaussian', 'triangular']
        ... )

        >>> # Sistema com universos customizados
        >>> fis = MamdaniSystem.create_automatic(
        ...     n_inputs=2,
        ...     n_outputs=1,
        ...     input_universes=[(0, 100), (-50, 50)],
        ...     output_universes=(0, 1),
        ...     input_names=['temperature', 'pressure'],
        ...     output_names=['valve']
        ... )

        >>> # Sistema complexo
        >>> fis = MamdaniSystem.create_automatic(
        ...     n_inputs=3,
        ...     n_outputs=2,
        ...     n_mfs=[5, 3, 4, 3, 3],  # 3 inputs + 2 outputs
        ...     mf_type='gaussian',
        ...     input_universes=[(0, 10), (0, 100), (-1, 1)],
        ...     output_universes=[(0, 1), (0, 100)]
        ... )

        Notes
        -----
        - MFs são distribuídas uniformemente com centros nos extremos
        - Para n_mfs=3: MFs em min, médio, max
        - Para n_mfs=5: MFs em min, 25%, 50%, 75%, max
        - Labels automáticos: "low", "medium", "high" (para n_mfs=3)
        - Para n_mfs > 3: "verylow", "low", "medium", "high", "veryhigh", etc.
        """

        # ==================== Validação de Parâmetros ====================
        if n_inputs < 1:
            raise ValueError(f"n_inputs deve ser >= 1, recebido: {n_inputs}")
        if n_outputs < 1:
            raise ValueError(f"n_outputs deve ser >= 1, recebido: {n_outputs}")

        total_vars = n_inputs + n_outputs

        # ==================== Processar n_mfs ====================
        if isinstance(n_mfs, int):
            n_mfs_list = [n_mfs] * total_vars
        elif isinstance(n_mfs, list):
            if len(n_mfs) != total_vars:
                raise ValueError(
                    f"Se n_mfs for lista, deve ter {total_vars} elementos "
                    f"({n_inputs} entradas + {n_outputs} saídas). "
                    f"Recebido: {len(n_mfs)}"
                )
            n_mfs_list = n_mfs
        else:
            raise TypeError(f"n_mfs deve ser int ou list, recebido: {type(n_mfs)}")

        # Validar número de MFs
        for i, n_mf in enumerate(n_mfs_list):
            if n_mf < 2:
                raise ValueError(f"Cada variável deve ter >= 2 MFs. Variável {i}: {n_mf}")

        # ==================== Processar mf_type ====================
        if isinstance(mf_type, str):
            mf_types_list = [mf_type] * total_vars
        elif isinstance(mf_type, list):
            if len(mf_type) != total_vars:
                raise ValueError(
                    f"Se mf_type for lista, deve ter {total_vars} elementos. "
                    f"Recebido: {len(mf_type)}"
                )
            mf_types_list = mf_type
        else:
            raise TypeError(f"mf_type deve ser str ou list, recebido: {type(mf_type)}")

        # Validar tipos de MF
        valid_types = ['triangular', 'gaussian', 'trapezoidal', 'bell', 'sigmoid', 'gauss2mf']
        for mf_t in mf_types_list:
            if mf_t not in valid_types:
                raise ValueError(
                    f"Tipo de MF inválido: '{mf_t}'. "
                    f"Válidos: {valid_types}"
                )

        # ==================== Processar universos ====================
        # Entradas
        if input_universes is None:
            input_universes_list = [(0.0, 1.0)] * n_inputs
        elif isinstance(input_universes, tuple) and len(input_universes) == 2:
            input_universes_list = [input_universes] * n_inputs
        elif isinstance(input_universes, list):
            if len(input_universes) != n_inputs:
                raise ValueError(
                    f"input_universes deve ter {n_inputs} elementos. "
                    f"Recebido: {len(input_universes)}"
                )
            input_universes_list = input_universes
        else:
            raise TypeError(
                f"input_universes deve ser tuple ou list of tuples"
            )

        # Saídas
        if output_universes is None:
            output_universes_list = [(0.0, 1.0)] * n_outputs
        elif isinstance(output_universes, tuple) and len(output_universes) == 2:
            output_universes_list = [output_universes] * n_outputs
        elif isinstance(output_universes, list):
            if len(output_universes) != n_outputs:
                raise ValueError(
                    f"output_universes deve ter {n_outputs} elementos. "
                    f"Recebido: {len(output_universes)}"
                )
            output_universes_list = output_universes
        else:
            raise TypeError(
                f"output_universes deve ser tuple ou list of tuples"
            )

        # ==================== Processar nomes ====================
        if input_names is None:
            input_names = [f"input_{i+1}" for i in range(n_inputs)]
        elif len(input_names) != n_inputs:
            raise ValueError(
                f"input_names deve ter {n_inputs} elementos. "
                f"Recebido: {len(input_names)}"
            )

        if output_names is None:
            output_names = [f"output_{i+1}" for i in range(n_outputs)]
        elif len(output_names) != n_outputs:
            raise ValueError(
                f"output_names deve ter {n_outputs} elementos. "
                f"Recebido: {len(output_names)}"
            )

        # ==================== Criar Sistema ====================
        system = cls(name=name, **kwargs)

        # ==================== Função para gerar labels ====================
        def _generate_labels(n: int) -> List[str]:
            """Gera labels linguísticos baseado no número de MFs."""
            if n == 2:
                return ['low', 'high']
            elif n == 3:
                return ['low', 'medium', 'high']
            elif n == 4:
                return ['low', 'medium_low', 'medium_high', 'high']
            elif n == 5:
                return ['very_low', 'low', 'medium', 'high', 'very_high']
            elif n == 7:
                return ['very_low', 'low', 'medium_low', 'medium', 
                       'medium_high', 'high', 'very_high']
            else:
                # Para n > 7 ou outros casos
                return [f'mf_{i+1}' for i in range(n)]

        # ==================== Função para gerar parâmetros de MF ====================
        def _generate_mf_params(universe: Tuple[float, float], 
                               n_mf: int, 
                               mf_type: str,
                               index: int) -> Tuple:
            """
            Gera parâmetros de MF igualmente espaçadas.
            Centros incluem os extremos do universo.
            """
            u_min, u_max = universe
            u_range = u_max - u_min

            # Centros igualmente espaçados (incluindo extremos)
            centers = np.linspace(u_min, u_max, n_mf)

            # Largura base para as MFs
            if n_mf > 1:
                width = u_range / (n_mf - 1)
            else:
                width = u_range / 2

            center = centers[index]

            if mf_type == 'triangular':
                # Triangular: [left, center, right]
                left = center - width
                right = center + width

                # Ajustar extremos para cobrir todo o universo
                if index == 0:
                    left = u_min
                if index == n_mf - 1:
                    right = u_max

                return (left, center, right)

            elif mf_type == 'trapezoidal':
                # Trapezoidal: [left, left_top, right_top, right]
                left = center - width
                left_top = center - width/4
                right_top = center + width/4
                right = center + width

                # Ajustar extremos
                if index == 0:
                    left = u_min
                    left_top = u_min
                if index == n_mf - 1:
                    right = u_max
                    right_top = u_max

                return (left, left_top, right_top, right)

            elif mf_type == 'gaussian':
                # Gaussian: [center, sigma]
                sigma = width / 3  # Regra empírica: 3*sigma cobre largura
                return (center, sigma)

            elif mf_type == 'bell':
                # Bell (Generalized Bell): [a, b, c]
                # a controla largura, b controla inclinação, c é o centro
                a = width / 2
                b = 2.0  # Inclinação padrão
                c = center
                return (a, b, c)

            elif mf_type == 'sigmoid':
                # Sigmoid: [a, c]
                # a controla inclinação, c é o centro
                a = 10 / width  # Inclinação inversamente proporcional à largura
                c = center
                return (a, c)

            elif mf_type == 'gauss2mf':
                # Gaussian combination: [mean1, sigma1, mean2, sigma2]
                sigma = width / 4
                return (center - width/4, sigma, center + width/4, sigma)

            else:
                raise ValueError(f"Tipo de MF não suportado: {mf_type}")

        # ==================== Adicionar Entradas ====================
        for i in range(n_inputs):
            var_name = input_names[i]
            universe = input_universes_list[i]
            n_mf = n_mfs_list[i]
            mf_type_var = mf_types_list[i]

            # Adicionar variável
            system.add_input(var_name, universe)

            # Gerar labels
            labels = _generate_labels(n_mf)

            # Adicionar MFs
            for j in range(n_mf):
                term_name = labels[j]
                params = _generate_mf_params(universe, n_mf, mf_type_var, j)
                system.add_term(var_name, term_name, mf_type_var, params)

        # ==================== Adicionar Saídas ====================
        for i in range(n_outputs):
            var_name = output_names[i]
            universe = output_universes_list[i]
            n_mf = n_mfs_list[n_inputs + i]  # Offset pelos inputs
            mf_type_var = mf_types_list[n_inputs + i]

            # Adicionar variável
            system.add_output(var_name, universe)

            # Gerar labels
            labels = _generate_labels(n_mf)

            # Adicionar MFs
            for j in range(n_mf):
                term_name = labels[j]
                params = _generate_mf_params(universe, n_mf, mf_type_var, j)
                system.add_term(var_name, term_name, mf_type_var, params)

        return system


class SugenoSystem(FuzzyInferenceSystem):
    """
    Sistema de Inferência Fuzzy tipo Sugeno (Takagi-Sugeno-Kang).

    Características:
    - Fuzzificação das entradas
    - Consequentes são funções (ordem 0 ou 1)
    - Saída é média ponderada
    - Não requer defuzzificação
    """

    def __init__(self,
                 name: str = "Sugeno FIS",
                 and_method: TNorm = TNorm.MIN,
                 or_method: SNorm = SNorm.MAX,
                 order: int = 0):
        """
        Inicializa o sistema Sugeno.

        Parâmetros:
            name: Nome do sistema
            and_method: T-norma para AND
            or_method: S-norma para OR
            order: Ordem do sistema (0=constantes, 1=linear)
        """
        super().__init__(name)
        self.inference_engine = SugenoInference(
            and_method=and_method,
            or_method=or_method,
            order=order
        )
        self.order = order

    def evaluate(self, *args, **kwargs) -> Dict[str, float]:
        """
        Avalia as saídas do sistema Sugeno.

        Aceita múltiplos formatos de entrada:
        - Dicionário: evaluate({'temperatura': 25})
        - Lista/Tupla: evaluate([25, 60])
        - Args diretos: evaluate(25, 60)
        - Kwargs: evaluate(temperatura=25, umidade=60)

        Parâmetros:
            *args: Valores de entrada (vários formatos)
            **kwargs: Valores de entrada como argumentos nomeados

        Retorna:
            Dicionário {variável_saída: valor}
        """
        # Normaliza entradas para dicionário
        inputs = self._normalize_inputs(*args, **kwargs)

        # Valida entradas
        for var_name in inputs:
            if var_name not in self.input_variables:
                raise ValueError(f"Variável de entrada '{var_name}' não definida no sistema")

        # 1. Fuzzificação
        fuzzified = {}
        for var_name, value in inputs.items():
            fuzzified[var_name] = self.input_variables[var_name].fuzzify(value)

        # 2. Inferência (já retorna valor crisp)
        # Em Sugeno, tipicamente há uma única saída
        # mas vamos suportar múltiplas saídas

        outputs = {}

        # Agrupa regras por variável de saída
        rules_by_output: Dict[str, List[FuzzyRule]] = {}

        for rule in self.rule_base.rules:
            for out_var in rule.consequent.keys():
                if out_var not in rules_by_output:
                    rules_by_output[out_var] = []
                rules_by_output[out_var].append(rule)

        # Computa saída para cada variável
        for out_var_name, rules in rules_by_output.items():
            output = self.inference_engine.infer(fuzzified, inputs, rules)
            outputs[out_var_name] = output

        return outputs

    def evaluate_detailed(self, *args, **kwargs) -> Dict:
        """
        Avalia as saídas com informações detalhadas do processo.

        Aceita os mesmos formatos de entrada que evaluate().

        Parâmetros:
            *args: Valores de entrada (vários formatos)
            **kwargs: Valores de entrada como argumentos nomeados

        Retorna:
            Dicionário com informações detalhadas
        """
        # Normaliza entradas
        inputs = self._normalize_inputs(*args, **kwargs)

        # Fuzzificação
        fuzzified = {}
        for var_name, value in inputs.items():
            fuzzified[var_name] = self.input_variables[var_name].fuzzify(value)

        # Informações sobre regras ativadas
        activated_rules = []
        for i, rule in enumerate(self.rule_base.rules):
            firing_strength = rule.evaluate_antecedent(
                fuzzified,
                self.inference_engine.fuzzy_op
            )

            rule_output = self.inference_engine._evaluate_consequent(rule, inputs)

            activated_rules.append({
                'rule_index': i,
                'rule': str(rule),
                'firing_strength': firing_strength,
                'rule_output': rule_output,
                'weighted_output': firing_strength * rule_output
            })

        # Computa saídas finais
        outputs = self.evaluate(inputs)

        return {
            'outputs': outputs,
            'fuzzified_inputs': fuzzified,
            'activated_rules': activated_rules
        }
    def add_output(self,
                    name_or_variable: Union[str, LinguisticVariable],
                    universe: Optional[Tuple[float, float]] = None) -> LinguisticVariable:
            """
            Adds output variable to Sugeno system.
            
            NOTE: For Sugeno systems, universe of discourse is OPTIONAL, since
            outputs are crisp functions (not fuzzy sets). If provided, it's used
            only for documentation and optional boundary validation.
            
            Parameters:
                name_or_variable: Variable name or LinguisticVariable object
                universe: Universe of discourse (optional for Sugeno)
            
            Returns:
                Created or provided linguistic variable
            
            Examples:
                # With universe (recommended for documentation)
                sugeno.add_output('temperature', (0, 100))
                
                # Without universe (valid for Sugeno)
                sugeno.add_output('temperature')
            """
            if isinstance(name_or_variable, LinguisticVariable):
                variable = name_or_variable
                self.output_variables[variable.name] = variable
                return variable
            
            name = name_or_variable
            
            # For Sugeno, universe is optional
            if universe is None:
                universe = (0.0, 1.0)  # Dummy placeholder
                print(f"Info: Output variable '{name}' created without defined universe. "
                    f"For Sugeno, outputs are direct crisp values.")
            
            variable = LinguisticVariable(name, universe)
            self.output_variables[name] = variable
            return variable

class TSKSystem(SugenoSystem):
    """
    Alias para SugenoSystem (Takagi-Sugeno-Kang).
    """
    pass


# ============================================================================
# Funções auxiliares para construção rápida de sistemas
# ============================================================================

def create_mamdani_system(
    input_specs: Dict[str, Tuple[Tuple[float, float], Dict[str, Tuple[str, Tuple]]]],
    output_specs: Dict[str, Tuple[Tuple[float, float], Dict[str, Tuple[str, Tuple]]]],
    rules: List[Tuple[Dict[str, str], Dict[str, str], str]],
    name: str = "Mamdani FIS",
    **kwargs
) -> MamdaniSystem:
    """
    Cria um sistema Mamdani de forma simplificada.

    Parâmetros:
        input_specs: {nome_var: (universo, {termo: (tipo_mf, params)})}
        output_specs: {nome_var: (universo, {termo: (tipo_mf, params)})}
        rules: Lista de (antecedentes, consequentes, operador)
        name: Nome do sistema
        **kwargs: Parâmetros adicionais para MamdaniSystem

    Retorna:
        Sistema Mamdani configurado

    Exemplo:
        >>> system = create_mamdani_system(
        ...     input_specs={
        ...         'temperatura': ((0, 100), {
        ...             'fria': ('triangular', (0, 0, 50)),
        ...             'quente': ('triangular', (50, 100, 100))
        ...         })
        ...     },
        ...     output_specs={
        ...         'ventilador': ((0, 100), {
        ...             'lento': ('triangular', (0, 0, 50)),
        ...             'rápido': ('triangular', (50, 100, 100))
        ...         })
        ...     },
        ...     rules=[
        ...         ({'temperatura': 'fria'}, {'ventilador': 'lento'}, 'AND'),
        ...         ({'temperatura': 'quente'}, {'ventilador': 'rápido'}, 'AND')
        ...     ]
        ... )
    """
    system = MamdaniSystem(name=name, **kwargs)

    # Cria variáveis de entrada
    for var_name, (universe, terms) in input_specs.items():
        var = LinguisticVariable(var_name, universe)
        for term_name, (mf_type, params) in terms.items():
            var.add_term(FuzzySet(term_name, mf_type, params))
        system.add_input(var)

    # Cria variáveis de saída
    for var_name, (universe, terms) in output_specs.items():
        var = LinguisticVariable(var_name, universe)
        for term_name, (mf_type, params) in terms.items():
            var.add_term(FuzzySet(term_name, mf_type, params))
        system.add_output(var)

    # Adiciona regras
    for antecedents, consequents, operator in rules:
        rule = FuzzyRule(antecedents, consequents, operator)
        system.add_rule(rule)

    return system


def create_sugeno_system(
    input_specs: Dict[str, Tuple[Tuple[float, float], Dict[str, Tuple[str, Tuple]]]],
    output_names: List[str],
    rules: List[Tuple[Dict[str, str], Dict[str, Union[float, Dict]], str]],
    name: str = "Sugeno FIS",
    order: int = 0,
    **kwargs
) -> SugenoSystem:
    """
    Cria um sistema Sugeno de forma simplificada.

    Parâmetros:
        input_specs: {nome_var: (universo, {termo: (tipo_mf, params)})}
        output_names: Lista de nomes das variáveis de saída
        rules: Lista de (antecedentes, {saída: valor/função}, operador)
        name: Nome do sistema
        order: Ordem do sistema (0 ou 1)
        **kwargs: Parâmetros adicionais para SugenoSystem

    Retorna:
        Sistema Sugeno configurado
    """
    system = SugenoSystem(name=name, order=order, **kwargs)

    # Cria variáveis de entrada
    for var_name, (universe, terms) in input_specs.items():
        var = LinguisticVariable(var_name, universe)
        for term_name, (mf_type, params) in terms.items():
            var.add_term(FuzzySet(term_name, mf_type, params))
        system.add_input(var)

    # Cria variáveis de saída (dummy, pois Sugeno não usa MFs de saída)
    for out_name in output_names:
        var = LinguisticVariable(out_name, (0, 1))
        system.add_output(var)

    # Adiciona regras
    for antecedents, consequents, operator in rules:
        rule = FuzzyRule(antecedents, consequents, operator)
        system.add_rule(rule)

    return system
