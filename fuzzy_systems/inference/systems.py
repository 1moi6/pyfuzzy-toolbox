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
                 *args,
                 operator: str = 'AND',
                 weight: float = 1.0,
                 **kwargs) -> None:
        """
        Adiciona uma regra ao sistema.

        Aceita múltiplos formatos:

        Forma 1 - Tupla/Lista Plana como argumentos (Mais Simples e Recomendado):
            >>> # Para 2 entradas e 2 saídas: add_rule(in1, in2, out1, out2)
            >>> system.add_rule('fria', 'seca', 'lento', 'baixa')
            >>> system.add_rule('quente', 'umida', 'rapido', 'alta')
            >>>
            >>> # Com operador e peso
            >>> system.add_rule('media', 'media', 'medio', 'media', operator='OR', weight=0.8)

        Forma 2 - Tupla/Lista Plana única:
            >>> system.add_rule(('quente', 'umida', 'rapido', 'alta'))
            >>> system.add_rule(['fria', 'seca', 'lento', 'baixa'])

        Forma 3 - Tuplas/Listas Separadas:
            >>> system.add_rule(['fria', 'seca'], ['lento'])
            >>> system.add_rule(('quente', 'umida'), ('rapido',))

        Forma 4 - Dicionários (Mais Explícito):
            >>> system.add_rule(
            ...     {'temperatura': 'fria', 'umidade': 'seca'},
            ...     {'ventilador': 'lento'}
            ... )

        Forma 5 - FuzzyRule completo (Compatibilidade):
            >>> system.add_rule(fis.FuzzyRule(...))

        Parâmetros:
            *args: Termos da regra (in1, in2, ..., out1, out2, ...)
                   ou (antecedents, consequents) ou FuzzyRule
            operator: 'AND' ou 'OR'
            weight: Peso da regra (0 a 1)

        Nota: Quando usar lista/tupla, a ordem segue a ordem de adição das variáveis.
              Para tupla plana: primeiros N elementos são entradas, restantes são saídas.
        """
        # Extrair antecedents e consequents de args
        if len(args) == 0:
            raise ValueError("add_rule requer pelo menos 1 argumento")

        # Forma 5: FuzzyRule completo
        if len(args) == 1 and isinstance(args[0], FuzzyRule):
            self.rule_base.add_rule(args[0])
            return

        input_vars = list(self.input_variables.keys())
        output_vars = list(self.output_variables.keys())
        n_inputs = len(input_vars)
        n_outputs = len(output_vars)
        total_expected = n_inputs + n_outputs

        # Forma 1: Múltiplos argumentos (tupla plana desempacotada)
        if len(args) >= total_expected and all(isinstance(arg, (str, int, float)) for arg in args[:total_expected]):
            # É tupla plana desempacotada!
            ant_terms = args[:n_inputs]
            cons_terms = args[n_inputs:total_expected]

            # Converte para dicionários
            ant_dict = {input_vars[i]: ant_terms[i] for i in range(n_inputs)}
            cons_dict = {output_vars[i]: cons_terms[i] for i in range(n_outputs)}

            rule = FuzzyRule(ant_dict, cons_dict, operator, weight)
            self.rule_base.add_rule(rule)
            return

        # Forma 2: Um argumento que é lista/tupla (tupla plana empacotada)
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            antecedents = args[0]

            # Checa se é tupla plana
            if len(antecedents) == total_expected:
                ant_terms = antecedents[:n_inputs]
                cons_terms = antecedents[n_inputs:]

                # Converte para dicionários
                ant_dict = {input_vars[i]: ant_terms[i] for i in range(n_inputs)}
                cons_dict = {output_vars[i]: cons_terms[i] for i in range(n_outputs)}

                rule = FuzzyRule(ant_dict, cons_dict, operator, weight)
                self.rule_base.add_rule(rule)
                return
            else:
                raise ValueError(
                    f"Tupla/lista deve ter {total_expected} elementos "
                    f"({n_inputs} entradas + {n_outputs} saídas). Recebeu: {len(antecedents)}"
                )

        # Forma 3: Dois argumentos (antecedents, consequents)
        if len(args) == 2:
            antecedents = args[0]
            consequents = args[1]

            # Listas/Tuplas separadas
            if isinstance(antecedents, (list, tuple)) and isinstance(consequents, (list, tuple, dict)):
                # Converte antecedentes
                if len(antecedents) != n_inputs:
                    raise ValueError(
                        f"Número de antecedentes ({len(antecedents)}) não corresponde "
                        f"ao número de variáveis de entrada ({n_inputs})"
                    )
                ant_dict = {input_vars[i]: antecedents[i] for i in range(n_inputs)}

                # Converte consequentes
                if isinstance(consequents, (list, tuple)):
                    if len(consequents) != n_outputs:
                        raise ValueError(
                            f"Número de consequentes ({len(consequents)}) não corresponde "
                            f"ao número de variáveis de saída ({n_outputs})"
                        )
                    cons_dict = {output_vars[i]: consequents[i] for i in range(n_outputs)}
                elif isinstance(consequents, dict):
                    cons_dict = consequents
                else:
                    # Valor único para Sugeno
                    if n_outputs != 1:
                        raise ValueError(
                            f"Consequente único mas sistema tem {n_outputs} saídas"
                        )
                    cons_dict = {output_vars[0]: consequents}

                rule = FuzzyRule(ant_dict, cons_dict, operator, weight)
                self.rule_base.add_rule(rule)
                return

            # Forma 4: Dicionários
            elif isinstance(antecedents, dict) and isinstance(consequents, (dict, int, float)):
                if isinstance(consequents, dict):
                    cons_dict = consequents
                else:
                    # Valor único - converte para dicionário
                    if n_outputs != 1:
                        raise ValueError(
                            f"Consequente único mas sistema tem {n_outputs} saídas"
                        )
                    cons_dict = {output_vars[0]: consequents}

                rule = FuzzyRule(antecedents, cons_dict, operator, weight)
                self.rule_base.add_rule(rule)
                return

        raise TypeError(
            f"Formato inválido. Use:\n"
            f"  - add_rule(in1, in2, ..., out1, out2, ...)\n"
            f"  - add_rule((in1, in2, ..., out1, out2, ...))\n"
            f"  - add_rule([in1, in2], [out1, out2])\n"
            f"  - add_rule({{'var': 'termo'}}, {{'var': 'termo'}})\n"
            f"Recebeu: {len(args)} argumentos"
        )

    def add_rules(self,
                  rules: Union[List[FuzzyRule], List[Tuple], List[List], List[Dict]]) -> None:
        """
        Adiciona múltiplas regras ao sistema.

        Aceita lista de:
        - Tuplas/Listas Planas (Mais Simples): [('B', 'B', 'crescimento', 'declinio'), ...]
        - Tuplas/Listas Separadas: [(antecedentes, consequentes), ...]
        - FuzzyRule objects
        - Dicionários: [{'if': {...}, 'then': {...}}, ...]

        Exemplos:
            >>> # Lista de tuplas PLANAS (RECOMENDADO - mais simples!)
            >>> system.add_rules([
            ...     ('B', 'B', 'crescimento', 'declinio'),
            ...     ('B', 'MB', 'crescimento', 'crescimento'),
            ...     ('MB', 'B', 'estavel', 'estavel')
            ... ])

            >>> # Com operadores e pesos diferentes
            >>> system.add_rules([
            ...     ('B', 'B', 'crescimento', 'declinio', 'AND', 1.0),
            ...     ('MB', 'MB', 'estavel', 'estavel', 'OR', 0.8)
            ... ])

            >>> # Lista de tuplas separadas (formato antigo)
            >>> system.add_rules([
            ...     (['fria', 'seca'], ['lento']),
            ...     (['quente', 'umida'], ['rapido'])
            ... ])

            >>> # Lista de dicionários
            >>> system.add_rules([
            ...     {'if': {'temperatura': 'fria'}, 'then': {'ventilador': 'lento'}},
            ...     {'if': {'temperatura': 'quente'}, 'then': {'ventilador': 'rapido'}}
            ... ])

        Parâmetros:
            rules: Lista de regras em diversos formatos
        """
        input_vars = list(self.input_variables.keys())
        output_vars = list(self.output_variables.keys())
        n_inputs = len(input_vars)
        n_outputs = len(output_vars)
        total_expected = n_inputs + n_outputs

        for rule_data in rules:
            if isinstance(rule_data, FuzzyRule):
                # Formato: FuzzyRule object
                self.rule_base.add_rule(rule_data)

            elif isinstance(rule_data, (tuple, list)):
                # Detectar se é tupla plana ou separada
                rule_len = len(rule_data)

                # Caso 1: Tupla plana (n_inputs + n_outputs) ou com operator/weight
                if rule_len == total_expected or rule_len == total_expected + 1 or rule_len == total_expected + 2:
                    # É tupla plana!
                    operator = rule_data[total_expected] if rule_len > total_expected else 'AND'
                    weight = rule_data[total_expected + 1] if rule_len > total_expected + 1 else 1.0

                    # Usar add_rule que já suporta tupla plana
                    self.add_rule(rule_data[:total_expected], operator=operator, weight=weight)

                # Caso 2: Tupla separada (antecedentes, consequentes[, operator[, weight]])
                elif rule_len >= 2:
                    antecedents = rule_data[0]
                    consequents = rule_data[1]
                    operator = rule_data[2] if rule_len > 2 else 'AND'
                    weight = rule_data[3] if rule_len > 3 else 1.0
                    self.add_rule(antecedents, consequents, operator, weight)
                else:
                    raise ValueError(
                        f"Tupla/lista deve ter pelo menos 2 elementos (antecedentes, consequentes) "
                        f"ou {total_expected} elementos (tupla plana). Recebeu: {rule_len}"
                    )

            elif isinstance(rule_data, dict):
                # Formato: {'if': {...}, 'then': {...}, 'op': '...', 'weight': ...}
                if 'if' not in rule_data or 'then' not in rule_data:
                    raise ValueError("Dicionário de regra deve ter 'if' e 'then'")
                antecedents = rule_data['if']
                consequents = rule_data['then']
                operator = rule_data.get('op', rule_data.get('operator', 'AND'))
                weight = rule_data.get('weight', 1.0)
                self.add_rule(antecedents, consequents, operator, weight)
            else:
                raise ValueError(f"Formato de regra inválido: {type(rule_data)}")

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

    def plot_variables(self, variables=None, var_type='all', **kwargs):
        """
        Plota as funções de pertinência das variáveis do sistema.

        Parâmetros:
            variables: Lista de nomes de variáveis específicas a plotar.
                      Se None, plota baseado em var_type.
                      Exemplos: ['temperatura', 'umidade']
                                ['temperatura']
            var_type: Tipo de variáveis a plotar ('input', 'output', 'all')
                     Usado apenas se variables=None
            **kwargs: Argumentos passados para LinguisticVariable.plot()
                - figsize: Tamanho da figura
                - num_points: Número de pontos para plotar

        Retorna:
            fig: Figura matplotlib

        Exemplos:
            >>> # Plotar todas as variáveis
            >>> system.plot_variables()

            >>> # Plotar apenas entradas
            >>> system.plot_variables(var_type='input')

            >>> # Plotar variáveis específicas
            >>> system.plot_variables(['temperatura', 'umidade'])
            >>> system.plot_variables(['ventilador'])
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "Matplotlib não está instalado. "
                "Instale com: pip install matplotlib"
            )

        # Seleciona variáveis a plotar
        variables_to_plot = []

        if variables is not None:
            # Plotar variáveis específicas
            if isinstance(variables, str):
                variables = [variables]  # Converter string única para lista

            for var_name in variables:
                # Buscar nas entradas
                if var_name in self.input_variables:
                    variables_to_plot.append(('Entrada', self.input_variables[var_name]))
                # Buscar nas saídas
                elif var_name in self.output_variables:
                    variables_to_plot.append(('Saída', self.output_variables[var_name]))
                else:
                    available = list(self.input_variables.keys()) + list(self.output_variables.keys())
                    raise ValueError(
                        f"Variável '{var_name}' não encontrada. "
                        f"Variáveis disponíveis: {available}"
                    )
        else:
            # Plotar baseado em var_type
            if var_type in ['input', 'all']:
                variables_to_plot.extend(
                    [('Entrada', var) for var in self.input_variables.values()]
                )

            if var_type in ['output', 'all']:
                variables_to_plot.extend(
                    [('Saída', var) for var in self.output_variables.values()]
                )

            if not variables_to_plot:
                raise ValueError(
                    f"Tipo de variável inválido: '{var_type}'. "
                    "Use 'input', 'output' ou 'all'"
                )

        # Calcula layout dos subplots
        n_vars = len(variables_to_plot)
        n_cols = min(2, n_vars)
        n_rows = (n_vars + n_cols - 1) // n_cols

        # Cria figura
        figsize = kwargs.get('figsize', (12, 4 * n_rows))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

        # Garante que axes seja sempre array
        if n_vars == 1:
            axes = np.array([axes])
        axes = axes.flatten() if n_vars > 1 else axes

        # Plota cada variável
        for i, (var_type_label, var) in enumerate(variables_to_plot):
            ax = axes[i] if n_vars > 1 else axes[0]
            title = f'{var_type_label}: {var.name}'
            var.plot(ax=ax, show=False, title=title, **kwargs)

        # Remove axes extras
        for i in range(n_vars, len(axes)):
            fig.delaxes(axes[i])

        plt.suptitle(f'Variáveis Linguísticas - {self.name}',
                    fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.show()

        return fig

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
