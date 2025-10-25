"""
Módulo de Inferência Fuzzy

Este módulo implementa os mecanismos de inferência fuzzy,
incluindo gerenciamento de regras e sistemas Mamdani e Sugeno.
"""

import numpy as np
from typing import Dict, List, Union, Tuple, Callable, Optional
from dataclasses import dataclass, field
from ..core.operators import FuzzyOperator, TNorm, SNorm, implication_mamdani
from ..core.fuzzification import LinguisticVariable


@dataclass
class FuzzyRule:
    """
    Representa uma regra fuzzy do tipo IF-THEN.

    Formato: IF (input1 IS term1) AND/OR (input2 IS term2) THEN (output IS term_out)
    """
    antecedents: Dict[str, str]  # {variável: termo}
    consequent: Dict[str, Union[str, float]]  # {variável: termo} ou {variável: função}
    operator: str = 'AND'  # 'AND' ou 'OR'
    weight: float = 1.0  # Peso da regra
    label: Optional[str] = None  # Rótulo opcional da regra

    def __post_init__(self):
        """Validação após inicialização."""
        if self.operator not in ['AND', 'OR']:
            raise ValueError(f"Operador deve ser 'AND' ou 'OR', recebido: '{self.operator}'")

        if not 0 <= self.weight <= 1:
            raise ValueError(f"Peso da regra deve estar em [0, 1], recebido: {self.weight}")

    def evaluate_antecedent(self,
                           fuzzified_inputs: Dict[str, Dict[str, float]],
                           fuzzy_op: FuzzyOperator) -> float:
        """
        Avalia o antecedente da regra (parte IF).

        Parâmetros:
            fuzzified_inputs: Dicionário {variável: {termo: grau}}
            fuzzy_op: Operador fuzzy a usar para AND/OR

        Retorna:
            Grau de ativação da regra (firing strength)
        """
        degrees = []

        for var_name, term_name in self.antecedents.items():
            if var_name not in fuzzified_inputs:
                raise ValueError(f"Variável '{var_name}' não encontrada nas entradas fuzzificadas")

            if term_name not in fuzzified_inputs[var_name]:
                raise ValueError(f"Termo '{term_name}' não encontrado para variável '{var_name}'")

            degrees.append(fuzzified_inputs[var_name][term_name])

        # Combina os graus usando operador AND ou OR
        if len(degrees) == 0:
            return 0.0
        elif len(degrees) == 1:
            result = degrees[0]
        else:
            result = degrees[0]
            for degree in degrees[1:]:
                if self.operator == 'AND':
                    result = fuzzy_op.AND(result, degree)
                else:  # OR
                    result = fuzzy_op.OR(result, degree)

        # Aplica o peso da regra
        return result * self.weight

    def __repr__(self) -> str:
        ant_str = f" {self.operator} ".join([f"{v} IS {t}" for v, t in self.antecedents.items()])
        cons_str = ", ".join([f"{v} IS {t}" for v, t in self.consequent.items()])
        label_str = f" [{self.label}]" if self.label else ""
        return f"IF {ant_str} THEN {cons_str}{label_str}"


class RuleBase:
    """
    Base de regras fuzzy - gerencia o conjunto de regras.
    """

    def __init__(self):
        """Inicializa a base de regras."""
        self.rules: List[FuzzyRule] = []

    def add_rule(self, rule: FuzzyRule) -> None:
        """
        Adiciona uma regra à base.

        Parâmetros:
            rule: Regra fuzzy a adicionar
        """
        self.rules.append(rule)

    def add_rules(self, rules: List[FuzzyRule]) -> None:
        """
        Adiciona múltiplas regras à base.

        Parâmetros:
            rules: Lista de regras fuzzy
        """
        self.rules.extend(rules)

    def create_rule(self,
                   antecedents: Dict[str, str],
                   consequent: Dict[str, Union[str, float]],
                   operator: str = 'AND',
                   weight: float = 1.0,
                   label: Optional[str] = None) -> FuzzyRule:
        """
        Cria e adiciona uma regra à base.

        Parâmetros:
            antecedents: Dicionário {variável: termo} da premissa
            consequent: Dicionário {variável: termo/função} da conclusão
            operator: 'AND' ou 'OR'
            weight: Peso da regra
            label: Rótulo opcional

        Retorna:
            Regra criada

        Exemplo:
            >>> rb = RuleBase()
            >>> rb.create_rule(
            ...     {'temperatura': 'alta', 'umidade': 'baixa'},
            ...     {'ventilador': 'rápido'},
            ...     operator='AND'
            ... )
        """
        rule = FuzzyRule(antecedents, consequent, operator, weight, label)
        self.add_rule(rule)
        return rule

    def clear(self) -> None:
        """Remove todas as regras da base."""
        self.rules.clear()

    def __len__(self) -> int:
        """Retorna o número de regras."""
        return len(self.rules)

    def __iter__(self):
        """Permite iterar sobre as regras."""
        return iter(self.rules)

    def __repr__(self) -> str:
        return f"RuleBase({len(self.rules)} rules)"


class MamdaniInference:
    """
    Motor de inferência Mamdani.

    Características:
    - Consequentes são conjuntos fuzzy
    - Usa implicação (min ou produto)
    - Agrega regras (max, soma, etc.)
    - Requer defuzzificação
    """

    def __init__(self,
                 and_method: TNorm = TNorm.MIN,
                 or_method: SNorm = SNorm.MAX,
                 implication_method: str = 'min',
                 aggregation_method: str = 'max'):
        """
        Inicializa o motor de inferência Mamdani.

        Parâmetros:
            and_method: T-norma para operador AND
            or_method: S-norma para operador OR
            implication_method: Método de implicação ('min' ou 'product')
            aggregation_method: Método de agregação ('max', 'sum', 'probabilistic')
        """
        self.fuzzy_op = FuzzyOperator(and_method, or_method)
        self.implication_method = implication_method
        self.aggregation_method = aggregation_method

    def infer(self,
              fuzzified_inputs: Dict[str, Dict[str, float]],
              rules: List[FuzzyRule],
              output_variable: LinguisticVariable,
              num_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Executa a inferência Mamdani.

        Parâmetros:
            fuzzified_inputs: Entradas fuzzificadas {variável: {termo: grau}}
            rules: Lista de regras a avaliar
            output_variable: Variável linguística de saída
            num_points: Número de pontos para discretizar o universo

        Retorna:
            Tupla (universo, função_pertinência_agregada)
        """
        # Cria universo de discurso da saída
        x = output_variable.get_universe_array(num_points)

        # Lista para armazenar saídas de cada regra
        rules_outputs = []

        # Avalia cada regra
        for rule in rules:
            # Calcula grau de ativação da regra
            firing_strength = rule.evaluate_antecedent(fuzzified_inputs, self.fuzzy_op)

            if firing_strength > 0:
                # Para cada variável de saída na regra
                for out_var_name, out_term_name in rule.consequent.items():
                    if out_var_name != output_variable.name:
                        continue

                    # Obtém o conjunto fuzzy de saída
                    if out_term_name not in output_variable.terms:
                        raise ValueError(f"Termo '{out_term_name}' não encontrado na variável de saída")

                    output_mf = output_variable.terms[out_term_name].membership(x)

                    # Aplica implicação
                    if self.implication_method == 'min':
                        implied_mf = np.minimum(firing_strength, output_mf)
                    elif self.implication_method == 'product':
                        implied_mf = firing_strength * output_mf
                    else:
                        raise ValueError(f"Método de implicação '{self.implication_method}' não suportado")

                    rules_outputs.append(implied_mf)

        # Agrega as saídas das regras
        if not rules_outputs:
            # Se nenhuma regra foi ativada, retorna conjunto vazio
            aggregated = np.zeros_like(x)
        else:
            if self.aggregation_method == 'max':
                aggregated = np.maximum.reduce(rules_outputs)
            elif self.aggregation_method == 'sum':
                aggregated = np.minimum(np.sum(rules_outputs, axis=0), 1.0)
            elif self.aggregation_method == 'probabilistic':
                aggregated = rules_outputs[0]
                for output in rules_outputs[1:]:
                    aggregated = aggregated + output - aggregated * output
            else:
                raise ValueError(f"Método de agregação '{self.aggregation_method}' não suportado")

        return x, aggregated


class SugenoInference:
    """
    Motor de inferência Sugeno (Takagi-Sugeno-Kang).

    Características:
    - Consequentes são funções lineares ou constantes
    - Saída é média ponderada
    - Não requer defuzzificação
    """

    def __init__(self,
                 and_method: TNorm = TNorm.MIN,
                 or_method: SNorm = SNorm.MAX,
                 order: int = 0):
        """
        Inicializa o motor de inferência Sugeno.

        Parâmetros:
            and_method: T-norma para operador AND
            or_method: S-norma para operador OR
            order: Ordem do sistema (0 = constantes, 1 = linear)
        """
        self.fuzzy_op = FuzzyOperator(and_method, or_method)
        self.order = order

    def infer(self,
              fuzzified_inputs: Dict[str, Dict[str, float]],
              crisp_inputs: Dict[str, float],
              rules: List[FuzzyRule]) -> float:
        """
        Executa a inferência Sugeno.

        Parâmetros:
            fuzzified_inputs: Entradas fuzzificadas {variável: {termo: grau}}
            crisp_inputs: Valores crisp das entradas {variável: valor}
            rules: Lista de regras a avaliar

        Retorna:
            Saída crisp (média ponderada)
        """
        firing_strengths = []
        rule_outputs = []

        # Avalia cada regra
        for rule in rules:
            # Calcula grau de ativação da regra
            firing_strength = rule.evaluate_antecedent(fuzzified_inputs, self.fuzzy_op)

            if firing_strength > 0:
                firing_strengths.append(firing_strength)

                # Calcula saída da regra
                # Em Sugeno, o consequente pode ser:
                # - Ordem 0: constante
                # - Ordem 1: função linear das entradas
                rule_output = self._evaluate_consequent(rule, crisp_inputs)
                rule_outputs.append(rule_output)

        # Média ponderada
        if not firing_strengths:
            return 0.0

        firing_strengths = np.array(firing_strengths)
        rule_outputs = np.array(rule_outputs)

        total_strength = np.sum(firing_strengths)

        if total_strength == 0:
            return 0.0

        return np.sum(firing_strengths * rule_outputs) / total_strength

    def _evaluate_consequent(self,
                            rule: FuzzyRule,
                            crisp_inputs: Dict[str, float]) -> float:
        """
        Avalia o consequente de uma regra Sugeno.

        Parâmetros:
            rule: Regra a avaliar
            crisp_inputs: Valores crisp das entradas

        Retorna:
            Valor de saída da regra
        """
        # Para Sugeno, assumimos que há apenas uma saída por regra
        # e que o consequente é um número (ordem 0) ou uma função (ordem 1)

        if len(rule.consequent) != 1:
            raise ValueError("Regras Sugeno devem ter exatamente uma saída")

        output_var, output_value = next(iter(rule.consequent.items()))

        # Ordem 0: constante
        if isinstance(output_value, (int, float)):
            return float(output_value)

        # Ordem 1: função linear com lista/tupla
        # Formato: [c0, c1, c2, ...] representa c0*x1 + c1*x2 + ... + c_n (constante no final)
        if isinstance(output_value, (list, tuple)):
            input_vars = list(crisp_inputs.keys())

            # Se tem mais coeficientes que variáveis, último é constante
            if len(output_value) > len(input_vars):
                result = output_value[-1]  # Constante
                coefs = output_value[:-1]
            else:
                result = 0.0
                coefs = output_value

            # Soma coef * input_value
            for i, coef in enumerate(coefs):
                if i < len(input_vars):
                    result += coef * crisp_inputs[input_vars[i]]

            return result

        # Ordem 1: função linear com dicionário
        # Formato esperado: dicionário com coeficientes
        # Ex: {'const': 5, 'x1': 2, 'x2': -1} representa 5 + 2*x1 - 1*x2
        if isinstance(output_value, dict):
            result = output_value.get('const', 0.0)

            for var_name, coef in output_value.items():
                if var_name != 'const':
                    if var_name in crisp_inputs:
                        result += coef * crisp_inputs[var_name]

            return result

        # Se for callable (função)
        if callable(output_value):
            return output_value(crisp_inputs)

        raise ValueError(f"Consequente Sugeno inválido: {output_value}")


class TSKInference(SugenoInference):
    """
    Alias para SugenoInference (Takagi-Sugeno-Kang).
    """
    pass


def create_rule_from_string(rule_str: str) -> FuzzyRule:
    """
    Cria uma regra a partir de uma string (parser simples).

    Formato: "IF var1 IS term1 AND var2 IS term2 THEN output IS term_out"

    Parâmetros:
        rule_str: String descrevendo a regra

    Retorna:
        Objeto FuzzyRule

    Nota: Esta é uma implementação simplificada.
    Para casos mais complexos, use construção direta de FuzzyRule.
    """
    # Remove espaços extras
    rule_str = ' '.join(rule_str.split())

    # Separa IF e THEN
    if ' THEN ' not in rule_str.upper():
        raise ValueError("Regra deve conter 'THEN'")

    parts = rule_str.upper().split(' THEN ')
    if len(parts) != 2:
        raise ValueError("Formato de regra inválido")

    antecedent_str, consequent_str = parts

    # Remove "IF" do início
    if antecedent_str.startswith('IF '):
        antecedent_str = antecedent_str[3:]

    # Determina operador (AND ou OR)
    if ' OR ' in antecedent_str:
        operator = 'OR'
        ant_parts = antecedent_str.split(' OR ')
    else:
        operator = 'AND'
        ant_parts = antecedent_str.split(' AND ')

    # Parse antecedentes
    antecedents = {}
    for part in ant_parts:
        tokens = part.strip().split(' IS ')
        if len(tokens) != 2:
            raise ValueError(f"Formato de antecedente inválido: {part}")
        var, term = tokens
        antecedents[var.strip().lower()] = term.strip().lower()

    # Parse consequente
    cons_parts = consequent_str.split(' AND ')
    consequent = {}
    for part in cons_parts:
        tokens = part.strip().split(' IS ')
        if len(tokens) != 2:
            raise ValueError(f"Formato de consequente inválido: {part}")
        var, term = tokens
        consequent[var.strip().lower()] = term.strip().lower()

    return FuzzyRule(antecedents, consequent, operator)
