"""
Módulo de Operadores Fuzzy

Este módulo implementa operadores fuzzy fundamentais como
AND, OR, NOT e suas variantes (t-normas e s-normas).
"""

import numpy as np
from typing import Union, Callable
from enum import Enum


class TNorm(Enum):
    """Enumeração de T-normas (operadores AND fuzzy)."""
    MIN = "min"
    PRODUCT = "product"
    LUKASIEWICZ = "lukasiewicz"
    DRASTIC = "drastic"
    HAMACHER = "hamacher"


class SNorm(Enum):
    """Enumeração de S-normas (operadores OR fuzzy)."""
    MAX = "max"
    PROBABILISTIC = "probabilistic"
    BOUNDED = "bounded"
    DRASTIC = "drastic"
    HAMACHER = "hamacher"


# ============================================================================
# T-Normas (Operadores AND)
# ============================================================================

def fuzzy_and_min(a: Union[float, np.ndarray],
                  b: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Operador AND usando mínimo (t-norma padrão de Zadeh).

    Parâmetros:
        a, b: Graus de pertinência

    Retorna:
        min(a, b)
    """
    return np.minimum(a, b)


def fuzzy_and_product(a: Union[float, np.ndarray],
                      b: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Operador AND usando produto algébrico.

    Parâmetros:
        a, b: Graus de pertinência

    Retorna:
        a * b
    """
    return a * b


def fuzzy_and_lukasiewicz(a: Union[float, np.ndarray],
                          b: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Operador AND de Łukasiewicz (t-norma limitada).

    Parâmetros:
        a, b: Graus de pertinência

    Retorna:
        max(0, a + b - 1)
    """
    return np.maximum(0, a + b - 1)


def fuzzy_and_drastic(a: Union[float, np.ndarray],
                      b: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Operador AND drástico.

    Parâmetros:
        a, b: Graus de pertinência

    Retorna:
        b se a == 1, a se b == 1, 0 caso contrário
    """
    a = np.asarray(a)
    b = np.asarray(b)

    result = np.where(a == 1, b,
                     np.where(b == 1, a, 0))

    return result if result.shape else float(result)


def fuzzy_and_hamacher(a: Union[float, np.ndarray],
                       b: Union[float, np.ndarray],
                       gamma: float = 2.0) -> Union[float, np.ndarray]:
    """
    Operador AND de Hamacher.

    Parâmetros:
        a, b: Graus de pertinência
        gamma: Parâmetro da família Hamacher (padrão: 2.0)

    Retorna:
        (a * b) / (gamma - (gamma - 1) * (a + b - a * b))
    """
    numerator = a * b
    denominator = gamma - (gamma - 1) * (a + b - a * b)

    # Evita divisão por zero
    denominator = np.where(denominator == 0, 1e-10, denominator)

    return numerator / denominator


# ============================================================================
# S-Normas (Operadores OR)
# ============================================================================

def fuzzy_or_max(a: Union[float, np.ndarray],
                 b: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Operador OR usando máximo (s-norma padrão de Zadeh).

    Parâmetros:
        a, b: Graus de pertinência

    Retorna:
        max(a, b)
    """
    return np.maximum(a, b)


def fuzzy_or_probabilistic(a: Union[float, np.ndarray],
                           b: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Operador OR probabilístico (soma algébrica).

    Parâmetros:
        a, b: Graus de pertinência

    Retorna:
        a + b - a * b
    """
    return a + b - a * b


def fuzzy_or_bounded(a: Union[float, np.ndarray],
                     b: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Operador OR limitado (s-norma de Łukasiewicz).

    Parâmetros:
        a, b: Graus de pertinência

    Retorna:
        min(1, a + b)
    """
    return np.minimum(1, a + b)


def fuzzy_or_drastic(a: Union[float, np.ndarray],
                     b: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Operador OR drástico.

    Parâmetros:
        a, b: Graus de pertinência

    Retorna:
        b se a == 0, a se b == 0, 1 caso contrário
    """
    a = np.asarray(a)
    b = np.asarray(b)

    result = np.where(a == 0, b,
                     np.where(b == 0, a, 1))

    return result if result.shape else float(result)


def fuzzy_or_hamacher(a: Union[float, np.ndarray],
                      b: Union[float, np.ndarray],
                      gamma: float = 2.0) -> Union[float, np.ndarray]:
    """
    Operador OR de Hamacher.

    Parâmetros:
        a, b: Graus de pertinência
        gamma: Parâmetro da família Hamacher (padrão: 2.0)

    Retorna:
        (a + b - a * b - (1 - gamma) * a * b) / (1 - (1 - gamma) * a * b)
    """
    numerator = a + b - a * b - (1 - gamma) * a * b
    denominator = 1 - (1 - gamma) * a * b

    # Evita divisão por zero
    denominator = np.where(denominator == 0, 1e-10, denominator)

    return numerator / denominator


# ============================================================================
# Operador NOT
# ============================================================================

def fuzzy_not(a: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Operador NOT fuzzy (complemento de Zadeh).

    Parâmetros:
        a: Grau de pertinência

    Retorna:
        1 - a
    """
    return 1 - a


def fuzzy_not_sugeno(a: Union[float, np.ndarray],
                     lambda_param: float = 1.0) -> Union[float, np.ndarray]:
    """
    Operador NOT de Sugeno (complemento).

    Parâmetros:
        a: Grau de pertinência
        lambda_param: Parâmetro λ > -1

    Retorna:
        (1 - a) / (1 + λ * a)
    """
    if lambda_param <= -1:
        raise ValueError("Parâmetro lambda deve ser > -1")

    return (1 - a) / (1 + lambda_param * a)


def fuzzy_not_yager(a: Union[float, np.ndarray],
                    w: float = 1.0) -> Union[float, np.ndarray]:
    """
    Operador NOT de Yager.

    Parâmetros:
        a: Grau de pertinência
        w: Parâmetro w > 0

    Retorna:
        (1 - a^w)^(1/w)
    """
    if w <= 0:
        raise ValueError("Parâmetro w deve ser > 0")

    return (1 - a ** w) ** (1 / w)


# ============================================================================
# Classes de Operadores Configuráveis
# ============================================================================

class FuzzyOperator:
    """
    Classe para gerenciar operadores fuzzy configuráveis.
    """

    def __init__(self,
                 and_method: TNorm = TNorm.MIN,
                 or_method: SNorm = SNorm.MAX):
        """
        Inicializa o operador fuzzy.

        Parâmetros:
            and_method: T-norma a usar para AND
            or_method: S-norma a usar para OR
        """
        self.and_method = and_method
        self.or_method = or_method

        self._and_ops = {
            TNorm.MIN: fuzzy_and_min,
            TNorm.PRODUCT: fuzzy_and_product,
            TNorm.LUKASIEWICZ: fuzzy_and_lukasiewicz,
            TNorm.DRASTIC: fuzzy_and_drastic,
            TNorm.HAMACHER: fuzzy_and_hamacher,
        }

        self._or_ops = {
            SNorm.MAX: fuzzy_or_max,
            SNorm.PROBABILISTIC: fuzzy_or_probabilistic,
            SNorm.BOUNDED: fuzzy_or_bounded,
            SNorm.DRASTIC: fuzzy_or_drastic,
            SNorm.HAMACHER: fuzzy_or_hamacher,
        }

    def AND(self, a: Union[float, np.ndarray],
            b: Union[float, np.ndarray],
            **kwargs) -> Union[float, np.ndarray]:
        """
        Aplica operador AND fuzzy.

        Parâmetros:
            a, b: Graus de pertinência
            **kwargs: Parâmetros adicionais para operadores específicos

        Retorna:
            Resultado do operador AND
        """
        op = self._and_ops[self.and_method]
        return op(a, b, **kwargs) if kwargs else op(a, b)

    def OR(self, a: Union[float, np.ndarray],
           b: Union[float, np.ndarray],
           **kwargs) -> Union[float, np.ndarray]:
        """
        Aplica operador OR fuzzy.

        Parâmetros:
            a, b: Graus de pertinência
            **kwargs: Parâmetros adicionais para operadores específicos

        Retorna:
            Resultado do operador OR
        """
        op = self._or_ops[self.or_method]
        return op(a, b, **kwargs) if kwargs else op(a, b)

    def NOT(self, a: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Aplica operador NOT fuzzy (complemento padrão de Zadeh).

        Parâmetros:
            a: Grau de pertinência

        Retorna:
            1 - a
        """
        return fuzzy_not(a)


# ============================================================================
# Funções de Implicação Fuzzy
# ============================================================================

def implication_mamdani(antecedent: Union[float, np.ndarray],
                        consequent: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Implicação de Mamdani (min).

    Parâmetros:
        antecedent: Grau de ativação da premissa
        consequent: Conjunto fuzzy da conclusão

    Retorna:
        min(antecedent, consequent)
    """
    return np.minimum(antecedent, consequent)


def implication_larsen(antecedent: Union[float, np.ndarray],
                       consequent: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Implicação de Larsen (produto).

    Parâmetros:
        antecedent: Grau de ativação da premissa
        consequent: Conjunto fuzzy da conclusão

    Retorna:
        antecedent * consequent
    """
    return antecedent * consequent


# ============================================================================
# Agregação de Regras
# ============================================================================

def aggregate_max(rules_output: list) -> np.ndarray:
    """
    Agrega saídas de regras usando máximo (padrão Mamdani).

    Parâmetros:
        rules_output: Lista de arrays representando saídas de regras

    Retorna:
        Agregação usando máximo
    """
    return np.maximum.reduce(rules_output)


def aggregate_sum(rules_output: list) -> np.ndarray:
    """
    Agrega saídas de regras usando soma (limitada a 1).

    Parâmetros:
        rules_output: Lista de arrays representando saídas de regras

    Retorna:
        Agregação usando soma limitada
    """
    result = np.sum(rules_output, axis=0)
    return np.minimum(result, 1.0)


def aggregate_probabilistic(rules_output: list) -> np.ndarray:
    """
    Agrega saídas de regras usando OR probabilístico.

    Parâmetros:
        rules_output: Lista de arrays representando saídas de regras

    Retorna:
        Agregação usando OR probabilístico
    """
    if not rules_output:
        raise ValueError("Lista de saídas vazia")

    result = rules_output[0]
    for output in rules_output[1:]:
        result = fuzzy_or_probabilistic(result, output)

    return result
