"""
Módulo de Defuzzificação

Este módulo implementa diversos métodos de defuzzificação
para converter conjuntos fuzzy em valores crisp.
"""

import numpy as np
from typing import Union, Tuple
from enum import Enum


class DefuzzMethod(Enum):
    """Enumeração de métodos de defuzzificação."""
    CENTROID = "centroid"
    BISECTOR = "bisector"
    MOM = "mom"  # Mean of Maximum
    SOM = "som"  # Smallest of Maximum
    LOM = "lom"  # Largest of Maximum
    WEIGHTED_AVERAGE = "weighted_average"  # Para Sugeno


def centroid(x: np.ndarray, mf: np.ndarray) -> float:
    """
    Método do centroide (centro de área/gravidade).
    Método mais comum em sistemas Mamdani.

    Parâmetros:
        x: Array do universo de discurso
        mf: Array com graus de pertinência correspondentes

    Retorna:
        Valor crisp defuzzificado

    Fórmula:
        COG = ∫ x·μ(x) dx / ∫ μ(x) dx
    """
    # Calcula área sob a curva usando integração trapezoidal
    numerator = np.trapz(x * mf, x)
    denominator = np.trapz(mf, x)

    if denominator == 0:
        # Se área é zero, retorna o centro do universo
        return (x[0] + x[-1]) / 2

    return numerator / denominator


def bisector(x: np.ndarray, mf: np.ndarray) -> float:
    """
    Método do bisector (divide área em duas partes iguais).

    Parâmetros:
        x: Array do universo de discurso
        mf: Array com graus de pertinência correspondentes

    Retorna:
        Valor crisp que divide a área em duas partes iguais
    """
    # Calcula área total
    total_area = np.trapz(mf, x)

    if total_area == 0:
        return (x[0] + x[-1]) / 2

    # Calcula área acumulada
    cumulative_area = np.zeros_like(x)
    for i in range(1, len(x)):
        cumulative_area[i] = cumulative_area[i-1] + np.trapz(mf[i-1:i+1], x[i-1:i+1])

    # Encontra ponto onde área acumulada é metade da área total
    half_area = total_area / 2
    idx = np.argmin(np.abs(cumulative_area - half_area))

    return x[idx]


def mean_of_maximum(x: np.ndarray, mf: np.ndarray, tolerance: float = 1e-6) -> float:
    """
    Método da média dos máximos (MOM).

    Parâmetros:
        x: Array do universo de discurso
        mf: Array com graus de pertinência correspondentes
        tolerance: Tolerância para considerar valores como máximos

    Retorna:
        Média dos valores x onde μ(x) é máximo
    """
    max_value = np.max(mf)

    if max_value == 0:
        return (x[0] + x[-1]) / 2

    # Encontra todos os índices onde mf está próximo do máximo
    max_indices = np.where(mf >= (max_value - tolerance))[0]

    return np.mean(x[max_indices])


def smallest_of_maximum(x: np.ndarray, mf: np.ndarray, tolerance: float = 1e-6) -> float:
    """
    Método do menor máximo (SOM).

    Parâmetros:
        x: Array do universo de discurso
        mf: Array com graus de pertinência correspondentes
        tolerance: Tolerância para considerar valores como máximos

    Retorna:
        Menor valor x onde μ(x) é máximo
    """
    max_value = np.max(mf)

    if max_value == 0:
        return x[0]

    max_indices = np.where(mf >= (max_value - tolerance))[0]

    return x[max_indices[0]]


def largest_of_maximum(x: np.ndarray, mf: np.ndarray, tolerance: float = 1e-6) -> float:
    """
    Método do maior máximo (LOM).

    Parâmetros:
        x: Array do universo de discurso
        mf: Array com graus de pertinência correspondentes
        tolerance: Tolerância para considerar valores como máximos

    Retorna:
        Maior valor x onde μ(x) é máximo
    """
    max_value = np.max(mf)

    if max_value == 0:
        return x[-1]

    max_indices = np.where(mf >= (max_value - tolerance))[0]

    return x[max_indices[-1]]


def weighted_average(values: np.ndarray, weights: np.ndarray) -> float:
    """
    Método da média ponderada (usado principalmente em Sugeno).

    Parâmetros:
        values: Array com valores crisp (saídas das regras)
        weights: Array com pesos (graus de ativação das regras)

    Retorna:
        Média ponderada dos valores

    Fórmula:
        WA = Σ(wi · zi) / Σ(wi)
        onde wi são os pesos e zi são os valores
    """
    weights = np.asarray(weights)
    values = np.asarray(values)

    total_weight = np.sum(weights)

    if total_weight == 0:
        # Se todos os pesos são zero, retorna média simples
        return np.mean(values) if len(values) > 0 else 0.0

    return np.sum(weights * values) / total_weight


def defuzzify(x: np.ndarray,
              mf: np.ndarray,
              method: Union[str, DefuzzMethod] = 'centroid') -> float:
    """
    Função genérica de defuzzificação.

    Parâmetros:
        x: Array do universo de discurso
        mf: Array com graus de pertinência correspondentes
        method: Método de defuzzificação a usar

    Retorna:
        Valor crisp defuzzificado

    Raises:
        ValueError: Se o método não for reconhecido

    Exemplo:
        >>> x = np.linspace(0, 10, 100)
        >>> mf = np.exp(-(x - 5)**2)  # Gaussiana centrada em 5
        >>> result = defuzzify(x, mf, 'centroid')
        >>> # result ≈ 5.0
    """
    # Converte string para enum se necessário
    if isinstance(method, str):
        method = method.lower()
        try:
            method = DefuzzMethod(method)
        except ValueError:
            pass

    # Mapeamento de métodos
    methods = {
        DefuzzMethod.CENTROID: centroid,
        'centroid': centroid,
        'cog': centroid,
        'center_of_gravity': centroid,

        DefuzzMethod.BISECTOR: bisector,
        'bisector': bisector,

        DefuzzMethod.MOM: mean_of_maximum,
        'mom': mean_of_maximum,
        'mean_of_maximum': mean_of_maximum,

        DefuzzMethod.SOM: smallest_of_maximum,
        'som': smallest_of_maximum,
        'smallest_of_maximum': smallest_of_maximum,

        DefuzzMethod.LOM: largest_of_maximum,
        'lom': largest_of_maximum,
        'largest_of_maximum': largest_of_maximum,
    }

    if method not in methods:
        available = ', '.join([m.value for m in DefuzzMethod if m != DefuzzMethod.WEIGHTED_AVERAGE])
        raise ValueError(f"Método '{method}' não reconhecido. Disponíveis: {available}")

    defuzz_func = methods[method]
    return defuzz_func(x, mf)


class Defuzzifier:
    """
    Classe para gerenciar defuzzificação com configurações.
    """

    def __init__(self, method: Union[str, DefuzzMethod] = DefuzzMethod.CENTROID):
        """
        Inicializa o defuzzificador.

        Parâmetros:
            method: Método de defuzzificação padrão
        """
        self.method = method

    def defuzzify(self, x: np.ndarray, mf: np.ndarray) -> float:
        """
        Defuzzifica usando o método configurado.

        Parâmetros:
            x: Array do universo de discurso
            mf: Array com graus de pertinência

        Retorna:
            Valor crisp
        """
        return defuzzify(x, mf, self.method)

    def set_method(self, method: Union[str, DefuzzMethod]) -> None:
        """
        Altera o método de defuzzificação.

        Parâmetros:
            method: Novo método
        """
        self.method = method

    def __repr__(self) -> str:
        return f"Defuzzifier(method='{self.method}')"


def sugeno_defuzzify(rule_outputs: np.ndarray,
                     firing_strengths: np.ndarray) -> float:
    """
    Defuzzificação específica para sistemas Sugeno (média ponderada).

    Parâmetros:
        rule_outputs: Array com saídas crisp das regras
        firing_strengths: Array com graus de ativação das regras

    Retorna:
        Saída crisp final

    Exemplo:
        >>> outputs = np.array([10, 20, 30])
        >>> strengths = np.array([0.8, 0.5, 0.3])
        >>> result = sugeno_defuzzify(outputs, strengths)
        >>> # result = (0.8*10 + 0.5*20 + 0.3*30) / (0.8 + 0.5 + 0.3) ≈ 16.875
    """
    return weighted_average(rule_outputs, firing_strengths)


def mamdani_defuzzify(x: np.ndarray,
                      aggregated_mf: np.ndarray,
                      method: Union[str, DefuzzMethod] = DefuzzMethod.CENTROID) -> float:
    """
    Defuzzificação específica para sistemas Mamdani.

    Parâmetros:
        x: Universo de discurso da variável de saída
        aggregated_mf: Função de pertinência agregada de todas as regras
        method: Método de defuzzificação

    Retorna:
        Saída crisp final

    Exemplo:
        >>> x = np.linspace(0, 100, 1000)
        >>> mf = np.maximum(triangular(x, (0, 25, 50)) * 0.7,
        ...                 triangular(x, (50, 75, 100)) * 0.3)
        >>> result = mamdani_defuzzify(x, mf, 'centroid')
    """
    return defuzzify(x, aggregated_mf, method)


def height_defuzzification(x: np.ndarray, mf: np.ndarray) -> float:
    """
    Método de defuzzificação por altura (usado em algumas variações).

    Similar ao MOM mas usa apenas o pico mais alto.

    Parâmetros:
        x: Array do universo de discurso
        mf: Array com graus de pertinência

    Retorna:
        Valor x correspondente ao pico máximo
    """
    max_idx = np.argmax(mf)
    return x[max_idx]


def modified_height_defuzzification(terms_centers: np.ndarray,
                                    terms_heights: np.ndarray) -> float:
    """
    Método de altura modificado (para conjuntos fuzzy de saída separados).

    Útil quando se tem múltiplos termos de saída e se quer ponderar
    pelos picos de cada um.

    Parâmetros:
        terms_centers: Array com centros dos termos fuzzy de saída
        terms_heights: Array com alturas (graus máximos) de cada termo

    Retorna:
        Média ponderada dos centros pelas alturas
    """
    return weighted_average(terms_centers, terms_heights)
