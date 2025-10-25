"""
Módulo de Funções de Pertinência (Membership Functions)

Este módulo implementa diferentes tipos de funções de pertinência
utilizadas em sistemas de inferência fuzzy.
"""

import numpy as np
from typing import Union, Tuple, Callable


def triangular(x: Union[float, np.ndarray],
               params: Tuple[float, float, float]) -> Union[float, np.ndarray]:
    """
    Função de pertinência triangular.

    Parâmetros:
        x: Valor(es) de entrada
        params: Tupla (a, b, c) onde:
            - a: limite inferior (grau 0)
            - b: pico (grau 1)
            - c: limite superior (grau 0)

    Retorna:
        Grau(s) de pertinência no intervalo [0, 1]

    Exemplo:
        >>> triangular(5, (0, 5, 10))
        1.0
        >>> triangular(2.5, (0, 5, 10))
        0.5
    """
    a, b, c = params
    x = np.asarray(x)

    # Inicializa com zeros
    result = np.zeros_like(x, dtype=float)

    # Rampa ascendente: de a até b
    mask1 = (x >= a) & (x <= b)
    if b != a:
        result[mask1] = (x[mask1] - a) / (b - a)

    # Rampa descendente: de b até c
    mask2 = (x > b) & (x <= c)
    if c != b:
        result[mask2] = (c - x[mask2]) / (c - b)

    return result if result.shape else float(result)


def trapezoidal(x: Union[float, np.ndarray],
                params: Tuple[float, float, float, float]) -> Union[float, np.ndarray]:
    """
    Função de pertinência trapezoidal.

    Parâmetros:
        x: Valor(es) de entrada
        params: Tupla (a, b, c, d) onde:
            - a: limite inferior (grau 0)
            - b: início do platô (grau 1)
            - c: fim do platô (grau 1)
            - d: limite superior (grau 0)

    Retorna:
        Grau(s) de pertinência no intervalo [0, 1]

    Exemplo:
        >>> trapezoidal(5, (0, 3, 7, 10))
        1.0
        >>> trapezoidal(1.5, (0, 3, 7, 10))
        0.5
    """
    a, b, c, d = params
    x = np.asarray(x)

    # Inicializa com zeros
    result = np.zeros_like(x, dtype=float)

    # Rampa ascendente: de a até b
    mask1 = (x >= a) & (x < b)
    if b != a:
        result[mask1] = (x[mask1] - a) / (b - a)

    # Platô: de b até c
    mask2 = (x >= b) & (x <= c)
    result[mask2] = 1.0

    # Rampa descendente: de c até d
    mask3 = (x > c) & (x <= d)
    if d != c:
        result[mask3] = (d - x[mask3]) / (d - c)

    return result if result.shape else float(result)


def gaussian(x: Union[float, np.ndarray],
             params: Tuple[float, float]) -> Union[float, np.ndarray]:
    """
    Função de pertinência gaussiana.

    Parâmetros:
        x: Valor(es) de entrada
        params: Tupla (mean, sigma) onde:
            - mean: média (centro da curva)
            - sigma: desvio padrão (controla a largura)

    Retorna:
        Grau(s) de pertinência no intervalo [0, 1]

    Exemplo:
        >>> gaussian(5, (5, 1))
        1.0
        >>> np.isclose(gaussian(6, (5, 1)), 0.6065, atol=0.001)
        True
    """
    mean, sigma = params
    x = np.asarray(x)

    result = np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))

    return result if result.shape else float(result)


def generalized_bell(x: Union[float, np.ndarray],
                     params: Tuple[float, float, float]) -> Union[float, np.ndarray]:
    """
    Função de pertinência sino generalizado (Generalized Bell).

    Parâmetros:
        x: Valor(es) de entrada
        params: Tupla (a, b, c) onde:
            - a: controla a largura
            - b: controla a inclinação (geralmente positivo)
            - c: centro da curva

    Retorna:
        Grau(s) de pertinência no intervalo [0, 1]

    Exemplo:
        >>> generalized_bell(5, (2, 4, 5))
        1.0
    """
    a, b, c = params
    x = np.asarray(x)

    result = 1 / (1 + np.abs((x - c) / a) ** (2 * b))

    return result if result.shape else float(result)


def sigmoid(x: Union[float, np.ndarray],
            params: Tuple[float, float]) -> Union[float, np.ndarray]:
    """
    Função de pertinência sigmoide.

    Parâmetros:
        x: Valor(es) de entrada
        params: Tupla (a, c) onde:
            - a: controla a inclinação
            - c: ponto de inflexão

    Retorna:
        Grau(s) de pertinência no intervalo [0, 1]

    Exemplo:
        >>> sigmoid(0, (1, 0))
        0.5
    """
    a, c = params
    x = np.asarray(x)

    result = 1 / (1 + np.exp(-a * (x - c)))

    return result if result.shape else float(result)


def singleton(x: Union[float, np.ndarray],
              value: float,
              tolerance: float = 1e-6) -> Union[float, np.ndarray]:
    """
    Função de pertinência singleton (valor discreto).

    Parâmetros:
        x: Valor(es) de entrada
        value: Valor do singleton
        tolerance: Tolerância para comparação de igualdade

    Retorna:
        1.0 se x == value (dentro da tolerância), 0.0 caso contrário

    Exemplo:
        >>> singleton(5.0, 5.0)
        1.0
        >>> singleton(5.001, 5.0, tolerance=0.01)
        1.0
        >>> singleton(4.0, 5.0)
        0.0
    """
    x = np.asarray(x)

    result = np.where(np.abs(x - value) <= tolerance, 1.0, 0.0)

    return result if result.shape else float(result)


def custom_membership(x: Union[float, np.ndarray],
                      func: Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]]) -> Union[float, np.ndarray]:
    """
    Permite definir uma função de pertinência customizada.

    Parâmetros:
        x: Valor(es) de entrada
        func: Função callable que recebe x e retorna grau de pertinência

    Retorna:
        Grau(s) de pertinência calculado pela função customizada

    Exemplo:
        >>> def my_func(x): return np.clip(x / 10, 0, 1)
        >>> custom_membership(5, my_func)
        0.5
    """
    return func(x)


# Dicionário para fácil acesso às funções
MEMBERSHIP_FUNCTIONS = {
    'triangular': triangular,
    'trapezoidal': trapezoidal,
    'gaussian': gaussian,
    'generalized_bell': generalized_bell,
    'gbellmf': generalized_bell,  # Alias
    'sigmoid': sigmoid,
    'sigmf': sigmoid,  # Alias
    'singleton': singleton,
    'trimf': triangular,  # Alias compatível com MATLAB
    'trapmf': trapezoidal,  # Alias compatível com MATLAB
    'gaussmf': gaussian,  # Alias compatível com MATLAB
}


def get_membership_function(name: str) -> Callable:
    """
    Retorna a função de pertinência pelo nome.

    Parâmetros:
        name: Nome da função de pertinência

    Retorna:
        Função de pertinência correspondente

    Raises:
        ValueError: Se o nome não for reconhecido
    """
    if name not in MEMBERSHIP_FUNCTIONS:
        available = ', '.join(MEMBERSHIP_FUNCTIONS.keys())
        raise ValueError(f"Função de pertinência '{name}' não encontrada. "
                        f"Disponíveis: {available}")

    return MEMBERSHIP_FUNCTIONS[name]
