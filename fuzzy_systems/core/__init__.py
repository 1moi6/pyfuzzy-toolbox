"""
Módulo Core - Componentes Fundamentais

Este módulo contém os componentes básicos reutilizáveis do fuzzy_systems:
- Funções de pertinência
- Operadores fuzzy (t-normas, s-normas)
- Fuzzificação (FuzzySet, LinguisticVariable)
- Defuzzificação
"""

# Funções de pertinência
from .membership import (
    triangular,
    trapezoidal,
    gaussian,
    generalized_bell,
    sigmoid,
    singleton,
    MEMBERSHIP_FUNCTIONS
)

# Operadores fuzzy
from .operators import (
    # T-normas (AND)
    fuzzy_and_min,
    fuzzy_and_product,
    fuzzy_and_lukasiewicz,
    fuzzy_and_drastic,
    fuzzy_and_hamacher,
    # S-normas (OR)
    fuzzy_or_max,
    fuzzy_or_probabilistic,
    fuzzy_or_bounded,
    fuzzy_or_drastic,
    fuzzy_or_hamacher,
    # Negação
    fuzzy_not,
    fuzzy_not_sugeno,
    fuzzy_not_yager,
    # Classes e enums
    TNorm,
    SNorm,
    FuzzyOperator,
    # Implicação e agregação
    implication_mamdani,
    implication_larsen,
    aggregate_max,
    aggregate_sum,
    aggregate_probabilistic,
)

# Fuzzificação
from .fuzzification import (
    FuzzySet,
    LinguisticVariable,
    Fuzzifier,
)

# Defuzzificação
from .defuzzification import (
    centroid,
    bisector,
    mean_of_maximum,
    smallest_of_maximum,
    largest_of_maximum,
    weighted_average,
    sugeno_defuzzify,
    mamdani_defuzzify,
    DefuzzMethod,
    Defuzzifier,
)

__all__ = [
    # Membership functions
    'triangular',
    'trapezoidal',
    'gaussian',
    'generalized_bell',
    'sigmoid',
    'singleton',
    'MEMBERSHIP_FUNCTIONS',
    # Operators
    'fuzzy_and_min',
    'fuzzy_and_product',
    'fuzzy_and_lukasiewicz',
    'fuzzy_and_drastic',
    'fuzzy_and_hamacher',
    'fuzzy_or_max',
    'fuzzy_or_probabilistic',
    'fuzzy_or_bounded',
    'fuzzy_or_drastic',
    'fuzzy_or_hamacher',
    'fuzzy_not',
    'fuzzy_not_sugeno',
    'fuzzy_not_yager',
    'TNorm',
    'SNorm',
    'FuzzyOperator',
    'implication_mamdani',
    'implication_larsen',
    'aggregate_max',
    'aggregate_sum',
    'aggregate_probabilistic',
    # Fuzzification
    'FuzzySet',
    'LinguisticVariable',
    'Fuzzifier',
    # Defuzzification
    'centroid',
    'bisector',
    'mean_of_maximum',
    'smallest_of_maximum',
    'largest_of_maximum',
    'weighted_average',
    'sugeno_defuzzify',
    'mamdani_defuzzify',
    'DefuzzMethod',
    'Defuzzifier',
]
