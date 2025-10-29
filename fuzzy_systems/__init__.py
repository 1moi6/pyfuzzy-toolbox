"""
Fuzzy Systems - Biblioteca Completa para Sistemas Fuzzy

Uma biblioteca Python moderna e abrangente para sistemas fuzzy, incluindo:
- Sistemas de Inferência Fuzzy (Mamdani, Sugeno/TSK)
- Aprendizado e Otimização (ANFIS, Wang-Mendel, GA, PSO)
- Sistemas Dinâmicos (EDOs fuzzy, p-fuzzy)
- Ferramentas e Utilidades

Desenvolvido com foco em aplicações didáticas e profissionais.
"""

# ============================================================================
# API PÚBLICA - Mantém compatibilidade retroativa total
# ============================================================================

# Core - Funções de Pertinência
from .core.membership import (
    triangular,
    trapezoidal,
    gaussian,
    generalized_bell,
    sigmoid,
    singleton,
    MEMBERSHIP_FUNCTIONS,
)

# Core - Operadores Fuzzy
from .core.operators import (
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
    # Enums e Classes
    TNorm,
    SNorm,
    FuzzyOperator,
    # Implicação e Agregação
    implication_mamdani,
    implication_larsen,
    aggregate_max,
    aggregate_sum,
    aggregate_probabilistic,
)

# Core - Fuzzificação
from .core.fuzzification import (
    FuzzySet,
    LinguisticVariable,
    Fuzzifier,
)

# Core - Defuzzificação
from .core.defuzzification import (
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

# Inference - Regras e Engines
from .inference.rules import (
    FuzzyRule,
    RuleBase,
    MamdaniInference,
    SugenoInference,
    TSKInference,
)

# Inference - Sistemas Completos
from .inference.systems import (
    FuzzyInferenceSystem,
    MamdaniSystem,
    SugenoSystem,
    TSKSystem,
    create_mamdani_system,
)

# Learning
from . import learning

# Dynamics - Sistemas Dinâmicos Fuzzy
from . import dynamics

# Utils - Em desenvolvimento
# from .utils import plot_membership, plot_surface, ...

# ============================================================================
# Metadata
# ============================================================================

__version__ = '1.0.4'
__author__ = 'Moiseis Cecconello'
__license__ = 'MIT'

# ============================================================================
# __all__ - Define a API pública
# ============================================================================

__all__ = [
    # === Core - Membership Functions ===
    'triangular',
    'trapezoidal',
    'gaussian',
    'generalized_bell',
    'sigmoid',
    'singleton',
    'MEMBERSHIP_FUNCTIONS',

    # === Core - Operators ===
    # T-normas
    'fuzzy_and_min',
    'fuzzy_and_product',
    'fuzzy_and_lukasiewicz',
    'fuzzy_and_drastic',
    'fuzzy_and_hamacher',
    # S-normas
    'fuzzy_or_max',
    'fuzzy_or_probabilistic',
    'fuzzy_or_bounded',
    'fuzzy_or_drastic',
    'fuzzy_or_hamacher',
    # Negação
    'fuzzy_not',
    'fuzzy_not_sugeno',
    'fuzzy_not_yager',
    # Classes e Enums
    'TNorm',
    'SNorm',
    'FuzzyOperator',
    # Implicação e Agregação
    'implication_mamdani',
    'implication_larsen',
    'aggregate_max',
    'aggregate_sum',
    'aggregate_probabilistic',

    # === Core - Fuzzification ===
    'FuzzySet',
    'LinguisticVariable',
    'Fuzzifier',

    # === Core - Defuzzification ===
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

    # === Inference - Rules and Engines ===
    'FuzzyRule',
    'RuleBase',
    'MamdaniInference',
    'SugenoInference',
    'TSKInference',

    # === Inference - Systems ===
    'FuzzyInferenceSystem',
    'MamdaniSystem',
    'SugenoSystem',
    'TSKSystem',
    'create_mamdani_system',

    # === Learning ===
    'learning',

    # === Dynamics ===
    'dynamics',

    # === Metadata ===
    '__version__',
    '__author__',
    '__license__',
]
