"""
Módulo Inference - Sistemas de Inferência Fuzzy

Este módulo contém implementações de sistemas de inferência fuzzy:
- Regras fuzzy e base de regras
- Engines de inferência (Mamdani, Sugeno)
- Sistemas completos (MamdaniSystem, SugenoSystem)
"""

# Import interno dos componentes
from .rules import (
    FuzzyRule,
    RuleBase,
    MamdaniInference,
    SugenoInference,
    TSKInference,
)

from .systems import (
    FuzzyInferenceSystem,
    MamdaniSystem,
    SugenoSystem,
    TSKSystem,
    create_mamdani_system,
)

__all__ = [
    # Rules
    'FuzzyRule',
    'RuleBase',
    # Inference engines
    'MamdaniInference',
    'SugenoInference',
    'TSKInference',
    # Systems
    'FuzzyInferenceSystem',
    'MamdaniSystem',
    'SugenoSystem',
    'TSKSystem',
    # Factory functions
    'create_mamdani_system',
]
