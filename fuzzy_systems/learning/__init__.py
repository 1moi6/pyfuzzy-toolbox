"""
Módulo Learning - Aprendizado e Otimização de Sistemas Fuzzy

Este módulo contém algoritmos para aprendizado e otimização:
- Wang-Mendel (geração automática de regras) ✅
- ANFIS (Adaptive Neuro-Fuzzy Inference System) ✅
- MamdaniLearning (Sistema Mamdani com aprendizado) ✅
- Metaheurísticas: PSO, DE, GA ✅
- Clustering fuzzy (FCM - Fuzzy C-Means) [TODO]

Status: Em desenvolvimento
"""

from .wang_mendel import WangMendelRegression, WangMendelClassification, WangMendelLearning
from .anfis import ANFIS
from .mandani_learning import MamdaniLearning
from .metaheuristics import PSO, DE, GA, get_optimizer

# TODO: Implementar FCM (Fuzzy C-Means Clustering)

__all__ = [
    'WangMendelLearning',
    'WangMendelRegression',
    'WangMendelClassification',
    'ANFIS',
    'MamdaniLearning',
    'PSO',
    'DE',
    'GA',
    'get_optimizer',
]

# Placeholder para desenvolvimento futuro
__version__ = '0.3.0-dev'
