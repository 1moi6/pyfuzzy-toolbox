"""
Módulo Dynamics - Sistemas Dinâmicos Fuzzy
===========================================

Este módulo contém ferramentas para análise de sistemas dinâmicos com incerteza fuzzy:

1. **Solver de EDOs Fuzzy (α-níveis)**:
   - FuzzyODESolver: Resolve EDOs com condições iniciais/parâmetros fuzzy
   - FuzzyNumber: Números fuzzy integrados com core
   - FuzzySolution: Visualização e exportação de soluções
   - Propagação de incerteza via α-níveis

2. **Sistemas p-Fuzzy**:
   - PFuzzyDiscrete: Sistemas dinâmicos discretos com regras fuzzy
   - PFuzzyContinuous: Sistemas dinâmicos contínuos com regras fuzzy
   - Integração completa com FIS (Mamdani/Sugeno)

Exemplos:
    >>> from fuzzy_systems.dynamics import FuzzyNumber, FuzzyODESolver
    >>>
    >>> # EDO Fuzzy: dy/dt = k*y
    >>> def growth(t, y, k):
    ...     return k * y
    >>>
    >>> y0 = FuzzyNumber.triangular(center=10, spread=2)
    >>> solver = FuzzyODESolver(
    ...     ode_func=growth,
    ...     t_span=(0, 10),
    ...     y0_fuzzy=[y0],
    ...     params={'k': 0.5}
    ... )
    >>> sol = solver.solve()
    >>> sol.plot()

    >>> # Sistema p-Fuzzy
    >>> from fuzzy_systems.dynamics import PFuzzyContinuous
    >>> from fuzzy_systems import MamdaniSystem
    >>>
    >>> fis = MamdaniSystem()
    >>> # ... configurar FIS ...
    >>> pfuzzy = PFuzzyContinuous(fis=fis, mode='absolute')
    >>> trajectory = pfuzzy.simulate(x0={'population': 10}, t_span=(0, 50))
    >>> pfuzzy.plot_trajectory()

Referências:
    Barros, L. C., Bassanezi, R. C., & Lodwick, W. A. (2017).
    "A First Course in Fuzzy Logic, Fuzzy Dynamical Systems, and Biomathematics"

Status: Implementado
Versão: 1.1.0
"""

from .fuzzy_ode import (
    FuzzyNumber,
    FuzzyODESolver,
    FuzzySolution
)

from .pfuzzy import (
    PFuzzySystem,
    PFuzzyDiscrete,
    PFuzzyContinuous
)

__all__ = [
    # EDO Fuzzy
    'FuzzyNumber',
    'FuzzyODESolver',
    'FuzzySolution',
    # Sistemas p-Fuzzy
    'PFuzzySystem',
    'PFuzzyDiscrete',
    'PFuzzyContinuous',
]

__version__ = '1.1.0'
