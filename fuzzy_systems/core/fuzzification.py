"""
Módulo de Fuzzificação

Este módulo contém classes e funções para o processo de fuzzificação,
convertendo valores crisp em conjuntos fuzzy.
"""

import numpy as np
from typing import Dict, List, Tuple, Union, Callable, Optional
from dataclasses import dataclass, field


@dataclass
class FuzzySet:
    """
    Representa um conjunto fuzzy com sua função de pertinência.

    Atributos:
        name: Nome do conjunto fuzzy (ex: "baixo", "médio", "alto")
        mf_type: Tipo da função de pertinência
        params: Parâmetros da função de pertinência
        mf_func: Função de pertinência (opcional, para funções customizadas)
    """
    name: str
    mf_type: str
    params: Tuple
    mf_func: Optional[Callable] = None

    def membership(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calcula o grau de pertinência de x neste conjunto fuzzy.

        Parâmetros:
            x: Valor(es) crisp de entrada

        Retorna:
            Grau(s) de pertinência
        """
        from .membership import get_membership_function

        if self.mf_func is not None:
            return self.mf_func(x)

        mf = get_membership_function(self.mf_type)
        return mf(x, self.params)

    def __repr__(self) -> str:
        return f"FuzzySet(name='{self.name}', type='{self.mf_type}', params={self.params})"


@dataclass
class LinguisticVariable:
    """
    Representa uma variável linguística com seus termos fuzzy.

    Atributos:
        name: Nome da variável (ex: "temperatura", "velocidade")
        universe: Universo de discurso [min, max]
        terms: Dicionário de termos fuzzy (nome -> FuzzySet)
    """
    name: str
    universe: Tuple[float, float]
    terms: Dict[str, FuzzySet] = field(default_factory=dict)

    def add_term(self,
                 name_or_fuzzy_set: Union[str, FuzzySet],
                 mf_type: Optional[str] = None,
                 params: Optional[Tuple] = None,
                 mf_func: Optional[Callable] = None) -> None:
        """
        Adiciona um termo fuzzy à variável linguística.

        Aceita duas formas de uso:

        Forma 1 (Direta - Recomendada):
            >>> var.add_term('baixo', 'triangular', (0, 0, 50))
            >>> var.add_term('alto', 'gaussiana', (75, 10))

        Forma 2 (Com FuzzySet):
            >>> var.add_term(FuzzySet('baixo', 'triangular', (0, 0, 50)))

        Parâmetros:
            name_or_fuzzy_set: Nome do termo (str) ou FuzzySet completo
            mf_type: Tipo da função de pertinência (apenas se name_or_fuzzy_set for str)
            params: Parâmetros da função (apenas se name_or_fuzzy_set for str)
            mf_func: Função customizada opcional
        """
        # Forma 2: Objeto FuzzySet completo
        if isinstance(name_or_fuzzy_set, FuzzySet):
            fuzzy_set = name_or_fuzzy_set
            self.terms[fuzzy_set.name] = fuzzy_set

        # Forma 1: Parâmetros diretos
        elif isinstance(name_or_fuzzy_set, str):
            if mf_type is None or params is None:
                raise ValueError(
                    "Quando passar nome como string, deve fornecer mf_type e params. "
                    "Exemplo: add_term('baixo', 'triangular', (0, 0, 50))"
                )

            fuzzy_set = FuzzySet(
                name=name_or_fuzzy_set,
                mf_type=mf_type,
                params=params,
                mf_func=mf_func
            )
            self.terms[fuzzy_set.name] = fuzzy_set

        else:
            raise TypeError(
                f"Primeiro parâmetro deve ser str ou FuzzySet, recebido: {type(name_or_fuzzy_set)}"
            )

    def fuzzify(self, value: float) -> Dict[str, float]:
        """
        Fuzzifica um valor crisp em todos os termos da variável.

        Parâmetros:
            value: Valor crisp a ser fuzzificado

        Retorna:
            Dicionário com graus de pertinência {termo: grau}
        """
        if not (self.universe[0] <= value <= self.universe[1]):
            import warnings
            warnings.warn(
                f"Valor {value} fora do universo de discurso {self.universe} "
                f"da variável '{self.name}'",
                UserWarning
            )

        return {
            term_name: fuzzy_set.membership(value)
            for term_name, fuzzy_set in self.terms.items()
        }

    def get_universe_array(self, num_points: int = 1000) -> np.ndarray:
        """
        Retorna um array com valores do universo de discurso.

        Parâmetros:
            num_points: Número de pontos a gerar

        Retorna:
            Array numpy com valores uniformemente espaçados
        """
        return np.linspace(self.universe[0], self.universe[1], num_points)

    def plot(self, ax=None, show=True, num_points=1000, **kwargs):
        """
        Plota todas as funções de pertinência da variável linguística.

        Parâmetros:
            ax: Matplotlib axes (opcional). Se None, cria nova figura
            show: Se True, mostra o plot (plt.show())
            num_points: Número de pontos para plotar
            **kwargs: Argumentos adicionais para personalização
                - figsize: Tamanho da figura (default: (10, 6))
                - colors: Lista de cores para os termos
                - linewidth: Espessura das linhas (default: 2)
                - alpha: Transparência (default: 0.7)
                - grid: Se True, mostra grid (default: True)
                - title: Título customizado

        Retorna:
            fig, ax: Figura e axes matplotlib (se show=False)

        Exemplo:
            >>> var = LinguisticVariable('temperatura', (0, 40))
            >>> var.add_term('fria', 'triangular', (0, 0, 20))
            >>> var.add_term('quente', 'triangular', (20, 40, 40))
            >>> var.plot()
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "Matplotlib não está instalado. "
                "Instale com: pip install matplotlib"
            )

        # Cria figura se não foi fornecida
        if ax is None:
            figsize = kwargs.get('figsize', (10, 6))
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        # Gera universo de discurso
        x = self.get_universe_array(num_points)

        # Configurações de estilo
        colors = kwargs.get('colors', None)
        linewidth = kwargs.get('linewidth', 2)
        alpha = kwargs.get('alpha', 0.7)

        # Plota cada termo
        for i, (term_name, fuzzy_set) in enumerate(self.terms.items()):
            # Calcula graus de pertinência
            y = fuzzy_set.membership(x)

            # Define cor
            if colors is not None and i < len(colors):
                color = colors[i]
            else:
                color = None  # Matplotlib escolhe automaticamente

            # Plota
            ax.plot(x, y, label=term_name, linewidth=linewidth,
                   alpha=alpha, color=color)
            ax.fill_between(x, 0, y, alpha=0.1, color=color)

        # Configurações do gráfico
        ax.set_xlabel(self.name, fontsize=12)
        ax.set_ylabel('Grau de Pertinência', fontsize=12)

        title = kwargs.get('title', f'Funções de Pertinência - {self.name}')
        ax.set_title(title, fontsize=14, fontweight='bold')

        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(self.universe[0], self.universe[1])

        if kwargs.get('grid', True):
            ax.grid(True, alpha=0.3, linestyle='--')

        ax.legend(loc='best', fontsize=10)

        if show:
            plt.tight_layout()
            plt.show()

        return fig, ax

    def __repr__(self) -> str:
        terms_str = ', '.join(self.terms.keys())
        return f"LinguisticVariable(name='{self.name}', universe={self.universe}, terms=[{terms_str}])"


class Fuzzifier:
    """
    Classe para gerenciar o processo de fuzzificação de múltiplas variáveis.
    """

    def __init__(self):
        """Inicializa o fuzzificador."""
        self.variables: Dict[str, LinguisticVariable] = {}

    def add_variable(self, variable: LinguisticVariable) -> None:
        """
        Adiciona uma variável linguística ao fuzzificador.

        Parâmetros:
            variable: Variável linguística a ser adicionada
        """
        self.variables[variable.name] = variable

    def create_variable(self,
                       name: str,
                       universe: Tuple[float, float],
                       terms: Optional[Dict[str, Tuple[str, Tuple]]] = None) -> LinguisticVariable:
        """
        Cria e adiciona uma variável linguística.

        Parâmetros:
            name: Nome da variável
            universe: Universo de discurso [min, max]
            terms: Dicionário opcional {nome_termo: (tipo_mf, params)}

        Retorna:
            Variável linguística criada

        Exemplo:
            >>> fuzz = Fuzzifier()
            >>> temp = fuzz.create_variable(
            ...     'temperatura',
            ...     (0, 100),
            ...     {
            ...         'baixa': ('triangular', (0, 0, 50)),
            ...         'média': ('triangular', (0, 50, 100)),
            ...         'alta': ('triangular', (50, 100, 100))
            ...     }
            ... )
        """
        variable = LinguisticVariable(name, universe)

        if terms:
            for term_name, (mf_type, params) in terms.items():
                fuzzy_set = FuzzySet(term_name, mf_type, params)
                variable.add_term(fuzzy_set)

        self.add_variable(variable)
        return variable

    def fuzzify(self, inputs: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """
        Fuzzifica múltiplas variáveis de entrada.

        Parâmetros:
            inputs: Dicionário {nome_variável: valor_crisp}

        Retorna:
            Dicionário aninhado {variável: {termo: grau}}

        Exemplo:
            >>> result = fuzz.fuzzify({'temperatura': 30})
            >>> # result: {'temperatura': {'baixa': 0.4, 'média': 0.6, 'alta': 0.0}}
        """
        result = {}

        for var_name, value in inputs.items():
            if var_name not in self.variables:
                raise ValueError(f"Variável '{var_name}' não encontrada no fuzzificador")

            result[var_name] = self.variables[var_name].fuzzify(value)

        return result

    def get_variable(self, name: str) -> LinguisticVariable:
        """
        Retorna uma variável linguística pelo nome.

        Parâmetros:
            name: Nome da variável

        Retorna:
            Variável linguística

        Raises:
            KeyError: Se a variável não existir
        """
        if name not in self.variables:
            raise KeyError(f"Variável '{name}' não encontrada")
        return self.variables[name]

    def __repr__(self) -> str:
        vars_str = ', '.join(self.variables.keys())
        return f"Fuzzifier(variables=[{vars_str}])"


def fuzzify_value(value: float,
                  fuzzy_sets: List[FuzzySet]) -> Dict[str, float]:
    """
    Função auxiliar para fuzzificar um valor em uma lista de conjuntos fuzzy.

    Parâmetros:
        value: Valor crisp a ser fuzzificado
        fuzzy_sets: Lista de conjuntos fuzzy

    Retorna:
        Dicionário {nome_conjunto: grau_pertinência}

    Exemplo:
        >>> sets = [
        ...     FuzzySet('baixo', 'triangular', (0, 0, 50)),
        ...     FuzzySet('alto', 'triangular', (50, 100, 100))
        ... ]
        >>> fuzzify_value(25, sets)
        {'baixo': 0.5, 'alto': 0.0}
    """
    return {fs.name: fs.membership(value) for fs in fuzzy_sets}


def create_fuzzy_partitions(universe: Tuple[float, float],
                           num_partitions: int,
                           overlap: float = 0.5,
                           mf_type: str = 'triangular') -> List[FuzzySet]:
    """
    Cria partições fuzzy uniformes sobre um universo de discurso.

    Parâmetros:
        universe: Universo de discurso [min, max]
        num_partitions: Número de partições a criar
        overlap: Fator de sobreposição (0 a 1)
        mf_type: Tipo de função de pertinência

    Retorna:
        Lista de conjuntos fuzzy representando as partições

    Exemplo:
        >>> partitions = create_fuzzy_partitions((0, 100), 3)
        >>> # Cria 3 partições: baixo, médio, alto
    """
    min_val, max_val = universe
    range_val = max_val - min_val
    step = range_val / (num_partitions - 1)

    fuzzy_sets = []
    names = _generate_partition_names(num_partitions)

    for i, name in enumerate(names):
        center = min_val + i * step

        if mf_type == 'triangular':
            # Ajusta a largura baseado no overlap
            width = step * (1 + overlap)
            a = max(min_val, center - width)
            b = center
            c = min(max_val, center + width)
            params = (a, b, c)
        elif mf_type == 'gaussian':
            sigma = step * overlap
            params = (center, sigma)
        else:
            raise ValueError(f"Tipo de função '{mf_type}' não suportado para partições automáticas")

        fuzzy_sets.append(FuzzySet(name, mf_type, params))

    return fuzzy_sets


def _generate_partition_names(num_partitions: int) -> List[str]:
    """
    Gera nomes padrão para partições fuzzy.

    Parâmetros:
        num_partitions: Número de partições

    Retorna:
        Lista de nomes
    """
    if num_partitions == 2:
        return ['baixo', 'alto']
    elif num_partitions == 3:
        return ['baixo', 'médio', 'alto']
    elif num_partitions == 5:
        return ['muito_baixo', 'baixo', 'médio', 'alto', 'muito_alto']
    elif num_partitions == 7:
        return ['muito_baixo', 'baixo', 'medio_baixo', 'médio',
                'medio_alto', 'alto', 'muito_alto']
    else:
        return [f'termo_{i+1}' for i in range(num_partitions)]
