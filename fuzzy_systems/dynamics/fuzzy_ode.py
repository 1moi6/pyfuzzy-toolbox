"""
Solver de Equações Diferenciais Ordinárias Fuzzy
==================================================

Implementação otimizada para resolver EDOs com condições iniciais e/ou
parâmetros fuzzy usando o método de α-níveis.

Algoritmo:
1. Gera n α-níveis (0 a 1)
2. Para cada α, extrai intervalos [min, max] dos números fuzzy
3. Constrói grid de pontos iniciais usando produto cartesiano
4. Resolve EDO para cada ponto do grid (vetorizado + paralelo)
5. Extrai envelope (min/max) em cada instante de tempo
6. Retorna solução fuzzy como conjunto de envelopes por α-nível

Integração completa com fuzzy_systems.core:
- Usa FuzzySet para representar números fuzzy
- Compatível com todas as funções de pertinência do core
- Suporta operadores fuzzy para propagação de incerteza

Autor: fuzzy_systems package
Versão: 1.0
"""

import numpy as np
from typing import Callable, List, Tuple, Union, Optional, Dict
from dataclasses import dataclass
from scipy.integrate import solve_ivp
from joblib import Parallel, delayed
import warnings

# Integração com core
from ..core import FuzzySet, triangular, trapezoidal, gaussian


@dataclass
class FuzzyNumber:
    """
    Número fuzzy baseado em FuzzySet do core.

    Representa um número fuzzy através de sua função de pertinência.
    Integrado completamente com fuzzy_systems.core.

    Atributos:
        fuzzy_set: FuzzySet do core (triangular, gaussiana, etc)
        support: Suporte do número fuzzy [min, max]
        name: Nome descritivo (opcional)

    Exemplos:
        >>> # Número triangular ~5
        >>> num1 = FuzzyNumber.triangular(center=5, spread=1)

        >>> # Número gaussiano ~10
        >>> num2 = FuzzyNumber.gaussian(mean=10, sigma=2)

        >>> # Número trapezoidal
        >>> num3 = FuzzyNumber.trapezoidal(a=1, b=2, c=3, d=4)
    """
    fuzzy_set: FuzzySet
    support: Tuple[float, float]
    name: str = "fuzzy_number"

    @classmethod
    def triangular(cls, center: float, spread: float,
                   name: str = "triangular") -> 'FuzzyNumber':
        """
        Cria número fuzzy triangular.

        Args:
            center: Centro (pico, μ=1)
            spread: Espalhamento (distância do centro aos extremos)
            name: Nome do número

        Returns:
            FuzzyNumber triangular
        """
        a = center - spread
        b = center
        c = center + spread

        fuzzy_set = FuzzySet(
            name=name,
            mf_type='triangular',
            params=(a, b, c)
        )

        return cls(
            fuzzy_set=fuzzy_set,
            support=(a, c),
            name=name
        )

    @classmethod
    def trapezoidal(cls, a: float, b: float, c: float, d: float,
                    name: str = "trapezoidal") -> 'FuzzyNumber':
        """
        Cria número fuzzy trapezoidal.

        Args:
            a: Limite inferior
            b: Início do plateau
            c: Fim do plateau
            d: Limite superior
            name: Nome do número

        Returns:
            FuzzyNumber trapezoidal
        """
        fuzzy_set = FuzzySet(
            name=name,
            mf_type='trapezoidal',
            params=(a, b, c, d)
        )

        return cls(
            fuzzy_set=fuzzy_set,
            support=(a, d),
            name=name
        )

    @classmethod
    def gaussian(cls, mean: float, sigma: float,
                 n_sigmas: float = 3.0,
                 name: str = "gaussian") -> 'FuzzyNumber':
        """
        Cria número fuzzy gaussiano.

        Args:
            mean: Média (centro)
            sigma: Desvio padrão
            n_sigmas: Quantos sigmas para definir suporte (padrão: 3)
            name: Nome do número

        Returns:
            FuzzyNumber gaussiano
        """
        fuzzy_set = FuzzySet(
            name=name,
            mf_type='gaussian',
            params=(mean, sigma)
        )

        support = (mean - n_sigmas * sigma, mean + n_sigmas * sigma)

        return cls(
            fuzzy_set=fuzzy_set,
            support=support,
            name=name
        )

    @classmethod
    def from_fuzzy_set(cls, fuzzy_set: FuzzySet, support: Tuple[float, float]) -> 'FuzzyNumber':
        """
        Cria FuzzyNumber a partir de um FuzzySet do core.

        Args:
            fuzzy_set: FuzzySet do core
            support: Suporte [min, max]

        Returns:
            FuzzyNumber
        """
        return cls(
            fuzzy_set=fuzzy_set,
            support=support,
            name=fuzzy_set.name
        )

    def alpha_cut(self, alpha: float, n_points: int = 100) -> Tuple[float, float]:
        """
        Extrai α-corte do número fuzzy.

        Args:
            alpha: Nível α (0 a 1)
            n_points: Pontos para busca numérica

        Returns:
            (min, max) do α-corte
        """
        if not (0 <= alpha <= 1):
            raise ValueError(f"Alpha deve estar em [0, 1], recebido: {alpha}")

        # Caso especial: α = 0 retorna suporte completo
        if alpha == 0:
            return self.support

        # Busca numérica dos pontos onde μ(x) >= alpha
        x = np.linspace(self.support[0], self.support[1], n_points)
        mu = self.fuzzy_set.membership(x)

        # Pontos que satisfazem μ(x) >= alpha
        valid_indices = np.where(mu >= alpha - 1e-10)[0]

        if len(valid_indices) == 0:
            # Alpha muito alto, retorna centro
            center = (self.support[0] + self.support[1]) / 2
            return (center, center)

        x_min = x[valid_indices[0]]
        x_max = x[valid_indices[-1]]

        return (x_min, x_max)

    def membership(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Calcula grau de pertinência."""
        return self.fuzzy_set.membership(x)

    def __repr__(self) -> str:
        return (f"FuzzyNumber(name='{self.name}', "
                f"type='{self.fuzzy_set.mf_type}', "
                f"support={self.support})")


@dataclass
class FuzzySolution:
    """
    Solução de uma EDO fuzzy.

    Contém envelopes (min/max) para cada α-nível em cada instante de tempo.

    Atributos:
        t: Array de tempos
        y_min: Array [n_alpha, n_vars, n_time] com envelope inferior
        y_max: Array [n_alpha, n_vars, n_time] com envelope superior
        alphas: Níveis α utilizados
        var_names: Nomes das variáveis
    """
    t: np.ndarray
    y_min: np.ndarray  # shape: (n_alpha, n_vars, n_time)
    y_max: np.ndarray  # shape: (n_alpha, n_vars, n_time)
    alphas: np.ndarray
    var_names: List[str] = None

    def __post_init__(self):
        if self.var_names is None:
            n_vars = self.y_min.shape[1]
            self.var_names = [f"y{i}" for i in range(n_vars)]

    def get_alpha_level(self, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retorna envelopes para um α-nível específico.

        Args:
            alpha: Nível α desejado

        Returns:
            (y_min, y_max) para o α mais próximo
        """
        idx = np.argmin(np.abs(self.alphas - alpha))
        return self.y_min[idx], self.y_max[idx]

    def plot(self, var_idx: int = 0, ax=None, alpha_levels=None,
             show=True, **kwargs):
        """
        Plota solução fuzzy com α-níveis.

        Args:
            var_idx: Índice da variável a plotar
            ax: Eixo matplotlib (None = criar novo)
            alpha_levels: Lista de αs a plotar (None = todos)
            show: Se True, chama plt.show()
            **kwargs: Argumentos para plt.plot
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        if alpha_levels is None:
            alpha_levels = self.alphas

        # Colormap para α-níveis
        cmap = plt.cm.Blues

        for i, alpha in enumerate(self.alphas):
            if alpha not in alpha_levels:
                continue

            y_min_alpha, y_max_alpha = self.get_alpha_level(alpha)

            # Intensidade da cor proporcional a α
            color = cmap(0.3 + 0.7 * alpha)

            # Plota envelope
            ax.fill_between(
                self.t,
                y_min_alpha[var_idx],
                y_max_alpha[var_idx],
                alpha=0.3,
                color=color,
                label=f'α={alpha:.2f}' if i % max(1, len(self.alphas) // 5) == 0 else None
            )

        ax.set_xlabel('Tempo', fontsize=12)
        ax.set_ylabel(self.var_names[var_idx], fontsize=12)
        ax.set_title(f'Solução Fuzzy: {self.var_names[var_idx]}',
                     fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        if show:
            plt.tight_layout()
            plt.show()

    def to_dataframe(self, alpha: Optional[float] = None):
        """
        Converte a solução fuzzy para pandas DataFrame.

        Args:
            alpha: Nível α específico (None = usa α=1.0, núcleo fuzzy)

        Returns:
            pandas.DataFrame com colunas:
                - time: Tempo
                - {var}_min: Envelope inferior para cada variável
                - {var}_max: Envelope superior para cada variável

        Raises:
            ImportError: Se pandas não está instalado

        Exemplo:
            >>> sol = solver.solve()
            >>> df = sol.to_dataframe(alpha=0.5)
            >>> df.head()
            >>>
            >>> # Exportar para CSV
            >>> df.to_csv('solucao_fuzzy.csv', index=False)
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas é necessário para to_dataframe(). "
                "Instale com: pip install pandas"
            )

        # Se alpha não fornecido, usa α=1.0 (núcleo)
        if alpha is None:
            alpha = 1.0

        # Busca α-nível mais próximo
        idx = np.argmin(np.abs(self.alphas - alpha))
        alpha_real = self.alphas[idx]

        y_min, y_max = self.get_alpha_level(alpha_real)

        # Constrói dicionário de dados
        data = {'time': self.t}

        for i, var_name in enumerate(self.var_names):
            data[f'{var_name}_min'] = y_min[i]
            data[f'{var_name}_max'] = y_max[i]

        df = pd.DataFrame(data)

        # Adiciona metadados como atributos
        df.attrs['alpha_level'] = float(alpha_real)
        df.attrs['n_alpha_levels'] = len(self.alphas)
        df.attrs['var_names'] = self.var_names

        return df

    def to_csv(self, filename: str, alpha: Optional[float] = None,
               sep: str = ',', decimal: str = '.', **kwargs):
        """
        Exporta solução fuzzy para arquivo CSV.

        Args:
            filename: Caminho do arquivo CSV
            alpha: Nível α (None = α=1.0)
            sep: Separador de colunas (padrão: ',')
            decimal: Separador decimal (padrão: '.' para internacional,
                    use ',' para formato brasileiro/europeu)
            **kwargs: Argumentos adicionais para pd.DataFrame.to_csv()

        Exemplo:
            >>> sol.to_csv('solucao.csv')
            >>>
            >>> # Formato brasileiro (Excel)
            >>> sol.to_csv('solucao.csv', sep=';', decimal=',')
            >>>
            >>> # α-nível específico
            >>> sol.to_csv('solucao_alpha05.csv', alpha=0.5)
        """
        df = self.to_dataframe(alpha=alpha)

        # Configurações padrão do CSV
        csv_kwargs = {
            'index': False,
            'sep': sep,
            'decimal': decimal,
        }
        csv_kwargs.update(kwargs)

        df.to_csv(filename, **csv_kwargs)

    def __repr__(self) -> str:
        return (f"FuzzySolution(n_vars={len(self.var_names)}, "
                f"n_alpha={len(self.alphas)}, "
                f"t_span=({self.t[0]:.2f}, {self.t[-1]:.2f}), "
                f"n_time={len(self.t)})")


class FuzzyODESolver:
    """
    Solver otimizado para EDOs com condições iniciais e parâmetros fuzzy.

    Usa método de α-níveis com vetorização e paralelização para máxima performance.
    Totalmente integrado com fuzzy_systems.core.

    Exemplo:
        >>> # Define EDO: dy/dt = k*y
        >>> def growth(t, y, k):
        ...     return k * y

        >>> # Condição inicial fuzzy
        >>> y0 = FuzzyNumber.triangular(center=10, spread=2)

        >>> # Parâmetro fuzzy
        >>> k = FuzzyNumber.triangular(center=0.5, spread=0.1)

        >>> # Resolver
        >>> solver = FuzzyODESolver(
        ...     ode_func=growth,
        ...     t_span=(0, 10),
        ...     y0_fuzzy=[y0],
        ...     params={'k': k},
        ...     n_alpha_cuts=11
        ... )
        >>> sol = solver.solve()
        >>> sol.plot()
    """

    def __init__(
        self,
        ode_func: Callable,
        t_span: Tuple[float, float],
        y0_fuzzy: List[Union[FuzzyNumber, float]],
        params: Optional[Dict[str, Union[FuzzyNumber, float]]] = None,
        n_alpha_cuts: int = 11,
        n_grid_points: int = 3,
        method: str = 'RK45',
        t_eval: Optional[np.ndarray] = None,
        n_jobs: int = -1,
        rtol: float = 1e-6,
        atol: float = 1e-9,
        var_names: Optional[List[str]] = None
    ):
        """
        Inicializa solver de EDO fuzzy.

        Args:
            ode_func: Função da EDO: dy/dt = f(t, y, **params)
            t_span: Intervalo de tempo (t0, tf)
            y0_fuzzy: Lista de condições iniciais (FuzzyNumber ou float)
            params: Dicionário de parâmetros {nome: FuzzyNumber ou float}
            n_alpha_cuts: Número de α-níveis (padrão: 11)
            n_grid_points: Pontos por dimensão no grid (padrão: 3 = extremos + centro)
            method: Método do solve_ivp ('RK45', 'DOP853', 'Radau', etc)
            t_eval: Tempos específicos para avaliar (None = automático)
            n_jobs: Cores para paralelização (-1 = todos)
            rtol: Tolerância relativa do ODE solver
            atol: Tolerância absoluta do ODE solver
            var_names: Nomes das variáveis (None = y0, y1, ...)
        """
        self.ode_func = ode_func
        self.t_span = t_span
        self.y0_fuzzy = y0_fuzzy
        self.params = params or {}
        self.n_alpha_cuts = n_alpha_cuts
        self.n_grid_points = n_grid_points
        self.method = method
        self.t_eval = t_eval
        self.n_jobs = n_jobs
        self.rtol = rtol
        self.atol = atol
        self.var_names = var_names

        # Dimensões
        self.n_vars = len(y0_fuzzy)
        self.n_params = len(self.params)

        # Valida
        self._validate_inputs()

    def _validate_inputs(self):
        """Valida entradas."""
        if self.n_vars == 0:
            raise ValueError("y0_fuzzy não pode ser vazio")

        if self.n_alpha_cuts < 2:
            raise ValueError("n_alpha_cuts deve ser >= 2")

        if self.n_grid_points < 2:
            raise ValueError("n_grid_points deve ser >= 2")

    def _generate_alpha_levels(self) -> np.ndarray:
        """Gera níveis α uniformemente espaçados."""
        return np.linspace(0, 1, self.n_alpha_cuts)

    def _extract_alpha_cuts(
        self,
        alpha: float
    ) -> Tuple[List[Tuple[float, float]], Dict[str, Tuple[float, float]]]:
        """
        Extrai α-cortes de todas as variáveis e parâmetros fuzzy.

        Args:
            alpha: Nível α

        Returns:
            (y0_intervals, params_intervals)
        """
        # α-cortes das condições iniciais
        y0_intervals = []
        for y0 in self.y0_fuzzy:
            if isinstance(y0, FuzzyNumber):
                interval = y0.alpha_cut(alpha)
            else:
                # Valor crisp
                interval = (float(y0), float(y0))
            y0_intervals.append(interval)

        # α-cortes dos parâmetros
        params_intervals = {}
        for param_name, param_value in self.params.items():
            if isinstance(param_value, FuzzyNumber):
                interval = param_value.alpha_cut(alpha)
            else:
                interval = (float(param_value), float(param_value))
            params_intervals[param_name] = interval

        return y0_intervals, params_intervals

    def _create_grid(
        self,
        y0_intervals: List[Tuple[float, float]],
        params_intervals: Dict[str, Tuple[float, float]]
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Cria grid de pontos iniciais e parâmetros (vetorizado).

        Args:
            y0_intervals: Intervalos [min, max] para cada y0
            params_intervals: Intervalos para cada parâmetro

        Returns:
            (y0_grid, params_grid)
            y0_grid: array (n_points, n_vars)
            params_grid: lista de dicts com parâmetros
        """
        # Cria pontos para cada dimensão
        y0_points = []
        for y_min, y_max in y0_intervals:
            if y_min == y_max:
                # Valor crisp
                points = np.array([y_min])
            else:
                # Pontos uniformemente espaçados no intervalo
                points = np.linspace(y_min, y_max, self.n_grid_points)
            y0_points.append(points)

        params_points = {}
        for param_name, (p_min, p_max) in params_intervals.items():
            if p_min == p_max:
                points = np.array([p_min])
            else:
                points = np.linspace(p_min, p_max, self.n_grid_points)
            params_points[param_name] = points

        # Produto cartesiano (grid completo)
        # Para y0
        y0_meshgrid = np.meshgrid(*y0_points, indexing='ij')
        y0_grid = np.stack([grid.flatten() for grid in y0_meshgrid], axis=1)

        # Para params
        if params_points:
            param_names = list(params_points.keys())
            param_values = [params_points[name] for name in param_names]
            param_meshgrid = np.meshgrid(*param_values, indexing='ij')

            # Repete para cada combinação de y0
            n_y0_combinations = y0_grid.shape[0]
            n_param_combinations = param_meshgrid[0].size

            # Expande y0_grid para incluir todas as combinações de parâmetros
            y0_grid_expanded = np.repeat(y0_grid, n_param_combinations, axis=0)

            # Cria lista de dicts de parâmetros
            params_grid = []
            for _ in range(n_y0_combinations):
                for idx in range(n_param_combinations):
                    param_dict = {
                        name: param_meshgrid[i].flatten()[idx]
                        for i, name in enumerate(param_names)
                    }
                    params_grid.append(param_dict)

            y0_grid = y0_grid_expanded
        else:
            # Sem parâmetros fuzzy
            params_grid = [{} for _ in range(y0_grid.shape[0])]

        return y0_grid, params_grid

    def _solve_single_ode(
        self,
        y0: np.ndarray,
        params: Dict
    ) -> np.ndarray:
        """
        Resolve uma única EDO com parâmetros específicos.

        Args:
            y0: Condição inicial
            params: Parâmetros

        Returns:
            Solução avaliada em t_eval
        """
        # Wrapper para incluir parâmetros
        def ode_wrapper(t, y):
            return self.ode_func(t, y, **params)

        # Resolve
        sol = solve_ivp(
            ode_wrapper,
            self.t_span,
            y0,
            method=self.method,
            t_eval=self.t_eval,
            rtol=self.rtol,
            atol=self.atol,
            dense_output=False
        )

        if not sol.success:
            warnings.warn(
                f"ODE solver falhou para y0={y0}, params={params}: {sol.message}",
                RuntimeWarning
            )
            # Retorna NaNs
            if self.t_eval is not None:
                return np.full((self.n_vars, len(self.t_eval)), np.nan)
            else:
                return np.full((self.n_vars, len(sol.t)), np.nan)

        return sol.y

    def _solve_alpha_level(
        self,
        alpha: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Resolve EDO para um α-nível específico.

        Args:
            alpha: Nível α

        Returns:
            (t, y_min, y_max)
        """
        # 1. Extrai α-cortes
        y0_intervals, params_intervals = self._extract_alpha_cuts(alpha)

        # 2. Cria grid
        y0_grid, params_grid = self._create_grid(y0_intervals, params_intervals)

        # 3. Resolve EDOs em paralelo
        solutions = Parallel(n_jobs=self.n_jobs, backend='loky')(
            delayed(self._solve_single_ode)(y0, params)
            for y0, params in zip(y0_grid, params_grid)
        )

        # 4. Empilha soluções (ignora NaNs)
        valid_solutions = [sol for sol in solutions if not np.any(np.isnan(sol))]

        if len(valid_solutions) == 0:
            raise RuntimeError(f"Nenhuma solução válida para α={alpha}")

        solutions_array = np.stack(valid_solutions, axis=0)  # (n_solutions, n_vars, n_time)

        # 5. Extrai envelopes (min/max)
        y_min = np.min(solutions_array, axis=0)  # (n_vars, n_time)
        y_max = np.max(solutions_array, axis=0)  # (n_vars, n_time)

        # 6. Extrai tempos (da primeira solução válida)
        if self.t_eval is not None:
            t = self.t_eval
        else:
            # Usa tempos da primeira solução
            sol_first = solve_ivp(
                lambda t, y: self.ode_func(t, y, **params_grid[0]),
                self.t_span,
                y0_grid[0],
                method=self.method,
                rtol=self.rtol,
                atol=self.atol
            )
            t = sol_first.t

        return t, y_min, y_max

    def solve(self, verbose: bool = True) -> FuzzySolution:
        """
        Resolve EDO fuzzy.

        Args:
            verbose: Se True, imprime progresso

        Returns:
            FuzzySolution com envelopes para cada α-nível
        """
        if verbose:
            print("=" * 70)
            print("SOLVER DE EDO FUZZY")
            print("=" * 70)
            print(f"Variáveis: {self.n_vars}")
            print(f"Parâmetros fuzzy: {self.n_params}")
            print(f"α-níveis: {self.n_alpha_cuts}")
            print(f"Pontos por dimensão: {self.n_grid_points}")
            print(f"Método: {self.method}")
            print(f"Paralelização: {self.n_jobs} jobs")
            print("=" * 70 + "\n")

        # Gera α-níveis
        alphas = self._generate_alpha_levels()

        # Resolve para cada α
        if verbose:
            print("Resolvendo para cada α-nível...")

        results = []
        for i, alpha in enumerate(alphas):
            if verbose:
                print(f"  α = {alpha:.3f} ({i+1}/{self.n_alpha_cuts})")

            t, y_min, y_max = self._solve_alpha_level(alpha)
            results.append((t, y_min, y_max))

        # Organiza resultados
        t_final = results[0][0]
        y_min_all = np.stack([res[1] for res in results], axis=0)  # (n_alpha, n_vars, n_time)
        y_max_all = np.stack([res[2] for res in results], axis=0)  # (n_alpha, n_vars, n_time)

        if verbose:
            print("\n✅ Solução fuzzy computada com sucesso!")
            print("=" * 70)

        return FuzzySolution(
            t=t_final,
            y_min=y_min_all,
            y_max=y_max_all,
            alphas=alphas,
            var_names=self.var_names
        )


# Mensagem de sucesso
print("✅ Módulo de EDO Fuzzy implementado com sucesso!")
print("\nCaracterísticas:")
print("  • Integração completa com fuzzy_systems.core")
print("  • Suporte a FuzzySet, triangular, gaussiana, trapezoidal")
print("  • Método de α-níveis vetorizado")
print("  • Paralelização automática (joblib)")
print("  • Condições iniciais e parâmetros fuzzy")
print("  • Visualização de envelopes por α-nível")
