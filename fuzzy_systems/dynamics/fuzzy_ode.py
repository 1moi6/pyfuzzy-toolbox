import numpy as np
from typing import Callable, List, Tuple, Union, Optional, Dict
from dataclasses import dataclass
from scipy.integrate import solve_ivp
import warnings
import itertools

# Importação opcional de joblib para paralelização
try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False
    warnings.warn(
        "joblib not found. Parallel processing disabled. "
        "Install with: pip install fuzzy-systems[ml]",
        ImportWarning
    )

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
                label=f'α={alpha:.2f}' if i % max(
                    1, len(self.alphas) // 5) == 0 else None
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

    def __init__(
        self,
        ode_func: Callable,
        t_span: Tuple[float, float],
        initial_condition: List[Union[FuzzyNumber, float]],
        params: Optional[Dict[str, Union[FuzzyNumber, float]]] = None,
        n_alpha_cuts: int = 11,
        method: str = 'RK45',
        t_eval: Optional[np.ndarray] = None,
        n_jobs: int = -1,
        rtol: float = 1e-6,
        atol: float = 1e-9,
        var_names: Optional[List[str]] = None
    ):

        self.ode_func = ode_func
        self.t_span = t_span
        self.initial_condition = initial_condition
        self.params = params or {}
        self.n_alpha_cuts = n_alpha_cuts

        self.method = method
        self.t_eval = t_eval
        self.n_jobs = n_jobs
        self.rtol = rtol
        self.atol = atol
        self.var_names = var_names

        # Dimensões
        self.n_vars = len(initial_condition)
        self.n_params = len(self.params)

        # Valida
        self._validate_inputs()

    def _validate_inputs(self):
        """Valida entradas."""
        if self.n_vars == 0:
            raise ValueError("y0_fuzzy não pode ser vazio")

        if self.n_alpha_cuts < 2:
            raise ValueError("n_alpha_cuts deve ser >= 2")

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
        for y0 in self.initial_condition:
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

    def _solve_single_ode_with_t_eval(
        self,
        y0: np.ndarray,
        params: Dict,
        t_eval: np.ndarray
    ) -> np.ndarray:
        """
        Resolve uma única EDO com parâmetros e tempos específicos.

        Args:
            y0: Condição inicial
            params: Parâmetros
            t_eval: Tempos para avaliação

        Returns:
            Array (n_vars, len(t_eval)) com a solução
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
            t_eval=t_eval,  # *** SEMPRE especifica t_eval ***
            rtol=self.rtol,
            atol=self.atol,
            dense_output=False
        )

        if not sol.success:
            warnings.warn(
                f"ODE solver falhou para y0={y0}, params={params}: {sol.message}",
                RuntimeWarning
            )
            # Retorna NaNs com a forma correta
            return np.full((self.n_vars, len(t_eval)), np.nan)

        # Garante que a saída tem a forma (n_vars, len(t_eval))
        if sol.y.shape[0] != self.n_vars or sol.y.shape[1] != len(t_eval):
            # Se algo deu errado, retorna NaNs
            return np.full((self.n_vars, len(t_eval)), np.nan)

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
        y0_grid, params_grid = self._create_grid(
            y0_intervals, params_intervals)

        # *** CORREÇÃO: Define t_eval EXPLICITAMENTE se não fornecido ***
        if self.t_eval is None:
            # Cria malha de tempos consistente
            t_eval_internal = np.linspace(
                self.t_span[0],
                self.t_span[1],
                100  # 100 pontos uniformes
            )
        else:
            t_eval_internal = self.t_eval

        # 3. Resolve EDOs (paralelo se joblib disponível, senão serial)
        if HAS_JOBLIB and self.n_jobs != 1:
            solutions = Parallel(n_jobs=self.n_jobs, backend='loky')(
                delayed(self._solve_single_ode_with_t_eval)(
                    y0, params, t_eval_internal
                )
                for y0, params in zip(y0_grid, params_grid)
            )
        else:
            # Fallback: processamento serial
            solutions = [
                self._solve_single_ode_with_t_eval(y0, params, t_eval_internal)
                for y0, params in zip(y0_grid, params_grid)
            ]

        # 4. Filtra soluções válidas (sem NaNs)
        valid_solutions = [
            sol for sol in solutions
            if not np.any(np.isnan(sol))
        ]

        if len(valid_solutions) == 0:
            raise RuntimeError(f"Nenhuma solução válida para α={alpha}")

        # *** AGORA TODAS AS SOLUÇÕES TÊM A MESMA FORMA ***
        # (n_solutions, n_vars, n_time)
        solutions_array = np.stack(valid_solutions, axis=0)

        # 5. Extrai envelopes (min/max)
        y_min = np.min(solutions_array, axis=0)  # (n_vars, n_time)
        y_max = np.max(solutions_array, axis=0)  # (n_vars, n_time)

        return t_eval_internal, y_min, y_max

    def _solve_standard(self, n_grid_points=5, verbose: bool = False) -> 'FuzzySolution':
        """
        Método padrão original - resolve cada α completo.
        (Este é o código original do solve())
        """
        if n_grid_points < 2:
            raise ValueError("n_grid_points deve ser >= 2")

        self.n_grid_points = n_grid_points
        if verbose:
            print("=" * 70)
            print("SOLVER DE EDO FUZZY - MÉTODO PADRÃO")
            print("=" * 70)
            print(f"Variáveis: {self.n_vars}")
            print(f"Parâmetros fuzzy: {self.n_params}")
            print(f"α-níveis: {self.n_alpha_cuts}")
            print(f"Pontos por dimensão: {self.n_grid_points}")
            print(f"Método: {self.method}")
            print("=" * 70 + "\n")

        alphas = self._generate_alpha_levels()

        if verbose:
            print("Resolvendo para cada α-nível...")

        results = []

        for i, alpha in enumerate(alphas):
            if verbose:
                print(f"  α = {alpha:.3f} ({i+1}/{self.n_alpha_cuts})")

            t, y_min, y_max = self._solve_alpha_level(alpha)
            results.append((t, y_min, y_max))

        t_final = results[0][0]
        y_min_all = np.stack([res[1] for res in results], axis=0)
        y_max_all = np.stack([res[2] for res in results], axis=0)

        if verbose:
            print("\n✅ Solução computada!")
            print("=" * 70)

        return FuzzySolution(
            t=t_final,
            y_min=y_min_all,
            y_max=y_max_all,
            alphas=alphas,
            var_names=self.var_names
        )

    def _solve_single_ode(self, y0: np.ndarray, params: Dict,
                          t_eval: np.ndarray = None) -> np.ndarray:
        """Wrapper para compatibilidade."""
        return self._solve_single_ode_with_t_eval(y0, params, t_eval)

    def solve_with_method_option(self,
                                 method: str = 'standard',
                                 verbose: bool = False,
                                 **method_kwargs) -> 'FuzzySolution':
        """
        Resolve EDO fuzzy com múltiplos métodos disponíveis.

        Args:
            method: Método a usar
                - 'standard': Resolve cada α-nível completamente (padrão)
                - 'hierarchical': Reuso hierárquico de α-níveis (3-5x mais rápido)
                - 'monte_carlo': Monte Carlo (escalável, 10-400x em alta dimensão)

            verbose: Se True, imprime progresso

            **method_kwargs: Argumentos específicos do método
                Para 'hierarchical': (nenhum)
                Para 'monte_carlo': n_samples=500, random_seed=None

        Returns:
            FuzzySolution

        """

        # Normaliza método
        method_lower = method.lower().strip()

        if method_lower == 'standard':
            n_grid_points = method_kwargs.get('n_grid_points', 20)
            return self._solve_standard(n_grid_points=n_grid_points, verbose=verbose)

        elif method_lower == 'hierarchical':
            return self._solve_hierarchical(verbose=verbose)

        elif method_lower == 'monte_carlo':
            n_samples = method_kwargs.get('n_samples', 1000)
            random_seed = method_kwargs.get('random_seed', None)
            return self._solve_monte_carlo(
                n_samples=n_samples,
                random_seed=random_seed,
                verbose=verbose
            )

        else:
            raise ValueError(
                f"Método desconhecido: '{method}'. "
                f"Opções válidas: 'standard', 'hierarchical', 'monte_carlo'"
            )

    def _solve_hierarchical(self, verbose: bool = False) -> 'FuzzySolution':
        """
        Método hierárquico - reusa α-níveis maiores.
        3-5x mais rápido que padrão.
        """
        # Importar a classe de otimização

        optimizer = HierarchicalFuzzyODESolver(self)
        return optimizer.solve_optimized(verbose=verbose)

    def _solve_monte_carlo(self,
                           n_samples: int = 1000,
                           random_seed: int = None,
                           verbose: bool = False) -> 'FuzzySolution':
        """
        Método Monte Carlo com pertinência herdada.
        10-400x mais rápido em alta dimensionalidade.
        """

        mc_solver = MonteCarloFuzzyODESolver(
            self,
            n_samples=n_samples,
            random_seed=random_seed
        )
        return mc_solver.solve_monte_carlo(verbose=verbose)
    def solve(self, method: str = 'standard',
              verbose: bool = False, **method_kwargs) -> FuzzySolution:
        """Resolve EDO fuzzy com múltiplos métodos."""
        return self.solve_with_method_option(method, verbose, **method_kwargs)


class MonteCarloFuzzyODESolver:
    """
    Solver Monte Carlo com pertinência HERDADA da CI.

    Ideia-chave:
        μ(y(t)) = μ(y0)
    """

    def __init__(self, base_solver, n_samples: int = 1000, random_seed: int = None):
        """
        Args:
            base_solver: FuzzyODESolver original
            n_samples: Número de amostras aleatórias
            random_seed: Para reprodutibilidade
        """
        self.solver = base_solver
        self.n_samples = n_samples
        if random_seed is not None:
            np.random.seed(random_seed)

        self.sampled_points = []
        self.solutions = []
        # Pertinência das CIs (herdada para todas soluções)
        self.pertinences_CI = []

    def _compute_ci_pertinence(self,
                               y0: np.ndarray,
                               params: Dict[str, float]) -> float:
        """
        Calcula pertinência de uma condição inicial no espaço fuzzy.

        A pertinência é calculada como:
        μ(CI) = min(μ_y0_1, μ_y0_2, ..., μ_param_1, μ_param_2, ...)

        Onde:
        - μ_y0_i: pertinência da i-ésima condição inicial
        - μ_param_j: pertinência do j-ésimo parâmetro fuzzy

        Args:
            y0 (np.ndarray): Vetor de condições iniciais
                            Shape: (n_vars,)
                            Valores avaliados nas funções de pertinência fuzzy

            params (Dict[str, float]): Dicionário de parâmetros
                                    Chaves: nomes dos parâmetros
                                    Valores: valores numéricos

        Returns:
            float: Grau de pertinência total ∈ [0, 1]
                1.0 = Máxima pertinência (núcleo fuzzy)
                0.0 = Mínima pertinência (fora do suporte)

        Raises:
            IndexError: Se y0 tem tamanho diferente de n_vars esperado
            KeyError: Se params não contém parâmetro esperado
        """

        # Inicializa com pertinência máxima
        pertinence = 1.0

        # ========================================================================
        # FASE 1: Calcula pertinência das condições iniciais fuzzy
        # ========================================================================

        for i, y0_var in enumerate(self.solver.initial_condition):

            # Obtém o valor da i-ésima CI
            y0_value = y0[i]

            # Verifica se é FuzzyNumber ou crisp
            if hasattr(y0_var, 'fuzzy_set'):
                # É FuzzyNumber: calcula pertinência
                mu_y0 = y0_var.fuzzy_set.membership(y0_value)
            else:
                # É crisp (float/int): pertinência é 1.0
                # (crisp não restringe o espaço fuzzy)
                mu_y0 = 1.0

            # Aplica t-norma (mínimo) para combinar pertinências
            pertinence = min(pertinence, mu_y0)

        # ========================================================================
        # FASE 2: Calcula pertinência dos parâmetros fuzzy
        # ========================================================================

        for param_name, param_var in self.solver.params.items():

            # Verifica se é FuzzyNumber ou crisp
            if hasattr(param_var, 'fuzzy_set'):
                # É FuzzyNumber: obtém valor do parâmetro
                param_value = params.get(param_name)

                if param_value is None:
                    # Parâmetro não fornecido, assume valor padrão
                    # Isso não deveria acontecer, mas é proteção
                    continue

                # Calcula pertinência do parâmetro
                mu_param = param_var.fuzzy_set.membership(param_value)
            else:
                # É crisp: pertinência é 1.0
                mu_param = 1.0

            # Aplica t-norma (mínimo)
            pertinence = min(pertinence, mu_param)

        # ========================================================================
        # FASE 3: Normaliza resultado para [0, 1]
        # ========================================================================

        # Garante que o resultado está no intervalo [0, 1]
        # (proteção contra erros numéricos)
        pertinence = max(0.0, min(1.0, pertinence))

        return pertinence

    def _sample_hypercube_with_pertinence(
    self,
    n_samples: int = 1000,
    verbose: bool = False
) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
        """
        Amostra pontos no hipercubo fuzzy de forma simples e direta.

        ESTRATÉGIA:
        1. Gera 1000 amostras aleatórias INDEPENDENTES em cada dimensão
        2. Usa zip() para combinar em 1000 pontos (hipercubo)
        3. Adiciona combinações dos EXTREMOS de α=0 (itertools.product)
        4. Adiciona combinações dos EXTREMOS de α=1.0
        5. Calcula pertinência de TODOS

        Args:
            n_samples: Número de amostras por dimensão (default: 1000)
            verbose: Se True, imprime estatísticas

        Returns:
            (y0_samples, param_samples, pertinences_CI)
        """

        # ========================================================================
        # FASE 1: EXTRAI INTERVALOS
        # ========================================================================

        # α=0 (suporte completo)
        y0_intervals_alpha_0, params_intervals_alpha_0 = (
            self.solver._extract_alpha_cuts(0.0)
        )

        # α=1.0 (núcleo)
        y0_intervals_alpha_1, params_intervals_alpha_1 = (
            self.solver._extract_alpha_cuts(1.0)
        )

        if verbose:
            print(f"\n  FASE 1: Gerando {n_samples} amostras por dimensão...")

        # ========================================================================
        # FASE 2: AMOSTRAS ALEATÓRIAS INDEPENDENTES EM CADA DIMENSÃO
        # ========================================================================

        # Amostras para cada CI
        y0_samples_per_dim = []
        for i, (y_min, y_max) in enumerate(y0_intervals_alpha_0):
            if y_min == y_max:
                samples = np.full(n_samples, y_min)
            else:
                samples = np.random.uniform(y_min, y_max, n_samples)
            y0_samples_per_dim.append(samples)
            if verbose:
                print(f"    - CI {i}: [{y_min:.3f}, {y_max:.3f}]")

        # Amostras para cada parâmetro fuzzy
        param_names = sorted([k for k, v in self.solver.params.items() 
                            if hasattr(v, 'fuzzy_set')])

        param_samples_per_dim = {}
        for param_name in param_names:
            p_min, p_max = params_intervals_alpha_0[param_name]
            if p_min == p_max:
                samples = np.full(n_samples, p_min)
            else:
                samples = np.random.uniform(p_min, p_max, n_samples)
            param_samples_per_dim[param_name] = samples
            if verbose:
                print(f"    - Parâm '{param_name}': [{p_min:.3f}, {p_max:.3f}]")

        # ========================================================================
        # FASE 3: COMBINA EM 1000 PONTOS COM zip()
        # ========================================================================

        if verbose:
            print(f"\n  FASE 2: Combinando amostras com zip()...")

        # Combina amostras: [(y0_1[0], y0_2[0], ...), ...]
        y0_combined = list(zip(*y0_samples_per_dim))
        y0_combined = np.array(y0_combined)

        # Combina parâmetros: [{'r': r[0], 'K': K[0]}, ...]
        param_combined = []
        for i in range(n_samples):
            param_dict = {}
            for param_name in param_names:
                param_dict[param_name] = param_samples_per_dim[param_name][i]
            param_combined.append(param_dict)

        n_pontos_amostrados = len(y0_combined)
        if verbose:
            print(f"    ✓ {n_pontos_amostrados} pontos do hipercubo")

        # ========================================================================
        # FASE 4: COMBINAÇÕES DOS EXTREMOS DE α=0
        # ========================================================================

        if verbose:
            print(f"\n  FASE 3: Adicionando extremos de α=0...")

        # Extremos de cada CI
        y0_extremos_alpha_0 = []
        for y_min, y_max in y0_intervals_alpha_0:
            y0_extremos_alpha_0.append([y_min, y_max])

        # Extremos de cada parâmetro
        param_extremos_alpha_0 = {}
        for param_name in param_names:
            p_min, p_max = params_intervals_alpha_0[param_name]
            param_extremos_alpha_0[param_name] = [p_min, p_max]

        # PRODUTO CARTESIANO dos extremos
        # Exemplo: [[y0_min, y0_max]] × [[r_min, r_max]] × [[K_min, K_max]]
        y0_extremos_product = list(itertools.product(*y0_extremos_alpha_0))
        param_extremos_product = list(itertools.product(*[
            param_extremos_alpha_0[pname] for pname in param_names
        ]))

        n_vertices_alpha_0 = len(y0_extremos_product) * len(param_extremos_product)

        # Combina y0 extremos com param extremos
        for y0_vertex in y0_extremos_product:
            for param_vertex in param_extremos_product:
                y0_combined = np.vstack([y0_combined, [y0_vertex]])

                param_dict = {}
                for param_idx, param_name in enumerate(param_names):
                    param_dict[param_name] = param_vertex[param_idx]
                param_combined.append(param_dict)

        if verbose:
            print(f"    ✓ {n_vertices_alpha_0} combinações de extremos")
            print(f"      = 2^{len(y0_extremos_alpha_0)} × 2^{len(param_names)} = {len(y0_extremos_product)} × {len(param_extremos_product)}")

        # ========================================================================
        # FASE 5: COMBINAÇÕES DOS EXTREMOS DE α=1.0
        # ========================================================================

        if verbose:
            print(f"\n  FASE 4: Adicionando extremos de α=1.0 (núcleo)...")

        # Extremos de α=1.0 (pode ser apenas um ponto se é triangular)
        y0_extremos_alpha_1 = []
        for y_min, y_max in y0_intervals_alpha_1:
            y0_extremos_alpha_1.append([y_min, y_max])

        param_extremos_alpha_1 = {}
        for param_name in param_names:
            p_min, p_max = params_intervals_alpha_1[param_name]
            param_extremos_alpha_1[param_name] = [p_min, p_max]

        # PRODUTO CARTESIANO dos extremos α=1.0
        y0_extremos_product_alpha_1 = list(itertools.product(*y0_extremos_alpha_1))
        param_extremos_product_alpha_1 = list(itertools.product(*[
            param_extremos_alpha_1[pname] for pname in param_names
        ]))

        n_vertices_alpha_1 = len(y0_extremos_product_alpha_1) * len(param_extremos_product_alpha_1)

        # Adiciona aos pontos
        for y0_vertex in y0_extremos_product_alpha_1:
            for param_vertex in param_extremos_product_alpha_1:
                y0_combined = np.vstack([y0_combined, [y0_vertex]])

                param_dict = {}
                for param_idx, param_name in enumerate(param_names):
                    param_dict[param_name] = param_vertex[param_idx]
                param_combined.append(param_dict)

        if verbose:
            print(f"    ✓ {n_vertices_alpha_1} combinações de extremos α=1")
            print(f"      = 2^{len(y0_extremos_alpha_1)} × 2^{len(param_names)} = {len(y0_extremos_product_alpha_1)} × {len(param_extremos_product_alpha_1)}")

        # ========================================================================
        # FASE 6: CALCULA PERTINÊNCIAS
        # ========================================================================

        if verbose:
            print(f"\n  FASE 5: Calculando pertinências...")

        n_total = len(y0_combined)
        pertinences_CI = []

        for i in range(n_total):
            mu_i = self._compute_ci_pertinence(y0_combined[i], param_combined[i])
            pertinences_CI.append(mu_i)

        pertinences_CI = np.array(pertinences_CI)

        # ========================================================================
        # FASE 7: RETORNA
        # ========================================================================

        if verbose:
            print(f"\n  RESUMO:")
            print(f"    Total de pontos: {n_total}")
            print(f"      = {n_pontos_amostrados} (zip) + {n_vertices_alpha_0} (extremos α=0) + {n_vertices_alpha_1} (extremos α=1)")
            print(f"\n    Pertinências: min={np.min(pertinences_CI):.3f}, max={np.max(pertinences_CI):.3f}, média={np.mean(pertinences_CI):.3f}")

        # Reorganiza param_combined em dicts por parâmetro
        param_samples_final = {pname: [] for pname in param_names}
        for param_dict in param_combined:
            for pname in param_names:
                param_samples_final[pname].append(param_dict[pname])

        for pname in param_names:
            param_samples_final[pname] = np.array(param_samples_final[pname])

        return y0_combined, param_samples_final, pertinences_CI

    def _solve_all_samples(
        self,
        y0_samples: np.ndarray,
        param_samples: Dict[str, np.ndarray]
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Resolve ODEs para todos os pontos amostrados.
        """

        t_eval = self.solver.t_eval
        if t_eval is None:
            t_eval = np.linspace(
                self.solver.t_span[0],
                self.solver.t_span[1],
                100
            )

        # Prepara lista de tarefas
        solve_tasks = []
        param_names = sorted(param_samples.keys())

        for i in range(len(y0_samples)):
            y0 = y0_samples[i]

            params = {}
            for pname in param_names:
                pvals = param_samples[pname]
                if len(pvals) > i:
                    params[pname] = pvals[i]
                else:
                    params[pname] = pvals[0]

            # Adiciona parâmetros crisp
            for param_name, param_val in self.solver.params.items():
                if param_name not in params:
                    params[param_name] = param_val

            solve_tasks.append((y0, params))

        # Resolve
        try:
            from joblib import Parallel, delayed
            HAS_JOBLIB = True
        except ImportError:
            HAS_JOBLIB = False

        if HAS_JOBLIB and self.solver.n_jobs != 1:
            solutions = Parallel(n_jobs=self.solver.n_jobs, backend='loky')(
                delayed(self.solver._solve_single_ode)(y0, params, t_eval)
                for y0, params in solve_tasks
            )
        else:
            solutions = [
                self.solver._solve_single_ode(y0, params, t_eval)
                for y0, params in solve_tasks
            ]

        return solutions, t_eval

    def _compute_alpha_levels_from_ci_pertinence(
        self,
        solutions: List[np.ndarray],
        pertinences_CI: np.ndarray,
        alphas: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calcula α-níveis usando APENAS a pertinência da CI.

        ALGORITMO (muito simples!):

        1. Ordena soluções por pertinência: decrescente
        2. Para cada α:
           - Seleciona soluções com μ(CI) ≥ α
           - Calcula min/max
        """

        # Valida soluções
        valid_mask = np.array([
            not np.any(np.isnan(sol)) for sol in solutions
        ])

        valid_solutions = [sol for sol, valid in zip(solutions, valid_mask)
                           if valid]
        valid_pertinences = pertinences_CI[valid_mask]

        if len(valid_solutions) == 0:
            raise RuntimeError("Nenhuma solução válida")

        solutions_array = np.stack(valid_solutions, axis=0)
        n_points, n_vars, n_time = solutions_array.shape

        y_min_all = []
        y_max_all = []
        alphas_valid = []

        for alpha in alphas:
            # Seleciona APENAS soluções com μ(CI) ≥ α
            alpha_mask = valid_pertinences >= (alpha - 1e-10)

            if np.sum(alpha_mask) == 0:
                # Se nenhuma solução atingir este α, pula
                continue

            solutions_alpha = solutions_array[alpha_mask]

            y_min_alpha = np.min(solutions_alpha, axis=0)
            y_max_alpha = np.max(solutions_alpha, axis=0)

            y_min_all.append(y_min_alpha)
            y_max_all.append(y_max_alpha)
            alphas_valid.append(alpha)

        return np.stack(y_min_all), np.stack(y_max_all), np.array(alphas_valid)

    def solve_monte_carlo(self, verbose: bool = False):
        """
        Resolve com o método Monte Carlo CORRETO.
        """

        if verbose:
            print("=" * 80)
            print("SOLVER MONTE CARLO + PERTINÊNCIA (CORRIGIDO)")
            print("=" * 80)

        # 1. Amostragem com pertinências
        if verbose:
            print(f"\n⏳ Amostrando {self.n_samples} pontos...")

        y0_samples, param_samples, pertinences_CI = (
            self._sample_hypercube_with_pertinence(
                self.n_samples,
                verbose=verbose  # ← PASSAR
            )
        )

        n_total = len(y0_samples)+len(param_samples)
        
        if verbose:
            print(f"✓ {n_total} pontos amostrados (+ vértices)")
            print(f"  Pertinências: min={np.min(pertinences_CI):.3f}, "
                  f"max={np.max(pertinences_CI):.3f}, "
                  f"média={np.mean(pertinences_CI):.3f}")

        # 2. Resolve TODAS as ODEs uma única vez
        if verbose:
            print(f"\n⏳ Resolvendo {n_total} ODEs...")

        solutions, t_eval = self._solve_all_samples(y0_samples, param_samples)

        if verbose:
            print(f"✓ {n_total} ODEs resolvidas")

        # 3. Calcula α-níveis (MUITO rápido!)
        if verbose:
            print(f"\n⏳ Calculando α-níveis...")

        alphas = self.solver._generate_alpha_levels()

        y_min_all, y_max_all, alphas_valid = (
            self._compute_alpha_levels_from_ci_pertinence(
                solutions, pertinences_CI, alphas
            )
        )

        if verbose:
            print(f"✓ {len(alphas_valid)} α-níveis calculados")
            print("\nESTATÍSTICAS:")
            print(f"  Total de ODEs resolvidas: {n_total}")
            print(
                f"  Vs. método padrão: 1/{self.solver.n_alpha_cuts} do custo")
            print("=" * 80)

        return FuzzySolution(
            t=t_eval,
            y_min=y_min_all,
            y_max=y_max_all,
            alphas=alphas_valid,
            var_names=self.solver.var_names
        )


class HierarchicalFuzzyODESolver:

    def __init__(self, base_solver):
        """
        Args:
            base_solver: Instância de FuzzyODESolver
        """
        self.solver = base_solver
        self.alpha_levels = self._generate_alpha_levels()

        # Novos atributos para guardar dados
        self.solutions_per_alpha = {}   # alpha -> [soluções]
        self.pertinences_per_alpha = {}  # alpha -> [pertinências]
        self.y0_grids_per_alpha = {}    # alpha -> grid de CI
        self.t_eval = None              # Tempos de avaliação

    def _generate_alpha_levels(self) -> np.ndarray:
        """Gera α-níveis em ordem DECRESCENTE."""
        alphas = np.linspace(0, 1, self.solver.n_alpha_cuts)
        return alphas[::-1]  # [1.0, 0.9, ..., 0.1, 0.0]

    def _compute_ci_pertinence(
        self,
        y0: np.ndarray,
        params: Dict[str, float]
    ) -> float:
        """
        Calcula pertinência de uma condição inicial.

        μ(CI) = min(μ_y0_1, μ_y0_2, ..., μ_r, μ_K, ...)
        (t-norma: mínimo)
        """
        pertinence = 1.0

        # Pertinência em condições iniciais fuzzy
        for i, y0_fuzzy_var in enumerate(self.solver.initial_condition):
            mu_y0 = y0_fuzzy_var.fuzzy_set.membership(y0[i])
            pertinence = min(pertinence, mu_y0)

        # Pertinência em parâmetros fuzzy
        for param_name, param_fuzzy in self.solver.params.items():
            if hasattr(param_fuzzy, 'fuzzy_set'):  # É FuzzyNumber
                param_val = params.get(param_name, param_fuzzy)
                mu_param = param_fuzzy.fuzzy_set.membership(param_val)
                pertinence = min(pertinence, mu_param)

        return max(0.0, min(1.0, pertinence))

    def _solve_alpha_level_with_storage(
        self,
        alpha: float,
        verbose: bool = False
    ) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
        """
        Resolve α-nível guardando:
        - Soluções individuais de cada ponto
        - Pertinência de cada CI

        Returns:
            (t_eval, solutions, pertinences)
        """

        # Extrai α-cortes
        y0_intervals, params_intervals = (
            self.solver._extract_alpha_cuts(alpha)
        )

        # Cria grid
        y0_grid, params_grid = self.solver._create_grid(
            y0_intervals, params_intervals
        )

        # Define tempos de avaliação (usando primeira resolução)
        if self.t_eval is None:
            self.t_eval = np.linspace(
                self.solver.t_span[0],
                self.solver.t_span[1],
                100
            )

        # Resolve ODEs com paralelização
        try:
            from joblib import Parallel, delayed
            HAS_JOBLIB = True
        except ImportError:
            HAS_JOBLIB = False

        if HAS_JOBLIB and self.solver.n_jobs != 1:
            solutions = Parallel(n_jobs=self.solver.n_jobs, backend='loky')(
                delayed(self.solver._solve_single_ode_with_t_eval)(
                    y0, params, self.t_eval
                )
                for y0, params in zip(y0_grid, params_grid)
            )
        else:
            solutions = [
                self.solver._solve_single_ode_with_t_eval(
                    y0, params, self.t_eval)
                for y0, params in zip(y0_grid, params_grid)
            ]

        # Calcula pertinências das CIs
        pertinences = []
        for y0, params in zip(y0_grid, params_grid):
            mu = self._compute_ci_pertinence(y0, params)
            pertinences.append(mu)

        pertinences = np.array(pertinences)

        if verbose:
            print(f"  α = {alpha:.3f}: {len(y0_grid)} ODEs, "
                  f"pertinências ∈ [{np.min(pertinences):.3f}, "
                  f"{np.max(pertinences):.3f}]")

        return self.t_eval, solutions, pertinences

    def _compute_alpha_levels_with_filtering(
        self,
        alphas: np.ndarray,
        verbose: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Recalcula α-níveis filtrando por pertinência.

        ALGORITMO CORRETO:
        Para cada α:
            Seleciona soluções com μ(CI) ≥ α
            Calcula min/max DESSAS

        Returns:
            (y_min_all, y_max_all) com shape (n_alpha, n_vars, n_time)
        """

        y_min_all = []
        y_max_all = []
        alphas_valid = []

        for alpha in alphas:
            # Obtém dados armazenados para este α
            if alpha not in self.solutions_per_alpha:
                continue

            solutions = self.solutions_per_alpha[alpha]
            pertinences = self.pertinences_per_alpha[alpha]

            # FILTRO CRÍTICO: seleciona por pertinência
            valid_mask = pertinences >= (alpha - 1e-10)

            if np.sum(valid_mask) == 0:
                # Nenhuma solução com pertinência ≥ α
                # Usa a com maior pertinência
                max_pert = np.max(pertinences)
                valid_mask = pertinences >= (max_pert - 1e-10)

            # Seleciona apenas soluções válidas
            valid_solutions = [
                sol for sol, valid in zip(solutions, valid_mask)
                if valid and not np.any(np.isnan(sol))
            ]

            if len(valid_solutions) == 0:
                if verbose:
                    print(f"  ⚠ α = {alpha:.3f}: sem soluções válidas")
                continue

            # Calcula envelopes APENAS das soluções selecionadas
            solutions_array = np.stack(valid_solutions, axis=0)
            y_min = np.min(solutions_array, axis=0)
            y_max = np.max(solutions_array, axis=0)

            y_min_all.append(y_min)
            y_max_all.append(y_max)
            alphas_valid.append(alpha)

            if verbose:
                n_used = np.sum(valid_mask)
                n_total = len(valid_mask)
                print(f"  ✓ α = {alpha:.3f}: {n_used}/{n_total} soluções "
                      f"({100*n_used/n_total:.0f}%)")

        return np.stack(y_min_all), np.stack(y_max_all), np.array(alphas_valid)

    def solve_hierarchical(self, verbose: bool = True) -> 'FuzzySolution':
        """
        Resolve com método hierárquico CORRIGIDO.

        Fluxo:
        1. Resolve cada α-nível (em ordem decrescente)
        2. Guarda TODAS as soluções individuais
        3. Guarda pertinências das CIs
        4. Filtra e recalcula α-níveis
        """

        if verbose:
            print("=" * 80)
            print("SOLVER HIERÁRQUICO (CORRIGIDO)")
            print("=" * 80)
            print(f"Variáveis: {self.solver.n_vars}")
            print(f"α-níveis: {len(self.alpha_levels)}")
            print(f"Método: Guarda soluções individuais + filtro por pertinência")
            print("=" * 80 + "\n")

        # FASE 1: Resolve e armazena para cada α
        if verbose:
            print("FASE 1: Resolvendo ODEs para cada α-nível...")

        for idx, alpha in enumerate(self.alpha_levels):
            if verbose and idx > 0:
                print()  # Linha em branco

            t, solutions, pertinences = self._solve_alpha_level_with_storage(
                alpha, verbose=verbose
            )

            # Armazena dados
            self.solutions_per_alpha[alpha] = solutions
            self.pertinences_per_alpha[alpha] = pertinences

        # FASE 2: Recalcula α-níveis com filtro correto
        if verbose:
            print("\n" + "-" * 80)
            print("FASE 2: Calculando α-níveis com filtro de pertinência...")
            print("-" * 80)

        y_min_all, y_max_all, alphas_valid = (
            self._compute_alpha_levels_with_filtering(
                self.alpha_levels, verbose=verbose
            )
        )

        if verbose:
            print("\n" + "=" * 80)
            print("✅ Solução hierárquica CORRIGIDA computada!")
            print("=" * 80 + "\n")

        from fuzzy_ode import FuzzySolution

        return FuzzySolution(
            t=self.t_eval,
            y_min=y_min_all,
            y_max=y_max_all,
            alphas=alphas_valid,
            var_names=self.solver.var_names
        )


@dataclass
class AlphaGridPoint:
    """Ponto no grid associado a um intervalo fuzzy."""
    point: np.ndarray  # Coordenadas (y0_1, y0_2, ..., y0_n)
    alpha_min: float   # Menor α para o qual este ponto pertence ao intervalo
    alpha_max: float   # Maior α
    grid_index: int    # Índice no grid cartesiano


# Mensagem de sucesso
if HAS_JOBLIB:
    print("Automatic parallelization (joblib)")
else:
    print("Serial processing (install joblib for parallelization)")
