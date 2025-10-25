"""
Sistemas p-Fuzzy - Sistemas Dinâmicos com Regras Fuzzy
========================================================

Implementa sistemas dinâmicos onde a função de evolução é definida
por regras fuzzy (Mamdani ou Sugeno).

Tipos de sistemas:
- Discretos: x_{n+1} = x_n + f(x_n) [absolute] ou x_{n+1} = x_n * f(x_n) [relative]
- Contínuos: dx/dt = f(x) [absolute] ou dx/dt = x * f(x) [relative]

Referências:
    Barros, L. C., Bassanezi, R. C., & Lodwick, W. A. (2017).
    "A First Course in Fuzzy Logic, Fuzzy Dynamical Systems, and Biomathematics"
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
from abc import ABC, abstractmethod
from ..inference.systems import MamdaniSystem, SugenoSystem, FuzzyInferenceSystem


class PFuzzySystem(ABC):
    """
    Classe base abstrata para sistemas p-fuzzy.

    Um sistema p-fuzzy é um sistema dinâmico onde a função de evolução
    é definida por um sistema de inferência fuzzy (Mamdani ou Sugeno).
    """

    def __init__(self,
                 fis: FuzzyInferenceSystem,
                 mode: str = 'absolute',
                 state_vars: Optional[List[str]] = None):
        """
        Inicializa um sistema p-fuzzy.

        Parâmetros:
            fis: Sistema de inferência fuzzy (Mamdani ou Sugeno)
            mode: Modo de evolução:
                 - 'absolute': Mudança absoluta (adição)
                 - 'relative': Mudança relativa (multiplicação/taxa)
            state_vars: Lista de nomes das variáveis de estado.
                       Se None, usa todas as variáveis de entrada do FIS.

        Nota:
            As saídas do FIS são interpretadas como taxas de mudança das
            variáveis de estado, na mesma ordem que as variáveis de estado.
            Exemplo: se state_vars=['x', 'y'], então a 1ª saída do FIS
            é a taxa de mudança de x, e a 2ª saída é a taxa de mudança de y.
        """
        self.fis = fis

        if mode not in ['absolute', 'relative']:
            raise ValueError(
                f"Modo '{mode}' inválido. Use 'absolute' ou 'relative'."
            )
        self.mode = mode

        # Identificar variáveis de estado
        if state_vars is None:
            self.state_vars = list(fis.input_variables.keys())
        else:
            self.state_vars = state_vars
            # Validar que variáveis de estado existem no FIS
            for var in state_vars:
                if var not in fis.input_variables:
                    raise ValueError(
                        f"Variável de estado '{var}' não encontrada no FIS. "
                        f"Variáveis disponíveis: {list(fis.input_variables.keys())}"
                    )

        self.n_vars = len(self.state_vars)

        # Obter lista de saídas do FIS
        self.output_vars = list(fis.output_variables.keys())

        # Validar que número de saídas corresponde ao número de estados
        if len(self.output_vars) != self.n_vars:
            raise ValueError(
                f"Número de saídas do FIS ({len(self.output_vars)}) deve "
                f"ser igual ao número de variáveis de estado ({self.n_vars})"
            )

        # Armazenar resultados de simulação
        self.trajectory = None
        self.time = None

    def _check_domain(self, x: np.ndarray) -> Tuple[bool, Optional[str]]:
        """
        Verifica se todas as variáveis de estado estão dentro de seus domínios.

        Parâmetros:
            x: Array com valores das variáveis de estado

        Retorna:
            (dentro_dominio, mensagem_erro)
            - dentro_dominio: True se todos os valores estão válidos
            - mensagem_erro: String descrevendo o problema (ou None se válido)
        """
        for i, var_name in enumerate(self.state_vars):
            value = x[i]
            var = self.fis.input_variables[var_name]
            min_val, max_val = var.universe

            if value < min_val or value > max_val:
                msg = (
                    f"Variável '{var_name}' = {value:.6f} está fora do domínio "
                    f"[{min_val}, {max_val}]. Simulação interrompida."
                )
                return False, msg

        return True, None

    def _evaluate_fis(self, state: Dict[str, float]) -> Dict[str, float]:
        """
        Avalia o FIS para um estado dado.

        Parâmetros:
            state: Dicionário com valores das variáveis de estado

        Retorna:
            Dicionário com saídas do FIS
        """
        return self.fis.evaluate(state)

    @abstractmethod
    def simulate(self, x0: Union[Dict[str, float], np.ndarray], **kwargs):
        """
        Simula o sistema p-fuzzy.

        Método abstrato - deve ser implementado pelas subclasses.
        """
        pass

    def plot_trajectory(self, variables=None, **kwargs):
        """
        Plota a trajetória temporal das variáveis de estado.

        Parâmetros:
            variables: Lista de variáveis a plotar. Se None, plota todas.
            **kwargs: Argumentos para customização do plot
                - figsize: Tamanho da figura
                - title: Título do gráfico
                - xlabel, ylabel: Rótulos dos eixos
        """
        if self.trajectory is None:
            raise RuntimeError("Execute simulate() antes de plotar!")

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Matplotlib necessário para plots")

        # Selecionar variáveis
        if variables is None:
            variables = self.state_vars
        elif isinstance(variables, str):
            variables = [variables]

        # Criar figura
        figsize = kwargs.get('figsize', (10, 6))
        fig, ax = plt.subplots(figsize=figsize)

        # Plotar cada variável
        for var in variables:
            if var not in self.state_vars:
                print(f"Aviso: Variável '{var}' não encontrada. Ignorando.")
                continue

            idx = self.state_vars.index(var)
            ax.plot(self.time, self.trajectory[:, idx],
                   label=var, linewidth=2, marker='o', markersize=3)

        # Configurações
        ax.set_xlabel(kwargs.get('xlabel', 'Tempo'), fontsize=12)
        ax.set_ylabel(kwargs.get('ylabel', 'Estado'), fontsize=12)
        ax.set_title(kwargs.get('title', 'Trajetória do Sistema p-Fuzzy'),
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return fig, ax

    def plot_phase_space(self, var_x, var_y, **kwargs):
        """
        Plota o espaço de fase (2D).

        Parâmetros:
            var_x: Nome da variável no eixo X
            var_y: Nome da variável no eixo Y
            **kwargs: Argumentos para customização
        """
        if self.trajectory is None:
            raise RuntimeError("Execute simulate() antes de plotar!")

        if var_x not in self.state_vars or var_y not in self.state_vars:
            raise ValueError(
                f"Variáveis devem estar em {self.state_vars}"
            )

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Matplotlib necessário para plots")

        idx_x = self.state_vars.index(var_x)
        idx_y = self.state_vars.index(var_y)

        figsize = kwargs.get('figsize', (8, 8))
        fig, ax = plt.subplots(figsize=figsize)

        # Trajetória
        ax.plot(self.trajectory[:, idx_x], self.trajectory[:, idx_y],
               'b-', linewidth=2, alpha=0.6, label='Trajetória')

        # Ponto inicial
        ax.plot(self.trajectory[0, idx_x], self.trajectory[0, idx_y],
               'go', markersize=10, label='Inicial')

        # Ponto final
        ax.plot(self.trajectory[-1, idx_x], self.trajectory[-1, idx_y],
               'ro', markersize=10, label='Final')

        ax.set_xlabel(var_x, fontsize=12)
        ax.set_ylabel(var_y, fontsize=12)
        ax.set_title(kwargs.get('title', f'Espaço de Fase: {var_x} vs {var_y}'),
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return fig, ax

    def to_csv(self, filename: str, sep: str = ',', decimal: str = '.') -> None:
        """
        Exporta a trajetória simulada para arquivo CSV.

        Parâmetros:
            filename: Caminho do arquivo CSV a ser criado
            sep: Separador de colunas (padrão: ',')
            decimal: Separador decimal (padrão: '.' para internacional,
                    use ',' para formato brasileiro/europeu)

        Raises:
            RuntimeError: Se não houver trajetória simulada

        Exemplo:
            >>> pfuzzy = fs.dynamic.PFuzzyContinuous(fis)
            >>> pfuzzy.simulate(x0=[10], t_span=(0, 50))
            >>> pfuzzy.to_csv('resultados.csv')
            >>>
            >>> # Para formato brasileiro (Excel)
            >>> pfuzzy.to_csv('resultados.csv', sep=';', decimal=',')
        """
        if self.trajectory is None or self.time is None:
            raise RuntimeError(
                "Nenhuma trajetória simulada. Execute simulate() primeiro."
            )

        # Criar cabeçalho: time, var1, var2, ...
        header = ['time'] + self.state_vars

        # Concatenar time e trajectory
        data = np.column_stack([self.time, self.trajectory])

        # Configurar formato numérico
        if decimal == ',':
            # Formato europeu/brasileiro: usar ; como separador
            fmt = '%.6f'
            # Converter pontos para vírgulas após salvar
            np.savetxt(
                filename,
                data,
                delimiter=sep,
                header=sep.join(header),
                comments='',
                fmt=fmt
            )
            # Substituir . por , no arquivo
            with open(filename, 'r') as f:
                content = f.read()
            with open(filename, 'w') as f:
                f.write(content.replace('.', ','))
        else:
            # Formato padrão internacional
            np.savetxt(
                filename,
                data,
                delimiter=sep,
                header=sep.join(header),
                comments='',
                fmt='%.6f'
            )


class PFuzzyDiscrete(PFuzzySystem):
    """
    Sistema p-fuzzy discreto.

    Modos:
    - absolute: x_{n+1} = x_n + f(x_n)  - mudança absoluta
    - relative: x_{n+1} = x_n * f(x_n) - mudança relativa (taxa)

    Exemplo:
        >>> import fuzzy_systems as fs
        >>>
        >>> # Criar FIS
        >>> fis = fs.MamdaniSystem()
        >>> fis.add_input('population', (0, 100))
        >>> fis.add_output('growth_rate', (-10, 10))
        >>> # ... adicionar termos e regras ...
        >>>
        >>> # Criar sistema p-fuzzy
        >>> pfuzzy = fs.dynamic.PFuzzyDiscrete(
        ...     fis=fis,
        ...     mode='absolute',
        ...     state_vars=['population']
        ... )
        >>>
        >>> # Simular
        >>> trajectory = pfuzzy.simulate(
        ...     x0={'population': 10},
        ...     n_steps=50
        ... )
        >>>
        >>> # Visualizar
        >>> pfuzzy.plot_trajectory()
    """

    def __init__(self,
                 fis: FuzzyInferenceSystem,
                 mode: str = 'absolute',
                 state_vars: Optional[List[str]] = None,
                 dt: float = 1.0):
        """
        Inicializa sistema p-fuzzy discreto.

        Parâmetros:
            fis: Sistema de inferência fuzzy
            mode: 'absolute' ou 'relative'
            state_vars: Variáveis de estado
            dt: Passo de tempo (para conversão tempo discreto -> contínuo)
        """
        super().__init__(fis, mode, state_vars)
        self.dt = dt

    def simulate(self,
                 x0: Union[Dict[str, float], np.ndarray, List[float], Tuple[float, ...]],
                 n_steps: int,
                 store_all: bool = True) -> np.ndarray:
        """
        Simula o sistema discreto.

        Parâmetros:
            x0: Condição inicial. Pode ser:
                - Dicionário: {'var1': val1, 'var2': val2}
                - Lista/tupla: [val1, val2] (ordem das state_vars)
                - Array numpy: np.array([val1, val2])
            n_steps: Número de iterações
            store_all: Se True, armazena toda a trajetória

        Retorna:
            Array com a trajetória (n_steps+1, n_vars)
        """
        # Converter condição inicial para array
        if isinstance(x0, dict):
            x_current = np.array([x0[var] for var in self.state_vars])
        elif isinstance(x0, (list, tuple)):
            x_current = np.array(x0, dtype=float)
        else:
            x_current = np.array(x0)

        if len(x_current) != self.n_vars:
            raise ValueError(
                f"Condição inicial deve ter {self.n_vars} valores, "
                f"recebeu {len(x_current)}"
            )

        # Validar condição inicial
        valid, msg = self._check_domain(x_current)
        if not valid:
            raise ValueError(f"Condição inicial inválida: {msg}")

        # Armazenar trajetória
        if store_all:
            trajectory = np.zeros((n_steps + 1, self.n_vars))
            trajectory[0] = x_current

        # Iterar
        for step in range(n_steps):
            # Criar dicionário de estado
            state = {self.state_vars[i]: x_current[i]
                    for i in range(self.n_vars)}

            # Avaliar FIS
            fis_output = self._evaluate_fis(state)

            # Extrair valores de saída na mesma ordem das variáveis de estado
            output_vals = np.array([
                fis_output[self.output_vars[i]] for i in range(self.n_vars)
            ])

            # Atualizar estado baseado no modo
            if self.mode == 'absolute':
                # x_{n+1} = x_n + f(x_n)
                x_current = x_current + output_vals
            else:  # relative
                # x_{n+1} = x_n * f(x_n)
                x_current = x_current * output_vals

            # Verificar se ainda está no domínio
            valid, msg = self._check_domain(x_current)
            if not valid:
                print(f"\n⚠️  AVISO: {msg}")
                print(f"    Passo: {step + 1}/{n_steps}")
                print(f"    Tempo: {(step + 1) * self.dt:.4f}")
                # Truncar trajetória no ponto onde saiu do domínio
                if store_all:
                    self.trajectory = trajectory[:step + 1]
                    self.time = np.arange(step + 1) * self.dt
                    return self.trajectory
                else:
                    return x_current

            if store_all:
                trajectory[step + 1] = x_current

        # Armazenar resultados
        if store_all:
            self.trajectory = trajectory
            self.time = np.arange(n_steps + 1) * self.dt
            return trajectory
        else:
            return x_current


class PFuzzyContinuous(PFuzzySystem):
    """
    Sistema p-fuzzy contínuo.

    Modos:
    - absolute: dx/dt = f(x)       - mudança absoluta
    - relative: dx/dt = x * f(x)   - mudança relativa (proporcional ao estado)

    Exemplo:
        >>> import fuzzy_systems as fs
        >>>
        >>> # Criar FIS
        >>> fis = fs.MamdaniSystem()
        >>> fis.add_input('temperature', (0, 100))
        >>> fis.add_output('cooling_rate', (-5, 5))
        >>> # ... adicionar termos e regras ...
        >>>
        >>> # Criar sistema p-fuzzy contínuo
        >>> pfuzzy = fs.dynamic.PFuzzyContinuous(
        ...     fis=fis,
        ...     mode='absolute',
        ...     state_vars=['temperature'],
        ...     method='rk4'
        ... )
        >>>
        >>> # Simular
        >>> trajectory = pfuzzy.simulate(
        ...     x0={'temperature': 80},
        ...     t_span=(0, 10),
        ...     dt=0.1
        ... )
        >>>
        >>> # Visualizar
        >>> pfuzzy.plot_trajectory()
    """

    def __init__(self,
                 fis: FuzzyInferenceSystem,
                 mode: str = 'absolute',
                 state_vars: Optional[List[str]] = None,
                 method: str = 'rk4'):
        """
        Inicializa sistema p-fuzzy contínuo.

        Parâmetros:
            fis: Sistema de inferência fuzzy
            mode: 'absolute' ou 'relative'
            state_vars: Variáveis de estado
            method: Método de integração numérica:
                   - 'euler': Método de Euler (simples, menos preciso)
                   - 'rk4': Runge-Kutta 4ª ordem (mais preciso)
        """
        super().__init__(fis, mode, state_vars)

        if method not in ['euler', 'rk4']:
            raise ValueError(
                f"Método '{method}' inválido. Use 'euler' ou 'rk4'."
            )
        self.method = method

    def _dynamics(self, x: np.ndarray) -> np.ndarray:
        """
        Calcula dx/dt = f(x) ou dx/dt = x * f(x).

        Parâmetros:
            x: Estado atual (array)

        Retorna:
            Derivada dx/dt (array)
        """
        # Criar dicionário de estado
        state = {self.state_vars[i]: x[i] for i in range(self.n_vars)}

        # Avaliar FIS
        fis_output = self._evaluate_fis(state)

        # Extrair valores de saída na mesma ordem das variáveis de estado
        f_x = np.array([fis_output[self.output_vars[i]] for i in range(self.n_vars)])

        # Calcular derivada baseado no modo
        if self.mode == 'absolute':
            # dx/dt = f(x)
            return f_x
        else:  # relative
            # dx/dt = x * f(x)
            return x * f_x

    def _step_euler(self, x: np.ndarray, dt: float) -> np.ndarray:
        """
        Um passo do método de Euler.

        x_{n+1} = x_n + dt * f(x_n)
        """
        return x + dt * self._dynamics(x)

    def _step_rk4(self, x: np.ndarray, dt: float) -> np.ndarray:
        """
        Um passo do método Runge-Kutta 4ª ordem.

        Mais preciso que Euler.
        """
        k1 = self._dynamics(x)
        k2 = self._dynamics(x + 0.5 * dt * k1)
        k3 = self._dynamics(x + 0.5 * dt * k2)
        k4 = self._dynamics(x + dt * k3)

        return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def simulate(self,
                 x0: Union[Dict[str, float], np.ndarray, List[float], Tuple[float, ...]],
                 t_span: Tuple[float, float],
                 dt: float = 0.01,
                 store_all: bool = True) -> np.ndarray:
        """
        Simula o sistema contínuo.

        Parâmetros:
            x0: Condição inicial. Pode ser:
                - Dicionário: {'var1': val1, 'var2': val2}
                - Lista/tupla: [val1, val2] (ordem das state_vars)
                - Array numpy: np.array([val1, val2])
            t_span: Tupla (t_inicial, t_final)
            dt: Passo de integração
            store_all: Se True, armazena toda a trajetória

        Retorna:
            Array com a trajetória
        """
        # Converter condição inicial
        if isinstance(x0, dict):
            x_current = np.array([x0[var] for var in self.state_vars])
        elif isinstance(x0, (list, tuple)):
            x_current = np.array(x0, dtype=float)
        else:
            x_current = np.array(x0)

        if len(x_current) != self.n_vars:
            raise ValueError(
                f"Condição inicial deve ter {self.n_vars} valores"
            )

        # Validar condição inicial
        valid, msg = self._check_domain(x_current)
        if not valid:
            raise ValueError(f"Condição inicial inválida: {msg}")

        # Tempos de integração
        t_start, t_end = t_span
        time_points = np.arange(t_start, t_end + dt, dt)
        n_steps = len(time_points)

        # Armazenar trajetória
        if store_all:
            trajectory = np.zeros((n_steps, self.n_vars))
            trajectory[0] = x_current

        # Selecionar método de integração
        step_func = self._step_rk4 if self.method == 'rk4' else self._step_euler

        # Integrar
        for i in range(1, n_steps):
            x_current = step_func(x_current, dt)

            # Verificar se ainda está no domínio
            valid, msg = self._check_domain(x_current)
            if not valid:
                print(f"\n⚠️  AVISO: {msg}")
                print(f"    Passo: {i}/{n_steps - 1}")
                print(f"    Tempo: {time_points[i]:.4f}")
                # Truncar trajetória no ponto onde saiu do domínio
                if store_all:
                    self.trajectory = trajectory[:i]
                    self.time = time_points[:i]
                    return self.trajectory
                else:
                    return x_current

            if store_all:
                trajectory[i] = x_current

        # Armazenar resultados
        if store_all:
            self.trajectory = trajectory
            self.time = time_points
            return trajectory
        else:
            return x_current
