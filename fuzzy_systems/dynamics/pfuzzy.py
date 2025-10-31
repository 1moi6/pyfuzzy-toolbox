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
import warnings 


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
        VERSÃO OTIMIZADA com operações vetorizadas NumPy.
        
        Parâmetros:
            x: Array com valores das variáveis de estado
            
        Retorna:
            (dentro_dominio, mensagem_erro)
            - dentro_dominio: True se todos os valores estão válidos
            - mensagem_erro: String descrevendo o problema (ou None se válido)
        """
        # Verificação vetorizada (muito mais rápida)
        min_vals = self._domain_limits[:, 0]
        max_vals = self._domain_limits[:, 1]
        
        out_of_bounds = (x < min_vals) | (x > max_vals)
        
        # Se todos estão dentro do domínio, retorna imediatamente
        if not np.any(out_of_bounds):
            return True, None
        
        # Encontrar primeira variável fora do domínio
        idx = np.where(out_of_bounds)[0][0]
        var_name = self.state_vars[idx]
        value = x[idx]
        min_val, max_val = self._domain_limits[idx]
        
        msg = (
            f"Variável '{var_name}' = {value:.6f} está fora do domínio "
            f"[{min_val}, {max_val}]. Simulação interrompida."
        )
        return False, msg


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
             state_vars: Optional[List[str]] = None):
        """
        Inicializa sistema p-fuzzy discreto.
        
        Parâmetros:
            fis: Sistema de inferência fuzzy
            mode: 'absolute' ou 'relative'
            state_vars: Variáveis de estado
        """
        super().__init__(fis, mode, state_vars)
        
        # OTIMIZAÇÃO 1: Pré-computar limites de domínio
        self._domain_limits = np.array([
            self.fis.input_variables[var].universe 
            for var in self.state_vars
        ])
        
        # OTIMIZAÇÃO 2: Dicionário reutilizável para estados
        self._state_dict = {var: 0.0 for var in self.state_vars}


    def simulate(self,
             x0: Union[Dict[str, float], np.ndarray, List[float], Tuple[float, ...]],
             n_steps: int,
             verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simula o sistema discreto p-fuzzy.
        
        Parâmetros:
            x0: Condição inicial. Pode ser:
                - Dicionário: {'var1': val1, 'var2': val2}
                - Lista/tupla: [val1, val2] (ordem das state_vars)
                - Array numpy: np.array([val1, val2])
            n_steps: Número de iterações
            verbose: Se True, imprime progresso da simulação
            
        Retorna:
            (iterations, trajectory): Tupla com arrays de iterações e estados
            
        Exemplo:
            >>> n, x = pfuzzy_disc.simulate(
            ...     x0={'populacao': 10.0},
            ...     n_steps=100,
            ...     verbose=True
            ... )
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
        
        # Inicializar trajetória (sempre armazena agora)
        trajectory = np.zeros((n_steps + 1, self.n_vars))
        trajectory[0] = x_current
        iterations = np.arange(n_steps + 1)
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"SIMULAÇÃO DISCRETA INICIADA")
            print(f"{'='*70}")
            print(f"Iterações: {n_steps}")
            print(f"Variáveis: {self.state_vars}")
            print(f"Modo: {self.mode}")
            print(f"{'='*70}\n")
        
        # Iterar
        for step in range(n_steps):
            # OTIMIZAÇÃO: Reutilizar dicionário ao invés de criar novo
            for i, var in enumerate(self.state_vars):
                self._state_dict[var] = float(x_current[i])
            
            # Avaliar FIS
            fis_output = self._evaluate_fis(self._state_dict)
            
            # Extrair valores de saída (vetorização)
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
            
            # Verificar se ainda está no domínio (versão otimizada)
            valid, msg = self._check_domain(x_current)
            if not valid:
                print(f"\n⚠️ AVISO: {msg}")
                print(f"   Iteração: {step + 1}/{n_steps}")
                
                # Truncar trajetória
                self.trajectory = trajectory[:step + 1]
                self.time = iterations[:step + 1]
                
                if verbose:
                    print(f"\n{'='*70}")
                    print(f"SIMULAÇÃO INTERROMPIDA")
                    print(f"{'='*70}")
                    print(f"Iterações completadas: {step + 1}/{n_steps}")
                    print(f"{'='*70}\n")
                
                return self.time, self.trajectory
            
            trajectory[step + 1] = x_current
            
            # Progresso (a cada 10% ou múltiplos de 100)
            if verbose and ((step + 1) % max(1, n_steps // 10) == 0 or (step + 1) % 100 == 0):
                progress = 100 * (step + 1) / n_steps
                print(f"Progresso: {progress:.1f}% | Iteração: {step + 1}/{n_steps}")
        
        # Armazenar resultados
        self.trajectory = trajectory
        self.time = iterations
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"SIMULAÇÃO CONCLUÍDA")
            print(f"{'='*70}")
            print(f"Iterações completadas: {n_steps}")
            print(f"Pontos na trajetória: {len(iterations)}")
            for i, var in enumerate(self.state_vars):
                print(f"{var}: {x_current[i]:.6f}")
            print(f"{'='*70}\n")
        
        return self.time, self.trajectory

    def step(self, x: Union[Dict[str, float], np.ndarray]) -> np.ndarray:
        """
        Executa um único passo da evolução discreta.
        
        Útil para análise manual ou estudos de estabilidade.
        
        Parâmetros:
            x: Estado atual (dict ou array)
            
        Retorna:
            Próximo estado (array)
            
        Exemplo:
            >>> x0 = np.array([10.0])
            >>> x1 = pfuzzy_disc.step(x0)
            >>> x2 = pfuzzy_disc.step(x1)
        """
        # Converter para array se necessário
        if isinstance(x, dict):
            x_current = np.array([x[var] for var in self.state_vars])
        else:
            x_current = np.array(x)
        
        # Validar domínio
        valid, msg = self._check_domain(x_current)
        if not valid:
            raise ValueError(f"Estado fora do domínio: {msg}")
        
        # Atualizar dicionário reutilizável
        for i, var in enumerate(self.state_vars):
            self._state_dict[var] = float(x_current[i])
        
        # Avaliar FIS
        fis_output = self._evaluate_fis(self._state_dict)
        
        # Extrair saídas
        output_vals = np.array([
            fis_output[self.output_vars[i]] for i in range(self.n_vars)
        ])
        
        # Calcular próximo estado
        if self.mode == 'absolute':
            return x_current + output_vals
        else:  # relative
            return x_current * output_vals


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
             method: str = 'euler'):
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
        
        # OTIMIZAÇÃO 1: Pré-computar limites de domínio (evita acessos repetidos)
        self._domain_limits = np.array([
            self.fis.input_variables[var].universe 
            for var in self.state_vars
        ])  # Shape: (n_vars, 2) -> [[min1, max1], [min2, max2], ...]
        
        # OTIMIZAÇÃO 2: Dicionário reutilizável para estados (evita criar novo a cada passo)
        self._state_dict = {var: 0.0 for var in self.state_vars}


    def _dynamics(self, x: np.ndarray) -> np.ndarray:
        """
        Calcula dx/dt = f(x) ou dx/dt = x * f(x).
        VERSÃO OTIMIZADA que reutiliza dicionário.
        
        Parâmetros:
            x: Estado atual (array)
            
        Retorna:
            Derivada dx/dt (array)
        """
        # OTIMIZAÇÃO: Atualizar valores no dicionário existente (não cria novo)
        for i, var in enumerate(self.state_vars):
            self._state_dict[var] = float(x[i])
        
        # Avaliar FIS
        fis_output = self._evaluate_fis(self._state_dict)
        
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
             dt: Optional[float] = None,
             adaptive: bool = False,
             tolerance: float = 1e-4,
             dt_min: float = 1e-5,
             dt_max: float = 1.0,
             verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simula o sistema p-fuzzy contínuo.
        
        Se adaptive=False, usa passo fixo (implementação original).
        Se adaptive=True, usa passo adaptativo (mais eficiente).
        
        Parâmetros:
            x0: Condição inicial
            t_span: (t_inicial, t_final)
            dt: Tamanho do passo (fixo ou inicial se adaptativo)
            adaptive: Se True, usa passo adaptativo
            tolerance: Tolerância de erro (apenas para adaptativo)
            dt_min, dt_max: Limites de dt (apenas para adaptativo)
            verbose: Imprimir estatísticas
            
        Retorna:
            (time_points, trajectory): Tupla com arrays de tempo e trajetória
        """
        if not adaptive:
            # Simulação com passo fixo (método original)
            if dt is None:
                dt = 0.05  # Padrão
            
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

            # Armazenar trajetória (SEMPRE armazena agora)
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
                    self.trajectory = trajectory[:i]
                    self.time = time_points[:i]
                    return self.time, self.trajectory  # ✓ Retorna tupla

                trajectory[i] = x_current

            # Armazenar resultados
            self.trajectory = trajectory
            self.time = time_points
            
            return self.time, self.trajectory  # ✓ Retorna tupla
        
        else:  # ✓ Indentação corrigida (alinhada com o if)
            # Simulação com passo adaptativo
            if dt is None:
                dt = 0.1  # Inicial maior para adaptativo
            
            return self.simulate_adaptive(
                x0=x0,
                t_span=t_span,
                dt_initial=dt,
                tolerance=tolerance,
                dt_min=dt_min,
                dt_max=dt_max,
                verbose=verbose
            )


    def _step_rk4_with_error(self, x: np.ndarray, dt: float) -> Tuple[np.ndarray, float]:
        """
        Um passo RK4 com estimativa de erro local.
        
        Calcula dois passos: um completo de tamanho dt e dois meios passos.
        A diferença fornece uma estimativa do erro local.
        
        Parâmetros:
            x: Estado atual
            dt: Tamanho do passo
            
        Retorna:
            (x_next, erro): Próximo estado e estimativa do erro
        """
        # Passo completo: x(t + dt)
        x_full = self._step_rk4(x, dt)
        
        # Dois meios passos: x(t + dt/2) e depois x(t + dt)
        x_half1 = self._step_rk4(x, dt/2)
        x_half2 = self._step_rk4(x_half1, dt/2)
        
        # Estimativa do erro local (método de Richardson)
        # O erro de RK4 é O(h^5), então a diferença entre
        # passo completo e meios passos é aproximadamente 15*erro
        error_estimate = np.linalg.norm(x_full - x_half2) / 15.0
        
        # Usar a solução mais precisa (meios passos)
        return x_half2, error_estimate

    def simulate_adaptive(self,
                     x0: Union[Dict[str, float], np.ndarray, List[float], Tuple[float, ...]],
                     t_span: Tuple[float, float],
                     dt_initial: float = 0.1,
                     tolerance: float = 1e-4,
                     dt_min: float = 1e-5,
                     dt_max: float = 1.0,
                     max_steps: int = 100000,
                     verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simula o sistema contínuo com passo adaptativo.
        
        O tamanho do passo é ajustado automaticamente baseado no erro local,
        permitindo passos maiores em regiões suaves e menores onde há mudanças
        rápidas. Funciona com Euler e RK4.
        
        NOTA: RK4 adaptativo é mais eficiente que Euler adaptativo para
        a maioria dos problemas (menos passos para mesma precisão).
        
        Parâmetros:
            x0: Condição inicial (dict, array, lista ou tupla)
            t_span: Tupla (t_inicial, t_final)
            dt_initial: Tamanho inicial do passo (padrão: 0.1)
            tolerance: Tolerância para o erro local (padrão: 1e-4)
                - Valores menores: maior precisão, mais passos
                - Valores maiores: menor precisão, menos passos
            dt_min: Tamanho mínimo permitido para o passo (padrão: 1e-5)
            dt_max: Tamanho máximo permitido para o passo (padrão: 1.0)
            max_steps: Número máximo de passos (prevenção de loops infinitos)
            verbose: Se True, imprime estatísticas durante a simulação
            
        Retorna:
            (time_points, trajectory): Arrays com tempos e estados
        """
        import warnings
        
        # Converter condição inicial
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
        
        # Inicializar
        t_start, t_end = t_span
        t_current = t_start
        dt = dt_initial
        
        trajectory = [x_current.copy()]
        time_points = [t_current]
        
        # Estatísticas
        n_accepted = 0
        n_rejected = 0
        dt_history = []
        
        # ESCOLHER função de passo com erro baseada no método
        if self.method == 'rk4':
            step_with_error = self._step_rk4_with_error
            order = 4  # Ordem do método (para ajuste de dt)
        else:  # euler
            step_with_error = self._step_euler_with_error
            order = 1  # Ordem do método
            if verbose:
                warnings.warn(
                    "Usando Euler adaptativo. Para melhor eficiência, "
                    "considere usar method='rk4'.",
                    UserWarning,
                    stacklevel=2
                )
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"SIMULAÇÃO ADAPTATIVA INICIADA")
            print(f"{'='*70}")
            print(f"Intervalo: [{t_start:.2f}, {t_end:.2f}]")
            print(f"Método: {self.method.upper()} (ordem {order})")
            print(f"dt inicial: {dt_initial:.6f}")
            print(f"Tolerância: {tolerance:.2e}")
            print(f"Limites dt: [{dt_min:.2e}, {dt_max:.2e}]")
            print(f"{'='*70}\n")
        
        # Loop de integração adaptativa
        step_count = 0
        while t_current < t_end and step_count < max_steps:
            step_count += 1
            
            # Ajustar dt se ultrapassar t_end
            if t_current + dt > t_end:
                dt = t_end - t_current
            
            # Tentar passo com estimativa de erro (usa método apropriado)
            x_next, error = step_with_error(x_current, dt)
            
            # Calcular erro relativo (normalizado)
            error_norm = error / (tolerance * (1.0 + np.linalg.norm(x_current)))
            
            if error_norm <= 1.0:
                # ✓ Passo aceito: erro dentro da tolerância
                
                # Verificar domínio
                valid, msg = self._check_domain(x_next)
                if not valid:
                    print(f"\n⚠️ AVISO: {msg}")
                    print(f"   Tempo: {t_current:.6f}")
                    print(f"   Passo: {step_count}")
                    break
                
                # Aceitar passo
                x_current = x_next
                t_current += dt
                
                trajectory.append(x_current.copy())
                time_points.append(t_current)
                dt_history.append(dt)
                
                n_accepted += 1
                
                if verbose and n_accepted % 100 == 0:
                    progress = 100 * (t_current - t_start) / (t_end - t_start)
                    print(f"Progresso: {progress:.1f}% | "
                        f"t={t_current:.4f} | "
                        f"dt={dt:.6f} | "
                        f"Aceitos: {n_accepted} | "
                        f"Rejeitados: {n_rejected}")
                
                # Aumentar dt para o próximo passo se erro for muito pequeno
                # Fator de segurança: 0.9
                # Regra: dt_new = dt * (tol/erro)^(1/(order+1))
                if error_norm > 0:
                    # Fórmula depende da ordem do método
                    exponent = 1.0 / (order + 1)
                    factor = 0.9 * (1.0 / error_norm) ** exponent
                    factor = min(factor, 2.0)  # Não aumentar mais que 2x
                    factor = max(factor, 0.5)   # Não diminuir mais que 0.5x
                    dt = dt * factor
                else:
                    # Erro zero ou negligível: aumentar dt
                    dt = min(dt * 1.5, dt_max)
                
                # Respeitar limites
                dt = np.clip(dt, dt_min, dt_max)
                
            else:
                # ✗ Passo rejeitado: erro muito grande
                n_rejected += 1
                
                # Reduzir dt e tentar novamente
                # Fator de segurança: 0.9
                exponent = 1.0 / (order + 1)
                factor = 0.9 * (1.0 / error_norm) ** exponent
                factor = max(factor, 0.1)  # Não reduzir mais que 10x
                dt = dt * factor
                
                # Respeitar limite mínimo
                if dt < dt_min:
                    print(f"\n⚠️ AVISO: dt atingiu limite mínimo ({dt_min:.2e})")
                    print(f"   Erro muito alto para tolerância especificada")
                    print(f"   Tempo: {t_current:.6f}")
                    break
        
        # Verificar se loop foi interrompido
        if step_count >= max_steps:
            warnings.warn(
                f"Número máximo de passos ({max_steps}) atingido. "
                f"Considere aumentar max_steps ou relaxar a tolerância.",
                UserWarning
            )
        
        # Converter para arrays
        self.time = np.array(time_points)
        self.trajectory = np.array(trajectory)
        
        # Imprimir estatísticas finais
        if verbose:
            print(f"\n{'='*70}")
            print(f"SIMULAÇÃO CONCLUÍDA")
            print(f"{'='*70}")
            print(f"Método: {self.method.upper()}")
            print(f"Passos aceitos: {n_accepted}")
            print(f"Passos rejeitados: {n_rejected}")
            print(f"Total de passos: {n_accepted + n_rejected}")
            if n_accepted + n_rejected > 0:
                print(f"Taxa de aceitação: {100*n_accepted/(n_accepted+n_rejected):.1f}%")
            print(f"Pontos na trajetória: {len(time_points)}")
            if len(dt_history) > 0:
                print(f"dt médio: {np.mean(dt_history):.6f}")
                print(f"dt mínimo: {np.min(dt_history):.6f}")
                print(f"dt máximo: {np.max(dt_history):.6f}")
            print(f"Tempo final: {t_current:.6f}")
            print(f"{'='*70}\n")
        
        return self.time, self.trajectory


    def _step_euler_with_error(self, x: np.ndarray, dt: float) -> Tuple[np.ndarray, float]:
        """
        Um passo Euler com estimativa de erro local.
        
        Usa dois meios passos para estimar o erro (menos preciso que RK4).
        
        Parâmetros:
            x: Estado atual
            dt: Tamanho do passo
            
        Retorna:
            (x_next, erro): Próximo estado e estimativa do erro
        """
        # Passo completo
        x_full = self._step_euler(x, dt)
        
        # Dois meios passos
        x_half1 = self._step_euler(x, dt/2)
        x_half2 = self._step_euler(x_half1, dt/2)
        
        # Estimativa do erro local
        # O erro de Euler é O(h^2), então a diferença é aproximadamente 2*erro
        error_estimate = np.linalg.norm(x_full - x_half2) / 2.0
        
        return x_half2, error_estimate
