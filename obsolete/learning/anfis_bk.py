"""
ANFIS - Adaptive Neuro-Fuzzy Inference System
==============================================

Implementação Python completa do ANFIS com:
- Aprendizado híbrido (LSE + Gradient Descent)
- Múltiplas funções de pertinência (gaussiana, sino generalizado, sigmoide)
- Regularização L1/L2 (Lasso, Ridge, Elastic Net)
- Minibatch training para datasets grandes
- Early stopping e validação
- Métricas detalhadas de treinamento
- Restrições de domínio adaptativas

Referências:
    Jang, J. S. (1993). "ANFIS: adaptive-network-based fuzzy inference system."
    IEEE transactions on systems, man, and cybernetics, 23(3), 665-685.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Union, Callable
import warnings
import time
import itertools

from ..core.membership import gaussian, generalized_bell, sigmoid


class MetricasANFIS:
    """
    Classe para armazenar e visualizar métricas de treinamento do ANFIS.

    Armazena RMSE, MAE, erro máximo e tempo de treinamento por época,
    além de permitir visualização da convergência.
    """

    def __init__(self):
        self.rmse_train = []
        self.rmse_val = []
        self.mae_train = []
        self.mae_val = []
        self.max_error_train = []
        self.max_error_val = []
        self.r2_train = []
        self.r2_val = []
        self.mape_train = []
        self.mape_val = []
        self.epoch_times = []
        self.gradientes_antecedentes = []
        self.learning_rates = []  # Taxa de aprendizado efetiva por época

    def adicionar_epoca(self, epoch_metrics: Dict):
        """
        Adiciona métricas de uma época ao histórico.

        Parâmetros:
            epoch_metrics: Dicionário com métricas da época
        """
        self.rmse_train.append(epoch_metrics.get('rmse_train', np.nan))
        self.rmse_val.append(epoch_metrics.get('rmse_val', np.nan))
        self.mae_train.append(epoch_metrics.get('mae_train', np.nan))
        self.mae_val.append(epoch_metrics.get('mae_val', np.nan))
        self.max_error_train.append(epoch_metrics.get('max_error_train', np.nan))
        self.max_error_val.append(epoch_metrics.get('max_error_val', np.nan))
        self.r2_train.append(epoch_metrics.get('r2_train', np.nan))
        self.r2_val.append(epoch_metrics.get('r2_val', np.nan))
        self.mape_train.append(epoch_metrics.get('mape_train', np.nan))
        self.mape_val.append(epoch_metrics.get('mape_val', np.nan))
        self.epoch_times.append(epoch_metrics.get('time', 0))
        self.gradientes_antecedentes.append(epoch_metrics.get('grad_norm', 0))
        self.learning_rates.append(epoch_metrics.get('learning_rate', np.nan))

    def plotar_convergencia(self, figsize=(18, 12)):
        """
        Visualiza a convergência do treinamento em múltiplos gráficos.

        Parâmetros:
            figsize: Tamanho da figura (largura, altura)

        Retorna:
            Figure do matplotlib com os gráficos
        """
        fig, axes = plt.subplots(3, 3, figsize=figsize)

        epochs = np.arange(1, len(self.rmse_train) + 1)

        # RMSE
        ax = axes[0, 0]
        ax.plot(epochs, self.rmse_train, 'b-', linewidth=2, label='Treino')
        if not np.all(np.isnan(self.rmse_val)):
            ax.plot(epochs, self.rmse_val, 'r--', linewidth=2, label='Validação')
        ax.set_xlabel('Época', fontsize=12)
        ax.set_ylabel('RMSE', fontsize=12)
        ax.set_title('Root Mean Square Error', fontsize=14, weight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # MAE
        ax = axes[0, 1]
        ax.plot(epochs, self.mae_train, 'b-', linewidth=2, label='Treino')
        if not np.all(np.isnan(self.mae_val)):
            ax.plot(epochs, self.mae_val, 'r--', linewidth=2, label='Validação')
        ax.set_xlabel('Época', fontsize=12)
        ax.set_ylabel('MAE', fontsize=12)
        ax.set_title('Mean Absolute Error', fontsize=14, weight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Erro Máximo
        ax = axes[0, 2]
        ax.plot(epochs, self.max_error_train, 'b-', linewidth=2, label='Treino')
        if not np.all(np.isnan(self.max_error_val)):
            ax.plot(epochs, self.max_error_val, 'r--', linewidth=2, label='Validação')
        ax.set_xlabel('Época', fontsize=12)
        ax.set_ylabel('Erro Máximo', fontsize=12)
        ax.set_title('Maximum Error', fontsize=14, weight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Log RMSE (para visualizar melhor pequenas mudanças)
        ax = axes[1, 0]
        ax.semilogy(epochs, self.rmse_train, 'b-', linewidth=2, label='Treino')
        if not np.all(np.isnan(self.rmse_val)):
            ax.semilogy(epochs, self.rmse_val, 'r--', linewidth=2, label='Validação')
        ax.set_xlabel('Época', fontsize=12)
        ax.set_ylabel('RMSE (escala log)', fontsize=12)
        ax.set_title('RMSE - Escala Logarítmica', fontsize=14, weight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, which='both')

        # Norma dos gradientes (antecedentes)
        ax = axes[1, 1]
        if len(self.gradientes_antecedentes) > 0 and not np.all(np.array(self.gradientes_antecedentes) == 0):
            ax.plot(epochs, self.gradientes_antecedentes, 'g-', linewidth=2)
            ax.set_xlabel('Época', fontsize=12)
            ax.set_ylabel('Norma do Gradiente', fontsize=12)
            ax.set_title('Magnitude dos Gradientes (Antecedentes)', fontsize=14, weight='bold')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Apenas consequentes\ntreinados',
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.set_title('Magnitude dos Gradientes', fontsize=14, weight='bold')

        # Tempo por época
        ax = axes[1, 2]
        if len(self.epoch_times) > 0:
            ax.bar(epochs, self.epoch_times, color='orange', alpha=0.7)
            ax.set_xlabel('Época', fontsize=12)
            ax.set_ylabel('Tempo (s)', fontsize=12)
            ax.set_title('Tempo de Treinamento por Época', fontsize=14, weight='bold')
            ax.grid(True, alpha=0.3, axis='y')

        # R² (Coeficiente de Determinação)
        ax = axes[2, 0]
        if len(self.r2_train) > 0 and not np.all(np.isnan(self.r2_train)):
            ax.plot(epochs, self.r2_train, 'b-', linewidth=2, label='Treino')
            if not np.all(np.isnan(self.r2_val)):
                ax.plot(epochs, self.r2_val, 'r--', linewidth=2, label='Validação')
            ax.set_xlabel('Época', fontsize=12)
            ax.set_ylabel('R²', fontsize=12)
            ax.set_title('Coeficiente de Determinação (R²)', fontsize=14, weight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.axhline(y=1.0, color='g', linestyle=':', alpha=0.5, label='Perfeito')
        else:
            ax.text(0.5, 0.5, 'R² não disponível', ha='center', va='center',
                   fontsize=12, transform=ax.transAxes)
            ax.set_title('R²', fontsize=14, weight='bold')

        # MAPE (Mean Absolute Percentage Error)
        ax = axes[2, 1]
        if len(self.mape_train) > 0 and not np.all(np.isnan(self.mape_train)):
            ax.plot(epochs, self.mape_train, 'b-', linewidth=2, label='Treino')
            if not np.all(np.isnan(self.mape_val)):
                ax.plot(epochs, self.mape_val, 'r--', linewidth=2, label='Validação')
            ax.set_xlabel('Época', fontsize=12)
            ax.set_ylabel('MAPE (%)', fontsize=12)
            ax.set_title('Mean Absolute Percentage Error', fontsize=14, weight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'MAPE não disponível', ha='center', va='center',
                   fontsize=12, transform=ax.transAxes)
            ax.set_title('MAPE', fontsize=14, weight='bold')

        # Learning Rate Efetivo
        ax = axes[2, 2]
        if len(self.learning_rates) > 0 and not np.all(np.isnan(self.learning_rates)):
            ax.plot(epochs, self.learning_rates, 'purple', linewidth=2)
            ax.set_xlabel('Época', fontsize=12)
            ax.set_ylabel('Learning Rate', fontsize=12)
            ax.set_title('Learning Rate Efetivo', fontsize=14, weight='bold')
            ax.grid(True, alpha=0.3)
            # Usar escala log se houver variação significativa
            lr_array = np.array(self.learning_rates)
            lr_array = lr_array[~np.isnan(lr_array)]
            if len(lr_array) > 0:
                lr_range = lr_array.max() / (lr_array.min() + 1e-10)
                if lr_range > 10:
                    ax.set_yscale('log')
        else:
            ax.text(0.5, 0.5, 'LR fixo ou não disponível', ha='center', va='center',
                   fontsize=12, transform=ax.transAxes)
            ax.set_title('Learning Rate', fontsize=14, weight='bold')

        plt.tight_layout()
        return fig


class ANFIS:
    """
    ANFIS - Adaptive Neuro-Fuzzy Inference System

    Sistema híbrido que combina redes neurais e lógica fuzzy para
    aprendizado supervisionado. Implementa:

    - Arquitetura de 5 camadas (fuzzificação, regras, normalização,
      consequentes, agregação)
    - Aprendizado híbrido: LSE para parâmetros consequentes + gradiente
      descendente para parâmetros antecedentes
    - Regularização L1/L2 para evitar overfitting
    - Minibatch training para eficiência computacional
    - Early stopping baseado em validação

    Exemplo:
        >>> import numpy as np
        >>> from fuzzy_systems.learning.anfis_moderno import ANFIS
        >>>
        >>> # Dados de treino
        >>> X_train = np.random.uniform(-3, 3, (100, 2))
        >>> y_train = np.sin(X_train[:, 0]) + np.cos(X_train[:, 1])
        >>>
        >>> # Criar e treinar ANFIS
        >>> anfis = ANFIS(n_inputs=2, n_mfs=[3, 3], mf_type='gaussmf',
        ...               learning_rate=0.01, lambda_l2=0.01, batch_size=32)
        >>> anfis.fit(X_train, y_train, epochs=100, verbose=True)
        >>>
        >>> # Predição
        >>> y_pred = anfis.predict(X_train)
        >>>
        >>> # Visualizar MFs e convergência
        >>> anfis.visualizar_mfs()
        >>> anfis.metricas.plotar_convergencia()
    """

    def __init__(self,
                 n_inputs: int,
                 n_mfs: Union[int, List[int]],
                 mf_type: str = 'gaussmf',
                 learning_rate: float = 0.01,
                 input_ranges: Optional[List[Tuple[float, float]]] = None,
                 lambda_l1: float = 0.0,
                 lambda_l2: float = 0.01,
                 batch_size: Optional[int] = None,
                 use_adaptive_lr: bool = False):
        """
        Inicializa o ANFIS com regularização e minibatch training.

        Parâmetros:
            n_inputs: Número de variáveis de entrada
            n_mfs: Número de funções de pertinência por entrada.
                   Pode ser int (mesmo número para todas) ou lista de ints
            mf_type: Tipo de função de pertinência:
                    'gaussmf' - Gaussiana (padrão)
                    'gbellmf' - Sino generalizado
                    'sigmf' - Sigmoide
            learning_rate: Taxa de aprendizado para gradiente descendente
            input_ranges: Lista com ranges (min, max) de cada entrada.
                         Se None, usa (-8, 8) para todas
            lambda_l1: Coeficiente de regularização L1 (Lasso) sobre larguras das MFs
            lambda_l2: Coeficiente de regularização L2 (Ridge) sobre larguras das MFs
            batch_size: Tamanho do batch para minibatch training
                       - None: Batch gradient descent (usa todos os dados)
                       - 1: Stochastic gradient descent
                       - 16-128: Minibatch gradient descent (recomendado)
            use_adaptive_lr: Se True, usa taxa de aprendizado adaptativa baseada
                            em Lyapunov. Garante convergência teórica mas pode
                            ser mais lento. Se False, usa learning_rate fixo.

        Nota sobre regularização:
            A regularização é aplicada APENAS nas larguras (sigmas) das funções
            de pertinência, não nos centros. Isso faz sentido porque:
            - Centros devem se adaptar livremente à posição dos dados
            - Larguras devem ser regularizadas para evitar overfitting

        Nota sobre Lyapunov:
            Quando use_adaptive_lr=True, a taxa de aprendizado é calculada
            dinamicamente para garantir estabilidade: η = 1.99/||∇E||²
            Isso garante convergência teórica segundo a teoria de Lyapunov.
        """
        self.n_inputs = n_inputs

        # Permitir n_mfs como int ou lista
        if isinstance(n_mfs, int):
            self.n_mfs = [n_mfs] * n_inputs
        else:
            if len(n_mfs) != n_inputs:
                raise ValueError(f"n_mfs deve ter {n_inputs} elementos")
            self.n_mfs = list(n_mfs)

        self.mf_type = mf_type
        self.lr = learning_rate
        self.n_regras = int(np.prod(self.n_mfs))
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.batch_size = batch_size
        self.use_adaptive_lr = use_adaptive_lr

        # Validar tipo de MF
        tipos_validos = ['gaussmf', 'gbellmf', 'sigmf']
        if mf_type not in tipos_validos:
            raise ValueError(f"mf_type deve ser um de {tipos_validos}")

        # Definir ranges
        if input_ranges is None:
            self.input_ranges = [(-8.0, 8.0)] * n_inputs
        else:
            if len(input_ranges) != n_inputs:
                raise ValueError(f"input_ranges deve ter {n_inputs} elementos")
            self.input_ranges = input_ranges

        # Inicializar parâmetros (será feito no fit com dados reais)
        self.params_mfs = None
        self.params_consequentes = None
        self.input_bounds = None

        # Métricas
        self.metricas = MetricasANFIS()

        # Cache de índices de regras
        self._cache_indices_regras = None

        # Histórico de penalidades
        self.historico_l1 = []
        self.historico_l2 = []
        self.historico_custo_total = []

    def _initialize_premise_params(self, X: np.ndarray):
        """
        Inicializa parâmetros das funções de pertinência baseado nos dados.

        Parâmetros:
            X: Dados de entrada (n_samples, n_inputs)
        """
        # Calcular bounds reais dos dados
        self.input_bounds = np.array([[X[:, i].min(), X[:, i].max()]
                                       for i in range(self.n_inputs)])

        self.params_mfs = []

        for i in range(self.n_inputs):
            x_min, x_max = self.input_bounds[i]
            x_range = x_max - x_min

            # Garantir range mínimo
            if x_range < 1e-6:
                x_range = 1.0
                x_min = x_min - 0.5
                x_max = x_max + 0.5

            mf_params = []
            n_mf = self.n_mfs[i]
            centers = np.linspace(x_min, x_max, n_mf)

            for j in range(n_mf):
                if self.mf_type == 'gaussmf':
                    # Parâmetros: [mean, sigma]
                    sigma = x_range / (2 * n_mf)
                    # Adicionar pequena perturbação aleatória
                    center = centers[j] + np.random.uniform(-0.1, 0.1) * sigma
                    params = np.array([center, sigma])

                elif self.mf_type == 'gbellmf':
                    # Parâmetros: [a (width), b (slope), c (center)]
                    width = x_range / (2 * n_mf)
                    center = centers[j] + np.random.uniform(-0.1, 0.1) * width
                    params = np.array([width, 2.0, center])

                elif self.mf_type == 'sigmf':
                    # Parâmetros: [a (slope), c (center)]
                    slope = 4.0 / (x_range / n_mf)
                    center = centers[j] + np.random.uniform(-0.1, 0.1) * (x_range / n_mf)
                    params = np.array([slope, center])

                mf_params.append(params)

            self.params_mfs.append(np.array(mf_params))

    def _apply_domain_constraints(self):
        """
        Aplica restrições de domínio aos parâmetros antecedentes para
        garantir que permaneçam em ranges válidos.
        """
        for i in range(self.n_inputs):
            x_min, x_max = self.input_bounds[i]

            for j in range(self.n_mfs[i]):
                if self.mf_type == 'gaussmf':
                    # [mean, sigma]
                    self.params_mfs[i][j, 0] = np.clip(self.params_mfs[i][j, 0], x_min, x_max)
                    self.params_mfs[i][j, 1] = np.maximum(self.params_mfs[i][j, 1], 1e-6)

                elif self.mf_type == 'gbellmf':
                    # [a, b, c]
                    self.params_mfs[i][j, 2] = np.clip(self.params_mfs[i][j, 2], x_min, x_max)
                    self.params_mfs[i][j, 0] = np.maximum(self.params_mfs[i][j, 0], 1e-6)
                    self.params_mfs[i][j, 1] = np.maximum(self.params_mfs[i][j, 1], 0.1)

                elif self.mf_type == 'sigmf':
                    # [a, c]
                    self.params_mfs[i][j, 1] = np.clip(self.params_mfs[i][j, 1], x_min, x_max)

    def _gerar_indices_regras(self) -> List[Tuple[int, ...]]:
        """
        Gera índices das regras (produto cartesiano das MFs).

        Retorna:
            Lista de tuplas com índices das MFs para cada regra
        """
        if self.n_inputs == 1:
            return [(i,) for i in range(self.n_mfs[0])]

        indices = list(itertools.product(*[range(n) for n in self.n_mfs]))
        return indices

    def _criar_batches(self, X: np.ndarray, y: np.ndarray,
                       batch_size: int, shuffle: bool = True) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Divide dados em batches para minibatch training.

        Parâmetros:
            X: Dados de entrada
            y: Dados de saída
            batch_size: Tamanho do batch
            shuffle: Se True, embaralha os dados antes de dividir

        Retorna:
            Lista de tuplas (X_batch, y_batch)
        """
        n_samples = len(X)
        indices = np.arange(n_samples)

        if shuffle:
            np.random.shuffle(indices)

        batches = []
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            batches.append((X[batch_indices], y[batch_indices]))

        return batches

    def _eval_mf(self, x: float, params: np.ndarray) -> float:
        """
        Avalia função de pertinência para um valor x.

        Parâmetros:
            x: Valor de entrada
            params: Parâmetros da MF

        Retorna:
            Grau de pertinência
        """
        if self.mf_type == 'gaussmf':
            return gaussian(x, tuple(params))
        elif self.mf_type == 'gbellmf':
            return generalized_bell(x, tuple(params))
        elif self.mf_type == 'sigmf':
            return sigmoid(x, tuple(params))

    def camada1_fuzzificacao(self, X: np.ndarray) -> List[np.ndarray]:
        """
        Camada 1: Fuzzificação - calcula graus de pertinência.

        Parâmetros:
            X: Vetor de entrada (n_inputs,)

        Retorna:
            Lista com graus de pertinência para cada entrada
        """
        mu = []
        for i in range(self.n_inputs):
            mu_i = np.array([self._eval_mf(X[i], params)
                           for params in self.params_mfs[i]])
            mu.append(mu_i)
        return mu

    def camada2_regras(self, mu: List[np.ndarray]) -> np.ndarray:
        """
        Camada 2: Força de disparo das regras (produto das MFs).

        Parâmetros:
            mu: Lista com graus de pertinência

        Retorna:
            Array com força de disparo de cada regra
        """
        w = np.zeros(self.n_regras)
        for regra_idx, indices_mf in enumerate(self._cache_indices_regras):
            w[regra_idx] = np.prod([mu[i][mf_idx]
                                   for i, mf_idx in enumerate(indices_mf)])
        return w

    def camada3_normalizacao(self, w: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Camada 3: Normalização das forças de disparo.

        Parâmetros:
            w: Forças de disparo

        Retorna:
            Tupla (w_norm, soma_w)
        """
        soma_w = np.sum(w) + 1e-10
        w_norm = w / soma_w
        return w_norm, soma_w

    def camada4_consequentes(self, X: np.ndarray, w_norm: np.ndarray) -> np.ndarray:
        """
        Camada 4: Cálculo das saídas dos consequentes (Takagi-Sugeno).

        Parâmetros:
            X: Vetor de entrada
            w_norm: Forças normalizadas

        Retorna:
            Saídas de cada regra
        """
        saidas = np.zeros(self.n_regras)
        for i in range(self.n_regras):
            params = self.params_consequentes[i]
            # f_i = p0 + p1*x1 + p2*x2 + ...
            f_i = params[0] + np.dot(params[1:], X)
            saidas[i] = w_norm[i] * f_i
        return saidas

    def camada5_agregacao(self, saidas: np.ndarray) -> float:
        """
        Camada 5: Agregação final (soma das saídas).

        Parâmetros:
            saidas: Saídas de cada regra

        Retorna:
            Saída final do ANFIS
        """
        return np.sum(saidas)

    def forward(self, X: np.ndarray) -> Tuple[float, Tuple]:
        """
        Propagação forward completa através das 5 camadas.

        Parâmetros:
            X: Vetor de entrada (n_inputs,)

        Retorna:
            Tupla (y_pred, cache) onde cache contém valores intermediários
        """
        mu = self.camada1_fuzzificacao(X)
        w = self.camada2_regras(mu)
        w_norm, soma_w = self.camada3_normalizacao(w)
        saidas = self.camada4_consequentes(X, w_norm)
        y_pred = self.camada5_agregacao(saidas)
        return y_pred, (mu, w, w_norm, soma_w)

    def forward_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Propagação forward vetorizada para múltiplas amostras.

        Parâmetros:
            X: Dados de entrada (n_samples, n_inputs)

        Retorna:
            Array com predições (n_samples,)
        """
        n_samples = X.shape[0]

        # Camada 1: Fuzzificação (vetorizada)
        # mu[i][j,k] = pertinência da amostra k na MF j da entrada i
        mu_batch = []
        for i in range(self.n_inputs):
            # Aplicar todas as MFs da entrada i a todas as amostras
            mu_i = np.array([self._eval_mf(X[:, i], params)
                           for params in self.params_mfs[i]])  # (n_mfs[i], n_samples)
            mu_batch.append(mu_i.T)  # (n_samples, n_mfs[i])

        # Camada 2: Força de disparo das regras (vetorizada)
        # w[k,j] = força da regra j para amostra k
        w_batch = np.ones((n_samples, self.n_regras))
        for regra_idx, indices_mf in enumerate(self._cache_indices_regras):
            for input_idx, mf_idx in enumerate(indices_mf):
                w_batch[:, regra_idx] *= mu_batch[input_idx][:, mf_idx]

        # Camada 3: Normalização (vetorizada)
        soma_w_batch = np.sum(w_batch, axis=1, keepdims=True) + 1e-10  # (n_samples, 1)
        w_norm_batch = w_batch / soma_w_batch  # (n_samples, n_regras)

        # Camada 4: Consequentes (vetorizada)
        # f_i = p0 + p1*x1 + p2*x2 + ... para cada regra
        X_extended = np.hstack([np.ones((n_samples, 1)), X])  # (n_samples, n_inputs+1)
        f_batch = X_extended @ self.params_consequentes.T  # (n_samples, n_regras)
        saidas_batch = w_norm_batch * f_batch  # (n_samples, n_regras)

        # Camada 5: Agregação (vetorizada)
        y_pred_batch = np.sum(saidas_batch, axis=1)  # (n_samples,)

        return y_pred_batch

    def _calcular_penalidade_l1(self) -> float:
        """
        Calcula penalidade L1 (Lasso) sobre larguras das MFs.

        Aplica regularização APENAS nas larguras (sigmas), não nos centros.
        """
        penalty = 0.0

        for input_idx in range(self.n_inputs):
            for mf_idx in range(self.n_mfs[input_idx]):
                params = self.params_mfs[input_idx][mf_idx]

                if self.mf_type == 'gaussmf':
                    # params = [centro, sigma] → regulariza apenas sigma
                    sigma = params[1]
                    penalty += np.abs(sigma)

                elif self.mf_type == 'gbellmf':
                    # params = [a, b, c] → regulariza a (width) e b (slope)
                    a, b = params[0], params[1]
                    penalty += np.abs(a) + np.abs(b)

                elif self.mf_type == 'sigmf':
                    # params = [a, c] → regulariza a (slope)
                    a = params[0]
                    penalty += np.abs(a)

        return penalty

    def _calcular_penalidade_l2(self) -> float:
        """
        Calcula penalidade L2 (Ridge) sobre larguras das MFs.

        Aplica regularização APENAS nas larguras (sigmas), não nos centros.
        """
        penalty = 0.0

        for input_idx in range(self.n_inputs):
            for mf_idx in range(self.n_mfs[input_idx]):
                params = self.params_mfs[input_idx][mf_idx]

                if self.mf_type == 'gaussmf':
                    # params = [centro, sigma] → regulariza apenas sigma
                    sigma = params[1]
                    penalty += sigma ** 2

                elif self.mf_type == 'gbellmf':
                    # params = [a, b, c] → regulariza a (width) e b (slope)
                    a, b = params[0], params[1]
                    penalty += a ** 2 + b ** 2

                elif self.mf_type == 'sigmf':
                    # params = [a, c] → regulariza a (slope)
                    a = params[0]
                    penalty += a ** 2

        return penalty

    def _calcular_custo_total(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
        """
        Calcula custo total = MSE + penalidades de regularização.

        Parâmetros:
            X: Dados de entrada
            y: Dados de saída

        Retorna:
            Tupla (mse, l1_penalty, l2_penalty)
        """
        y_pred = self.predict(X)
        mse = np.mean((y - y_pred) ** 2)

        l1_penalty = self._calcular_penalidade_l1()
        l2_penalty = self._calcular_penalidade_l2()

        return mse, l1_penalty, l2_penalty

    def _ajustar_consequentes_least_squares(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Ajusta parâmetros consequentes usando Least Squares (método híbrido).

        Este é o método mais eficiente para ajustar os consequentes,
        calculando a solução analítica ótima.

        Parâmetros:
            X: Dados de entrada (n_samples, n_inputs)
            y: Dados de saída (n_samples,)

        Retorna:
            RMSE após ajuste
        """
        n_samples = len(X)
        n_params = self.n_regras * (self.n_inputs + 1)

        # Construir matriz A do sistema linear
        A = np.zeros((n_samples, n_params))
        y_target = y.copy()

        for i in range(n_samples):
            _, (mu, w, w_norm, soma_w) = self.forward(X[i])
            for j in range(self.n_regras):
                start_idx = j * (self.n_inputs + 1)
                end_idx = start_idx + (self.n_inputs + 1)
                # [w_norm_j, w_norm_j*x1, w_norm_j*x2, ...]
                A[i, start_idx:end_idx] = w_norm[j] * np.concatenate([[1], X[i]])

        # Resolver com regularização de Tikhonov para estabilidade numérica
        try:
            lambda_reg = 1e-6
            ATA = A.T @ A + lambda_reg * np.eye(A.shape[1])
            ATy = A.T @ y_target
            p_flat = np.linalg.solve(ATA, ATy)
            self.params_consequentes = p_flat.reshape(self.n_regras, self.n_inputs + 1)
        except np.linalg.LinAlgError:
            warnings.warn("Erro ao resolver sistema linear para consequentes")

        # Calcular RMSE
        y_pred = self.predict(X)
        rmse = np.sqrt(np.mean((y_target - y_pred) ** 2))
        return rmse

    def _gradiente_l1(self, parametro: float) -> float:
        """Subgradiente da penalidade L1."""
        if parametro > 0:
            return 1.0
        elif parametro < 0:
            return -1.0
        else:
            return 0.0

    def _gradiente_l2(self, parametro: float) -> float:
        """Gradiente da penalidade L2."""
        return 2.0 * parametro

    def _compute_adaptive_learning_rate(self, gradient: np.ndarray, max_lr: float = 0.01) -> float:
        """
        Calcula taxa de aprendizado adaptativa baseada em estabilidade de Lyapunov.

        A teoria de estabilidade de Lyapunov garante que o algoritmo converge
        se a taxa de aprendizado satisfaz: 0 < η < 2/||∇E||²

        Para garantir estabilidade, usamos: η = 1.99/||∇E||²
        limitado por um valor máximo para evitar passos muito grandes.

        Parâmetros:
            gradient: Vetor de gradientes
            max_lr: Taxa de aprendizado máxima permitida

        Retorna:
            Taxa de aprendizado adaptativa que garante estabilidade

        Referência:
            Jang, J. S. (1993). "ANFIS: adaptive-network-based fuzzy inference system."
            IEEE transactions on systems, man, and cybernetics, 23(3), 665-685.
        """
        grad_norm_squared = np.sum(gradient ** 2)

        if grad_norm_squared < 1e-10:
            # Gradiente muito pequeno → usar lr máximo
            return max_lr

        # Critério de estabilidade de Lyapunov: η < 2/||∇E||²
        # Usamos 1.99 para margem de segurança
        stable_lr = 1.99 / grad_norm_squared

        # Limitar pelo máximo especificado
        return min(stable_lr, max_lr)


    def _ajustar_antecedentes_gradiente(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """
        Ajusta parâmetros antecedentes usando gradiente descendente com regularização.

        Calcula gradientes analiticamente e aplica regularização L1/L2.

        Parâmetros:
            X: Dados de entrada (n_samples, n_inputs)
            y: Dados de saída (n_samples,)

        Retorna:
            Tupla (norma do gradiente total, learning rate efetivo)
        """
        n_samples = len(X)

        # Acumuladores de gradiente
        grad_acumulado = {}
        for input_idx in range(self.n_inputs):
            grad_acumulado[input_idx] = {}
            for mf_idx in range(self.n_mfs[input_idx]):
                n_params = len(self.params_mfs[input_idx][mf_idx])
                grad_acumulado[input_idx][mf_idx] = np.zeros(n_params)

        # Acumular gradientes de todas as amostras do batch
        for sample_idx in range(n_samples):
            X_sample = X[sample_idx]
            target = y[sample_idx]

            # Forward pass
            y_pred, (mu, w, w_norm, soma_w) = self.forward(X_sample)
            erro = target - y_pred

            # Contribuições dos consequentes
            contribuicoes = np.zeros(self.n_regras)
            for j in range(self.n_regras):
                params = self.params_consequentes[j]
                contribuicoes[j] = params[0] + np.dot(params[1:], X_sample)

            # Calcular gradientes para cada MF
            for input_idx in range(self.n_inputs):
                x_val = X_sample[input_idx]

                for regra_idx, indices_mf in enumerate(self._cache_indices_regras):
                    mf_idx = indices_mf[input_idx]
                    params = self.params_mfs[input_idx][mf_idx]

                    # Gradientes dependem do tipo de MF
                    mu_val = mu[input_idx][mf_idx]

                    # Componentes comuns da regra da cadeia
                    dw_dmu = w[regra_idx] / (mu_val + 1e-10)
                    dy_dwn = contribuicoes[regra_idx]
                    dwn_dw = (1 / soma_w) if soma_w > 1e-10 else 0

                    # Fator comum: -erro * dy/dwn * dwn/dw * dw/dmu
                    chain_common = -erro * dy_dwn * dwn_dw * dw_dmu

                    if self.mf_type == 'gaussmf':
                        # μ(x) = exp(-(x-c)²/(2σ²))
                        # Parâmetros: [centro, sigma]
                        centro, sigma = params

                        dmu_dc = mu_val * (x_val - centro) / (sigma ** 2)
                        dmu_ds = mu_val * ((x_val - centro) ** 2) / (sigma ** 3)

                        grad_acumulado[input_idx][mf_idx][0] += chain_common * dmu_dc
                        grad_acumulado[input_idx][mf_idx][1] += chain_common * dmu_ds

                    elif self.mf_type == 'gbellmf':
                        # μ(x) = 1 / (1 + |((x-c)/a)|^(2b))
                        # Parâmetros: [a (width), b (slope), c (center)]
                        a, b, c = params

                        diff = x_val - c
                        abs_ratio = np.abs(diff / (a + 1e-10))
                        denominator = 1 + abs_ratio ** (2 * b)

                        if abs_ratio > 1e-10:  # Evitar divisão por zero
                            # ∂μ/∂a
                            dmu_da = (2 * b * mu_val ** 2 * abs_ratio ** (2 * b)) / (a + 1e-10)

                            # ∂μ/∂b
                            log_ratio = np.log(abs_ratio + 1e-10)
                            dmu_db = -2 * mu_val ** 2 * abs_ratio ** (2 * b) * log_ratio

                            # ∂μ/∂c
                            sign_diff = np.sign(diff)
                            dmu_dc = -2 * b * mu_val ** 2 * abs_ratio ** (2 * b) * sign_diff / (a + 1e-10)

                            grad_acumulado[input_idx][mf_idx][0] += chain_common * dmu_da
                            grad_acumulado[input_idx][mf_idx][1] += chain_common * dmu_db
                            grad_acumulado[input_idx][mf_idx][2] += chain_common * dmu_dc

                    elif self.mf_type == 'sigmf':
                        # μ(x) = 1 / (1 + exp(-a·(x-c)))
                        # Parâmetros: [a (slope), c (center)]
                        a, c = params

                        # ∂μ/∂a = μ·(1-μ)·(x-c)
                        dmu_da = mu_val * (1 - mu_val) * (x_val - c)

                        # ∂μ/∂c = -a·μ·(1-μ)
                        dmu_dc = -a * mu_val * (1 - mu_val)

                        grad_acumulado[input_idx][mf_idx][0] += chain_common * dmu_da
                        grad_acumulado[input_idx][mf_idx][1] += chain_common * dmu_dc

        # Atualizar parâmetros usando gradientes médios do batch
        grad_norm_total = 0
        all_gradients = []  # Para calcular learning rate adaptativo

        for input_idx in range(self.n_inputs):
            for mf_idx in range(self.n_mfs[input_idx]):
                params = self.params_mfs[input_idx][mf_idx]
                n_params = len(params)

                for param_idx in range(n_params):
                    param_val = params[param_idx]

                    # Gradiente médio do MSE
                    grad_mse = grad_acumulado[input_idx][mf_idx][param_idx] / n_samples

                    # Determinar se este parâmetro é uma largura (deve ser regularizado)
                    # ou um centro (NÃO deve ser regularizado)
                    is_largura = False

                    if self.mf_type == 'gaussmf':
                        # [centro, sigma] → param_idx=1 é sigma (largura)
                        is_largura = (param_idx == 1)
                    elif self.mf_type == 'gbellmf':
                        # [a, b, c] → param_idx=0,1 são larguras, param_idx=2 é centro
                        is_largura = (param_idx in [0, 1])
                    elif self.mf_type == 'sigmf':
                        # [a, c] → param_idx=0 é largura, param_idx=1 é centro
                        is_largura = (param_idx == 0)

                    # Aplicar regularização APENAS nas larguras
                    if is_largura:
                        grad_l1 = self.lambda_l1 * self._gradiente_l1(param_val)
                        grad_l2 = self.lambda_l2 * self._gradiente_l2(param_val)
                        grad_total = grad_mse + grad_l1 + grad_l2
                    else:
                        # Centros: sem regularização
                        grad_total = grad_mse

                    all_gradients.append(grad_total)
                    grad_norm_total += grad_total ** 2

        # Calcular learning rate (fixo ou adaptativo)
        if self.use_adaptive_lr:
            # Taxa adaptativa baseada em Lyapunov
            lr_efetivo = self._compute_adaptive_learning_rate(
                np.array(all_gradients), max_lr=self.lr
            )
        else:
            # Taxa fixa
            lr_efetivo = self.lr

        # Aplicar atualização com learning rate calculado
        grad_idx = 0
        for input_idx in range(self.n_inputs):
            for mf_idx in range(self.n_mfs[input_idx]):
                n_params = len(self.params_mfs[input_idx][mf_idx])
                for param_idx in range(n_params):
                    grad_total = all_gradients[grad_idx]
                    self.params_mfs[input_idx][mf_idx][param_idx] -= lr_efetivo * grad_total
                    grad_idx += 1

        # Aplicar restrições de domínio
        self._apply_domain_constraints()

        return np.sqrt(grad_norm_total), lr_efetivo

    def _validar_entrada(self, X: np.ndarray, y: Optional[np.ndarray] = None,
                        nome_X: str = "X", nome_y: str = "y") -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Valida dados de entrada de forma robusta.

        Parâmetros:
            X: Dados de entrada
            y: Dados de saída (opcional)
            nome_X: Nome da variável X para mensagens de erro
            nome_y: Nome da variável y para mensagens de erro

        Retorna:
            Tupla (X validado, y validado)

        Levanta:
            TypeError: Se tipos não são numpy arrays
            ValueError: Se há valores inválidos ou dimensões incorretas
        """
        # Validar tipo de X
        if not isinstance(X, np.ndarray):
            raise TypeError(f"{nome_X} deve ser numpy.ndarray, recebeu {type(X)}")

        # Validar dimensões de X
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim != 2:
            raise ValueError(f"{nome_X} deve ter 1 ou 2 dimensões, recebeu {X.ndim}")

        # Validar número de features
        if hasattr(self, 'n_inputs'):
            if X.shape[1] != self.n_inputs:
                raise ValueError(
                    f"{nome_X} deve ter {self.n_inputs} colunas, recebeu {X.shape[1]}"
                )

        # Validar valores NaN/Inf
        if np.any(np.isnan(X)):
            raise ValueError(f"{nome_X} contém valores NaN")
        if np.any(np.isinf(X)):
            raise ValueError(f"{nome_X} contém valores Inf")

        # Validar y se fornecido
        if y is not None:
            if not isinstance(y, np.ndarray):
                raise TypeError(f"{nome_y} deve ser numpy.ndarray, recebeu {type(y)}")

            # Aceitar 1D ou 2D
            if y.ndim == 2:
                if y.shape[1] == 1:
                    y = y.ravel()
                else:
                    raise ValueError(f"{nome_y} deve ter 1 coluna, recebeu {y.shape[1]}")
            elif y.ndim != 1:
                raise ValueError(f"{nome_y} deve ter 1 ou 2 dimensões, recebeu {y.ndim}")

            # Validar comprimento compatível
            if y.shape[0] != X.shape[0]:
                raise ValueError(
                    f"{nome_X} e {nome_y} devem ter mesmo número de amostras. "
                    f"{nome_X}: {X.shape[0]}, {nome_y}: {y.shape[0]}"
                )

            # Validar valores NaN/Inf
            if np.any(np.isnan(y)):
                raise ValueError(f"{nome_y} contém valores NaN")
            if np.any(np.isinf(y)):
                raise ValueError(f"{nome_y} contém valores Inf")

        return X, y

    def _calcular_metricas(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Calcula métricas de desempenho.

        Parâmetros:
            X: Dados de entrada
            y: Dados de saída

        Retorna:
            Dicionário com RMSE, MAE, erro máximo, R² e MAPE
        """
        y_pred = self.predict(X)
        erros = y - y_pred

        # Métricas básicas
        mse = np.mean(erros ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(erros))
        max_error = np.max(np.abs(erros))

        # R² (coeficiente de determinação)
        ss_res = np.sum(erros ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-10))

        # MAPE (Mean Absolute Percentage Error)
        # Evita divisão por zero
        mape = np.mean(np.abs(erros / (y + 1e-10))) * 100

        return {
            'rmse': rmse,
            'mae': mae,
            'max_error': max_error,
            'r2': r2,
            'mape': mape
        }

    def fit(self, X: np.ndarray, y: np.ndarray,
            epochs: int = 100,
            verbose: bool = True,
            treinar_antecedentes: bool = True,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            early_stopping_patience: int = 20):
        """
        Treina o modelo ANFIS usando aprendizado híbrido.

        Algoritmo:
        1. Inicializa parâmetros baseado nos dados
        2. Para cada época:
           a. Divide dados em batches (se minibatch)
           b. Para cada batch:
              - Ajusta consequentes com LSE
              - Ajusta antecedentes com gradiente (se habilitado)
           c. Calcula métricas no conjunto completo
           d. Verifica early stopping

        Parâmetros:
            X: Dados de entrada (n_samples, n_inputs)
            y: Dados de saída (n_samples,)
            epochs: Número de épocas de treinamento
            verbose: Se True, exibe progresso
            treinar_antecedentes: Se True, treina também os antecedentes (MFs)
            X_val: Dados de validação (opcional)
            y_val: Saídas de validação (opcional)
            early_stopping_patience: Número de épocas sem melhora antes de parar

        Retorna:
            self (para encadeamento)
        """
        # Validar entradas
        X, y = self._validar_entrada(X, y, "X", "y")

        # Validar dados de validação se fornecidos
        if X_val is not None and y_val is not None:
            X_val, y_val = self._validar_entrada(X_val, y_val, "X_val", "y_val")

        # Inicializar parâmetros
        self._initialize_premise_params(X)
        self.params_consequentes = np.random.randn(self.n_regras, self.n_inputs + 1) * 0.1
        self._cache_indices_regras = self._gerar_indices_regras()

        # Reset métricas
        self.metricas = MetricasANFIS()
        self.historico_l1 = []
        self.historico_l2 = []
        self.historico_custo_total = []

        # Early stopping
        melhor_rmse_val = np.inf
        epochs_sem_melhora = 0
        melhor_estado = None

        # Determinar batch_size efetivo
        n_samples = len(X)
        batch_size_efetivo = self.batch_size if self.batch_size is not None else n_samples
        batch_size_efetivo = min(batch_size_efetivo, n_samples)
        n_batches = int(np.ceil(n_samples / batch_size_efetivo))

        if verbose:
            print(f"\n{'='*70}")
            print(f"ANFIS - Treinamento")
            print(f"{'='*70}")
            print(f"Arquitetura:")
            print(f"  • Entradas: {self.n_inputs}")
            print(f"  • MFs por entrada: {self.n_mfs}")
            print(f"  • Regras: {self.n_regras}")
            print(f"  • Tipo MF: {self.mf_type}")
            print(f"\nTreinamento:")
            print(f"  • Modo: {'Minibatch' if batch_size_efetivo < n_samples else 'Batch'} GD")
            print(f"  • Samples: {n_samples}")
            print(f"  • Batch size: {batch_size_efetivo}")
            print(f"  • Batches/época: {n_batches}")
            print(f"\nRegularização:")
            print(f"  • Tipo: {self._get_tipo_reg()}")
            print(f"  • λ₁ (L1): {self.lambda_l1}")
            print(f"  • λ₂ (L2): {self.lambda_l2}")
            print(f"{'='*70}\n")

        # Inicialização dos consequentes
        if verbose:
            print("Inicializando consequentes com LSE...")
        rmse_inicial = self._ajustar_consequentes_least_squares(X, y)
        if verbose:
            print(f"RMSE inicial: {rmse_inicial:.6f}\n")

        # Loop de treinamento
        for epoch in range(epochs):
            start_time = time.time()

            # Criar batches
            usar_shuffle = (batch_size_efetivo < n_samples)
            batches = self._criar_batches(X, y, batch_size_efetivo, shuffle=usar_shuffle)

            # Processar cada batch
            grad_norm = 0
            lr_efetivo = self.lr  # Learning rate efetivo (pode ser adaptativo)
            for X_batch, y_batch in batches:
                # Fase 1: Ajustar consequentes
                self._ajustar_consequentes_least_squares(X_batch, y_batch)

                # Fase 2: Ajustar antecedentes
                if treinar_antecedentes:
                    grad_norm, lr_efetivo = self._ajustar_antecedentes_gradiente(X_batch, y_batch)

            # Calcular métricas no conjunto completo
            mse, l1_penalty, l2_penalty = self._calcular_custo_total(X, y)
            custo_total = mse + self.lambda_l1 * l1_penalty + self.lambda_l2 * l2_penalty

            self.historico_l1.append(l1_penalty)
            self.historico_l2.append(l2_penalty)
            self.historico_custo_total.append(custo_total)

            metricas_train = self._calcular_metricas(X, y)
            metricas_val = {}

            # Validação
            if X_val is not None and y_val is not None:
                metricas_val = self._calcular_metricas(X_val, y_val)

                if metricas_val['rmse'] < melhor_rmse_val:
                    melhor_rmse_val = metricas_val['rmse']
                    epochs_sem_melhora = 0
                    melhor_estado = {
                        'params_mfs': [p.copy() for p in self.params_mfs],
                        'params_consequentes': self.params_consequentes.copy()
                    }
                else:
                    epochs_sem_melhora += 1

                if epochs_sem_melhora >= early_stopping_patience:
                    if verbose:
                        print(f"\nEarly stopping na época {epoch+1}")
                        print(f"Melhor RMSE validação: {melhor_rmse_val:.6f}")
                    if melhor_estado is not None:
                        self.params_mfs = melhor_estado['params_mfs']
                        self.params_consequentes = melhor_estado['params_consequentes']
                    break

            epoch_time = time.time() - start_time

            # Armazenar métricas
            self.metricas.adicionar_epoca({
                'rmse_train': metricas_train['rmse'],
                'rmse_val': metricas_val.get('rmse', np.nan),
                'mae_train': metricas_train['mae'],
                'mae_val': metricas_val.get('mae', np.nan),
                'max_error_train': metricas_train['max_error'],
                'max_error_val': metricas_val.get('max_error', np.nan),
                'r2_train': metricas_train['r2'],
                'r2_val': metricas_val.get('r2', np.nan),
                'mape_train': metricas_train['mape'],
                'mape_val': metricas_val.get('mape', np.nan),
                'time': epoch_time,
                'grad_norm': grad_norm if treinar_antecedentes else 0,
                'learning_rate': lr_efetivo
            })

            # Exibir progresso
            if verbose and ((epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1):
                msg = f"Época {epoch+1:3d}/{epochs} - "
                msg += f"Train RMSE: {metricas_train['rmse']:.6f}"
                if X_val is not None:
                    msg += f", Val RMSE: {metricas_val['rmse']:.6f}"
                msg += f" | Custo: {custo_total:.6f}"
                print(msg)

        if verbose:
            print(f"\n{'='*70}")
            print("Treinamento concluído!")
            print(f"{'='*70}\n")

        return self

    def fit_metaheuristic(self, X: np.ndarray, y: np.ndarray,
                          optimizer: str = 'pso',
                          n_particles: int = 30,
                          n_iterations: int = 100,
                          verbose: bool = True,
                          **optimizer_kwargs) -> 'ANFIS':
        """
        Treina o ANFIS usando otimização metaheurística global.

        Diferente do fit() tradicional (LSE + Gradiente), este método usa
        algoritmos metaheurísticos (PSO, DE, GA) para otimizar TODOS os
        parâmetros (antecedentes + consequentes) simultaneamente.

        Vantagens:
        - Otimização global (evita mínimos locais)
        - Não requer gradientes (funciona com qualquer MF)
        - Robusto a configurações iniciais

        Desvantagens:
        - Mais lento que fit() tradicional
        - Requer mais iterações para convergir

        Parâmetros:
            X: Dados de entrada (n_samples, n_inputs)
            y: Dados de saída (n_samples,)
            optimizer: Tipo de otimizador ('pso', 'de', 'ga')
            n_particles: Tamanho da população/enxame
            n_iterations: Número de iterações
            verbose: Exibe progresso
            **optimizer_kwargs: Parâmetros específicos do otimizador
                Para PSO: w_max, w_min, c1, c2
                Para DE: F, CR
                Para GA: elite_ratio, mutation_rate, tournament_size

        Retorna:
            self (para encadeamento)

        Exemplo:
            >>> # PSO (recomendado para maioria dos casos)
            >>> anfis.fit_metaheuristic(X, y, optimizer='pso',
            ...                          n_particles=30, n_iterations=100)
            >>>
            >>> # DE (bom para espaços complexos)
            >>> anfis.fit_metaheuristic(X, y, optimizer='de',
            ...                          n_particles=50, n_iterations=150,
            ...                          F=0.8, CR=0.9)
            >>>
            >>> # GA (bom para exploração ampla)
            >>> anfis.fit_metaheuristic(X, y, optimizer='ga',
            ...                          n_particles=50, n_iterations=100,
            ...                          elite_ratio=0.1, mutation_rate=0.1)
        """
        from .metaheuristics import get_optimizer

        # Validar entradas
        X, y = self._validar_entrada(X, y, "X", "y")

        # Inicializar parâmetros
        self._initialize_premise_params(X)
        self.params_consequentes = np.random.randn(self.n_regras, self.n_inputs + 1) * 0.1
        self._cache_indices_regras = self._gerar_indices_regras()

        # Reset métricas
        self.metricas = MetricasANFIS()

        # Converter parâmetros para vetor
        param_vector = self._params_to_vector()
        bounds = self._create_optimization_bounds(X)

        # Função objetivo
        def objective(params_vec):
            try:
                self._vector_to_params(params_vec)
                y_pred = self.forward_batch(X)
                mse = np.mean((y - y_pred) ** 2)
                return mse
            except:
                return 1e10  # Penalidade para parâmetros inválidos

        # Criar otimizador
        if optimizer.lower() == 'pso':
            opt_params = {'n_particles': n_particles, 'n_iterations': n_iterations}
        elif optimizer.lower() in ['de', 'ga']:
            opt_params = {'pop_size': n_particles, 'max_iter': n_iterations}
            if optimizer.lower() == 'ga':
                opt_params['max_gen'] = opt_params.pop('max_iter')
        else:
            raise ValueError(f"Otimizador '{optimizer}' desconhecido. Use 'pso', 'de' ou 'ga'")

        opt_params.update(optimizer_kwargs)
        opt = get_optimizer(optimizer, **opt_params)

        if verbose:
            print(f"\n{'='*70}")
            print(f"ANFIS - Treinamento com Otimização Metaheurística ({optimizer.upper()})")
            print(f"{'='*70}")
            print(f"Arquitetura:")
            print(f"  • Entradas: {self.n_inputs}")
            print(f"  • MFs por entrada: {self.n_mfs}")
            print(f"  • Regras: {self.n_regras}")
            print(f"  • Tipo MF: {self.mf_type}")
            print(f"\nOtimização:")
            print(f"  • Algoritmo: {optimizer.upper()}")
            print(f"  • População: {n_particles}")
            print(f"  • Iterações: {n_iterations}")
            print(f"  • Parâmetros totais: {len(param_vector)}")
            print(f"    - Antecedentes: {sum([len(self.params_mfs[i]) * len(self.params_mfs[i][0]) for i in range(self.n_inputs)])}")
            print(f"    - Consequentes: {self.n_regras * (self.n_inputs + 1)}")
            print(f"{'='*70}\n")

        # Otimizar
        best_params, best_fitness, history = opt.optimize(
            objective, bounds, minimize=True, verbose=verbose
        )

        # Atualizar com melhores parâmetros
        self._vector_to_params(best_params)

        # Calcular métricas finais
        metricas_finais = self._calcular_metricas(X, y)

        if verbose:
            print(f"\n{'='*70}")
            print("Otimização concluída!")
            print(f"  • MSE final: {best_fitness:.6f}")
            print(f"  • RMSE: {metricas_finais['rmse']:.6f}")
            print(f"  • R²: {metricas_finais['r2']:.4f}")
            print(f"  • MAPE: {metricas_finais['mape']:.2f}%")
            print(f"{'='*70}\n")

        # Armazenar histórico de otimização
        self.optimization_history = history

        return self

    def _params_to_vector(self) -> np.ndarray:
        """Converte parâmetros ANFIS para vetor 1D"""
        vector = []

        # Parâmetros antecedentes
        for i in range(self.n_inputs):
            for j in range(self.n_mfs[i]):
                params = self.params_mfs[i][j]
                vector.extend(params)

        # Parâmetros consequentes
        vector.extend(self.params_consequentes.flatten())

        return np.array(vector)

    def _vector_to_params(self, vector: np.ndarray):
        """Converte vetor 1D para parâmetros ANFIS"""
        idx = 0

        # Parâmetros antecedentes
        for i in range(self.n_inputs):
            for j in range(self.n_mfs[i]):
                n_params = len(self.params_mfs[i][j])
                self.params_mfs[i][j] = vector[idx:idx+n_params].copy()
                idx += n_params

        # Parâmetros consequentes
        n_conseq = self.n_regras * (self.n_inputs + 1)
        self.params_consequentes = vector[idx:idx+n_conseq].reshape(
            self.n_regras, self.n_inputs + 1
        )

    def _create_optimization_bounds(self, X: np.ndarray) -> np.ndarray:
        """Cria limites para otimização metaheurística"""
        bounds = []

        # Bounds para parâmetros antecedentes
        for i in range(self.n_inputs):
            x_min, x_max = self.input_bounds[i]
            x_range = x_max - x_min

            for j in range(self.n_mfs[i]):
                if self.mf_type == 'gaussmf':
                    # [centro, sigma]
                    bounds.append([x_min, x_max])  # centro
                    bounds.append([x_range * 0.05, x_range * 2.0])  # sigma

                elif self.mf_type == 'gbellmf':
                    # [a, b, c]
                    bounds.append([x_range * 0.05, x_range * 2.0])  # a (width)
                    bounds.append([0.5, 5.0])  # b (slope)
                    bounds.append([x_min, x_max])  # c (center)

                elif self.mf_type == 'sigmf':
                    # [a, c]
                    bounds.append([-10.0, 10.0])  # a (slope)
                    bounds.append([x_min, x_max])  # c (center)

        # Bounds para parâmetros consequentes
        for _ in range(self.n_regras * (self.n_inputs + 1)):
            bounds.append([-10.0, 10.0])

        return np.array(bounds)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Realiza predições para novos dados de forma vetorizada.

        Parâmetros:
            X: Dados de entrada (n_samples, n_inputs) ou (n_inputs,)

        Retorna:
            Array com predições (n_samples,) ou escalar se X for 1D
        """
        # Validar entrada
        entrada_1d = (X.ndim == 1)
        X, _ = self._validar_entrada(X, None, "X", "y")

        # Predição vetorizada
        predicoes = self.forward_batch(X)

        # Retornar escalar se entrada era 1D
        return predicoes[0] if entrada_1d else predicoes

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Retorna o coeficiente de determinação R² da predição.

        Método compatível com a API do scikit-learn.

        Parâmetros:
            X: Dados de entrada (n_samples, n_inputs)
            y: Valores verdadeiros (n_samples,)

        Retorna:
            R² score (melhor valor = 1.0, pode ser negativo se modelo for pior que baseline)
        """
        X, y = self._validar_entrada(X, y, "X", "y")
        metricas = self._calcular_metricas(X, y)
        return metricas['r2']

    def save(self, filepath: str):
        """
        Salva o modelo treinado em arquivo.

        Parâmetros:
            filepath: Caminho do arquivo (será adicionado .npz se não tiver extensão)
        """
        import os

        # Adicionar extensão se não tiver
        if not filepath.endswith('.npz'):
            filepath = filepath + '.npz'

        # Preparar dados para salvar
        save_dict = {
            # Arquitetura
            'n_inputs': self.n_inputs,
            'n_mfs': np.array(self.n_mfs),
            'n_regras': self.n_regras,
            'mf_type': self.mf_type,

            # Hiperparâmetros
            'learning_rate': self.lr,
            'lambda_l1': self.lambda_l1,
            'lambda_l2': self.lambda_l2,
            'batch_size': self.batch_size if self.batch_size is not None else -1,
            'use_adaptive_lr': self.use_adaptive_lr,

            # Parâmetros treinados
            'params_consequentes': self.params_consequentes,
            'input_bounds': self.input_bounds,

            # Índices de regras
            'indices_regras': np.array([list(idx) for idx in self._cache_indices_regras])
        }

        # Salvar params_mfs (lista de arrays com tamanhos diferentes)
        for i in range(self.n_inputs):
            save_dict[f'params_mfs_{i}'] = self.params_mfs[i]

        # Salvar
        np.savez_compressed(filepath, **save_dict)
        print(f"Modelo salvo em: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'ANFIS':
        """
        Carrega modelo treinado de arquivo.

        Parâmetros:
            filepath: Caminho do arquivo

        Retorna:
            Instância de ANFIS carregada
        """
        import os

        # Adicionar extensão se não tiver
        if not filepath.endswith('.npz'):
            filepath = filepath + '.npz'

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")

        # Carregar dados
        data = np.load(filepath, allow_pickle=True)

        # Reconstruir n_mfs
        n_mfs = data['n_mfs'].tolist()
        if isinstance(n_mfs, int):
            n_mfs = [n_mfs]

        # Criar instância
        batch_size = int(data['batch_size'])
        if batch_size == -1:
            batch_size = None

        modelo = cls(
            n_inputs=int(data['n_inputs']),
            n_mfs=n_mfs,
            mf_type=str(data['mf_type']),
            learning_rate=float(data['learning_rate']),
            lambda_l1=float(data['lambda_l1']),
            lambda_l2=float(data['lambda_l2']),
            batch_size=batch_size,
            use_adaptive_lr=bool(data['use_adaptive_lr'])
        )

        # Restaurar parâmetros treinados
        modelo.params_consequentes = data['params_consequentes']
        modelo.input_bounds = data['input_bounds']

        # Restaurar params_mfs
        modelo.params_mfs = []
        for i in range(modelo.n_inputs):
            modelo.params_mfs.append(data[f'params_mfs_{i}'])

        # Restaurar índices de regras
        indices_array = data['indices_regras']
        modelo._cache_indices_regras = [tuple(row) for row in indices_array]

        print(f"Modelo carregado de: {filepath}")
        return modelo

    def visualizar_mfs(self, figsize_por_input=(6, 4)):
        """
        Visualiza as funções de pertinência aprendidas.

        Parâmetros:
            figsize_por_input: Tamanho de cada subplot

        Retorna:
            Figure do matplotlib
        """
        n_cols = min(3, self.n_inputs)
        n_rows = int(np.ceil(self.n_inputs / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols,
                                figsize=(figsize_por_input[0]*n_cols,
                                        figsize_por_input[1]*n_rows))

        if self.n_inputs == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for input_idx in range(self.n_inputs):
            ax = axes[input_idx]
            x_min, x_max = self.input_bounds[input_idx]
            x_range = np.linspace(x_min, x_max, 200)

            for mf_idx, params in enumerate(self.params_mfs[input_idx]):
                mu = self._eval_mf(x_range, params)

                if self.mf_type == 'gaussmf':
                    label = f'MF_{mf_idx+1} (μ={params[0]:.2f}, σ={params[1]:.2f})'
                elif self.mf_type == 'gbellmf':
                    label = f'MF_{mf_idx+1} (a={params[0]:.2f}, b={params[1]:.2f}, c={params[2]:.2f})'
                elif self.mf_type == 'sigmf':
                    label = f'MF_{mf_idx+1} (a={params[0]:.2f}, c={params[1]:.2f})'

                ax.plot(x_range, mu, linewidth=2, label=label)

            ax.set_xlabel(f'Entrada {input_idx+1}', fontsize=12)
            ax.set_ylabel('Pertinência', fontsize=12)
            ax.set_title(f'MFs - Entrada {input_idx+1} ({self.mf_type})',
                        fontsize=13, weight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([-0.05, 1.1])

        for idx in range(self.n_inputs, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        return fig

    def plotar_regularizacao(self, figsize=(16, 5)):
        """
        Plota evolução das penalidades de regularização.

        Parâmetros:
            figsize: Tamanho da figura

        Retorna:
            Figure do matplotlib
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        epochs = np.arange(1, len(self.historico_custo_total) + 1)

        # Custo total
        ax = axes[0]
        ax.plot(epochs, self.historico_custo_total, 'b-', linewidth=2)
        ax.set_xlabel('Época', fontsize=12)
        ax.set_ylabel('Custo Total', fontsize=12)
        ax.set_title('J = MSE + λ₁·L1 + λ₂·L2', fontsize=13, weight='bold')
        ax.grid(True, alpha=0.3)

        # Penalidade L1
        ax = axes[1]
        ax.plot(epochs, self.historico_l1, 'r-', linewidth=2)
        ax.set_xlabel('Época', fontsize=12)
        ax.set_ylabel('Penalidade L1', fontsize=12)
        ax.set_title(f'||θ||₁ (λ₁={self.lambda_l1})', fontsize=13, weight='bold')
        ax.grid(True, alpha=0.3)

        # Penalidade L2
        ax = axes[2]
        ax.plot(epochs, self.historico_l2, 'g-', linewidth=2)
        ax.set_xlabel('Época', fontsize=12)
        ax.set_ylabel('Penalidade L2', fontsize=12)
        ax.set_title(f'||θ||² (λ₂={self.lambda_l2})', fontsize=13, weight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def _get_tipo_reg(self) -> str:
        """Retorna descrição do tipo de regularização usado."""
        if self.lambda_l1 > 0 and self.lambda_l2 > 0:
            return "Elastic Net (L1 + L2)"
        elif self.lambda_l1 > 0:
            return "Lasso (L1)"
        elif self.lambda_l2 > 0:
            return "Ridge (L2)"
        else:
            return "Sem regularização"

    def resumo(self):
        """Exibe resumo da arquitetura e configuração do modelo."""
        n_params = self.params_consequentes.size + sum(p.size for p in self.params_mfs)
        batch_size_str = str(self.batch_size) if self.batch_size is not None else "Full batch"

        print("=" * 70)
        print("ANFIS MODERNO - Resumo do Modelo")
        print("=" * 70)
        print(f"Arquitetura:")
        print(f"  • Entradas:              {self.n_inputs}")
        print(f"  • MFs por entrada:       {self.n_mfs}")
        print(f"  • Total de regras:       {self.n_regras}")
        print(f"  • Tipo de MF:            {self.mf_type}")
        print(f"  • Total de parâmetros:   {n_params}")
        print(f"\nTreinamento:")
        print(f"  • Batch size:            {batch_size_str}")
        print(f"  • Learning rate:         {self.lr}")
        print(f"\nRegularização (aplicada apenas em larguras):")
        print(f"  • Tipo:                  {self._get_tipo_reg()}")
        print(f"  • λ₁ (L1):               {self.lambda_l1}")
        print(f"  • λ₂ (L2):               {self.lambda_l2}")
        print(f"  • Centros:               Livres (não regularizados)")
        print(f"  • Larguras:              Regularizadas")
        print("=" * 70)


# ================ EXEMPLO DE USO ================

if __name__ == "__main__":
    # Gera dados sintéticos
    np.random.seed(42)

    n_samples = 300
    X_train = np.random.uniform(-3, 3, (n_samples, 2))
    y_train = (np.sin(X_train[:, 0]) +
               np.cos(X_train[:, 1]) +
               0.5 * X_train[:, 0] * X_train[:, 1] +
               np.random.normal(0, 0.1, n_samples))

    # Dividir em treino e validação
    split_idx = int(0.8 * n_samples)
    X_tr, X_val = X_train[:split_idx], X_train[split_idx:]
    y_tr, y_val = y_train[:split_idx], y_train[split_idx:]

    # Criar e treinar modelo
    print("=" * 70)
    print("ANFIS Moderno - Exemplo de Uso")
    print("=" * 70)

    model = ANFIS(n_inputs=2, n_mfs=[3, 3], mf_type='gaussmf',
                  learning_rate=0.01, lambda_l2=0.01, batch_size=32)

    model.resumo()

    model.fit(X_tr, y_tr, epochs=50, verbose=True,
              X_val=X_val, y_val=y_val, early_stopping_patience=10)

    # Avalia
    y_pred = model.predict(X_val)
    mse = np.mean((y_val - y_pred) ** 2)
    r2 = 1 - (np.sum((y_val - y_pred) ** 2) / np.sum((y_val - np.mean(y_val)) ** 2))

    print(f"\n{'=' * 70}")
    print(f"Resultados Finais (Validação):")
    print(f"  • MSE:  {mse:.6f}")
    print(f"  • RMSE: {np.sqrt(mse):.6f}")
    print(f"  • R²:   {r2:.6f}")
    print(f"{'=' * 70}")

    # Visualizar
    model.visualizar_mfs()
    model.metricas.plotar_convergencia()
    model.plotar_regularizacao()
    plt.show()
