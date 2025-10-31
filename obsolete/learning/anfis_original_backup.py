"""
ANFIS com Regularização e Suporte a Minibatch Training

Implementação de ANFIS com:
- Regularização L1/L2
- Minibatch gradient descent
- Shuffle de dados entre épocas

Minibatch Training:
- Divide dados em batches pequenos
- Múltiplas atualizações por época
- Convergência mais rápida
- Melhor generalização (ruído estocástico)

Comparação de métodos:
1. Batch GD:         batch_size = len(X_train)  (1 atualização/época)
2. Minibatch GD:     batch_size = 16-128        (múltiplas atualizações/época)
3. Stochastic GD:    batch_size = 1             (máximo atualizações/época)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import warnings


class MetricasANFIS:
    """Classe para armazenar e visualizar métricas de treinamento"""

    def __init__(self):
        self.rmse_train = []
        self.rmse_val = []
        self.mae_train = []
        self.mae_val = []
        self.max_error_train = []
        self.max_error_val = []
        self.epoch_times = []
        self.gradientes_antecedentes = []

    def adicionar_epoca(self, epoch_metrics: Dict):
        """Adiciona métricas de uma época"""
        self.rmse_train.append(epoch_metrics.get('rmse_train', np.nan))
        self.rmse_val.append(epoch_metrics.get('rmse_val', np.nan))
        self.mae_train.append(epoch_metrics.get('mae_train', np.nan))
        self.mae_val.append(epoch_metrics.get('mae_val', np.nan))
        self.max_error_train.append(epoch_metrics.get('max_error_train', np.nan))
        self.max_error_val.append(epoch_metrics.get('max_error_val', np.nan))
        self.epoch_times.append(epoch_metrics.get('time', 0))
        self.gradientes_antecedentes.append(epoch_metrics.get('grad_norm', 0))

    def plotar_convergencia(self, figsize=(16, 10)):
        """Visualiza a convergência do treinamento"""
        fig, axes = plt.subplots(2, 3, figsize=figsize)

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

        plt.tight_layout()
        return fig


class ANFIS:
    """ANFIS com regularização L1/L2 e suporte a minibatch training"""

    def __init__(self, n_inputs: int, n_mfs: int = 3,
                 learning_rate: float = 1.0e-5,
                 input_ranges: Optional[List[Tuple[float, float]]] = None,
                 lambda_l1: float = 0.0,
                 lambda_l2: float = 1,
                 regularizar_centros: bool = False,
                 regularizar_sigmas: bool = True,
                 batch_size: Optional[int] = None):
        """
        Inicializa o ANFIS com regularização e minibatch

        Parâmetros:
        -----------
        n_inputs : int
            Número de variáveis de entrada
        n_mfs : int
            Número de funções de pertinência por variável
        learning_rate : float
            Taxa de aprendizado base
        input_ranges : list of tuples, optional
            Lista com os ranges (min, max) de cada entrada
        lambda_l1 : float
            Coeficiente de regularização L1 (Lasso)
        lambda_l2 : float
            Coeficiente de regularização L2 (Ridge/Weight Decay)
        regularizar_centros : bool
            Se True, aplica regularização aos centros das MFs
        regularizar_sigmas : bool
            Se True, aplica regularização aos sigmas das MFs
        batch_size : int, optional
            Tamanho do batch para minibatch training
            - None ou >= n_samples: Batch gradient descent (default)
            - 1: Stochastic gradient descent
            - 16-128: Minibatch gradient descent (recomendado)
        """
        self.n_inputs = n_inputs
        self.n_mfs = n_mfs
        self.lr = learning_rate
        self.n_regras = n_mfs ** n_inputs
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.regularizar_centros = regularizar_centros
        self.regularizar_sigmas = regularizar_sigmas
        self.batch_size = batch_size

        # Definir ranges
        if input_ranges is None:
            self.input_ranges = [(-8.0, 8.0)] * n_inputs
        else:
            if len(input_ranges) != n_inputs:
                raise ValueError(f"input_ranges deve ter {n_inputs} elementos")
            self.input_ranges = input_ranges

        # Inicializar parâmetros das MFs
        self.params_mfs = [self._init_mf_params(i) for i in range(n_inputs)]

        # Parâmetros consequentes
        n_params_por_regra = n_inputs + 1
        self.params_consequentes = np.random.randn(self.n_regras, n_params_por_regra) * 0.1

        # Métricas
        self.metricas = MetricasANFIS()

        # Cache
        self._cache_indices_regras = self._gerar_indices_regras()

        # Histórico de penalidades
        self.historico_l1 = []
        self.historico_l2 = []
        self.historico_custo_total = []

        # Histórico de batches (para análise)
        self.historico_batches = []

    def _init_mf_params(self, input_idx: int) -> np.ndarray:
        """Inicializa parâmetros das MFs"""
        min_val, max_val = self.input_ranges[input_idx]
        centros = np.linspace(min_val, max_val, self.n_mfs)
        centros = np.array([c + np.random.uniform(-0.1, 0.1) for c in centros])
        sigma = (max_val - min_val) / (self.n_mfs - 1) * 0.9
        return np.array([[c, sigma] for c in centros])

    def _gerar_indices_regras(self) -> List[Tuple[int, ...]]:
        """Gera índices das regras"""
        if self.n_inputs == 1:
            return [(i,) for i in range(self.n_mfs)]

        indices = []
        ranges = [range(self.n_mfs) for _ in range(self.n_inputs)]

        def produto_cartesiano(arrays, current=[]):
            if not arrays:
                indices.append(tuple(current))
            else:
                for item in arrays[0]:
                    produto_cartesiano(arrays[1:], current + [item])

        produto_cartesiano(ranges)
        return indices

    def _criar_batches(self, X: np.ndarray, y: np.ndarray,
                       batch_size: int, shuffle: bool = True) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Divide dados em batches

        Parâmetros:
        -----------
        X : np.ndarray
            Dados de entrada
        y : np.ndarray
            Targets
        batch_size : int
            Tamanho do batch
        shuffle : bool
            Se True, embaralha os dados antes de dividir

        Retorna:
        --------
        List[Tuple[np.ndarray, np.ndarray]]
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

    def gaussian_mf(self, x: float, centro: float, sigma: float) -> float:
        """Função de pertinência gaussiana"""
        return np.exp(-((x - centro) ** 2) / (2 * sigma ** 2))

    def _calcular_penalidade_l1(self) -> float:
        """Calcula penalidade L1 (Lasso)"""
        penalty = 0.0

        for input_idx in range(self.n_inputs):
            for mf_idx in range(self.n_mfs):
                centro, sigma = self.params_mfs[input_idx][mf_idx]

                if self.regularizar_centros:
                    penalty += np.abs(centro)

                if self.regularizar_sigmas:
                    penalty += np.abs(sigma)

        return penalty

    def _calcular_penalidade_l2(self) -> float:
        """Calcula penalidade L2 (Ridge / Weight Decay)"""
        penalty = 0.0

        for input_idx in range(self.n_inputs):
            for mf_idx in range(self.n_mfs):
                centro, sigma = self.params_mfs[input_idx][mf_idx]

                if self.regularizar_centros:
                    penalty += centro ** 2

                if self.regularizar_sigmas:
                    penalty += sigma ** 2

        return penalty

    def _calcular_custo_total(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
        """Calcula custo total = MSE + penalidades"""
        # MSE
        y_pred = self.prever(X)
        mse = np.mean((y - y_pred) ** 2)

        # Penalidades
        l1_penalty = self._calcular_penalidade_l1()
        l2_penalty = self._calcular_penalidade_l2()

        return mse, l1_penalty, l2_penalty

    def camada1_fuzzificacao(self, X: np.ndarray) -> List[np.ndarray]:
        """Camada 1: Fuzzificação"""
        mu = []
        for i in range(self.n_inputs):
            mu_i = np.array([self.gaussian_mf(X[i], c, s)
                           for c, s in self.params_mfs[i]])
            mu.append(mu_i)
        return mu

    def camada2_regras(self, mu: List[np.ndarray]) -> np.ndarray:
        """Camada 2: Regras"""
        w = np.zeros(self.n_regras)
        for regra_idx, indices_mf in enumerate(self._cache_indices_regras):
            w[regra_idx] = np.prod([mu[i][mf_idx]
                                   for i, mf_idx in enumerate(indices_mf)])
        return w

    def camada3_normalizacao(self, w: np.ndarray) -> Tuple[np.ndarray, float]:
        """Camada 3: Normalização"""
        soma_w = np.sum(w) + 1e-10
        w_norm = w / soma_w
        return w_norm, soma_w

    def camada4_consequentes(self, X: np.ndarray, w_norm: np.ndarray) -> np.ndarray:
        """Camada 4: Consequentes"""
        saidas = np.zeros(self.n_regras)
        for i in range(self.n_regras):
            params = self.params_consequentes[i]
            f_i = params[0] + np.dot(params[1:], X)
            saidas[i] = w_norm[i] * f_i
        return saidas

    def camada5_agregacao(self, saidas: np.ndarray) -> float:
        """Camada 5: Agregação"""
        return np.sum(saidas)

    def forward(self, X: np.ndarray) -> Tuple[float, Tuple]:
        """Forward pass completo"""
        mu = self.camada1_fuzzificacao(X)
        w = self.camada2_regras(mu)
        w_norm, soma_w = self.camada3_normalizacao(w)
        saidas = self.camada4_consequentes(X, w_norm)
        y_pred = self.camada5_agregacao(saidas)
        return y_pred, (mu, w, w_norm, soma_w)

    def _ajustar_consequentes_least_squares(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Ajusta consequentes usando least squares

        NOTA: Este método usa TODO o batch fornecido para calcular
              os parâmetros consequentes de forma ótima (least squares).
        """
        n_samples = len(X)
        n_params = self.n_regras * (self.n_inputs + 1)

        A = np.zeros((n_samples, n_params))
        y_target = y.copy()

        for i in range(n_samples):
            _, (mu, w, w_norm, soma_w) = self.forward(X[i])
            for j in range(self.n_regras):
                start_idx = j * (self.n_inputs + 1)
                end_idx = start_idx + (self.n_inputs + 1)
                A[i, start_idx:end_idx] = w_norm[j] * np.concatenate([[1], X[i]])

        try:
            p_flat, residuals, rank, s = np.linalg.lstsq(A, y_target, rcond=None)
            self.params_consequentes = p_flat.reshape(self.n_regras, self.n_inputs + 1)
        except np.linalg.LinAlgError:
            warnings.warn("Erro ao resolver sistema linear.")

        y_pred = self.prever(X)
        rmse = np.sqrt(np.mean((y_target - y_pred) ** 2))
        return rmse

    def _gradiente_l1(self, parametro: float) -> float:
        """Gradiente (subgradiente) da penalidade L1"""
        if parametro > 0:
            return 1.0
        elif parametro < 0:
            return -1.0
        else:
            return 0.0

    def _gradiente_l2(self, parametro: float) -> float:
        """Gradiente da penalidade L2"""
        return 2.0 * parametro

    def _ajustar_antecedentes_gradiente(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Ajusta antecedentes com gradiente + regularização

        Este método processa TODO o batch fornecido e acumula os gradientes
        antes de atualizar os parâmetros.
        """
        n_samples = len(X)

        # Acumuladores de gradiente
        grad_acumulado = {}
        for input_idx in range(self.n_inputs):
            grad_acumulado[input_idx] = {}
            for mf_idx in range(self.n_mfs):
                grad_acumulado[input_idx][mf_idx] = {'centro': 0.0, 'sigma': 0.0}

        # Acumular gradientes de todas as amostras do batch
        for sample_idx in range(n_samples):
            X_sample = X[sample_idx]
            target = y[sample_idx]

            # Forward pass
            y_pred, (mu, w, w_norm, soma_w) = self.forward(X_sample)

            # Erro
            erro = target - y_pred

            # Contribuições
            contribuicoes = np.zeros(self.n_regras)
            for j in range(self.n_regras):
                params = self.params_consequentes[j]
                contribuicoes[j] = params[0] + np.dot(params[1:], X_sample)

            # Calcular gradientes para cada MF
            for input_idx in range(self.n_inputs):
                x_val = X_sample[input_idx]

                for regra_idx, indices_mf in enumerate(self._cache_indices_regras):
                    mf_idx = indices_mf[input_idx]

                    centro, sigma = self.params_mfs[input_idx][mf_idx]

                    # Gradiente do MSE
                    dmu_dc = mu[input_idx][mf_idx] * (x_val - centro) / (sigma ** 2)
                    dmu_ds = mu[input_idx][mf_idx] * ((x_val - centro) ** 2) / (sigma ** 3)

                    dw_dmu = w[regra_idx] / (mu[input_idx][mf_idx] + 1e-10)
                    dy_dwn = contribuicoes[regra_idx]
                    dwn_dw = (1 / soma_w) if soma_w > 1e-10 else 0

                    grad_mse_centro = -erro * dy_dwn * dwn_dw * dw_dmu * dmu_dc
                    grad_mse_sigma = -erro * dy_dwn * dwn_dw * dw_dmu * dmu_ds

                    # Acumular gradientes
                    grad_acumulado[input_idx][mf_idx]['centro'] += grad_mse_centro
                    grad_acumulado[input_idx][mf_idx]['sigma'] += grad_mse_sigma

        # Atualizar parâmetros usando gradientes médios do batch
        grad_norm_total = 0

        for input_idx in range(self.n_inputs):
            for mf_idx in range(self.n_mfs):
                centro, sigma = self.params_mfs[input_idx][mf_idx]

                # Gradiente médio do MSE (dividir por n_samples)
                grad_mse_centro = grad_acumulado[input_idx][mf_idx]['centro'] / n_samples
                grad_mse_sigma = grad_acumulado[input_idx][mf_idx]['sigma'] / n_samples

                # Gradientes das penalidades (não dependem do batch)
                grad_l1_centro = self.lambda_l1 * self._gradiente_l1(centro) if self.regularizar_centros else 0
                grad_l1_sigma = self.lambda_l1 * self._gradiente_l1(sigma) if self.regularizar_sigmas else 0

                grad_l2_centro = self.lambda_l2 * self._gradiente_l2(centro) if self.regularizar_centros else 0
                grad_l2_sigma = self.lambda_l2 * self._gradiente_l2(sigma) if self.regularizar_sigmas else 0

                # Gradiente total
                grad_total_centro = grad_mse_centro + grad_l1_centro + grad_l2_centro
                grad_total_sigma = grad_mse_sigma + grad_l1_sigma + grad_l2_sigma

                # Atualizar parâmetros
                self.params_mfs[input_idx][mf_idx, 0] -= self.lr * grad_total_centro
                self.params_mfs[input_idx][mf_idx, 1] -= self.lr * grad_total_sigma

                # Limitar sigma mínimo
                min_val, max_val = self.input_ranges[input_idx]
                sigma_min = (max_val - min_val) * 0.05
                self.params_mfs[input_idx][mf_idx, 1] = max(
                    self.params_mfs[input_idx][mf_idx, 1], sigma_min
                )

                # Acumular norma
                grad_norm_total += grad_total_centro**2 + grad_total_sigma**2

        return np.sqrt(grad_norm_total)

    def _calcular_metricas(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Calcula métricas"""
        y_pred = self.prever(X)
        erros = y - y_pred
        return {
            'rmse': np.sqrt(np.mean(erros ** 2)),
            'mae': np.mean(np.abs(erros)),
            'max_error': np.max(np.abs(erros))
        }

    def treinar(self, X_train: np.ndarray, y_train: np.ndarray,
                epochs: int = 100, verbose: bool = True,
                treinar_antecedentes: bool = True,
                X_val: Optional[np.ndarray] = None,
                y_val: Optional[np.ndarray] = None,
                early_stopping_patience: int = 20):
        """
        Treinamento com minibatch e regularização

        Se batch_size foi especificado:
        - Divide dados em batches
        - Embaralha a cada época
        - Múltiplas atualizações por época

        Se batch_size é None:
        - Usa batch gradient descent (comportamento padrão)
        """
        import time

        self.metricas = MetricasANFIS()
        self.historico_l1 = []
        self.historico_l2 = []
        self.historico_custo_total = []
        self.historico_batches = []

        melhor_rmse_val = np.inf
        epochs_sem_melhora = 0
        melhor_estado = None

        # Determinar batch_size efetivo
        n_samples = len(X_train)
        batch_size_efetivo = self.batch_size if self.batch_size is not None else n_samples
        batch_size_efetivo = min(batch_size_efetivo, n_samples)

        n_batches = int(np.ceil(n_samples / batch_size_efetivo))

        if verbose:
            print(f"\nTreinamento com {'Minibatch' if batch_size_efetivo < n_samples else 'Batch'} GD")
            print(f"  Samples: {n_samples}")
            print(f"  Batch size: {batch_size_efetivo}")
            print(f"  Batches por época: {n_batches}")
            print(f"  Atualizações por época: {n_batches}")

        # Inicialização dos consequentes com least squares (importante!)
        if verbose:
            print(f"\nInicializando consequentes com least squares...")
        rmse_inicial = self._ajustar_consequentes_least_squares(X_train, y_train)
        if verbose:
            print(f"RMSE após inicialização: {rmse_inicial:.4f}\n")

        for epoch in range(epochs):
            start_time = time.time()

            # Criar batches (com shuffle se minibatch)
            usar_shuffle = (batch_size_efetivo < n_samples)
            batches = self._criar_batches(X_train, y_train, batch_size_efetivo, shuffle=usar_shuffle)

            # Processar cada batch
            rmse_batches = []
            for batch_idx, (X_batch, y_batch) in enumerate(batches):
                # Fase 1: Ajustar consequentes no batch
                rmse_ls = self._ajustar_consequentes_least_squares(X_batch, y_batch)
                rmse_batches.append(rmse_ls)

                # Fase 2: Ajustar antecedentes no batch (com regularização)
                if treinar_antecedentes:
                    grad_norm = self._ajustar_antecedentes_gradiente(X_batch, y_batch)

            # Calcular métricas no conjunto completo (após todos os batches)
            mse, l1_penalty, l2_penalty = self._calcular_custo_total(X_train, y_train)
            custo_total = mse + self.lambda_l1 * l1_penalty + self.lambda_l2 * l2_penalty

            self.historico_l1.append(l1_penalty)
            self.historico_l2.append(l2_penalty)
            self.historico_custo_total.append(custo_total)
            self.historico_batches.append(np.mean(rmse_batches))

            # Métricas
            metricas_train = self._calcular_metricas(X_train, y_train)
            metricas_val = {}

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
                        print(f"Melhor RMSE validação: {melhor_rmse_val:.4f}")
                    if melhor_estado is not None:
                        self.params_mfs = melhor_estado['params_mfs']
                        self.params_consequentes = melhor_estado['params_consequentes']
                    break

            epoch_time = time.time() - start_time

            self.metricas.adicionar_epoca({
                'rmse_train': metricas_train['rmse'],
                'rmse_val': metricas_val.get('rmse', np.nan),
                'mae_train': metricas_train['mae'],
                'mae_val': metricas_val.get('mae', np.nan),
                'max_error_train': metricas_train['max_error'],
                'max_error_val': metricas_val.get('max_error', np.nan),
                'time': epoch_time,
                'grad_norm': grad_norm if treinar_antecedentes else 0
            })

            if verbose and ((epoch + 1) % 5 == 0 or epoch == 0):
                modo = f"Minibatch (bs={batch_size_efetivo})" if batch_size_efetivo < n_samples else "Batch"
                msg = f"Época {epoch+1:3d}/{epochs} - Train: {metricas_train['rmse']:.4f}"
                if X_val is not None:
                    msg += f", Val: {metricas_val['rmse']:.4f}"
                msg += f" | Custo: {custo_total:.4f}"
                msg += f" [{modo}]"
                print(msg)

    def prever(self, X: np.ndarray) -> np.ndarray:
        """Predições"""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        predicoes = np.array([self.forward(x)[0] for x in X])
        return predicoes

    def prever_classe(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predições de classe"""
        predicoes = self.prever(X)
        return (predicoes >= threshold).astype(int)

    def visualizar_mfs(self, figsize_por_input=(6, 4)):
        """Visualiza MFs"""
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
            min_val, max_val = self.input_ranges[input_idx]
            x_range = np.linspace(min_val, max_val, 200)

            for mf_idx, (centro, sigma) in enumerate(self.params_mfs[input_idx]):
                mu = self.gaussian_mf(x_range, centro, sigma)
                ax.plot(x_range, mu, linewidth=2,
                       label=f'MF_{mf_idx+1} (c={centro:.2f}, σ={sigma:.2f})')

            ax.set_xlabel(f'Entrada {input_idx+1}', fontsize=12)
            ax.set_ylabel('Pertinência', fontsize=12)
            ax.set_title(f'MFs - Entrada {input_idx+1}', fontsize=13, weight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([-0.05, 1.1])

        for idx in range(self.n_inputs, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        return fig

    def plotar_regularizacao(self, figsize=(16, 5)):
        """Plota evolução das penalidades"""
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
        ax.plot(epochs, self.historico_l1, 'r-', linewidth=2, label='L1')
        ax.set_xlabel('Época', fontsize=12)
        ax.set_ylabel('Penalidade L1', fontsize=12)
        ax.set_title(f'||θ||₁ (λ₁={self.lambda_l1})', fontsize=13, weight='bold')
        ax.grid(True, alpha=0.3)

        # Penalidade L2
        ax = axes[2]
        ax.plot(epochs, self.historico_l2, 'g-', linewidth=2, label='L2')
        ax.set_xlabel('Época', fontsize=12)
        ax.set_ylabel('Penalidade L2', fontsize=12)
        ax.set_title(f'||θ||² (λ₂={self.lambda_l2})', fontsize=13, weight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def get_info(self) -> Dict:
        """Info do modelo"""
        n_samples_estimado = len(self.metricas.rmse_train) if len(self.metricas.rmse_train) > 0 else 0
        batch_size_efetivo = self.batch_size if self.batch_size is not None else "Full batch"

        info = {
            'n_inputs': self.n_inputs,
            'n_mfs': self.n_mfs,
            'n_regras': self.n_regras,
            'learning_rate': self.lr,
            'lambda_l1': self.lambda_l1,
            'lambda_l2': self.lambda_l2,
            'regularizar_centros': self.regularizar_centros,
            'regularizar_sigmas': self.regularizar_sigmas,
            'batch_size': batch_size_efetivo,
            'tipo_regularizacao': self._get_tipo_reg(),
            'n_params_total': self.params_consequentes.size + sum(p.size for p in self.params_mfs)
        }
        return info

    def _get_tipo_reg(self) -> str:
        """Retorna tipo de regularização"""
        if self.lambda_l1 > 0 and self.lambda_l2 > 0:
            return "Elastic Net (L1 + L2)"
        elif self.lambda_l1 > 0:
            return "Lasso (L1)"
        elif self.lambda_l2 > 0:
            return "Ridge (L2)"
        else:
            return "Sem regularização"

    def resumo(self):
        """Resumo do modelo"""
        info = self.get_info()
        print("=" * 70)
        print("ANFIS COM REGULARIZAÇÃO E MINIBATCH")
        print("=" * 70)
        print(f"Entradas:                {info['n_inputs']}")
        print(f"MFs por entrada:         {info['n_mfs']}")
        print(f"Total de regras:         {info['n_regras']}")
        print(f"Total de parâmetros:     {info['n_params_total']}")
        print(f"\nTreinamento:")
        print(f"  Batch size:            {info['batch_size']}")
        print(f"  Learning rate:         {info['learning_rate']}")
        print(f"\nRegularização:")
        print(f"  Tipo:                  {info['tipo_regularizacao']}")
        print(f"  λ₁ (L1):               {info['lambda_l1']}")
        print(f"  λ₂ (L2):               {info['lambda_l2']}")
        print(f"  Regularizar centros:   {info['regularizar_centros']}")
        print(f"  Regularizar sigmas:    {info['regularizar_sigmas']}")
        print("=" * 70)
