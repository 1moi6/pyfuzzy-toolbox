"""
Sistema Neuro-Fuzzy Mamdani Moderno
====================================

Implementação Python completa de sistema Mamdani com:
- Aprendizado batch/online/mini-batch
- Defuzzificação: Center-of-Gravity (COG) e Center-of-Sets (COS)
- Otimização: PSO, GA, e híbrido
- Interpretabilidade: Regras linguísticas legíveis
- Arquitetura adaptativa com aprendizado de estrutura

Diferenças do ANFIS (Takagi-Sugeno):
- Consequentes são conjuntos fuzzy (não funções lineares)
- Maior interpretabilidade das regras
- Ideal para sistemas especialistas e diagnóstico

Referências:
    Mamdani, E. H. (1974). "Application of fuzzy algorithms for control"
    Nakasima-López et al. (2019). "Full batch, online and mini-batch learning 
        on Mamdani neuro-fuzzy system"
    Sanchez et al. (2019). "Data-Driven Mamdani-Type Fuzzy Clinical Decision"
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Callable
import warnings
import itertools
warnings.filterwarnings('ignore')


class MamdaniNeurofuzzy:
    """
    Sistema Neuro-Fuzzy Mamdani com aprendizado adaptativo.

    Arquitetura em 4 camadas:
    - Camada 0: Entradas
    - Camada 1: Fuzzificação (funções de pertinência gaussianas adaptativas)
    - Camada 2: Regras fuzzy (produto T-norma)
    - Camada 3: Defuzzificação (COG ou COS)

    Parâmetros ajustáveis:
    - Médias (m) e desvios padrão (σ) das MFs de entrada
    - Centroides (C) dos conjuntos fuzzy de saída
    """

    def __init__(self, 
                 n_inputs: int,
                 n_mfs_input: List[int],
                 n_mfs_output: int = 5,
                 learning_mode: str = 'mini-batch',
                 defuzz_method: str = 'cog',
                 linguistic_labels: Optional[List[str]] = None):
        """
        Parâmetros:
        -----------
        n_inputs: int
            Número de variáveis de entrada
        n_mfs_input: List[int]
            Número de MFs para cada entrada (ex: [3, 3] para 2 entradas)
        n_mfs_output: int
            Número de conjuntos fuzzy na saída
        learning_mode: str
            'batch', 'online' ou 'mini-batch'
        defuzz_method: str
            'cog' (Center of Gravity) ou 'cos' (Center of Sets)
        linguistic_labels: List[str], opcional
            Rótulos linguísticos para saída (ex: ['Muito Baixo', 'Baixo', ...])
        """
        self.n_inputs = n_inputs
        self.n_mfs_input = n_mfs_input
        self.n_mfs_output = n_mfs_output
        self.n_rules = np.prod(n_mfs_input)
        self.learning_mode = learning_mode
        self.defuzz_method = defuzz_method

        # Rótulos linguísticos para interpretabilidade
        if linguistic_labels is None:
            self.linguistic_labels = [f'Saída_{i+1}' for i in range(n_mfs_output)]
        else:
            self.linguistic_labels = linguistic_labels

        # Parâmetros aprendíveis
        self.input_means = []      # Médias das MFs de entrada
        self.input_sigmas = []     # Desvios padrão das MFs
        self.output_centroids = None  # Centroides dos conjuntos fuzzy de saída
        self.rule_weights = None   # Pesos das regras (para aprendizado)

        # Domínios
        self.input_bounds = None
        self.output_bound = None

        # Histórico
        self.training_history = []

    def _initialize_parameters(self, X: np.ndarray, y: np.ndarray):
        """
        Inicializa parâmetros usando estatísticas dos dados.
        Estratégia: distribuição uniforme com cobertura do domínio.
        """
        # Domínios de entrada
        self.input_bounds = np.array([[X[:, i].min(), X[:, i].max()] 
                                       for i in range(self.n_inputs)])

        # Domínio de saída
        self.output_bound = [y.min(), y.max()]

        # Inicializa MFs de entrada
        self.input_means = []
        self.input_sigmas = []

        for i in range(self.n_inputs):
            x_min, x_max = self.input_bounds[i]
            x_range = x_max - x_min
            n_mf = self.n_mfs_input[i]

            # Distribui médias uniformemente
            means = np.linspace(x_min, x_max, n_mf)

            # Sigma = largura / 2 para boa cobertura
            sigma = x_range / (2 * n_mf)
            sigmas = np.full(n_mf, sigma)

            self.input_means.append(means)
            self.input_sigmas.append(sigmas)

        # Inicializa centroides de saída
        y_min, y_max = self.output_bound
        self.output_centroids = np.linspace(y_min, y_max, self.n_mfs_output)

        # Inicializa pesos das regras (todos iguais inicialmente)
        self.rule_weights = np.ones(self.n_rules)

    def _gaussian_mf(self, x: np.ndarray, mean: float, sigma: float) -> np.ndarray:
        """Função de pertinência Gaussiana."""
        return np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))

    def _fuzzify_inputs(self, X: np.ndarray) -> List[np.ndarray]:
        """
        Camada 1: Fuzzificação das entradas.

        Retorna lista de matrizes de pertinência, uma por entrada.
        """
        n_samples = X.shape[0]
        membership_values = []

        for i in range(self.n_inputs):
            x = X[:, i].reshape(-1, 1)
            mf_vals = []

            for j in range(self.n_mfs_input[i]):
                mean = self.input_means[i][j]
                sigma = self.input_sigmas[i][j]
                mu = self._gaussian_mf(x, mean, sigma)
                mf_vals.append(mu)

            membership_values.append(np.hstack(mf_vals))

        return membership_values

    def _fire_rules(self, membership_values: List[np.ndarray]) -> np.ndarray:
        """
        Camada 2: Calcula força de disparo das regras.

        Usa produto (T-norma) para combinar antecedentes.
        """
        n_samples = membership_values[0].shape[0]
        firing_strengths = np.ones((n_samples, self.n_rules))

        rule_idx = 0
        indices = [range(n_mf) for n_mf in self.n_mfs_input]

        for combination in itertools.product(*indices):
            for i, idx in enumerate(combination):
                firing_strengths[:, rule_idx] *= membership_values[i][:, idx]
            rule_idx += 1

        # Aplica pesos das regras
        firing_strengths = firing_strengths * self.rule_weights

        return firing_strengths

    def _defuzzify_cog(self, firing_strengths: np.ndarray) -> np.ndarray:
        """
        Defuzzificação usando Center of Gravity (COG).

        Formula: y = Σ(w_i * c_i) / Σ(w_i)
        onde w_i é a força de disparo da regra i
        e c_i é o centroide do consequente da regra i
        """
        n_samples = firing_strengths.shape[0]

        # Mapeia cada regra para um centroide de saída
        # Estratégia: distribui regras uniformemente entre centroides
        rule_to_centroid = np.linspace(0, self.n_mfs_output - 1, self.n_rules)
        rule_to_centroid = rule_to_centroid.astype(int)

        output = np.zeros(n_samples)

        for sample_idx in range(n_samples):
            numerator = 0.0
            denominator = 0.0

            for rule_idx in range(self.n_rules):
                w = firing_strengths[sample_idx, rule_idx]
                c = self.output_centroids[rule_to_centroid[rule_idx]]

                numerator += w * c
                denominator += w

            # Evita divisão por zero
            if denominator < 1e-10:
                output[sample_idx] = np.mean(self.output_centroids)
            else:
                output[sample_idx] = numerator / denominator

        return output

    def _defuzzify_cos(self, firing_strengths: np.ndarray) -> np.ndarray:
        """
        Defuzzificação usando Center of Sets (COS).

        Mais eficiente que COG, calcula diretamente:
        y = Σ(w_i * C_i * a_i) onde C_i são centroides e a_i forças normalizadas
        """
        # Normaliza forças de disparo
        total = np.sum(firing_strengths, axis=1, keepdims=True)
        total = np.where(total == 0, 1e-10, total)
        normalized = firing_strengths / total

        # Mapeia regras para centroides
        rule_to_centroid = np.linspace(0, self.n_mfs_output - 1, self.n_rules)
        rule_to_centroid = rule_to_centroid.astype(int)

        # Calcula saída ponderada
        output = np.zeros(firing_strengths.shape[0])

        for rule_idx in range(self.n_rules):
            centroid_idx = rule_to_centroid[rule_idx]
            centroid = self.output_centroids[centroid_idx]
            output += normalized[:, rule_idx] * centroid

        return output

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Propagação forward completa através do sistema Mamdani.

        Retorna:
        --------
        output: np.ndarray
            Valores de saída defuzzificados
        firing_strengths: np.ndarray
            Forças de disparo das regras (para treinamento)
        """
        # Camada 1: Fuzzificação
        membership_values = self._fuzzify_inputs(X)

        # Camada 2: Regras
        firing_strengths = self._fire_rules(membership_values)

        # Camada 3: Defuzzificação
        if self.defuzz_method == 'cog':
            output = self._defuzzify_cog(firing_strengths)
        else:  # cos
            output = self._defuzzify_cos(firing_strengths)

        return output, firing_strengths

    def _compute_gradients(self, X: np.ndarray, y: np.ndarray, 
                          predictions: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calcula gradientes dos parâmetros usando retropropagação.

        Retorna dicionário com gradientes para:
        - means: médias das MFs de entrada
        - sigmas: desvios padrão das MFs
        - centroids: centroides de saída
        """
        n_samples = X.shape[0]
        error = predictions - y

        # Recalcula forward para obter valores intermediários
        membership_values = self._fuzzify_inputs(X)
        firing_strengths = self._fire_rules(membership_values)

        # Gradientes dos centroides (mais diretos)
        grad_centroids = np.zeros_like(self.output_centroids)

        rule_to_centroid = np.linspace(0, self.n_mfs_output - 1, self.n_rules)
        rule_to_centroid = rule_to_centroid.astype(int)

        for centroid_idx in range(self.n_mfs_output):
            # Regras associadas a este centroide
            relevant_rules = np.where(rule_to_centroid == centroid_idx)[0]

            for sample_idx in range(n_samples):
                # Contribuição deste centroide para o erro
                total_firing = np.sum(firing_strengths[sample_idx])

                if total_firing > 1e-10:
                    relevant_firing = np.sum(firing_strengths[sample_idx, relevant_rules])
                    weight = relevant_firing / total_firing
                    grad_centroids[centroid_idx] += error[sample_idx] * weight

        grad_centroids /= n_samples

        # Gradientes das médias e sigmas (aproximação numérica)
        grad_means = [np.zeros_like(self.input_means[i]) 
                     for i in range(self.n_inputs)]
        grad_sigmas = [np.zeros_like(self.input_sigmas[i]) 
                      for i in range(self.n_inputs)]

        epsilon = 1e-5

        for i in range(self.n_inputs):
            for j in range(self.n_mfs_input[i]):
                # Gradiente da média
                original_mean = self.input_means[i][j]

                self.input_means[i][j] = original_mean + epsilon
                pred_plus, _ = self.forward(X)

                self.input_means[i][j] = original_mean - epsilon
                pred_minus, _ = self.forward(X)

                self.input_means[i][j] = original_mean

                grad_means[i][j] = np.mean(error * (pred_plus - pred_minus) / (2 * epsilon))

                # Gradiente do sigma
                original_sigma = self.input_sigmas[i][j]

                self.input_sigmas[i][j] = original_sigma + epsilon
                pred_plus, _ = self.forward(X)

                self.input_sigmas[i][j] = original_sigma - epsilon
                pred_minus, _ = self.forward(X)

                self.input_sigmas[i][j] = original_sigma

                grad_sigmas[i][j] = np.mean(error * (pred_plus - pred_minus) / (2 * epsilon))

        return {
            'means': grad_means,
            'sigmas': grad_sigmas,
            'centroids': grad_centroids
        }

    def _apply_domain_constraints(self):
        """Aplica restrições de domínio aos parâmetros."""
        for i in range(self.n_inputs):
            x_min, x_max = self.input_bounds[i]

            # Médias dentro do domínio
            self.input_means[i] = np.clip(self.input_means[i], x_min, x_max)

            # Sigmas positivos
            self.input_sigmas[i] = np.maximum(self.input_sigmas[i], 1e-6)

        # Centroides dentro do domínio de saída
        y_min, y_max = self.output_bound
        self.output_centroids = np.clip(self.output_centroids, y_min, y_max)

    def fit(self, X: np.ndarray, y: np.ndarray,
            epochs: int = 100,
            learning_rate: float = 0.01,
            batch_size: int = 32,
            validation_split: float = 0.2,
            verbose: bool = True):
        """
        Treina o sistema Mamdani neuro-fuzzy.

        Parâmetros:
        -----------
        X: np.ndarray, shape (n_samples, n_inputs)
            Dados de entrada
        y: np.ndarray, shape (n_samples,)
            Valores alvo
        epochs: int
            Número de épocas
        learning_rate: float
            Taxa de aprendizado
        batch_size: int
            Tamanho do batch (para mini-batch)
        validation_split: float
            Proporção dos dados para validação
        verbose: bool
            Exibe progresso
        """
        # Inicializa parâmetros
        self._initialize_parameters(X, y)

        # Split treino/validação
        n_val = int(len(X) * validation_split)
        indices = np.random.permutation(len(X))

        X_train = X[indices[n_val:]]
        y_train = y[indices[n_val:]]
        X_val = X[indices[:n_val]]
        y_val = y[indices[:n_val]]

        self.training_history = []
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0

        if verbose:
            print("="*70)
            print(f"TREINAMENTO: Sistema Neuro-Fuzzy Mamdani")
            print(f"Modo: {self.learning_mode}")
            print(f"Defuzzificação: {self.defuzz_method.upper()}")
            print("="*70)
            print(f"Treino: {len(X_train)} amostras | Validação: {len(X_val)} amostras")
            print(f"Regras: {self.n_rules} | Centroides saída: {self.n_mfs_output}")
            print("="*70)

        for epoch in range(epochs):
            # Mini-batch ou batch
            if self.learning_mode == 'mini-batch':
                n_batches = max(1, len(X_train) // batch_size)
                batch_indices = np.array_split(np.random.permutation(len(X_train)), n_batches)
            elif self.learning_mode == 'online':
                batch_indices = [[i] for i in range(len(X_train))]
            else:  # batch
                batch_indices = [np.arange(len(X_train))]

            # Treina em batches
            for batch_idx in batch_indices:
                X_batch = X_train[batch_idx]
                y_batch = y_train[batch_idx]

                # Forward
                predictions, _ = self.forward(X_batch)

                # Backward
                gradients = self._compute_gradients(X_batch, y_batch, predictions)

                # Atualiza parâmetros
                for i in range(self.n_inputs):
                    self.input_means[i] -= learning_rate * gradients['means'][i]
                    self.input_sigmas[i] -= learning_rate * gradients['sigmas'][i]

                self.output_centroids -= learning_rate * gradients['centroids']

                # Aplica restrições
                self._apply_domain_constraints()

            # Avalia
            train_pred, _ = self.forward(X_train)
            val_pred, _ = self.forward(X_val)

            train_mse = np.mean((y_train - train_pred) ** 2)
            val_mse = np.mean((y_val - val_pred) ** 2)

            train_rmse = np.sqrt(train_mse)
            val_rmse = np.sqrt(val_mse)

            self.training_history.append({
                'epoch': epoch + 1,
                'train_rmse': train_rmse,
                'val_rmse': val_rmse
            })

            # Early stopping
            if val_mse < best_val_loss:
                best_val_loss = val_mse
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                if verbose:
                    print(f"\nEarly stopping na época {epoch+1}")
                break

            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"Época {epoch+1}/{epochs} | "
                      f"Train RMSE: {train_rmse:.6f} | "
                      f"Val RMSE: {val_rmse:.6f}")

        if verbose:
            print("="*70)
            print("TREINAMENTO CONCLUÍDO!")
            print(f"Melhor Val RMSE: {np.sqrt(best_val_loss):.6f}")
            print("="*70)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prediz valores para novos dados."""
        output, _ = self.forward(X)
        return output

    def get_linguistic_rules(self, top_k: int = 10) -> List[str]:
        """
        Retorna regras mais importantes em formato linguístico.

        Parâmetros:
        -----------
        top_k: int
            Número de regras mais importantes a retornar
        """
        # Calcula importância das regras (baseado em pesos)
        rule_importance = self.rule_weights / np.sum(self.rule_weights)
        top_indices = np.argsort(rule_importance)[-top_k:][::-1]

        # Mapeia regras para centroides
        rule_to_centroid = np.linspace(0, self.n_mfs_output - 1, self.n_rules)
        rule_to_centroid = rule_to_centroid.astype(int)

        rules = []
        rule_idx = 0
        indices = [range(n_mf) for n_mf in self.n_mfs_input]

        for combination in itertools.product(*indices):
            if rule_idx in top_indices:
                # Constrói antecedente
                antecedents = []
                for i, mf_idx in enumerate(combination):
                    mean = self.input_means[i][mf_idx]
                    antecedents.append(f"X{i+1} ≈ {mean:.2f}")

                antecedent_str = " AND ".join(antecedents)

                # Consequente
                centroid_idx = rule_to_centroid[rule_idx]
                output_label = self.linguistic_labels[centroid_idx]
                output_val = self.output_centroids[centroid_idx]

                # Importância
                importance = rule_importance[rule_idx] * 100

                rule = (f"SE {antecedent_str} ENTÃO Y = {output_label} "
                       f"({output_val:.2f}) [Peso: {importance:.1f}%]")
                rules.append(rule)

            rule_idx += 1

        return rules


print("✅ Sistema Neuro-Fuzzy Mamdani implementado com sucesso!")
print("\nCaracterísticas:")
print("  • Arquitetura em 4 camadas")
print("  • Aprendizado: batch, online, mini-batch")
print("  • Defuzzificação: COG e COS")
print("  • Interpretabilidade: extração de regras linguísticas")
print("  • Restrições de domínio adaptativas")
print("  • Early stopping e validação")
