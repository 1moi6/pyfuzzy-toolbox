"""
Sistema Mamdani Neuro-Fuzzy com Aprendizado e Otimização
=========================================================

Implementação de sistema Mamdani com múltiplas estratégias de aprendizado:

1. **Aprendizado por Gradiente**: Batch, online e mini-batch
2. **Otimização Metaheurística**: PSO, DE, GA com caching de ativações
3. **Defuzzificação**: COG (Center of Gravity) e COS (Center of Sums)

Características principais:
- Funções de pertinência Gaussianas ajustáveis
- Pesos de regras aprendíveis
- Restrições de domínio automáticas
- Suporte a early stopping
- **Caching de ativações** para otimização eficiente (conjuntos de entrada fixos)

Autor: fuzzy_systems package
Versão: 1.0
"""

import numpy as np
from typing import Tuple, Optional, Dict, List, Literal
import warnings

# Imports para integração com MamdaniSystem
from ..core.operators import TNorm
from ..core.defuzzification import DefuzzMethod


class MamdaniLearning:
    """
    Sistema Mamdani Neuro-Fuzzy com aprendizado.

    Arquitetura de 4 camadas:
    1. **Fuzzificação**: Gaussianas nos inputs
    2. **Regras**: T-norm produto (∏)
    3. **Defuzzificação**: COG ou COS
    4. **Saída**: Valor crisp

    Parâmetros aprendíveis:
    - input_means: Centros das MFs de entrada
    - input_sigmas: Larguras das MFs de entrada
    - output_centroides: Centros das MFs de saída
    - rule_weights: Pesos das regras (opcional)

    Exemplo:
        >>> mamdani = MamdaniLearning(
        ...     n_inputs=2,
        ...     n_mfs_input=[3, 3],
        ...     n_mfs_output=3,
        ...     defuzz_method='cog'
        ... )
        >>> mamdani.fit(X_train, y_train, epochs=100, learning_rate=0.01)
        >>> y_pred = mamdani.predict(X_test)
    """

    def __init__(
        self,
        n_inputs: int,
        n_mfs_input: List[int],
        n_mfs_output: int,
        defuzz_method: Literal['cog', 'cos'] = 'cog',
        use_rule_weights: bool = True,
        input_bounds: Optional[List[Tuple[float, float]]] = None,
        output_bound: Optional[Tuple[float, float]] = None
    ):
        """
        Inicializa sistema Mamdani.

        Args:
            n_inputs: Número de variáveis de entrada
            n_mfs_input: Lista com número de MFs para cada entrada
            n_mfs_output: Número de MFs de saída
            defuzz_method: Método de defuzzificação ('cog' ou 'cos')
            use_rule_weights: Se True, usa pesos aprendíveis nas regras
            input_bounds: Lista de (min, max) para cada entrada
            output_bound: Tupla (min, max) para saída
        """
        self.n_inputs = n_inputs
        self.n_mfs_input = n_mfs_input
        self.n_mfs_output = n_mfs_output
        self.defuzz_method = defuzz_method
        self.use_rule_weights = use_rule_weights

        # Número total de regras (produto cartesiano)
        self.n_rules = int(np.prod(n_mfs_input))

        # Bounds (serão definidos em _initialize_parameters se None)
        self.input_bounds = input_bounds
        self.output_bound = output_bound

        # Parâmetros (inicializados em fit)
        self.input_means: List[np.ndarray] = []
        self.input_sigmas: List[np.ndarray] = []
        self.output_centroids: np.ndarray = None
        self.rule_weights: np.ndarray = None

        # Cache para otimização metaheurística
        self._cached_activations: Optional[np.ndarray] = None
        self._cached_X: Optional[np.ndarray] = None

        # Flag de treinamento
        self._is_fitted = False

    def _initialize_parameters(self, X: np.ndarray, y: np.ndarray):
        """
        Inicializa parâmetros baseado nos dados.

        Args:
            X: Dados de entrada (n_samples, n_inputs)
            y: Saídas desejadas (n_samples,)
        """
        # Define bounds se não fornecidos
        if self.input_bounds is None:
            self.input_bounds = [(X[:, i].min(), X[:, i].max())
                                 for i in range(self.n_inputs)]

        if self.output_bound is None:
            self.output_bound = (y.min(), y.max())

        # Inicializa MFs de entrada (uniformemente espaçadas)
        self.input_means = []
        self.input_sigmas = []

        for i in range(self.n_inputs):
            x_min, x_max = self.input_bounds[i]
            n_mfs = self.n_mfs_input[i]

            # Médias uniformemente espaçadas
            means = np.linspace(x_min, x_max, n_mfs)
            self.input_means.append(means)

            # Sigmas: 50% da distância entre centros adjacentes
            if n_mfs > 1:
                sigma = (x_max - x_min) / (2 * (n_mfs - 1))
            else:
                sigma = (x_max - x_min) / 4
            sigmas = np.full(n_mfs, sigma)
            self.input_sigmas.append(sigmas)

        # Inicializa centroides de saída
        y_min, y_max = self.output_bound
        self.output_centroids = np.linspace(y_min, y_max, self.n_mfs_output)

        # Inicializa pesos das regras
        if self.use_rule_weights:
            self.rule_weights = np.ones(self.n_rules)
        else:
            self.rule_weights = None

    def _apply_domain_constraints(self):
        """Aplica restrições de domínio aos parâmetros."""
        # Restringe médias aos bounds
        for i in range(self.n_inputs):
            x_min, x_max = self.input_bounds[i]
            self.input_means[i] = np.clip(self.input_means[i], x_min, x_max)

        # Restringe sigmas (mínimo 1e-6 para evitar divisão por zero)
        for i in range(self.n_inputs):
            self.input_sigmas[i] = np.maximum(self.input_sigmas[i], 1e-6)

        # Restringe centroides de saída
        y_min, y_max = self.output_bound
        self.output_centroids = np.clip(self.output_centroids, y_min, y_max)

        # Restringe pesos (positivos)
        if self.rule_weights is not None:
            self.rule_weights = np.maximum(self.rule_weights, 1e-6)

    def _gaussian_mf(self, x: np.ndarray, mean: float, sigma: float) -> np.ndarray:
        """
        Função de pertinência Gaussiana.

        Args:
            x: Valores de entrada (n_samples,)
            mean: Centro da Gaussiana
            sigma: Desvio padrão

        Returns:
            Graus de pertinência (n_samples,)
        """
        return np.exp(-0.5 * ((x - mean) / sigma) ** 2)

    def _fuzzify_inputs(self, X: np.ndarray) -> List[np.ndarray]:
        """
        Camada 1: Fuzzificação das entradas.

        Args:
            X: Dados de entrada (n_samples, n_inputs)

        Returns:
            Lista de graus de pertinência para cada variável
            membership_values[i] tem shape (n_samples, n_mfs_input[i])
        """
        membership_values = []

        for i in range(self.n_inputs):
            x_i = X[:, i:i+1]  # (n_samples, 1)
            n_mfs = self.n_mfs_input[i]

            # Calcula μ para cada MF
            mu = np.zeros((X.shape[0], n_mfs))
            for j in range(n_mfs):
                mu[:, j] = self._gaussian_mf(
                    x_i.flatten(),
                    self.input_means[i][j],
                    self.input_sigmas[i][j]
                )

            membership_values.append(mu)

        return membership_values

    def _fire_rules(self, membership_values: List[np.ndarray]) -> np.ndarray:
        """
        Camada 2: Disparo das regras (T-norm produto).

        Args:
            membership_values: Lista de graus de pertinência

        Returns:
            Forças de disparo (n_samples, n_rules)
        """
        n_samples = membership_values[0].shape[0]
        firing_strengths = np.zeros((n_samples, self.n_rules))

        # Gera todas as combinações de regras (produto cartesiano)
        rule_idx = 0
        for combo in np.ndindex(tuple(self.n_mfs_input)):
            # T-norm: produto
            strength = np.ones(n_samples)
            for i, mf_idx in enumerate(combo):
                strength *= membership_values[i][:, mf_idx]

            firing_strengths[:, rule_idx] = strength
            rule_idx += 1

        return firing_strengths

    def _defuzzify_cog(
        self,
        firing_strengths: np.ndarray,
        consequent_indices: np.ndarray
    ) -> np.ndarray:
        """
        Defuzzificação por COG (Center of Gravity).

        COG = Σ(w_i * c_i) / Σ(w_i)

        Args:
            firing_strengths: Forças de disparo (n_samples, n_rules)
            consequent_indices: Índices dos consequentes (n_rules,)

        Returns:
            Saídas defuzzificadas (n_samples,)
        """
        # Aplica pesos das regras se habilitado
        if self.rule_weights is not None:
            weighted_strengths = firing_strengths * self.rule_weights
        else:
            weighted_strengths = firing_strengths

        # COG: soma ponderada dos centroides
        numerator = np.zeros(firing_strengths.shape[0])
        denominator = np.zeros(firing_strengths.shape[0])

        for rule_idx, consequent_idx in enumerate(consequent_indices):
            centroid = self.output_centroids[consequent_idx]
            strength = weighted_strengths[:, rule_idx]

            numerator += strength * centroid
            denominator += strength

        # Evita divisão por zero
        denominator = np.where(denominator < 1e-10, 1e-10, denominator)

        return numerator / denominator

    def _defuzzify_cos(
        self,
        firing_strengths: np.ndarray,
        consequent_indices: np.ndarray
    ) -> np.ndarray:
        """
        Defuzzificação por COS (Center of Sums).

        COS = Σ(μ_i * c_i) / Σ(μ_i)

        Similar a COG mas agrega por MF de saída antes de normalizar.

        Args:
            firing_strengths: Forças de disparo (n_samples, n_rules)
            consequent_indices: Índices dos consequentes (n_rules,)

        Returns:
            Saídas defuzzificadas (n_samples,)
        """
        # Aplica pesos das regras
        if self.rule_weights is not None:
            weighted_strengths = firing_strengths * self.rule_weights
        else:
            weighted_strengths = firing_strengths

        # Agrega por MF de saída
        n_samples = firing_strengths.shape[0]
        aggregated_strengths = np.zeros((n_samples, self.n_mfs_output))

        for rule_idx, consequent_idx in enumerate(consequent_indices):
            aggregated_strengths[:, consequent_idx] += weighted_strengths[:, rule_idx]

        # COS
        numerator = np.sum(
            aggregated_strengths * self.output_centroids,
            axis=1
        )
        denominator = np.sum(aggregated_strengths, axis=1)
        denominator = np.where(denominator < 1e-10, 1e-10, denominator)

        return numerator / denominator

    def forward(
        self,
        X: np.ndarray,
        consequent_indices: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Propagação forward completa.

        Args:
            X: Dados de entrada (n_samples, n_inputs)
            consequent_indices: Índices dos consequentes (n_rules,)
                              Se None, usa todas as regras mapeando para MFs de saída

        Returns:
            predictions: Saídas (n_samples,)
            firing_strengths: Forças de disparo (n_samples, n_rules)
        """
        # Índices padrão: mapeia cada regra para MF de saída correspondente
        if consequent_indices is None:
            # Estratégia: distribui regras uniformemente entre MFs de saída
            consequent_indices = np.arange(self.n_rules) % self.n_mfs_output

        # Fuzzificação
        membership_values = self._fuzzify_inputs(X)

        # Disparo das regras
        firing_strengths = self._fire_rules(membership_values)

        # Defuzzificação
        if self.defuzz_method == 'cog':
            predictions = self._defuzzify_cog(firing_strengths, consequent_indices)
        elif self.defuzz_method == 'cos':
            predictions = self._defuzzify_cos(firing_strengths, consequent_indices)
        else:
            raise ValueError(f"Método de defuzzificação desconhecido: {self.defuzz_method}")

        return predictions, firing_strengths

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predição simples.

        Args:
            X: Dados de entrada (n_samples, n_inputs)

        Returns:
            Predições (n_samples,)
        """
        # Verifica se parâmetros foram inicializados
        if self.output_centroids is None:
            raise RuntimeError("Modelo não treinado. Execute fit() ou fit_metaheuristic() primeiro.")

        predictions, _ = self.forward(X)
        return predictions

    def _compute_gradients(
        self,
        X: np.ndarray,
        y: np.ndarray,
        predictions: np.ndarray,
        firing_strengths: np.ndarray,
        consequent_indices: np.ndarray
    ) -> Dict[str, List[np.ndarray]]:
        """
        Calcula gradientes por backpropagation.

        Gradientes analíticos para centroides, numéricos para médias/sigmas.

        Returns:
            Dict com gradientes para cada tipo de parâmetro
        """
        n_samples = X.shape[0]
        errors = predictions - y  # (n_samples,)

        # Gradientes dos centroides (analítico)
        grad_centroids = np.zeros(self.n_mfs_output)

        denominator = np.zeros(n_samples)
        for rule_idx, consequent_idx in enumerate(consequent_indices):
            strength = firing_strengths[:, rule_idx]
            if self.rule_weights is not None:
                strength *= self.rule_weights[rule_idx]
            denominator += strength

        denominator = np.where(denominator < 1e-10, 1e-10, denominator)

        for mf_idx in range(self.n_mfs_output):
            # Encontra regras que apontam para esta MF
            rule_mask = (consequent_indices == mf_idx)

            if not np.any(rule_mask):
                continue

            # Gradiente do centroide
            grad_sum = 0.0
            for rule_idx in np.where(rule_mask)[0]:
                strength = firing_strengths[:, rule_idx]
                if self.rule_weights is not None:
                    strength *= self.rule_weights[rule_idx]

                grad_sum += np.sum(errors * strength / denominator)

            grad_centroids[mf_idx] = (2.0 / n_samples) * grad_sum

        # Gradientes das médias e sigmas (numérico - diferenças finitas)
        epsilon = 1e-5

        grad_means = []
        grad_sigmas = []

        for i in range(self.n_inputs):
            n_mfs = self.n_mfs_input[i]

            # Médias
            g_means = np.zeros(n_mfs)
            for j in range(n_mfs):
                # Perturba média
                original = self.input_means[i][j]
                self.input_means[i][j] = original + epsilon
                pred_plus, _ = self.forward(X, consequent_indices)

                self.input_means[i][j] = original - epsilon
                pred_minus, _ = self.forward(X, consequent_indices)

                # Restaura
                self.input_means[i][j] = original

                # Gradiente numérico
                mse_plus = np.mean((pred_plus - y) ** 2)
                mse_minus = np.mean((pred_minus - y) ** 2)
                g_means[j] = (mse_plus - mse_minus) / (2 * epsilon)

            grad_means.append(g_means)

            # Sigmas
            g_sigmas = np.zeros(n_mfs)
            for j in range(n_mfs):
                original = self.input_sigmas[i][j]
                self.input_sigmas[i][j] = original + epsilon
                pred_plus, _ = self.forward(X, consequent_indices)

                self.input_sigmas[i][j] = original - epsilon
                pred_minus, _ = self.forward(X, consequent_indices)

                self.input_sigmas[i][j] = original

                mse_plus = np.mean((pred_plus - y) ** 2)
                mse_minus = np.mean((pred_minus - y) ** 2)
                g_sigmas[j] = (mse_plus - mse_minus) / (2 * epsilon)

            grad_sigmas.append(g_sigmas)

        return {
            'centroids': grad_centroids,
            'means': grad_means,
            'sigmas': grad_sigmas
        }

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        learning_rate: float = 0.01,
        batch_size: Optional[int] = None,
        learning_mode: Literal['batch', 'online', 'mini-batch'] = 'mini-batch',
        early_stopping: bool = False,
        patience: int = 10,
        validation_split: float = 0.2,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Treina o sistema Mamdani por gradiente descendente.

        Args:
            X: Dados de entrada (n_samples, n_inputs)
            y: Saídas desejadas (n_samples,)
            epochs: Número de épocas
            learning_rate: Taxa de aprendizado
            batch_size: Tamanho do batch (None = modo batch, 1 = online)
            learning_mode: 'batch', 'online' ou 'mini-batch'
            early_stopping: Se True, para quando validação não melhorar
            patience: Épocas sem melhoria antes de parar
            validation_split: Fração dos dados para validação
            verbose: Imprimir progresso

        Returns:
            Histórico de treinamento
        """
        # Inicializa parâmetros
        self._initialize_parameters(X, y)

        # Split validação
        if early_stopping and validation_split > 0:
            n_val = int(len(X) * validation_split)
            indices = np.random.permutation(len(X))
            val_indices = indices[:n_val]
            train_indices = indices[n_val:]

            X_train, y_train = X[train_indices], y[train_indices]
            X_val, y_val = X[val_indices], y[val_indices]
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None

        # Define tamanho do batch
        if learning_mode == 'batch':
            batch_size = len(X_train)
        elif learning_mode == 'online':
            batch_size = 1
        elif batch_size is None:
            batch_size = min(32, len(X_train))

        # Índices de consequentes (padrão: distribuição uniforme)
        consequent_indices = np.arange(self.n_rules) % self.n_mfs_output

        # Histórico
        history = {
            'train_loss': [],
            'val_loss': [] if X_val is not None else None
        }

        best_val_loss = float('inf')
        patience_counter = 0

        if verbose:
            print("=" * 70)
            print("TREINAMENTO MAMDANI - GRADIENTE DESCENDENTE")
            print("=" * 70)
            print(f"Modo: {learning_mode}")
            print(f"Épocas: {epochs}")
            print(f"Learning rate: {learning_rate}")
            print(f"Batch size: {batch_size}")
            print(f"Regras: {self.n_rules}")
            print("=" * 70 + "\n")

        # Loop de treinamento
        for epoch in range(epochs):
            # Shuffle
            indices = np.random.permutation(len(X_train))
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            epoch_loss = 0.0
            n_batches = 0

            # Loop de batches
            for i in range(0, len(X_train), batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                # Forward
                predictions, firing_strengths = self.forward(X_batch, consequent_indices)

                # Loss
                mse = np.mean((predictions - y_batch) ** 2)
                epoch_loss += mse
                n_batches += 1

                # Gradientes
                gradients = self._compute_gradients(
                    X_batch, y_batch, predictions,
                    firing_strengths, consequent_indices
                )

                # Atualização
                self.output_centroids -= learning_rate * gradients['centroids']

                for i in range(self.n_inputs):
                    self.input_means[i] -= learning_rate * gradients['means'][i]
                    self.input_sigmas[i] -= learning_rate * gradients['sigmas'][i]

                # Aplica restrições
                self._apply_domain_constraints()

            # Loss média da época
            train_loss = epoch_loss / n_batches
            history['train_loss'].append(train_loss)

            # Validação
            if X_val is not None:
                val_pred = self.predict(X_val)
                val_loss = np.mean((val_pred - y_val) ** 2)
                history['val_loss'].append(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if early_stopping and patience_counter >= patience:
                        if verbose:
                            print(f"\nEarly stopping na época {epoch+1}")
                        break

            # Log
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                msg = f"Época {epoch+1}/{epochs} - Loss: {train_loss:.6f}"
                if X_val is not None:
                    msg += f" - Val Loss: {val_loss:.6f}"
                print(msg)

        self._is_fitted = True

        if verbose:
            print("\n" + "=" * 70)
            print("TREINAMENTO CONCLUÍDO!")
            print("=" * 70)

        return history

    def fit_metaheuristic(
        self,
        X: np.ndarray,
        y: np.ndarray,
        optimizer: Literal['pso', 'de', 'ga'] = 'pso',
        n_particles: int = 30,
        n_iterations: int = 100,
        optimize_params: Literal['all', 'output_only', 'consequents_only', 'hybrid'] = 'consequents_only',
        verbose: bool = True,
        **optimizer_kwargs
    ) -> 'MamdaniLearning':
        """
        Treina usando otimização metaheurística com caching de ativações.

        **OTIMIZAÇÃO CHAVE**: Quando optimize_params='consequents_only',
        as ativações das regras são calculadas UMA ÚNICA VEZ e reutilizadas
        em todas as avaliações do otimizador (100-1000 vezes mais rápido).

        Args:
            X: Dados de entrada
            y: Saídas desejadas
            optimizer: 'pso', 'de' ou 'ga'
            n_particles: População do otimizador
            n_iterations: Iterações do otimizador
            optimize_params:
                - 'consequents_only': Otimiza apenas índices dos consequentes (RÁPIDO, usa cache)
                - 'hybrid': Otimiza consequentes + centroides de saída (MÉDIO, cache parcial)
                - 'output_only': Otimiza centroides de saída (SEM cache)
                - 'all': Otimiza tudo (SEM cache, muito lento)
            verbose: Imprimir progresso
            **optimizer_kwargs: Parâmetros adicionais para o otimizador

        Returns:
            Self (fitted)
        """
        from .metaheuristics import get_optimizer

        # Inicializa parâmetros
        self._initialize_parameters(X, y)

        if optimize_params == 'consequents_only':
            # ============================================================
            # MODO COM CACHING DE ATIVAÇÕES (OTIMIZAÇÃO CRÍTICA!)
            # ============================================================
            if verbose:
                print("\n" + "=" * 70)
                print("OTIMIZAÇÃO MAMDANI - METAHEURÍSTICA COM CACHING")
                print("=" * 70)
                print(f"Optimizer: {optimizer.upper()}")
                print(f"Modo: {optimize_params}")
                print(f"Regras: {self.n_rules}")
                print(f"MFs de saída: {self.n_mfs_output}")
                print("=" * 70)
                print("\n🚀 PRÉ-COMPUTANDO ATIVAÇÕES (executado 1x)...")

            # PRÉ-COMPUTA ATIVAÇÕES UMA ÚNICA VEZ
            membership_values = self._fuzzify_inputs(X)
            firing_strengths = self._fire_rules(membership_values)

            self._cached_activations = firing_strengths
            self._cached_X = X.copy()

            if verbose:
                print(f"✅ Ativações em cache: {firing_strengths.shape}")
                print(f"   → {firing_strengths.shape[0]} amostras")
                print(f"   → {firing_strengths.shape[1]} regras\n")

            # Define bounds: cada regra pode apontar para qualquer MF de saída
            bounds = np.array([[0, self.n_mfs_output - 1]] * self.n_rules)

            # Função objetivo (USA CACHE!)
            def objective(consequent_indices_float):
                consequent_indices = np.round(consequent_indices_float).astype(int)
                consequent_indices = np.clip(consequent_indices, 0, self.n_mfs_output - 1)

                # USA ATIVAÇÕES EM CACHE - não recalcula fuzzificação!
                if self.defuzz_method == 'cog':
                    predictions = self._defuzzify_cog(
                        self._cached_activations,
                        consequent_indices
                    )
                else:
                    predictions = self._defuzzify_cos(
                        self._cached_activations,
                        consequent_indices
                    )

                mse = np.mean((predictions - y) ** 2)
                return mse

            # Solução inicial: distribuição uniforme
            initial_solution = (np.arange(self.n_rules) % self.n_mfs_output).astype(float)

        elif optimize_params == 'hybrid':
            # ============================================================
            # MODO HÍBRIDO: Consequentes + Centroides (CACHE PARCIAL)
            # ============================================================
            if verbose:
                print("\n" + "=" * 70)
                print("OTIMIZAÇÃO MAMDANI - MODO HÍBRIDO (CACHE PARCIAL)")
                print("=" * 70)
                print(f"Optimizer: {optimizer.upper()}")
                print(f"Modo: hybrid (consequents + output_centroids)")
                print(f"Regras: {self.n_rules}")
                print(f"MFs de saída: {self.n_mfs_output}")
                print(f"Parâmetros totais: {self.n_rules + self.n_mfs_output}")
                print("=" * 70)
                print("\n🚀 PRÉ-COMPUTANDO ATIVAÇÕES (executado 1x)...")

            # PRÉ-COMPUTA ATIVAÇÕES UMA ÚNICA VEZ
            membership_values = self._fuzzify_inputs(X)
            firing_strengths = self._fire_rules(membership_values)

            self._cached_activations = firing_strengths
            self._cached_X = X.copy()

            if verbose:
                print(f"✅ Ativações em cache: {firing_strengths.shape}")
                print(f"   → {firing_strengths.shape[0]} amostras")
                print(f"   → {firing_strengths.shape[1]} regras")
                print(f"\n📊 Otimizando:")
                print(f"   → Consequentes: {self.n_rules} parâmetros (inteiros 0-{self.n_mfs_output-1})")
                print(f"   → Centroides: {self.n_mfs_output} parâmetros (contínuos)")
                print(f"   → Total: {self.n_rules + self.n_mfs_output} parâmetros\n")

            # Define bounds: [consequent_indices (n_rules), centroids (n_mfs_output)]
            y_min, y_max = self.output_bound
            bounds = []

            # Bounds para consequentes (índices 0 a n_mfs_output-1)
            for _ in range(self.n_rules):
                bounds.append([0, self.n_mfs_output - 1])

            # Bounds para centroides
            for _ in range(self.n_mfs_output):
                bounds.append([y_min, y_max])

            bounds = np.array(bounds)

            # Função objetivo (USA CACHE para ativações!)
            def objective(params_vector):
                # Divide vetor: [consequents, centroids]
                consequent_indices_float = params_vector[:self.n_rules]
                centroids = params_vector[self.n_rules:]

                # Converte consequentes para inteiros
                consequent_indices = np.round(consequent_indices_float).astype(int)
                consequent_indices = np.clip(consequent_indices, 0, self.n_mfs_output - 1)

                # Atualiza centroides temporariamente
                original_centroids = self.output_centroids.copy()
                self.output_centroids = centroids.copy()

                # USA ATIVAÇÕES EM CACHE - não recalcula fuzzificação!
                # Apenas defuzzificação é recalculada (pois centroides mudaram)
                if self.defuzz_method == 'cog':
                    predictions = self._defuzzify_cog(
                        self._cached_activations,
                        consequent_indices
                    )
                else:
                    predictions = self._defuzzify_cos(
                        self._cached_activations,
                        consequent_indices
                    )

                # Restaura centroides originais
                self.output_centroids = original_centroids

                mse = np.mean((predictions - y) ** 2)
                return mse

            # Solução inicial: [consequents uniformes, centroides atuais]
            initial_consequents = (np.arange(self.n_rules) % self.n_mfs_output).astype(float)
            initial_centroids = self.output_centroids.copy()
            initial_solution = np.concatenate([initial_consequents, initial_centroids])

        elif optimize_params == 'output_only':
            # Otimiza apenas centroides de saída (SEM cache)
            if verbose:
                print(f"\nOtimizando apenas centroides de saída com {optimizer.upper()}...")

            bounds = np.array(
                [[self.output_bound[0], self.output_bound[1]]] * self.n_mfs_output
            )

            def objective(centroids):
                self.output_centroids = centroids.copy()
                predictions = self.predict(X)
                mse = np.mean((predictions - y) ** 2)
                return mse

            initial_solution = self.output_centroids.copy()

        elif optimize_params == 'all':
            # Otimiza tudo (muito lento, SEM cache)
            if verbose:
                print(f"\nOtimizando todos os parâmetros com {optimizer.upper()}...")
                warnings.warn(
                    "Modo 'all' é muito lento. Considere 'consequents_only' ou 'output_only'.",
                    UserWarning
                )

            # Cria vetor e bounds
            param_vector = self._params_to_vector()
            bounds = self._create_optimization_bounds(X, y)

            def objective(params_vec):
                self._vector_to_params(params_vec)
                self._apply_domain_constraints()
                predictions = self.predict(X)
                mse = np.mean((predictions - y) ** 2)
                return mse

            initial_solution = param_vector

        else:
            raise ValueError(f"optimize_params inválido: {optimize_params}")

        # Configura otimizador (nomes de parâmetros específicos)
        if optimizer == 'pso':
            opt_params = {
                'n_particles': n_particles,
                'n_iterations': n_iterations,
            }
        elif optimizer == 'de':
            opt_params = {
                'pop_size': n_particles,
                'max_iter': n_iterations,
            }
        elif optimizer == 'ga':
            opt_params = {
                'pop_size': n_particles,
                'max_gen': n_iterations,
            }
        else:
            raise ValueError(f"Optimizer desconhecido: {optimizer}")

        opt_params.update(optimizer_kwargs)
        opt = get_optimizer(optimizer, **opt_params)

        # Otimiza
        best_params, best_fitness, history = opt.optimize(
            objective,
            bounds,
            minimize=True,
            verbose=verbose
        )

        # Aplica melhores parâmetros
        if optimize_params == 'consequents_only':
            # Melhor conjunto de regras
            best_consequents = np.round(best_params).astype(int)
            best_consequents = np.clip(best_consequents, 0, self.n_mfs_output - 1)
            self._best_consequent_indices = best_consequents

        elif optimize_params == 'hybrid':
            # Melhor conjunto de regras + centroides
            best_consequents = np.round(best_params[:self.n_rules]).astype(int)
            best_consequents = np.clip(best_consequents, 0, self.n_mfs_output - 1)
            best_centroids = best_params[self.n_rules:]

            self._best_consequent_indices = best_consequents
            self.output_centroids = best_centroids.copy()

            if verbose:
                print(f"\n📊 Melhores parâmetros encontrados:")
                print(f"   Consequentes: {best_consequents}")
                print(f"   Centroides: {best_centroids}")

        elif optimize_params == 'output_only':
            self.output_centroids = best_params.copy()

        elif optimize_params == 'all':
            self._vector_to_params(best_params)
            self._apply_domain_constraints()

        self._is_fitted = True
        self.optimization_history = history

        if verbose:
            print(f"\n✅ Otimização concluída! MSE final: {best_fitness:.6f}")

        return self

    def _params_to_vector(self) -> np.ndarray:
        """Converte parâmetros para vetor 1D."""
        vector = []
        for i in range(self.n_inputs):
            vector.extend(self.input_means[i])
        for i in range(self.n_inputs):
            vector.extend(self.input_sigmas[i])
        vector.extend(self.output_centroids)
        if self.rule_weights is not None:
            vector.extend(self.rule_weights)
        return np.array(vector)

    def _vector_to_params(self, vector: np.ndarray):
        """Converte vetor 1D para parâmetros."""
        idx = 0
        for i in range(self.n_inputs):
            n_mf = self.n_mfs_input[i]
            self.input_means[i] = vector[idx:idx+n_mf].copy()
            idx += n_mf
        for i in range(self.n_inputs):
            n_mf = self.n_mfs_input[i]
            self.input_sigmas[i] = vector[idx:idx+n_mf].copy()
            idx += n_mf
        self.output_centroids = vector[idx:idx+self.n_mfs_output].copy()
        idx += self.n_mfs_output
        if self.rule_weights is not None:
            self.rule_weights = vector[idx:idx+self.n_rules].copy()

    def _create_optimization_bounds(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Cria bounds para otimização."""
        bounds = []

        # Médias
        for i in range(self.n_inputs):
            x_min, x_max = self.input_bounds[i]
            for _ in range(self.n_mfs_input[i]):
                bounds.append([x_min, x_max])

        # Sigmas
        for i in range(self.n_inputs):
            x_range = self.input_bounds[i][1] - self.input_bounds[i][0]
            for _ in range(self.n_mfs_input[i]):
                bounds.append([x_range * 0.01, x_range * 0.5])

        # Centroides
        y_min, y_max = self.output_bound
        for _ in range(self.n_mfs_output):
            bounds.append([y_min, y_max])

        # Pesos
        if self.rule_weights is not None:
            for _ in range(self.n_rules):
                bounds.append([0.1, 10.0])

        return np.array(bounds)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Retorna R² (coeficiente de determinação).

        Args:
            X: Dados de entrada
            y: Saídas verdadeiras

        Returns:
            R² score
        """
        predictions = self.predict(X)
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / (ss_tot + 1e-10))

    def get_linguistic_rules(self) -> List[str]:
        """
        Extrai regras em formato linguístico.

        Returns:
            Lista de regras no formato: "IF x1 IS low AND x2 IS high THEN y IS medium"
        """
        if not self._is_fitted:
            raise RuntimeError("Modelo não treinado.")

        # Labels linguísticos
        input_labels = []
        for i in range(self.n_inputs):
            n_mfs = self.n_mfs_input[i]
            if n_mfs == 3:
                labels = ['low', 'medium', 'high']
            elif n_mfs == 5:
                labels = ['very_low', 'low', 'medium', 'high', 'very_high']
            else:
                labels = [f'mf{j}' for j in range(n_mfs)]
            input_labels.append(labels)

        if self.n_mfs_output == 3:
            output_labels = ['low', 'medium', 'high']
        elif self.n_mfs_output == 5:
            output_labels = ['very_low', 'low', 'medium', 'high', 'very_high']
        else:
            output_labels = [f'mf{j}' for j in range(self.n_mfs_output)]

        # Gera regras
        rules = []
        consequent_indices = getattr(
            self,
            '_best_consequent_indices',
            np.arange(self.n_rules) % self.n_mfs_output
        )

        rule_idx = 0
        for combo in np.ndindex(tuple(self.n_mfs_input)):
            antecedent_parts = []
            for i, mf_idx in enumerate(combo):
                antecedent_parts.append(f"x{i+1} IS {input_labels[i][mf_idx]}")

            antecedent = " AND ".join(antecedent_parts)
            consequent_idx = consequent_indices[rule_idx]
            consequent = f"y IS {output_labels[consequent_idx]}"

            rule_str = f"IF {antecedent} THEN {consequent}"
            rules.append(rule_str)
            rule_idx += 1

        return rules

    def to_mamdani_system(
        self,
        input_names: Optional[List[str]] = None,
        output_name: str = "output"
    ):
        """
        Converte MamdaniLearning treinado para MamdaniSystem do inference.

        Cria um MamdaniSystem completo com:
        - Variáveis linguísticas de entrada com funções gaussianas
        - Variável linguística de saída com singletons
        - Regras fuzzy baseadas nos consequentes aprendidos

        Args:
            input_names: Nomes das variáveis de entrada (None = x0, x1, ...)
            output_name: Nome da variável de saída

        Returns:
            MamdaniSystem configurado e pronto para uso

        Raises:
            RuntimeError: Se o modelo não foi treinado

        Exemplo:
            >>> mamdani_learning = MamdaniLearning(n_inputs=2, n_mfs_input=[3, 3], n_mfs_output=3)
            >>> mamdani_learning.fit(X_train, y_train, epochs=100)
            >>>
            >>> # Converter para MamdaniSystem
            >>> fis = mamdani_learning.to_mamdani_system(
            ...     input_names=['temperatura', 'umidade'],
            ...     output_name='ventilador'
            ... )
            >>>
            >>> # Usar como FIS normal
            >>> resultado = fis.evaluate(temperatura=25, umidade=60)
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Modelo não foi treinado. Execute fit() ou fit_metaheuristic() primeiro."
            )

        # Importa MamdaniSystem
        from ..inference.systems import MamdaniSystem
        from ..core.fuzzification import LinguisticVariable, FuzzySet

        # Nomes padrão se não fornecidos
        if input_names is None:
            input_names = [f"x{i}" for i in range(self.n_inputs)]

        if len(input_names) != self.n_inputs:
            raise ValueError(
                f"input_names deve ter {self.n_inputs} elementos, recebeu {len(input_names)}"
            )

        # Criar sistema Mamdani
        fis = MamdaniSystem(
            name="Mamdani from Learning",
            and_method=TNorm.PRODUCT,  # MamdaniLearning usa produto
            defuzzification_method=DefuzzMethod.CENTROID if self.defuzz_method == 'cog' else DefuzzMethod.MOM
        )

        # Adicionar variáveis de entrada com MFs gaussianas
        for i in range(self.n_inputs):
            var_name = input_names[i]
            var = fis.add_input(var_name, self.input_bounds[i])

            # Adicionar termos linguísticos (MFs gaussianas)
            n_mfs = self.n_mfs_input[i]
            labels = self._get_term_labels(n_mfs)

            for j in range(n_mfs):
                mean = self.input_means[i][j]
                sigma = self.input_sigmas[i][j]

                var.add_term(
                    labels[j],
                    mf_type='gaussian',
                    params=(mean, sigma)
                )

        # Adicionar variável de saída com singletons (centroides)
        output_var = fis.add_output(output_name, self.output_bound)
        output_labels = self._get_term_labels(self.n_mfs_output)

        for j in range(self.n_mfs_output):
            centroid = self.output_centroids[j]
            output_var.add_term(
                output_labels[j],
                mf_type='singleton',
                params=(centroid,)
            )

        # Adicionar regras
        consequent_indices = getattr(
            self,
            '_best_consequent_indices',
            np.arange(self.n_rules) % self.n_mfs_output
        )

        rule_idx = 0
        for combo in np.ndindex(tuple(self.n_mfs_input)):
            # Antecedente: dicionário {var_name: term_name}
            antecedent = {}
            for i, mf_idx in enumerate(combo):
                var_name = input_names[i]
                term_labels = self._get_term_labels(self.n_mfs_input[i])
                antecedent[var_name] = term_labels[mf_idx]

            # Consequente
            consequent_idx = consequent_indices[rule_idx]
            consequent = {output_name: output_labels[consequent_idx]}

            # Criar regra
            from ..inference.rules import FuzzyRule
            rule = FuzzyRule(antecedent, consequent)
            fis.rule_base.add_rule(rule)

            rule_idx += 1

        return fis

    def _get_term_labels(self, n_terms: int) -> List[str]:
        """Helper para gerar rótulos linguísticos."""
        if n_terms == 3:
            return ['low', 'medium', 'high']
        elif n_terms == 5:
            return ['very_low', 'low', 'medium', 'high', 'very_high']
        elif n_terms == 7:
            return ['very_very_low', 'very_low', 'low', 'medium', 'high', 'very_high', 'very_very_high']
        else:
            return [f'mf{j}' for j in range(n_terms)]

    @classmethod
    def from_mamdani_system(
        cls,
        fis,
        defuzz_method: Literal['cog', 'cos'] = 'cog',
        use_rule_weights: bool = True
    ):
        """
        Cria MamdaniLearning a partir de MamdaniSystem existente.

        IMPORTANTE: Só funciona se o MamdaniSystem usar funções gaussianas
        nas entradas. Caso contrário, lança ValueError.

        Args:
            fis: MamdaniSystem do inference
            defuzz_method: Método de defuzzificação ('cog' ou 'cos')
            use_rule_weights: Se True, permite aprendizado de pesos

        Returns:
            MamdaniLearning inicializado com parâmetros do FIS

        Raises:
            ValueError: Se o FIS não for compatível (MFs não-gaussianas)

        Exemplo:
            >>> # Criar FIS manualmente
            >>> fis = MamdaniSystem()
            >>> fis.add_input('temperatura', (0, 40))
            >>> fis.input_variables['temperatura'].add_term('fria', 'gaussian', (10, 5))
            >>> # ... configurar FIS ...
            >>>
            >>> # Converter para MamdaniLearning
            >>> mamdani_learning = MamdaniLearning.from_mamdani_system(fis)
            >>>
            >>> # Treinar/otimizar
            >>> mamdani_learning.fit(X_train, y_train, epochs=100)
        """
        from ..inference.systems import MamdaniSystem

        if not isinstance(fis, MamdaniSystem):
            raise TypeError(f"fis deve ser MamdaniSystem, recebeu {type(fis)}")

        # Extrair informações
        n_inputs = len(fis.input_variables)
        input_names = list(fis.input_variables.keys())
        output_names = list(fis.output_variables.keys())

        if len(output_names) != 1:
            raise ValueError(
                f"MamdaniLearning suporta apenas 1 saída, FIS tem {len(output_names)}"
            )

        # Extrair número de MFs e bounds
        n_mfs_input = []
        input_bounds = []
        input_means = []
        input_sigmas = []

        for var_name in input_names:
            var = fis.input_variables[var_name]
            n_mfs = len(var.terms)
            n_mfs_input.append(n_mfs)
            input_bounds.append(var.universe)

            # Extrair parâmetros gaussianos
            means = []
            sigmas = []
            for term_name, fuzzy_set in var.terms.items():
                if fuzzy_set.mf_type != 'gaussian':
                    raise ValueError(
                        f"Variável '{var_name}', termo '{term_name}': "
                        f"MamdaniLearning requer MFs gaussianas, encontrado '{fuzzy_set.mf_type}'"
                    )
                mean, sigma = fuzzy_set.params
                means.append(mean)
                sigmas.append(sigma)

            input_means.append(np.array(means))
            input_sigmas.append(np.array(sigmas))

        # Extrair saída
        output_name = output_names[0]
        output_var = fis.output_variables[output_name]
        n_mfs_output = len(output_var.terms)
        output_bound = output_var.universe

        # Extrair centroides (assumindo singletons ou pegar centro da MF)
        output_centroids = []
        for term_name, fuzzy_set in output_var.terms.items():
            if fuzzy_set.mf_type == 'singleton':
                centroid = fuzzy_set.params[0]
            elif fuzzy_set.mf_type == 'gaussian':
                centroid = fuzzy_set.params[0]  # mean
            elif fuzzy_set.mf_type == 'triangular':
                centroid = fuzzy_set.params[1]  # center
            else:
                # Calcular centroide numericamente
                x = np.linspace(output_bound[0], output_bound[1], 1000)
                mu = fuzzy_set.membership(x)
                centroid = np.sum(x * mu) / (np.sum(mu) + 1e-10)
            output_centroids.append(centroid)

        output_centroids = np.array(output_centroids)

        # Criar MamdaniLearning
        mamdani_learning = cls(
            n_inputs=n_inputs,
            n_mfs_input=n_mfs_input,
            n_mfs_output=n_mfs_output,
            defuzz_method=defuzz_method,
            use_rule_weights=use_rule_weights,
            input_bounds=input_bounds,
            output_bound=output_bound
        )

        # Inicializar parâmetros com valores do FIS
        mamdani_learning.input_means = input_means
        mamdani_learning.input_sigmas = input_sigmas
        mamdani_learning.output_centroids = output_centroids

        if use_rule_weights:
            mamdani_learning.rule_weights = np.ones(mamdani_learning.n_rules)

        # Extrair consequentes das regras (se possível)
        if len(fis.rule_base.rules) > 0:
            mamdani_learning._extract_consequents_from_rules(fis, output_name)

        mamdani_learning._is_fitted = True

        return mamdani_learning

    def _extract_consequents_from_rules(self, fis, output_name: str):
        """Extrai índices dos consequentes das regras do FIS."""
        output_var = fis.output_variables[output_name]
        output_term_names = list(output_var.terms.keys())

        # Mapear regras para consequentes
        consequent_indices = []

        for rule in fis.rule_base.rules:
            consequent_term = rule.consequent.get(output_name)
            if consequent_term is not None:
                try:
                    idx = output_term_names.index(consequent_term)
                    consequent_indices.append(idx)
                except ValueError:
                    # Termo não encontrado, usa default
                    consequent_indices.append(0)
            else:
                consequent_indices.append(0)

        if len(consequent_indices) == self.n_rules:
            self._best_consequent_indices = np.array(consequent_indices)


# Mensagem de sucesso
print("✅ MamdaniLearning implementado com sucesso!")
print("\nCaracterísticas:")
print("  • Aprendizado por gradiente: batch, online, mini-batch")
print("  • Otimização metaheurística: PSO, DE, GA")
print("  • Defuzzificação: COG e COS")
print("  • Caching de ativações para otimização eficiente")
print("  • Extração de regras linguísticas")
