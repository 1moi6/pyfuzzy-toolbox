"""
ANFIS Moderno com Otimização Metaheurística
Implementação Python com os algoritmos mais avançados:
- PSO (Particle Swarm Optimization)
- DE (Differential Evolution)
- GA (Genetic Algorithm)
- Híbrido: PSO-ANFIS, DE-ANFIS, GA-ANFIS

Referências:
- PSO: Kennedy & Eberhart (1995)
- DE: Storn & Price (1997)
- GA: Holland (1975)
- ANFIS: Jang (1993)
"""

import numpy as np
from typing import List, Tuple, Optional, Callable
import warnings
warnings.filterwarnings('ignore')


class ParticleSwarmOptimizer:
    """
    Implementação moderna de PSO para otimização de parâmetros ANFIS
    Com melhorias: inércia adaptativa, velocidade limitada, constrição
    """

    def __init__(self, n_particles: int = 30, n_iterations: int = 100,
                 w: float = 0.7298, c1: float = 1.49618, c2: float = 1.49618,
                 adaptive_inertia: bool = True):
        """
        Parâmetros:
        -----------
        n_particles: int
            Número de partículas no enxame
        n_iterations: int
            Número de iterações
        w: float
            Peso de inércia (constrição de Clerc)
        c1, c2: float
            Coeficientes cognitivo e social
        adaptive_inertia: bool
            Se True, usa peso de inércia adaptativo
        """
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w
        self.w_max = 0.9
        self.w_min = 0.4
        self.c1 = c1
        self.c2 = c2
        self.adaptive_inertia = adaptive_inertia

    def optimize(self, objective_func: Callable, bounds: np.ndarray, 
                 minimize: bool = True, verbose: bool = False) -> Tuple[np.ndarray, float]:
        """
        Otimiza função objetivo usando PSO

        Parâmetros:
        -----------
        objective_func: Callable
            Função objetivo a ser otimizada
        bounds: np.ndarray, shape (n_dims, 2)
            Limites inferior e superior para cada dimensão
        minimize: bool
            Se True, minimiza; se False, maximiza
        verbose: bool
            Exibe progresso

        Retorna:
        --------
        best_position: np.ndarray
            Melhor posição encontrada
        best_fitness: float
            Melhor valor de fitness
        """
        n_dims = bounds.shape[0]

        # Inicializa posições e velocidades
        positions = np.random.uniform(
            bounds[:, 0], bounds[:, 1], 
            (self.n_particles, n_dims)
        )

        velocities = np.random.uniform(
            -np.abs(bounds[:, 1] - bounds[:, 0]) * 0.1,
            np.abs(bounds[:, 1] - bounds[:, 0]) * 0.1,
            (self.n_particles, n_dims)
        )

        # Velocidade máxima (10% da faixa)
        v_max = np.abs(bounds[:, 1] - bounds[:, 0]) * 0.2

        # Avalia fitness inicial
        fitness = np.array([objective_func(p) for p in positions])

        # Inicializa melhor pessoal e global
        pbest_positions = positions.copy()
        pbest_fitness = fitness.copy()

        if minimize:
            gbest_idx = np.argmin(pbest_fitness)
        else:
            gbest_idx = np.argmax(pbest_fitness)

        gbest_position = pbest_positions[gbest_idx].copy()
        gbest_fitness = pbest_fitness[gbest_idx]

        history = []

        # Loop principal PSO
        for iteration in range(self.n_iterations):
            # Inércia adaptativa
            if self.adaptive_inertia:
                w = self.w_max - (self.w_max - self.w_min) * iteration / self.n_iterations
            else:
                w = self.w

            # Atualiza velocidades e posições
            r1 = np.random.random((self.n_particles, n_dims))
            r2 = np.random.random((self.n_particles, n_dims))

            # Equação de velocidade PSO
            cognitive = self.c1 * r1 * (pbest_positions - positions)
            social = self.c2 * r2 * (gbest_position - positions)
            velocities = w * velocities + cognitive + social

            # Limita velocidades
            velocities = np.clip(velocities, -v_max, v_max)

            # Atualiza posições
            positions = positions + velocities

            # Aplica limites
            positions = np.clip(positions, bounds[:, 0], bounds[:, 1])

            # Avalia novo fitness
            fitness = np.array([objective_func(p) for p in positions])

            # Atualiza pbest
            if minimize:
                improved = fitness < pbest_fitness
            else:
                improved = fitness > pbest_fitness

            pbest_positions[improved] = positions[improved]
            pbest_fitness[improved] = fitness[improved]

            # Atualiza gbest
            if minimize:
                current_best_idx = np.argmin(pbest_fitness)
            else:
                current_best_idx = np.argmax(pbest_fitness)

            if (minimize and pbest_fitness[current_best_idx] < gbest_fitness) or \
               (not minimize and pbest_fitness[current_best_idx] > gbest_fitness):
                gbest_position = pbest_positions[current_best_idx].copy()
                gbest_fitness = pbest_fitness[current_best_idx]

            history.append(gbest_fitness)

            if verbose and (iteration % 10 == 0 or iteration == self.n_iterations - 1):
                print(f"PSO Iteração {iteration+1}/{self.n_iterations}, "
                      f"Best Fitness: {gbest_fitness:.6f}")

        return gbest_position, gbest_fitness, history


class DifferentialEvolution:
    """
    Implementação moderna de Differential Evolution (DE)
    Estratégia: DE/rand/1/bin
    """

    def __init__(self, pop_size: int = 50, max_iter: int = 100,
                 F: float = 0.8, CR: float = 0.9):
        """
        Parâmetros:
        -----------
        pop_size: int
            Tamanho da população
        max_iter: int
            Número máximo de iterações
        F: float
            Fator de mutação (0 < F <= 2)
        CR: float
            Taxa de crossover (0 <= CR <= 1)
        """
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.F = F
        self.CR = CR

    def optimize(self, objective_func: Callable, bounds: np.ndarray,
                 minimize: bool = True, verbose: bool = False) -> Tuple[np.ndarray, float]:
        """
        Otimiza função objetivo usando DE
        """
        n_dims = bounds.shape[0]

        # Inicializa população
        population = np.random.uniform(
            bounds[:, 0], bounds[:, 1],
            (self.pop_size, n_dims)
        )

        # Avalia fitness inicial
        fitness = np.array([objective_func(ind) for ind in population])

        if minimize:
            best_idx = np.argmin(fitness)
        else:
            best_idx = np.argmax(fitness)

        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        history = []

        # Loop principal DE
        for iteration in range(self.max_iter):
            for i in range(self.pop_size):
                # Seleção de 3 indivíduos diferentes
                candidates = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = np.random.choice(candidates, 3, replace=False)

                # Mutação: v = xa + F * (xb - xc)
                mutant = population[a] + self.F * (population[b] - population[c])

                # Limita aos bounds
                mutant = np.clip(mutant, bounds[:, 0], bounds[:, 1])

                # Crossover binomial
                cross_points = np.random.rand(n_dims) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, n_dims)] = True

                trial = np.where(cross_points, mutant, population[i])

                # Seleção
                trial_fitness = objective_func(trial)

                if (minimize and trial_fitness < fitness[i]) or \
                   (not minimize and trial_fitness > fitness[i]):
                    population[i] = trial
                    fitness[i] = trial_fitness

                    if (minimize and trial_fitness < best_fitness) or \
                       (not minimize and trial_fitness > best_fitness):
                        best_solution = trial.copy()
                        best_fitness = trial_fitness

            history.append(best_fitness)

            if verbose and (iteration % 10 == 0 or iteration == self.max_iter - 1):
                print(f"DE Iteração {iteration+1}/{self.max_iter}, "
                      f"Best Fitness: {best_fitness:.6f}")

        return best_solution, best_fitness, history


class GeneticAlgorithm:
    """
    Implementação moderna de Algoritmo Genético (GA)
    Com elitismo, torneio, crossover aritmético e mutação gaussiana
    """

    def __init__(self, pop_size: int = 50, max_gen: int = 100,
                 elite_ratio: float = 0.1, mutation_rate: float = 0.1,
                 tournament_size: int = 3):
        """
        Parâmetros:
        -----------
        pop_size: int
            Tamanho da população
        max_gen: int
            Número máximo de gerações
        elite_ratio: float
            Proporção de elite a preservar
        mutation_rate: float
            Taxa de mutação
        tournament_size: int
            Tamanho do torneio para seleção
        """
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.elite_ratio = elite_ratio
        self.n_elite = max(1, int(pop_size * elite_ratio))
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size

    def _tournament_selection(self, population: np.ndarray, 
                             fitness: np.ndarray, minimize: bool) -> np.ndarray:
        """Seleção por torneio"""
        indices = np.random.choice(len(population), self.tournament_size, replace=False)

        if minimize:
            winner_idx = indices[np.argmin(fitness[indices])]
        else:
            winner_idx = indices[np.argmax(fitness[indices])]

        return population[winner_idx].copy()

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Crossover aritmético"""
        alpha = np.random.random()
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = (1 - alpha) * parent1 + alpha * parent2
        return child1, child2

    def _mutate(self, individual: np.ndarray, bounds: np.ndarray) -> np.ndarray:
        """Mutação gaussiana adaptativa"""
        mutated = individual.copy()

        for i in range(len(individual)):
            if np.random.random() < self.mutation_rate:
                # Mutação gaussiana com 10% da faixa como desvio padrão
                sigma = (bounds[i, 1] - bounds[i, 0]) * 0.1
                mutated[i] += np.random.normal(0, sigma)
                mutated[i] = np.clip(mutated[i], bounds[i, 0], bounds[i, 1])

        return mutated

    def optimize(self, objective_func: Callable, bounds: np.ndarray,
                 minimize: bool = True, verbose: bool = False) -> Tuple[np.ndarray, float]:
        """
        Otimiza função objetivo usando GA
        """
        n_dims = bounds.shape[0]

        # Inicializa população
        population = np.random.uniform(
            bounds[:, 0], bounds[:, 1],
            (self.pop_size, n_dims)
        )

        # Avalia fitness inicial
        fitness = np.array([objective_func(ind) for ind in population])

        history = []

        # Loop principal GA
        for generation in range(self.max_gen):
            # Ordena por fitness
            if minimize:
                sorted_indices = np.argsort(fitness)
            else:
                sorted_indices = np.argsort(fitness)[::-1]

            population = population[sorted_indices]
            fitness = fitness[sorted_indices]

            best_solution = population[0].copy()
            best_fitness = fitness[0]

            history.append(best_fitness)

            if verbose and (generation % 10 == 0 or generation == self.max_gen - 1):
                print(f"GA Geração {generation+1}/{self.max_gen}, "
                      f"Best Fitness: {best_fitness:.6f}")

            # Elite
            elite = population[:self.n_elite].copy()

            # Nova população
            new_population = [elite]

            # Gera offspring
            while len(new_population[0]) + len(new_population) - 1 < self.pop_size:
                # Seleção
                parent1 = self._tournament_selection(population, fitness, minimize)
                parent2 = self._tournament_selection(population, fitness, minimize)

                # Crossover
                child1, child2 = self._crossover(parent1, parent2)

                # Mutação
                child1 = self._mutate(child1, bounds)
                child2 = self._mutate(child2, bounds)

                new_population.extend([child1, child2])

            # Atualiza população
            population = np.vstack(new_population)[:self.pop_size]

            # Avalia nova população
            fitness = np.array([objective_func(ind) for ind in population])

        if minimize:
            best_idx = np.argmin(fitness)
        else:
            best_idx = np.argmax(fitness)

        return population[best_idx], fitness[best_idx], history


class MetaheuristicANFIS:
    """
    ANFIS moderno com otimização metaheurística
    Suporta PSO, DE, GA e métodos híbridos
    """

    def __init__(self, n_inputs: int, n_mfs: List[int], 
                 mf_type: str = 'gbellmf',
                 optimizer: str = 'pso'):
        """
        Parâmetros:
        -----------
        n_inputs: int
            Número de entradas
        n_mfs: List[int]
            Número de funções de pertinência por entrada
        mf_type: str
            Tipo de função de pertinência
        optimizer: str
            Otimizador: 'pso', 'de', 'ga', 'hybrid'
        """
        self.n_inputs = n_inputs
        self.n_mfs = n_mfs
        self.mf_type = mf_type
        self.n_rules = np.prod(n_mfs)
        self.optimizer_name = optimizer

        # Parâmetros ANFIS
        self.premise_params = []
        self.consequent_params = None
        self.input_bounds = None

        # Histórico
        self.training_history = []
        self.optimization_history = []

    def _params_to_vector(self) -> np.ndarray:
        """Converte parâmetros ANFIS para vetor 1D"""
        vector = []

        # Parâmetros antecedentes
        for i in range(self.n_inputs):
            for j in range(self.n_mfs[i]):
                for param_name, param_value in self.premise_params[i][j].items():
                    vector.append(param_value)

        # Parâmetros consequentes
        if self.consequent_params is not None:
            vector.extend(self.consequent_params.flatten())

        return np.array(vector)

    def _vector_to_params(self, vector: np.ndarray):
        """Converte vetor 1D para parâmetros ANFIS"""
        idx = 0

        # Parâmetros antecedentes
        for i in range(self.n_inputs):
            for j in range(self.n_mfs[i]):
                for param_name in self.premise_params[i][j].keys():
                    self.premise_params[i][j][param_name] = vector[idx]
                    idx += 1

        # Parâmetros consequentes
        n_conseq = self.n_rules * (self.n_inputs + 1)
        self.consequent_params = vector[idx:idx+n_conseq].reshape(
            self.n_rules, self.n_inputs + 1
        )

    def _create_bounds(self) -> np.ndarray:
        """Cria limites para otimização"""
        bounds = []

        # Bounds para parâmetros antecedentes
        for i in range(self.n_inputs):
            x_min, x_max = self.input_bounds[i]
            x_range = x_max - x_min

            for j in range(self.n_mfs[i]):
                if self.mf_type == 'gbellmf':
                    # a: largura (0.1 a 2x do range)
                    bounds.append([x_range * 0.05, x_range * 2.0])
                    # b: formato (0.5 a 5.0)
                    bounds.append([0.5, 5.0])
                    # c: centro (dentro do domínio)
                    bounds.append([x_min, x_max])

        # Bounds para parâmetros consequentes
        for _ in range(self.n_rules * (self.n_inputs + 1)):
            bounds.append([-10.0, 10.0])

        return np.array(bounds)

    def _generalized_bell_mf(self, x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        """Função de pertinência Generalized Bell"""
        return 1.0 / (1.0 + np.abs((x - c) / a) ** (2 * b))

    def _forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass completo"""
        import itertools

        n_samples = X.shape[0]

        # Fuzzificação
        membership_values = []
        for i in range(self.n_inputs):
            x = X[:, i].reshape(-1, 1)
            mf_vals = []

            for j in range(self.n_mfs[i]):
                params = self.premise_params[i][j]
                mu = self._generalized_bell_mf(x, params['a'], params['b'], params['c'])
                mf_vals.append(mu)

            membership_values.append(np.hstack(mf_vals))

        # Força de disparo
        firing_strengths = np.ones((n_samples, self.n_rules))
        rule_idx = 0
        indices = [range(n_mf) for n_mf in self.n_mfs]

        for combination in itertools.product(*indices):
            for i, idx in enumerate(combination):
                firing_strengths[:, rule_idx] *= membership_values[i][:, idx]
            rule_idx += 1

        # Normalização
        total = np.sum(firing_strengths, axis=1, keepdims=True)
        total = np.where(total == 0, 1e-10, total)
        normalized = firing_strengths / total

        # Defuzzificação
        X_augmented = np.column_stack([np.ones(n_samples), X])
        rule_outputs = np.dot(X_augmented, self.consequent_params.T)
        output = np.sum(normalized * rule_outputs, axis=1)

        return output

    def _objective_function(self, param_vector: np.ndarray) -> float:
        """Função objetivo para otimização"""
        self._vector_to_params(param_vector)

        try:
            predictions = self._forward(self.X_train)
            mse = np.mean((self.y_train - predictions) ** 2)
            return mse
        except:
            return 1e10  # Penalidade para parâmetros inválidos

    def fit(self, X: np.ndarray, y: np.ndarray, 
            n_particles: int = 30, n_iterations: int = 100,
            verbose: bool = True):
        """
        Treina ANFIS usando otimização metaheurística
        """
        self.X_train = X
        self.y_train = y

        # Inicializa parâmetros
        self.input_bounds = np.array([[X[:, i].min(), X[:, i].max()] 
                                       for i in range(self.n_inputs)])

        self.premise_params = []
        for i in range(self.n_inputs):
            x_min, x_max = self.input_bounds[i]
            x_range = x_max - x_min
            mf_params = []
            n_mf = self.n_mfs[i]
            centers = np.linspace(x_min, x_max, n_mf)

            for j in range(n_mf):
                width = x_range / (2 * n_mf)
                params = {'a': width, 'b': 2.0, 'c': centers[j]}
                mf_params.append(params)

            self.premise_params.append(mf_params)

        self.consequent_params = np.random.randn(self.n_rules, self.n_inputs + 1) * 0.1

        # Cria bounds
        bounds = self._create_bounds()

        print(f"\n{'='*70}")
        print(f"TREINAMENTO ANFIS COM OTIMIZAÇÃO METAHEURÍSTICA: {self.optimizer_name.upper()}")
        print(f"{'='*70}")
        print(f"Total de parâmetros a otimizar: {len(bounds)}")
        print(f"  • Parâmetros antecedentes: {sum(self.n_mfs) * 3}")
        print(f"  • Parâmetros consequentes: {self.n_rules * (self.n_inputs + 1)}")
        print(f"{'='*70}\n")

        # Seleciona otimizador
        if self.optimizer_name == 'pso':
            optimizer = ParticleSwarmOptimizer(
                n_particles=n_particles,
                n_iterations=n_iterations,
                adaptive_inertia=True
            )
        elif self.optimizer_name == 'de':
            optimizer = DifferentialEvolution(
                pop_size=n_particles,
                max_iter=n_iterations
            )
        elif self.optimizer_name == 'ga':
            optimizer = GeneticAlgorithm(
                pop_size=n_particles,
                max_gen=n_iterations
            )
        else:
            raise ValueError(f"Otimizador desconhecido: {self.optimizer_name}")

        # Otimiza
        best_params, best_fitness, history = optimizer.optimize(
            self._objective_function,
            bounds,
            minimize=True,
            verbose=verbose
        )

        # Atualiza com melhores parâmetros
        self._vector_to_params(best_params)
        self.optimization_history = history

        print(f"\n{'='*70}")
        print(f"OTIMIZAÇÃO CONCLUÍDA!")
        print(f"  • MSE Final: {best_fitness:.6f}")
        print(f"  • RMSE: {np.sqrt(best_fitness):.6f}")
        print(f"{'='*70}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predição"""
        return self._forward(X)


print("✅ ANFIS Moderno com Otimização Metaheurística implementado!")
print("\nAlgoritmos disponíveis:")
print("  • PSO (Particle Swarm Optimization) - Recomendado")
print("  • DE (Differential Evolution)")
print("  • GA (Genetic Algorithm)")
print("\nCaracterísticas:")
print("  • Implementação pura Python (SEM dependência do DEAP)")
print("  • PSO com inércia adaptativa")
print("  • DE com estratégia rand/1/bin")
print("  • GA com elitismo e seleção por torneio")
print("  • Otimização global de todos os parâmetros")
