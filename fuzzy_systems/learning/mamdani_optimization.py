"""
Otimização Metaheurística para Sistema Mamdani
===============================================

Implementação de otimização usando:
- PSO (Particle Swarm Optimization)
- GA (Genetic Algorithm)  
- Híbrido (PSO-GA)

Para otimizar parâmetros de sistemas Mamdani neuro-fuzzy.
"""

import numpy as np
from typing import Tuple, Callable
from mamdani_neuro_fuzzy import MamdaniNeurofuzzy


class MamdaniPSOOptimizer:
    """
    Otimização PSO específica para sistema Mamdani.
    Otimiza: médias, sigmas das MFs de entrada e centroides de saída.
    """

    def __init__(self, n_particles: int = 30, n_iterations: int = 100,
                 w: float = 0.7298, c1: float = 1.49618, c2: float = 1.49618):
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def _params_to_vector(self, mamdani: MamdaniNeurofuzzy) -> np.ndarray:
        """Converte parâmetros Mamdani para vetor 1D."""
        vector = []

        # Médias
        for i in range(mamdani.n_inputs):
            vector.extend(mamdani.input_means[i])

        # Sigmas
        for i in range(mamdani.n_inputs):
            vector.extend(mamdani.input_sigmas[i])

        # Centroides
        vector.extend(mamdani.output_centroids)

        return np.array(vector)

    def _vector_to_params(self, vector: np.ndarray, mamdani: MamdaniNeurofuzzy):
        """Converte vetor 1D para parâmetros Mamdani."""
        idx = 0

        # Médias
        for i in range(mamdani.n_inputs):
            n_mf = mamdani.n_mfs_input[i]
            mamdani.input_means[i] = vector[idx:idx+n_mf].copy()
            idx += n_mf

        # Sigmas
        for i in range(mamdani.n_inputs):
            n_mf = mamdani.n_mfs_input[i]
            mamdani.input_sigmas[i] = vector[idx:idx+n_mf].copy()
            idx += n_mf

        # Centroides
        mamdani.output_centroids = vector[idx:idx+mamdani.n_mfs_output].copy()

    def _create_bounds(self, mamdani: MamdaniNeurofuzzy) -> np.ndarray:
        """Cria limites para otimização."""
        bounds = []

        # Bounds para médias
        for i in range(mamdani.n_inputs):
            x_min, x_max = mamdani.input_bounds[i]
            for _ in range(mamdani.n_mfs_input[i]):
                bounds.append([x_min, x_max])

        # Bounds para sigmas
        for i in range(mamdani.n_inputs):
            x_range = mamdani.input_bounds[i][1] - mamdani.input_bounds[i][0]
            for _ in range(mamdani.n_mfs_input[i]):
                bounds.append([x_range * 0.01, x_range * 0.5])

        # Bounds para centroides
        y_min, y_max = mamdani.output_bound
        for _ in range(mamdani.n_mfs_output):
            bounds.append([y_min, y_max])

        return np.array(bounds)

    def optimize(self, mamdani: MamdaniNeurofuzzy, 
                 X_train: np.ndarray, y_train: np.ndarray,
                 verbose: bool = True) -> Tuple[MamdaniNeurofuzzy, list]:
        """
        Otimiza parâmetros do sistema Mamdani usando PSO.
        """
        # Inicializa modelo base
        mamdani._initialize_parameters(X_train, y_train)

        # Cria bounds
        bounds = self._create_bounds(mamdani)
        n_dims = len(bounds)

        # Inicializa enxame
        positions = np.random.uniform(
            bounds[:, 0], bounds[:, 1],
            (self.n_particles, n_dims)
        )

        velocities = np.random.uniform(
            -np.abs(bounds[:, 1] - bounds[:, 0]) * 0.1,
            np.abs(bounds[:, 1] - bounds[:, 0]) * 0.1,
            (self.n_particles, n_dims)
        )

        v_max = np.abs(bounds[:, 1] - bounds[:, 0]) * 0.2

        # Função objetivo
        def objective(params_vector):
            self._vector_to_params(params_vector, mamdani)
            mamdani._apply_domain_constraints()
            predictions, _ = mamdani.forward(X_train)
            mse = np.mean((y_train - predictions) ** 2)
            return mse

        # Avalia enxame inicial
        fitness = np.array([objective(p) for p in positions])

        pbest_positions = positions.copy()
        pbest_fitness = fitness.copy()

        gbest_idx = np.argmin(pbest_fitness)
        gbest_position = pbest_positions[gbest_idx].copy()
        gbest_fitness = pbest_fitness[gbest_idx]

        history = []

        if verbose:
            print("\n" + "="*70)
            print("OTIMIZAÇÃO PSO - SISTEMA MAMDANI")
            print("="*70)
            print(f"Partículas: {self.n_particles}")
            print(f"Iterações: {self.n_iterations}")
            print(f"Parâmetros: {n_dims}")
            print("="*70 + "\n")

        # Loop PSO
        for iteration in range(self.n_iterations):
            # Inércia adaptativa
            w = self.w * (1 - iteration / self.n_iterations)

            # Atualiza velocidades e posições
            r1 = np.random.random((self.n_particles, n_dims))
            r2 = np.random.random((self.n_particles, n_dims))

            cognitive = self.c1 * r1 * (pbest_positions - positions)
            social = self.c2 * r2 * (gbest_position - positions)
            velocities = w * velocities + cognitive + social

            velocities = np.clip(velocities, -v_max, v_max)
            positions = positions + velocities
            positions = np.clip(positions, bounds[:, 0], bounds[:, 1])

            # Avalia
            fitness = np.array([objective(p) for p in positions])

            # Atualiza pbest
            improved = fitness < pbest_fitness
            pbest_positions[improved] = positions[improved]
            pbest_fitness[improved] = fitness[improved]

            # Atualiza gbest
            current_best_idx = np.argmin(pbest_fitness)
            if pbest_fitness[current_best_idx] < gbest_fitness:
                gbest_position = pbest_positions[current_best_idx].copy()
                gbest_fitness = pbest_fitness[current_best_idx]

            history.append(gbest_fitness)

            if verbose and (iteration % 10 == 0 or iteration == self.n_iterations - 1):
                print(f"Iteração {iteration+1}/{self.n_iterations}, "
                      f"Melhor MSE: {gbest_fitness:.6f}, "
                      f"RMSE: {np.sqrt(gbest_fitness):.6f}")

        # Atualiza modelo com melhores parâmetros
        self._vector_to_params(gbest_position, mamdani)
        mamdani._apply_domain_constraints()

        if verbose:
            print("\n" + "="*70)
            print("OTIMIZAÇÃO CONCLUÍDA!")
            print(f"MSE Final: {gbest_fitness:.6f}")
            print(f"RMSE Final: {np.sqrt(gbest_fitness):.6f}")
            print("="*70)

        return mamdani, history


class MamdaniGAOptimizer:
    """
    Otimização GA (Algoritmo Genético) para sistema Mamdani.
    """

    def __init__(self, pop_size: int = 50, n_generations: int = 100,
                 elite_ratio: float = 0.1, mutation_rate: float = 0.1):
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.elite_ratio = elite_ratio
        self.n_elite = max(1, int(pop_size * elite_ratio))
        self.mutation_rate = mutation_rate

    def _params_to_vector(self, mamdani: MamdaniNeurofuzzy) -> np.ndarray:
        """Converte parâmetros para vetor."""
        vector = []
        for i in range(mamdani.n_inputs):
            vector.extend(mamdani.input_means[i])
        for i in range(mamdani.n_inputs):
            vector.extend(mamdani.input_sigmas[i])
        vector.extend(mamdani.output_centroids)
        return np.array(vector)

    def _vector_to_params(self, vector: np.ndarray, mamdani: MamdaniNeurofuzzy):
        """Converte vetor para parâmetros."""
        idx = 0
        for i in range(mamdani.n_inputs):
            n_mf = mamdani.n_mfs_input[i]
            mamdani.input_means[i] = vector[idx:idx+n_mf].copy()
            idx += n_mf
        for i in range(mamdani.n_inputs):
            n_mf = mamdani.n_mfs_input[i]
            mamdani.input_sigmas[i] = vector[idx:idx+n_mf].copy()
            idx += n_mf
        mamdani.output_centroids = vector[idx:idx+mamdani.n_mfs_output].copy()

    def _create_bounds(self, mamdani: MamdaniNeurofuzzy) -> np.ndarray:
        """Cria limites."""
        bounds = []
        for i in range(mamdani.n_inputs):
            x_min, x_max = mamdani.input_bounds[i]
            for _ in range(mamdani.n_mfs_input[i]):
                bounds.append([x_min, x_max])
        for i in range(mamdani.n_inputs):
            x_range = mamdani.input_bounds[i][1] - mamdani.input_bounds[i][0]
            for _ in range(mamdani.n_mfs_input[i]):
                bounds.append([x_range * 0.01, x_range * 0.5])
        y_min, y_max = mamdani.output_bound
        for _ in range(mamdani.n_mfs_output):
            bounds.append([y_min, y_max])
        return np.array(bounds)

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Crossover aritmético."""
        alpha = np.random.random()
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = (1 - alpha) * parent1 + alpha * parent2
        return child1, child2

    def _mutate(self, individual: np.ndarray, bounds: np.ndarray) -> np.ndarray:
        """Mutação gaussiana."""
        mutated = individual.copy()
        for i in range(len(individual)):
            if np.random.random() < self.mutation_rate:
                sigma = (bounds[i, 1] - bounds[i, 0]) * 0.1
                mutated[i] += np.random.normal(0, sigma)
                mutated[i] = np.clip(mutated[i], bounds[i, 0], bounds[i, 1])
        return mutated

    def optimize(self, mamdani: MamdaniNeurofuzzy,
                 X_train: np.ndarray, y_train: np.ndarray,
                 verbose: bool = True) -> Tuple[MamdaniNeurofuzzy, list]:
        """Otimiza com GA."""
        mamdani._initialize_parameters(X_train, y_train)
        bounds = self._create_bounds(mamdani)
        n_dims = len(bounds)

        # População inicial
        population = np.random.uniform(
            bounds[:, 0], bounds[:, 1],
            (self.pop_size, n_dims)
        )

        def objective(params):
            self._vector_to_params(params, mamdani)
            mamdani._apply_domain_constraints()
            predictions, _ = mamdani.forward(X_train)
            return np.mean((y_train - predictions) ** 2)

        history = []

        if verbose:
            print("\n" + "="*70)
            print("OTIMIZAÇÃO GA - SISTEMA MAMDANI")
            print("="*70)
            print(f"População: {self.pop_size}")
            print(f"Gerações: {self.n_generations}")
            print(f"Parâmetros: {n_dims}")
            print("="*70 + "\n")

        for generation in range(self.n_generations):
            # Avalia
            fitness = np.array([objective(ind) for ind in population])

            # Ordena
            sorted_indices = np.argsort(fitness)
            population = population[sorted_indices]
            fitness = fitness[sorted_indices]

            best_fitness = fitness[0]
            history.append(best_fitness)

            if verbose and (generation % 10 == 0 or generation == self.n_generations - 1):
                print(f"Geração {generation+1}/{self.n_generations}, "
                      f"Melhor MSE: {best_fitness:.6f}, "
                      f"RMSE: {np.sqrt(best_fitness):.6f}")

            # Elite
            elite = population[:self.n_elite].copy()

            # Nova população
            new_population = [elite]

            while len(np.vstack(new_population)) < self.pop_size:
                # Seleção por torneio
                idx1, idx2 = np.random.choice(len(population), 2, replace=False)
                parent1 = population[idx1] if fitness[idx1] < fitness[idx2] else population[idx2]

                idx1, idx2 = np.random.choice(len(population), 2, replace=False)
                parent2 = population[idx1] if fitness[idx1] < fitness[idx2] else population[idx2]

                # Crossover
                child1, child2 = self._crossover(parent1, parent2)

                # Mutação
                child1 = self._mutate(child1, bounds)
                child2 = self._mutate(child2, bounds)

                new_population.extend([child1, child2])

            population = np.vstack(new_population)[:self.pop_size]

        # Melhor solução
        fitness = np.array([objective(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_params = population[best_idx]
        best_fitness = fitness[best_idx]

        self._vector_to_params(best_params, mamdani)
        mamdani._apply_domain_constraints()

        if verbose:
            print("\n" + "="*70)
            print("OTIMIZAÇÃO CONCLUÍDA!")
            print(f"MSE Final: {best_fitness:.6f}")
            print(f"RMSE Final: {np.sqrt(best_fitness):.6f}")
            print("="*70)

        return mamdani, history


class MamdaniHybridOptimizer:
    """
    Otimização híbrida PSO-GA para sistema Mamdani.

    Estratégia: PSO para exploração global rápida, 
    seguido de GA para refinamento local.
    """

    def __init__(self, pso_iterations: int = 50, ga_generations: int = 50):
        self.pso_optimizer = MamdaniPSOOptimizer(n_iterations=pso_iterations)
        self.ga_optimizer = MamdaniGAOptimizer(n_generations=ga_generations)

    def optimize(self, mamdani: MamdaniNeurofuzzy,
                 X_train: np.ndarray, y_train: np.ndarray,
                 verbose: bool = True) -> Tuple[MamdaniNeurofuzzy, dict]:
        """Otimiza usando abordagem híbrida."""

        if verbose:
            print("\n" + "="*70)
            print("OTIMIZAÇÃO HÍBRIDA PSO-GA - SISTEMA MAMDANI")
            print("="*70)
            print("Fase 1: PSO (exploração global)")
            print("Fase 2: GA (refinamento local)")
            print("="*70)

        # Fase 1: PSO
        if verbose:
            print("\n>>> FASE 1: PSO <<<")
        mamdani, pso_history = self.pso_optimizer.optimize(
            mamdani, X_train, y_train, verbose=verbose
        )

        # Fase 2: GA (começa do resultado do PSO)
        if verbose:
            print("\n>>> FASE 2: GA <<<")
        mamdani, ga_history = self.ga_optimizer.optimize(
            mamdani, X_train, y_train, verbose=verbose
        )

        history = {
            'pso': pso_history,
            'ga': ga_history,
            'combined': pso_history + ga_history
        }

        return mamdani, history


print("✅ Otimizadores Mamdani implementados com sucesso!")
print("\nAlgoritmos disponíveis:")
print("  • MamdaniPSOOptimizer - Particle Swarm Optimization")
print("  • MamdaniGAOptimizer - Genetic Algorithm")
print("  • MamdaniHybridOptimizer - Híbrido PSO-GA")
