"""
Metaheurísticas para Otimização de Sistemas Fuzzy
==================================================

Implementação de algoritmos metaheurísticos para otimização global:
- PSO (Particle Swarm Optimization) - Otimização por Enxame de Partículas
- DE (Differential Evolution) - Evolução Diferencial
- GA (Genetic Algorithm) - Algoritmo Genético

Todos otimizadores seguem a mesma interface:
    optimizer.optimize(objective_func, bounds, minimize=True, verbose=False)

Referências:
    - PSO: Kennedy & Eberhart (1995)
    - DE: Storn & Price (1997)
    - GA: Holland (1975)
"""

import numpy as np
from typing import Callable, Tuple, List
from abc import ABC, abstractmethod


class BaseOptimizer(ABC):
    """Classe base para todos os otimizadores metaheurísticos"""

    @abstractmethod
    def optimize(self, objective_func: Callable, bounds: np.ndarray,
                 minimize: bool = True, verbose: bool = False) -> Tuple[np.ndarray, float, List[float]]:
        """
        Otimiza função objetivo.

        Parâmetros:
            objective_func: Função objetivo f(x) -> float
            bounds: Array (n_dims, 2) com [min, max] para cada dimensão
            minimize: Se True minimiza, se False maximiza
            verbose: Exibe progresso

        Retorna:
            (melhor_solução, melhor_fitness, histórico)
        """
        pass


class PSO(BaseOptimizer):
    """
    Particle Swarm Optimization (PSO) - Otimização por Enxame de Partículas

    Implementação moderna com:
    - Inércia adaptativa (decresce linearmente)
    - Constrição de Clerc & Kennedy
    - Limitação de velocidade
    - Convergência garantida

    Parâmetros:
        n_particles: Número de partículas no enxame (padrão: 30)
        n_iterations: Número de iterações (padrão: 100)
        w_max: Peso de inércia máximo (padrão: 0.9)
        w_min: Peso de inércia mínimo (padrão: 0.4)
        c1: Coeficiente cognitivo (padrão: 1.49618)
        c2: Coeficiente social (padrão: 1.49618)

    Exemplo:
        >>> pso = PSO(n_particles=30, n_iterations=100)
        >>> bounds = np.array([[-5, 5], [-5, 5]])
        >>> def sphere(x): return np.sum(x**2)
        >>> best_x, best_f, history = pso.optimize(sphere, bounds)
    """

    def __init__(self, n_particles: int = 30, n_iterations: int = 100,
                 w_max: float = 0.9, w_min: float = 0.4,
                 c1: float = 1.49618, c2: float = 1.49618):
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w_max = w_max
        self.w_min = w_min
        self.c1 = c1
        self.c2 = c2

    def optimize(self, objective_func: Callable, bounds: np.ndarray,
                 minimize: bool = True, verbose: bool = False) -> Tuple[np.ndarray, float, List[float]]:
        """Otimiza função objetivo usando PSO"""
        n_dims = bounds.shape[0]

        # Inicializa posições aleatórias
        positions = np.random.uniform(
            bounds[:, 0], bounds[:, 1],
            (self.n_particles, n_dims)
        )

        # Inicializa velocidades
        v_range = np.abs(bounds[:, 1] - bounds[:, 0]) * 0.1
        velocities = np.random.uniform(-v_range, v_range, (self.n_particles, n_dims))

        # Velocidade máxima (20% da faixa)
        v_max = np.abs(bounds[:, 1] - bounds[:, 0]) * 0.2

        # Avalia fitness inicial
        fitness = np.array([objective_func(p) for p in positions])

        # Melhor pessoal (pbest)
        pbest_positions = positions.copy()
        pbest_fitness = fitness.copy()

        # Melhor global (gbest)
        gbest_idx = np.argmin(pbest_fitness) if minimize else np.argmax(pbest_fitness)
        gbest_position = pbest_positions[gbest_idx].copy()
        gbest_fitness = pbest_fitness[gbest_idx]

        history = []

        # Loop principal do PSO
        for iteration in range(self.n_iterations):
            # Inércia adaptativa (decai linearmente)
            w = self.w_max - (self.w_max - self.w_min) * iteration / self.n_iterations

            # Componentes aleatórios
            r1 = np.random.random((self.n_particles, n_dims))
            r2 = np.random.random((self.n_particles, n_dims))

            # Atualização de velocidade (equação clássica do PSO)
            cognitive = self.c1 * r1 * (pbest_positions - positions)
            social = self.c2 * r2 * (gbest_position - positions)
            velocities = w * velocities + cognitive + social

            # Limita velocidades
            velocities = np.clip(velocities, -v_max, v_max)

            # Atualiza posições
            positions = positions + velocities
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
            current_best_idx = np.argmin(pbest_fitness) if minimize else np.argmax(pbest_fitness)
            if (minimize and pbest_fitness[current_best_idx] < gbest_fitness) or \
               (not minimize and pbest_fitness[current_best_idx] > gbest_fitness):
                gbest_position = pbest_positions[current_best_idx].copy()
                gbest_fitness = pbest_fitness[current_best_idx]

            history.append(gbest_fitness)

            if verbose and (iteration % 10 == 0 or iteration == self.n_iterations - 1):
                print(f"  PSO [{iteration+1:3d}/{self.n_iterations}] "
                      f"Best: {gbest_fitness:.6f}, w={w:.3f}")

        return gbest_position, gbest_fitness, history


class DE(BaseOptimizer):
    """
    Differential Evolution (DE) - Evolução Diferencial

    Estratégia: DE/rand/1/bin (mais comum e robusta)

    Parâmetros:
        pop_size: Tamanho da população (padrão: 50)
        max_iter: Número máximo de iterações (padrão: 100)
        F: Fator de mutação, 0 < F <= 2 (padrão: 0.8)
        CR: Taxa de crossover, 0 <= CR <= 1 (padrão: 0.9)

    Exemplo:
        >>> de = DE(pop_size=50, max_iter=100, F=0.8, CR=0.9)
        >>> bounds = np.array([[-5, 5], [-5, 5]])
        >>> best_x, best_f, history = de.optimize(lambda x: np.sum(x**2), bounds)
    """

    def __init__(self, pop_size: int = 50, max_iter: int = 100,
                 F: float = 0.8, CR: float = 0.9):
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.F = F
        self.CR = CR

    def optimize(self, objective_func: Callable, bounds: np.ndarray,
                 minimize: bool = True, verbose: bool = False) -> Tuple[np.ndarray, float, List[float]]:
        """Otimiza função objetivo usando DE"""
        n_dims = bounds.shape[0]

        # Inicializa população
        population = np.random.uniform(
            bounds[:, 0], bounds[:, 1],
            (self.pop_size, n_dims)
        )

        # Avalia fitness inicial
        fitness = np.array([objective_func(ind) for ind in population])

        best_idx = np.argmin(fitness) if minimize else np.argmax(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        history = []

        # Loop principal do DE
        for iteration in range(self.max_iter):
            for i in range(self.pop_size):
                # Seleciona 3 indivíduos distintos diferentes de i
                candidates = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = np.random.choice(candidates, 3, replace=False)

                # Mutação: v = xa + F * (xb - xc)
                mutant = population[a] + self.F * (population[b] - population[c])
                mutant = np.clip(mutant, bounds[:, 0], bounds[:, 1])

                # Crossover binomial
                cross_points = np.random.rand(n_dims) < self.CR
                # Garante pelo menos um gene do mutante
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, n_dims)] = True

                trial = np.where(cross_points, mutant, population[i])

                # Seleção (greedy)
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
                print(f"  DE  [{iteration+1:3d}/{self.max_iter}] "
                      f"Best: {best_fitness:.6f}")

        return best_solution, best_fitness, history


class GA(BaseOptimizer):
    """
    Genetic Algorithm (GA) - Algoritmo Genético

    Implementação com:
    - Elitismo (preserva os melhores)
    - Seleção por torneio
    - Crossover aritmético
    - Mutação gaussiana

    Parâmetros:
        pop_size: Tamanho da população (padrão: 50)
        max_gen: Número máximo de gerações (padrão: 100)
        elite_ratio: Proporção de elite (padrão: 0.1)
        mutation_rate: Taxa de mutação (padrão: 0.1)
        tournament_size: Tamanho do torneio (padrão: 3)

    Exemplo:
        >>> ga = GA(pop_size=50, max_gen=100)
        >>> bounds = np.array([[-5, 5], [-5, 5]])
        >>> best_x, best_f, history = ga.optimize(lambda x: np.sum(x**2), bounds)
    """

    def __init__(self, pop_size: int = 50, max_gen: int = 100,
                 elite_ratio: float = 0.1, mutation_rate: float = 0.1,
                 tournament_size: int = 3):
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
        """Mutação gaussiana"""
        mutated = individual.copy()

        for i in range(len(individual)):
            if np.random.random() < self.mutation_rate:
                # Desvio padrão = 10% da faixa
                sigma = (bounds[i, 1] - bounds[i, 0]) * 0.1
                mutated[i] += np.random.normal(0, sigma)
                mutated[i] = np.clip(mutated[i], bounds[i, 0], bounds[i, 1])

        return mutated

    def optimize(self, objective_func: Callable, bounds: np.ndarray,
                 minimize: bool = True, verbose: bool = False) -> Tuple[np.ndarray, float, List[float]]:
        """Otimiza função objetivo usando GA"""
        n_dims = bounds.shape[0]

        # Inicializa população
        population = np.random.uniform(
            bounds[:, 0], bounds[:, 1],
            (self.pop_size, n_dims)
        )

        # Avalia fitness inicial
        fitness = np.array([objective_func(ind) for ind in population])

        history = []

        # Loop principal do GA
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
                print(f"  GA  [{generation+1:3d}/{self.max_gen}] "
                      f"Best: {best_fitness:.6f}")

            # Elite (os melhores sobrevivem)
            elite = population[:self.n_elite].copy()

            # Gera nova população
            new_population = [elite]

            # Cria offspring até completar população
            while len(np.vstack(new_population if len(new_population) > 1 else [new_population[0]])) < self.pop_size:
                # Seleção por torneio
                parent1 = self._tournament_selection(population, fitness, minimize)
                parent2 = self._tournament_selection(population, fitness, minimize)

                # Crossover
                child1, child2 = self._crossover(parent1, parent2)

                # Mutação
                child1 = self._mutate(child1, bounds)
                child2 = self._mutate(child2, bounds)

                new_population.extend([child1, child2])

            # Atualiza população (mantém tamanho correto)
            population = np.vstack(new_population)[:self.pop_size]

            # Avalia nova população
            fitness = np.array([objective_func(ind) for ind in population])

        # Retorna melhor solução final
        best_idx = np.argmin(fitness) if minimize else np.argmax(fitness)
        return population[best_idx], fitness[best_idx], history


# Factory function para facilitar uso
def get_optimizer(name: str, **kwargs) -> BaseOptimizer:
    """
    Factory para criar otimizador pelo nome.

    Parâmetros:
        name: Nome do otimizador ('pso', 'de', 'ga')
        **kwargs: Parâmetros específicos do otimizador

    Retorna:
        Instância do otimizador

    Exemplo:
        >>> opt = get_optimizer('pso', n_particles=30, n_iterations=100)
        >>> best, fitness, history = opt.optimize(func, bounds)
    """
    optimizers = {
        'pso': PSO,
        'de': DE,
        'ga': GA
    }

    name_lower = name.lower()
    if name_lower not in optimizers:
        available = ', '.join(optimizers.keys())
        raise ValueError(f"Otimizador '{name}' desconhecido. Disponíveis: {available}")

    return optimizers[name_lower](**kwargs)
