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
from abc import ABC, abstractmethod
import random
import math
from typing import Callable, Dict, List, Tuple, Literal, Optional, Any



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

class BinaryGA:
    """
    Binary/Integer Genetic Algorithm for discrete optimization.
    
    Optimized for problems where decision variables are integers (e.g., rule indices).
    Uses uniform crossover (gene-by-gene) which is superior for independent variables.
    
    Parameters
    ----------
    pop_size : int, default=100
        Population size. Recommended: 50-200.
        
    max_gen : int, default=500
        Maximum number of generations.
        
    elite_ratio : float, default=0.15
        Fraction of best individuals preserved each generation.
        Range: 0.05-0.25.
        
    crossover_rate : float, default=0.8
        Probability of crossover. Range: 0.6-0.95.
        
    crossover_type : str, default='uniform'
        Crossover type:
        - 'uniform': Gene-by-gene (RECOMMENDED for independent variables)
        - 'one_point': Single crossover point
        - 'two_point': Two crossover points
        - 'multi_point': Multiple crossover points
        - 'arithmetic': Weighted average
        
    mutation_rate : float, default=0.05
        Per-gene mutation probability. Range: 0.01-0.15.
        
    tournament_size : int, default=5
        Tournament selection size. Range: 2-7.
        
    crossover_points : int, default=2
        Number of points for multi_point crossover.
        
    adaptive_mutation : bool, default=True
        Increase mutation when convergence stagnates.
        
    plateau_generations : int, default=50
        Generations without improvement before adapting mutation.
        
    mutation_boost_factor : float, default=2.0
        Factor to increase mutation during plateau.
        
    Examples
    --------
    >>> import numpy as np
    >>> from metaheuristics import BinaryGA
    >>> 
    >>> # Minimize sum of squared deviations from target [2,2,2,...,2]
    >>> def objective(x):
    ...     return np.sum((x - 2)**2)
    >>> 
    >>> bounds = np.array([[0, 4]] * 50)  # 50 variables in [0,4]
    >>> ga = BinaryGA(pop_size=100, max_gen=200, crossover_type='uniform')
    >>> best_solution, best_fitness, history = ga.optimize(objective, bounds, verbose=True)
    >>> print(f"Best solution: {best_solution}")
    >>> print(f"Best fitness: {best_fitness:.6f}")
    """
    
    def __init__(
        self,
        pop_size: int = 100,
        max_gen: int = 500,
        elite_ratio: float = 0.15,
        crossover_rate: float = 0.8,
        crossover_type: str = 'uniform',
        mutation_rate: float = 0.05,
        tournament_size: int = 5,
        crossover_points: int = 2,
        adaptive_mutation: bool = True,
        plateau_generations: int = 50,
        mutation_boost_factor: float = 2.0
    ):
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.elite_ratio = elite_ratio
        self.crossover_rate = crossover_rate
        self.crossover_type = crossover_type
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.crossover_points = crossover_points
        self.adaptive_mutation = adaptive_mutation
        self.plateau_generations = plateau_generations
        self.mutation_boost_factor = mutation_boost_factor
        
        # Validation
        valid_crossover_types = ['uniform', 'one_point', 'two_point', 'multi_point', 'arithmetic']
        if crossover_type not in valid_crossover_types:
            raise ValueError(f"crossover_type must be one of {valid_crossover_types}")
        
        if not 0 < elite_ratio < 0.5:
            raise ValueError("elite_ratio must be in (0, 0.5)")
        if not 0 < crossover_rate <= 1.0:
            raise ValueError("crossover_rate must be in (0, 1]")
        if not 0 < mutation_rate < 1.0:
            raise ValueError("mutation_rate must be in (0, 1)")
        if tournament_size < 2:
            raise ValueError("tournament_size must be >= 2")
    
    def optimize(
        self, 
        objective_func: Callable, 
        bounds: np.ndarray,
        minimize: bool = True, 
        verbose: bool = False,
        initial_population: Optional[List[np.ndarray]] = None  # ← NOVO
    ) -> Tuple[np.ndarray, float, dict]:
        """
        Optimize with optional pre-initialized population.
        
        Parameters
        ----------
        initial_population : List[np.ndarray], optional
            Pre-initialized population. If provided, skips random initialization.
        """
        n_dims = bounds.shape[0]
        lower_bounds = bounds[:, 0].astype(int)
        upper_bounds = bounds[:, 1].astype(int)
        n_elite = max(1, int(self.pop_size * self.elite_ratio))
        
        # ========== Initialize or use provided population ==========
        if initial_population is not None:
            if len(initial_population) != self.pop_size:
                raise ValueError(f"initial_population size {len(initial_population)} != pop_size {self.pop_size}")
            population = [ind.copy() for ind in initial_population]
        else:
            # Random initialization
            population = []
            for _ in range(self.pop_size):
                individual = np.array([
                    np.random.randint(lower_bounds[i], upper_bounds[i] + 1)
                    for i in range(n_dims)
                ])
                population.append(individual)
        
        # Evaluate initial population
        fitness_sign = 1 if minimize else -1
        fitness = np.array([fitness_sign * objective_func(ind) for ind in population])
        
        best_idx = np.argmin(fitness)
        best_chromosome = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        initial_fitness = best_fitness
        
        history = {
            'generation': [],
            'best_fitness': [],
            'mean_fitness': [],
            'std_fitness': [],
            'diversity': []
        }
        
        generations_without_improvement = 0
        current_mutation_rate = self.mutation_rate
        
        if verbose:
            print("="*70)
            print("BinaryGA Optimization")
            print("="*70)
            print(f"  Population:      {self.pop_size}")
            print(f"  Generations:     {self.max_gen}")
            print(f"  Dimensions:      {n_dims}")
            print(f"  Elite:           {n_elite} ({self.elite_ratio*100:.1f}%)")
            print(f"  Crossover:       {self.crossover_type} (rate={self.crossover_rate})")
            print(f"  Mutation:        {self.mutation_rate} (adaptive={self.adaptive_mutation})")
            print(f"  Tournament:      {self.tournament_size}")
            print(f"  Initial best:    {fitness_sign * best_fitness:.6f}\n")
        
        # ============================================================
        # GA Evolution Loop
        # ============================================================
        for generation in range(self.max_gen):
            # --- Elitism: preserve best individuals ---
            elite_indices = np.argsort(fitness)[:n_elite]
            new_population = [population[i].copy() for i in elite_indices]
            
            # --- Generate offspring ---
            while len(new_population) < self.pop_size:
                # Tournament selection
                parent1 = self._tournament_selection(population, fitness)
                parent2 = self._tournament_selection(population, fitness)
                
                # Crossover
                child1, child2 = self._crossover(parent1, parent2, lower_bounds, upper_bounds)
                
                # Mutation
                child1 = self._mutate(child1, lower_bounds, upper_bounds, current_mutation_rate)
                child2 = self._mutate(child2, lower_bounds, upper_bounds, current_mutation_rate)
                
                new_population.append(child1)
                if len(new_population) < self.pop_size:
                    new_population.append(child2)
            
            # Update population
            population = new_population[:self.pop_size]
            fitness = np.array([fitness_sign * objective_func(ind) for ind in population])
            
            # --- Update best ---
            gen_best_idx = np.argmin(fitness)
            gen_best_fitness = fitness[gen_best_idx]
            
            if gen_best_fitness < best_fitness:
                best_chromosome = population[gen_best_idx].copy()
                best_fitness = gen_best_fitness
                generations_without_improvement = 0
                
                if verbose:
                    improvement = (1 - best_fitness / initial_fitness) * 100 if initial_fitness != 0 else 0
                    print(f"Gen {generation:4d} | New best! {fitness_sign * best_fitness:.6f} "
                          f"| Improvement: {improvement:+.2f}%")
            else:
                generations_without_improvement += 1
            
            # --- Adaptive mutation ---
            if self.adaptive_mutation and generations_without_improvement >= self.plateau_generations:
                current_mutation_rate = min(0.3, self.mutation_rate * self.mutation_boost_factor)
            else:
                current_mutation_rate = self.mutation_rate
            
            # --- Diversity metric ---
            diversity = self._calculate_diversity(population)
            
            # --- History ---
            history['generation'].append(generation)
            history['best_fitness'].append(fitness_sign * best_fitness)
            history['mean_fitness'].append(fitness_sign * np.mean(fitness))
            history['std_fitness'].append(np.std(fitness))
            history['diversity'].append(diversity)
            
            # --- Verbose output ---
            if verbose:
                log_interval = max(1, self.max_gen // 10)
                if generation % log_interval == 0 and generation > 0:
                    improvement = (1 - best_fitness / initial_fitness) * 100 if initial_fitness != 0 else 0
                    print(f"Gen {generation:4d}/{self.max_gen} | "
                          f"Best: {fitness_sign * best_fitness:.6f} | "
                          f"Mean: {fitness_sign * np.mean(fitness):.6f} | "
                          f"Diversity: {diversity:.1f} | "
                          f"Improvement: {improvement:+.2f}%")
        
        if verbose:
            improvement = (1 - best_fitness / initial_fitness) * 100 if initial_fitness != 0 else 0
            print(f"\n{'='*70}")
            print(f"Optimization completed!")
            print(f"  Final best:      {fitness_sign * best_fitness:.6f}")
            print(f"  Improvement:     {improvement:+.2f}%")
            print(f"  Generations:     {self.max_gen}")
            print(f"{'='*70}\n")
        
        return best_chromosome, fitness_sign * best_fitness, history
    
    def _tournament_selection(self, population: List[np.ndarray], fitness: np.ndarray) -> np.ndarray:
        """Tournament selection: pick best from random subset."""
        indices = np.random.choice(len(population), self.tournament_size, replace=False)
        winner_idx = indices[np.argmin(fitness[indices])]
        return population[winner_idx].copy()
    
    def _crossover(
        self, 
        parent1: np.ndarray, 
        parent2: np.ndarray,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Crossover operation based on self.crossover_type."""
        n_genes = len(parent1)
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        if random.random() >= self.crossover_rate:
            return child1, child2
        
        if self.crossover_type == 'uniform':
            # Uniform crossover: each gene independently from either parent
            mask = np.random.rand(n_genes) < 0.5
            child1[mask] = parent2[mask]
            child2[mask] = parent1[mask]
        
        elif self.crossover_type == 'one_point':
            # Single-point crossover
            if n_genes > 1:
                point = np.random.randint(1, n_genes)
                child1[point:] = parent2[point:]
                child2[point:] = parent1[point:]
        
        elif self.crossover_type == 'two_point':
            # Two-point crossover
            if n_genes > 2:
                points = sorted(np.random.choice(range(1, n_genes), 2, replace=False))
                child1[points[0]:points[1]] = parent2[points[0]:points[1]]
                child2[points[0]:points[1]] = parent1[points[0]:points[1]]
        
        elif self.crossover_type == 'multi_point':
            # Multi-point crossover
            if n_genes > 1:
                n_points = min(self.crossover_points, n_genes - 1)
                points = sorted(random.sample(range(1, n_genes), n_points))
                
                for i in range(len(points)):
                    if i % 2 == 0:
                        start = points[i]
                        end = points[i + 1] if i + 1 < len(points) else n_genes
                        child1[start:end], child2[start:end] = child2[start:end].copy(), child1[start:end].copy()
        
        elif self.crossover_type == 'arithmetic':
            # Arithmetic crossover: weighted average (for continuous-like problems)
            alpha = np.random.rand()
            child1 = np.round(alpha * parent1 + (1 - alpha) * parent2).astype(int)
            child2 = np.round(alpha * parent2 + (1 - alpha) * parent1).astype(int)
        
        # Ensure bounds
        child1 = np.clip(child1, lower_bounds, upper_bounds)
        child2 = np.clip(child2, lower_bounds, upper_bounds)
        
        return child1, child2
    
    def _mutate(
        self, 
        chromosome: np.ndarray, 
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
        mutation_rate: float
    ) -> np.ndarray:
        """Flip genes with given probability (uniform random replacement)."""
        mutated = chromosome.copy()
        
        for i in range(len(chromosome)):
            if random.random() < mutation_rate:
                # Replace with random valid value
                mutated[i] = np.random.randint(lower_bounds[i], upper_bounds[i] + 1)
        
        return mutated
    
    def _calculate_diversity(self, population: List[np.ndarray]) -> float:
        """
        Calculate genetic diversity as average Hamming distance.
        
        Higher diversity means population is more spread out in search space.
        """
        if len(population) < 2:
            return 0.0
        
        # Sample subset for efficiency
        sample_size = min(20, len(population))
        distances = []
        
        for i in range(sample_size):
            for j in range(i + 1, sample_size):
                # Hamming distance: number of differing genes
                distance = np.sum(population[i] != population[j])
                distances.append(distance)
        
        return np.mean(distances) if distances else 0.0


"""
Initialization Strategies for Discrete Optimization
====================================================

Intelligent initialization methods for discrete/integer optimization problems.
"""


def initialize_population_discrete(
    pop_size: int,
    n_dims: int,
    bounds: np.ndarray,
    method: Literal['random', 'uniform', 'gradient', 'mixed'] = 'random',
    objective_func: Optional[Callable] = None,
    gradient_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    noise_level: float = 0.1,
    verbose: bool = False,
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Initialize population for discrete optimization with optional data-driven strategy.
    
    Parameters
    ----------
    pop_size : int
        Population size.
        
    n_dims : int
        Number of dimensions (genes per individual).
        
    bounds : np.ndarray, shape (n_dims, 2)
        Integer bounds [[min, max], ...] for each dimension.
        
    method : str, default='random'
        Initialization method:
        - 'random': Fully random integers
        - 'uniform': All individuals start at middle value
        - 'gradient': Data-driven (requires gradient_data)
        - 'mixed': 50% gradient + 50% random
        
    objective_func : callable, optional
        Function to evaluate fitness. Required to return fitness values.
        
    gradient_data : tuple of (activations, targets, centroids), optional
        Data for gradient-based initialization:
        - activations: np.ndarray, shape (n_samples, n_dims)
          Rule activation strengths (firing strengths)
        - targets: np.ndarray, shape (n_samples,)
          Target outputs
        - value_range: tuple (min, max)
          Output value range
        
    noise_level : float, default=0.1
        Fraction of genes to randomly perturb (for gradient/uniform methods).
        
    verbose : bool, default=False
        Print initialization statistics.
        
    Returns
    -------
    population : List[np.ndarray]
        List of integer chromosomes.
        
    fitness_values : List[float]
        Fitness of each individual (empty list if objective_func is None).
        
    Examples
    --------
    >>> import numpy as np
    >>> from metaheuristics import initialize_population_discrete
    >>> 
    >>> bounds = np.array([[0, 4]] * 50)
    >>> 
    >>> # Random initialization
    >>> pop, _ = initialize_population_discrete(100, 50, bounds, method='random')
    >>> 
    >>> # Gradient-based (requires data)
    >>> activations = np.random.rand(100, 50)  # Rule activations
    >>> targets = np.random.rand(100)  # Target outputs
    >>> gradient_data = (activations, targets, (0, 1))
    >>> 
    >>> def objective(x):
    ...     return np.sum((x - 2)**2)
    >>> 
    >>> pop, fitness = initialize_population_discrete(
    ...     100, 50, bounds, 
    ...     method='gradient',
    ...     objective_func=objective,
    ...     gradient_data=gradient_data
    ... )
    """
    lower_bounds = bounds[:, 0].astype(int)
    upper_bounds = bounds[:, 1].astype(int)
    n_values = upper_bounds[0] - lower_bounds[0] + 1  # Assumes same range for all dims
    
    population = []
    
    # ========== Helper Functions ==========
    def create_random():
        """Random integer chromosome."""
        return np.array([
            np.random.randint(lower_bounds[i], upper_bounds[i] + 1)
            for i in range(n_dims)
        ])
    
    def create_uniform():
        """All genes at middle value."""
        mid_values = ((lower_bounds + upper_bounds) // 2).astype(int)
        return mid_values.copy()
    
    def create_gradient():
        """Data-driven initialization using activations and targets."""
        if gradient_data is None:
            raise ValueError("gradient_data required for gradient initialization")
        
        activations, targets, value_range = gradient_data
        out_min, out_max = value_range
        
        chromosome = np.zeros(n_dims, dtype=int)
        
        # For each dimension (rule), find best discrete value
        for dim_idx in range(n_dims):
            dim_activations = activations[:, dim_idx]
            
            # Only consider samples where this dimension is active
            threshold = 0.1
            mask = dim_activations > threshold
            
            if mask.sum() > 0:
                # Weighted average output
                avg_output = np.average(targets[mask], weights=dim_activations[mask])
                
                # Normalize to [0, 1]
                normalized = (avg_output - out_min) / (out_max - out_min + 1e-10)
                normalized = np.clip(normalized, 0, 1)
                
                # Map to discrete index
                value_idx = int(np.round(normalized * (n_values - 1)))
                chromosome[dim_idx] = lower_bounds[dim_idx] + value_idx
                chromosome[dim_idx] = np.clip(chromosome[dim_idx], lower_bounds[dim_idx], upper_bounds[dim_idx])
            else:
                # Dimension never active - random value
                chromosome[dim_idx] = np.random.randint(lower_bounds[dim_idx], upper_bounds[dim_idx] + 1)
        
        return chromosome
    
    def add_noise(chromosome, noise_level):
        """Add random perturbations to chromosome."""
        if noise_level <= 0:
            return chromosome
        
        mutated = chromosome.copy()
        n_mutations = max(1, int(n_dims * noise_level))
        mutation_indices = np.random.choice(n_dims, n_mutations, replace=False)
        
        for idx in mutation_indices:
            mutated[idx] = np.random.randint(lower_bounds[idx], upper_bounds[idx] + 1)
        
        return mutated
    
    # ========== Generate Population ==========
    if method == 'random':
        for _ in range(pop_size):
            population.append(create_random())
    
    elif method == 'uniform':
        for _ in range(pop_size):
            chromosome = create_uniform()
            chromosome = add_noise(chromosome, noise_level)
            population.append(chromosome)
    
    elif method == 'gradient':
        for _ in range(pop_size):
            chromosome = create_gradient()
            chromosome = add_noise(chromosome, noise_level)
            population.append(chromosome)
    
    elif method == 'mixed':
        n_gradient = pop_size // 2
        
        # Half gradient
        for _ in range(n_gradient):
            chromosome = create_gradient()
            chromosome = add_noise(chromosome, noise_level)
            population.append(chromosome)
        
        # Half random
        for _ in range(pop_size - n_gradient):
            population.append(create_random())
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # ========== Evaluate Population ==========
    fitness_values = []
    if objective_func is not None:
        for chromosome in population:
            fitness_values.append(objective_func(chromosome))
        
        if verbose:
            print(f"Population initialized: {method}")
            print(f"  Size: {pop_size}")
            print(f"  Best fitness:  {min(fitness_values):.6f}")
            print(f"  Mean fitness:  {np.mean(fitness_values):.6f}")
            print(f"  Std fitness:   {np.std(fitness_values):.6f}")
    
    return population, fitness_values



class SimulatedAnnealing:
    """
    Generic Simulated Annealing algorithm implementation.
    
    Simulated Annealing is an optimization metaheuristic inspired by the
    annealing process in metallurgy. It allows escaping local optima
    through probabilistic acceptance of worse solutions.
    """
    
    def __init__(self,
                 cost_function: Callable[[Any], float],
                 neighbor_function: Callable[[Any], Any],
                 initial_solution_function: Callable[[], Any],
                 temperature_init: float = 100.0,
                 temperature_min: float = 0.01,
                 cooling_rate: float = 0.95,
                 max_iterations: int = 1000,
                 plateau_iterations: int = 50,
                 cooling_schedule: str = 'exponential',
                 verbose: bool = True):
        """
        Initialize the Simulated Annealing optimizer.
        
        Parameters:
            cost_function: Function that calculates the cost of a solution
            neighbor_function: Function that generates a neighboring solution
            initial_solution_function: Function that generates the initial solution
            temperature_init: Initial temperature
            temperature_min: Minimum temperature (stopping criterion)
            cooling_rate: Cooling rate (alpha) for exponential schedule
            max_iterations: Maximum number of iterations
            plateau_iterations: Iterations without improvement before stopping
            cooling_schedule: Type of cooling ('exponential', 'linear', 
                            'logarithmic', 'adaptive')
            verbose: If True, prints progress
        """
        self.cost_function = cost_function
        self.neighbor_function = neighbor_function
        self.initial_solution_function = initial_solution_function
        self.temperature_init = temperature_init
        self.temperature_min = temperature_min
        self.cooling_rate = cooling_rate
        self.max_iterations = max_iterations
        self.plateau_iterations = plateau_iterations
        self.cooling_schedule = cooling_schedule
        self.verbose = verbose
        
        # Optimization history
        self.history = {
            'iteration': [],
            'temperature': [],
            'current_cost': [],
            'best_cost': [],
            'acceptance_rate': []
        }
    
    def acceptance_probability(self, 
                             current_cost: float, 
                             new_cost: float, 
                             temperature: float) -> float:
        """
        Metropolis acceptance criterion.
        
        Parameters:
            current_cost: Cost of current solution
            new_cost: Cost of new solution
            temperature: Current temperature
            
        Returns:
            Acceptance probability [0, 1]
        """
        if new_cost < current_cost:
            return 1.0
        else:
            # Accept worse solutions with probability exp(-ΔE/T)
            delta_e = new_cost - current_cost
            return math.exp(-delta_e / temperature)
    
    def update_temperature(self, temperature: float, iteration: int) -> float:
        """
        Update temperature according to chosen schedule.
        
        Parameters:
            temperature: Current temperature
            iteration: Current iteration
            
        Returns:
            New temperature
        """
        if self.cooling_schedule == 'exponential':
            # T_new = alpha * T_current
            return temperature * self.cooling_rate
        
        elif self.cooling_schedule == 'linear':
            # T_new = T_init - (iteration / max_iter) * T_init
            return self.temperature_init * (1 - iteration / self.max_iterations)
        
        elif self.cooling_schedule == 'logarithmic':
            # T_new = T_init / log(1 + iteration)
            return self.temperature_init / math.log(2 + iteration)
        
        elif self.cooling_schedule == 'adaptive':
            # Adaptive cooling based on acceptance rate
            if iteration > 0 and iteration % 10 == 0:
                recent_acceptance = np.mean(self.history['acceptance_rate'][-10:])
                if recent_acceptance > 0.5:
                    return temperature * 0.9  # Cool faster
                elif recent_acceptance < 0.2:
                    return temperature * 0.98  # Cool slower
            return temperature * self.cooling_rate
        
        else:
            raise ValueError(f"Unknown schedule: {self.cooling_schedule}")
    
    def optimize(self) -> Tuple[Any, float, Dict]:
        """
        Execute the Simulated Annealing algorithm.
        
        Returns:
            Tuple containing:
            - best_solution: Best solution found
            - best_cost: Best cost achieved
            - history: Optimization history
        """
        # Initialization
        current_solution = self.initial_solution_function()
        current_cost = self.cost_function(current_solution)
        
        # Store initial cost for improvement calculation
        initial_cost = current_cost
        
        best_solution = current_solution
        best_cost = current_cost
        temperature = self.temperature_init
        iterations_without_improvement = 0
        
        if self.verbose:
            print("=" * 60)
            print("Simulated Annealing - Optimization")
            print("=" * 60)
            print(f"Initial cost: {current_cost:.6f}")
            print(f"Initial temperature: {temperature:.2f}")
            print("=" * 60)
        
        # Main loop
        for iteration in range(self.max_iterations):
            # Generate neighbor
            neighbor_solution = self.neighbor_function(current_solution)
            neighbor_cost = self.cost_function(neighbor_solution)
            
            # Calculate acceptance probability
            accept_prob = self.acceptance_probability(
                current_cost, neighbor_cost, temperature
            )
            
            # Decide whether to accept neighbor
            accepted = False
            if random.random() < accept_prob:
                current_solution = neighbor_solution
                current_cost = neighbor_cost
                accepted = True
            
            # Update best solution if necessary
            if current_cost < best_cost:
                best_solution = current_solution
                best_cost = current_cost
                iterations_without_improvement = 0
                
                if self.verbose:
                    print(f"Iter {iteration:4d} | T={temperature:6.2f} | "
                          f"New best solution! Cost: {best_cost:.6f}")
            
            # Record history
            self.history['iteration'].append(iteration)
            self.history['temperature'].append(temperature)
            self.history['current_cost'].append(current_cost)
            self.history['best_cost'].append(best_cost)
            self.history['acceptance_rate'].append(1.0 if accepted else 0.0)
            
            # Update temperature
            temperature = self.update_temperature(temperature, iteration)
            
            # Check stopping criteria
            iterations_without_improvement += 1
            
            if temperature < self.temperature_min:
                if self.verbose:
                    print(f"\nMinimum temperature reached: {temperature:.4f}")
                break
            
            if iterations_without_improvement >= self.plateau_iterations:
                if self.verbose:
                    print(f"\nPlateau detected: {iterations_without_improvement} "
                          f"iterations without improvement")
                break
            
            # Periodic logging
            if self.verbose and iteration % 100 == 0 and iteration > 0:
                recent_acceptance = np.mean(self.history['acceptance_rate'][-100:]) * 100
                print(f"Iter {iteration:4d} | T={temperature:6.2f} | "
                      f"Current cost: {current_cost:.6f} | "
                      f"Best: {best_cost:.6f} | "
                      f"Acceptance rate: {recent_acceptance:.1f}%")
        
        if self.verbose:
            print("=" * 60)
            print(f"Optimization completed!")
            print(f"Iterations executed: {iteration + 1}")
            print(f"Best cost: {best_cost:.6f}")
            print(f"Initial cost: {initial_cost:.6f}")
            if initial_cost > 0:
                print(f"Improvement: {(1 - best_cost/initial_cost)*100:.2f}%")
            print("=" * 60)
        
        return best_solution, best_cost, self.history
