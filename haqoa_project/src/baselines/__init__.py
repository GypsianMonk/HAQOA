"""
Baseline Algorithms for HAQOA Comparison
=========================================

Implements standard optimization algorithms:
- Genetic Algorithm (GA)
- Simulated Annealing (SA)
- Particle Swarm Optimization (PSO)
- Ant Colony Optimization (ACO)

All algorithms are implemented for TSP to ensure fair comparison.
"""

import numpy as np
from typing import Tuple, List, Dict, Any
import random


class GeneticAlgorithm:
    """
    Genetic Algorithm for TSP
    
    Standard evolutionary baseline with:
    - Tournament selection
    - Order crossover (OX1)
    - Swap mutation
    - Elitism
    """
    
    def __init__(
        self,
        population_size: int = 50,
        generations: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        elitism_count: int = 2,
        tournament_size: int = 3,
        seed: int = None
    ):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_count = elitism_count
        self.tournament_size = tournament_size
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
    
    def _initialize_population(self, n_cities: int) -> List[np.ndarray]:
        """Create initial population of random tours"""
        population = []
        for _ in range(self.population_size):
            tour = np.random.permutation(n_cities)
            population.append(tour)
        return population
    
    def _calculate_fitness(self, tour: np.ndarray, distance_matrix: np.ndarray) -> float:
        """Calculate fitness (inverse of distance for maximization)"""
        total_distance = 0
        n_cities = len(tour)
        for i in range(n_cities):
            from_city = tour[i]
            to_city = tour[(i + 1) % n_cities]
            total_distance += distance_matrix[from_city, to_city]
        return 1.0 / total_distance if total_distance > 0 else float('inf')
    
    def _tournament_selection(
        self, 
        population: List[np.ndarray], 
        fitnesses: List[float]
    ) -> np.ndarray:
        """Select parent using tournament selection"""
        indices = np.random.choice(len(population), self.tournament_size, replace=False)
        best_idx = max(indices, key=lambda i: fitnesses[i])
        return population[best_idx].copy()
    
    def _order_crossover(
        self, 
        parent1: np.ndarray, 
        parent2: np.ndarray
    ) -> np.ndarray:
        """Order crossover (OX1) for TSP"""
        size = len(parent1)
        start, end = sorted(np.random.choice(size, 2, replace=False))
        
        child = np.full(size, -1, dtype=int)
        child[start:end+1] = parent1[start:end+1]
        
        # Fill remaining positions from parent2
        parent2_genes = [g for g in parent2 if g not in child[start:end+1]]
        child_idx = 0
        parent2_idx = 0
        
        while child_idx < size:
            if child[child_idx] == -1:
                child[child_idx] = parent2_genes[parent2_idx]
                parent2_idx += 1
            child_idx += 1
        
        return child
    
    def _swap_mutation(self, tour: np.ndarray) -> np.ndarray:
        """Swap two cities in the tour"""
        mutated = tour.copy()
        if np.random.random() < self.mutation_rate:
            i, j = np.random.choice(len(tour), 2, replace=False)
            mutated[i], mutated[j] = mutated[j], mutated[i]
        return mutated
    
    def optimize(
        self, 
        distance_matrix: np.ndarray,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Run GA optimization"""
        n_cities = distance_matrix.shape[0]
        population = self._initialize_population(n_cities)
        
        best_solution = None
        best_fitness = -float('inf')
        history = []
        
        for gen in range(self.generations):
            # Evaluate fitness
            fitnesses = [self._calculate_fitness(tour, distance_matrix) 
                        for tour in population]
            
            # Track best solution
            gen_best_idx = np.argmax(fitnesses)
            gen_best_fitness = fitnesses[gen_best_idx]
            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_solution = population[gen_best_idx].copy()
            
            # Record history
            if gen % 10 == 0 or gen == self.generations - 1:
                history.append({
                    'generation': gen,
                    'best_fitness': gen_best_fitness,
                    'mean_fitness': np.mean(fitnesses),
                    'best_distance': 1.0 / gen_best_fitness
                })
            
            # Create next generation
            new_population = []
            
            # Elitism
            elite_indices = np.argsort(fitnesses)[-self.elitism_count:]
            for idx in elite_indices:
                new_population.append(population[idx].copy())
            
            # Generate rest of population
            while len(new_population) < self.population_size:
                parent1 = self._tournament_selection(population, fitnesses)
                parent2 = self._tournament_selection(population, fitnesses)
                
                if np.random.random() < self.crossover_rate:
                    child = self._order_crossover(parent1, parent2)
                else:
                    child = parent1.copy()
                
                child = self._swap_mutation(child)
                new_population.append(child)
            
            population = new_population
            
            if verbose and gen % 10 == 0:
                print(f"Gen {gen:4d} | Best Distance: {1.0/gen_best_fitness:.6f} | "
                      f"Mean Fitness: {np.mean(fitnesses):.6f}")
        
        # Calculate final distance
        best_distance = 0
        n_cities = len(best_solution)
        for i in range(n_cities):
            from_city = best_solution[i]
            to_city = best_solution[(i + 1) % n_cities]
            best_distance += distance_matrix[from_city, to_city]
        
        return {
            'best_tour': best_solution,
            'best_distance': best_distance,
            'best_fitness': best_fitness,
            'history': history,
            'algorithm': 'GA'
        }


class SimulatedAnnealing:
    """
    Simulated Annealing for TSP
    
    Probabilistic baseline with temperature-based acceptance.
    """
    
    def __init__(
        self,
        initial_temp: float = 1000.0,
        cooling_rate: float = 0.995,
        min_temp: float = 1e-8,
        iterations_per_temp: int = 100,
        seed: int = None
    ):
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.iterations_per_temp = iterations_per_temp
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
    
    def _calculate_distance(self, tour: np.ndarray, distance_matrix: np.ndarray) -> float:
        """Calculate total tour distance"""
        total_distance = 0
        n_cities = len(tour)
        for i in range(n_cities):
            from_city = tour[i]
            to_city = tour[(i + 1) % n_cities]
            total_distance += distance_matrix[from_city, to_city]
        return total_distance
    
    def _neighbor(self, tour: np.ndarray) -> np.ndarray:
        """Generate neighboring solution via 2-opt swap"""
        neighbor = tour.copy()
        i, j = np.random.choice(len(tour), 2, replace=False)
        if i > j:
            i, j = j, i
        neighbor[i:j+1] = neighbor[i:j+1][::-1]
        return neighbor
    
    def optimize(
        self, 
        distance_matrix: np.ndarray,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Run SA optimization"""
        n_cities = distance_matrix.shape[0]
        
        # Initialize with random tour
        current_tour = np.random.permutation(n_cities)
        current_distance = self._calculate_distance(current_tour, distance_matrix)
        
        best_tour = current_tour.copy()
        best_distance = current_distance
        
        temp = self.initial_temp
        iteration = 0
        history = []
        
        while temp > self.min_temp:
            for _ in range(self.iterations_per_temp):
                neighbor_tour = self._neighbor(current_tour)
                neighbor_distance = self._calculate_distance(neighbor_tour, distance_matrix)
                
                delta = neighbor_distance - current_distance
                
                # Accept or reject
                if delta < 0 or np.random.random() < np.exp(-delta / temp):
                    current_tour = neighbor_tour
                    current_distance = neighbor_distance
                    
                    if current_distance < best_distance:
                        best_distance = current_distance
                        best_tour = current_tour.copy()
                
                iteration += 1
            
            # Cool down
            temp *= self.cooling_rate
            
            if verbose and iteration % 1000 == 0:
                print(f"Iter {iteration:5d} | Temp: {temp:8.4f} | "
                      f"Best Distance: {best_distance:.6f}")
            
            # Record history periodically
            if iteration % 5000 == 0:
                history.append({
                    'iteration': iteration,
                    'temperature': temp,
                    'best_distance': best_distance
                })
        
        history.append({
            'iteration': iteration,
            'temperature': temp,
            'best_distance': best_distance
        })
        
        return {
            'best_tour': best_tour,
            'best_distance': best_distance,
            'total_iterations': iteration,
            'history': history,
            'algorithm': 'SA'
        }


class ParticleSwarmOptimization:
    """
    Particle Swarm Optimization for TSP
    
    Adapted for discrete permutation problems.
    Uses position-based encoding and velocity-inspired operators.
    """
    
    def __init__(
        self,
        swarm_size: int = 30,
        generations: int = 100,
        inertia_weight: float = 0.7,
        cognitive_coefficient: float = 1.5,
        social_coefficient: float = 1.5,
        seed: int = None
    ):
        self.swarm_size = swarm_size
        self.generations = generations
        self.inertia_weight = inertia_weight
        self.cognitive_coefficient = cognitive_coefficient
        self.social_coefficient = social_coefficient
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
    
    def _initialize_swarm(self, n_cities: int) -> List[np.ndarray]:
        """Initialize particle positions (tours)"""
        swarm = []
        for _ in range(self.swarm_size):
            tour = np.random.permutation(n_cities)
            swarm.append(tour)
        return swarm
    
    def _calculate_distance(self, tour: np.ndarray, distance_matrix: np.ndarray) -> float:
        """Calculate total tour distance"""
        total_distance = 0
        n_cities = len(tour)
        for i in range(n_cities):
            from_city = tour[i]
            to_city = tour[(i + 1) % n_cities]
            total_distance += distance_matrix[from_city, to_city]
        return total_distance
    
    def _apply_velocity(
        self, 
        tour: np.ndarray, 
        velocity: float,
        personal_best: np.ndarray,
        global_best: np.ndarray,
        distance_matrix: np.ndarray
    ) -> np.ndarray:
        """Apply velocity-inspired operations"""
        new_tour = tour.copy()
        n_cities = len(tour)
        
        # Number of swaps based on velocity
        n_swaps = max(1, int(velocity * n_cities * 0.1))
        
        for _ in range(n_swaps):
            # Random swap component (inertia)
            if np.random.random() < self.inertia_weight:
                i, j = np.random.choice(n_cities, 2, replace=False)
                new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
            
            # Cognitive component (move toward personal best)
            if np.random.random() < self.cognitive_coefficient * 0.3:
                # Find a city that's in different position in personal best
                diff_indices = np.where(new_tour != personal_best)[0]
                if len(diff_indices) > 0:
                    idx = np.random.choice(diff_indices)
                    target_pos = np.where(personal_best == new_tour[idx])[0][0]
                    current_pos = np.where(new_tour == new_tour[idx])[0][0]
                    new_tour[target_pos], new_tour[current_pos] = \
                        new_tour[current_pos], new_tour[target_pos]
            
            # Social component (move toward global best)
            if np.random.random() < self.social_coefficient * 0.3:
                diff_indices = np.where(new_tour != global_best)[0]
                if len(diff_indices) > 0:
                    idx = np.random.choice(diff_indices)
                    target_pos = np.where(global_best == new_tour[idx])[0][0]
                    current_pos = np.where(new_tour == new_tour[idx])[0][0]
                    new_tour[target_pos], new_tour[current_pos] = \
                        new_tour[current_pos], new_tour[target_pos]
        
        return new_tour
    
    def optimize(
        self, 
        distance_matrix: np.ndarray,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Run PSO optimization"""
        n_cities = distance_matrix.shape[0]
        
        # Initialize swarm
        swarm = self._initialize_swarm(n_cities)
        personal_bests = [tour.copy() for tour in swarm]
        personal_best_distances = [
            self._calculate_distance(tour, distance_matrix) for tour in swarm
        ]
        
        # Find global best
        global_best_idx = np.argmin(personal_best_distances)
        global_best = swarm[global_best_idx].copy()
        global_best_distance = personal_best_distances[global_best_idx]
        
        # Initialize velocities
        velocities = np.random.uniform(0.5, 1.0, self.swarm_size)
        
        history = []
        
        for gen in range(self.generations):
            for i in range(self.swarm_size):
                # Update velocity
                r1, r2 = np.random.random(2)
                velocities[i] = (self.inertia_weight * velocities[i] +
                               self.cognitive_coefficient * r1 * 0.5 +
                               self.social_coefficient * r2 * 0.5)
                velocities[i] = np.clip(velocities[i], 0.1, 2.0)
                
                # Apply velocity to get new position
                new_tour = self._apply_velocity(
                    swarm[i], velocities[i],
                    personal_bests[i], global_best,
                    distance_matrix
                )
                
                new_distance = self._calculate_distance(new_tour, distance_matrix)
                
                # Update personal best
                if new_distance < personal_best_distances[i]:
                    personal_best_distances[i] = new_distance
                    personal_bests[i] = new_tour.copy()
                    
                    # Update global best
                    if new_distance < global_best_distance:
                        global_best_distance = new_distance
                        global_best = new_tour.copy()
                
                swarm[i] = new_tour
            
            if verbose and gen % 10 == 0:
                print(f"Gen {gen:4d} | Global Best Distance: {global_best_distance:.6f}")
            
            history.append({
                'generation': gen,
                'best_distance': global_best_distance,
                'mean_distance': np.mean(personal_best_distances)
            })
        
        return {
            'best_tour': global_best,
            'best_distance': global_best_distance,
            'history': history,
            'algorithm': 'PSO'
        }


class AntColonyOptimization:
    """
    Ant Colony Optimization for TSP
    
    Bio-inspired algorithm using pheromone trails.
    """
    
    def __init__(
        self,
        n_ants: int = 20,
        generations: int = 100,
        alpha: float = 1.0,      # Pheromone importance
        beta: float = 2.0,       # Heuristic importance
        evaporation_rate: float = 0.5,
        q: float = 1.0,          # Pheromone deposit factor
        seed: int = None
    ):
        self.n_ants = n_ants
        self.generations = generations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.q = q
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
    
    def _select_next_city(
        self, 
        current_city: int, 
        visited: set,
        pheromone: np.ndarray,
        heuristic: np.ndarray,
        distance_matrix: np.ndarray
    ) -> int:
        """Select next city based on pheromone and heuristic"""
        n_cities = distance_matrix.shape[0]
        available_cities = [c for c in range(n_cities) if c not in visited]
        
        if not available_cities:
            return visited.pop()  # Should not happen in valid TSP
        
        # Calculate probabilities
        probabilities = []
        for city in available_cities:
            tau = pheromone[current_city, city] ** self.alpha
            eta = heuristic[current_city, city] ** self.beta
            probabilities.append(tau * eta)
        
        probabilities = np.array(probabilities)
        probabilities /= probabilities.sum()
        
        # Select city
        selected_idx = np.random.choice(len(available_cities), p=probabilities)
        return available_cities[selected_idx]
    
    def _construct_tour(
        self, 
        pheromone: np.ndarray,
        heuristic: np.ndarray,
        distance_matrix: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Construct a complete tour for one ant"""
        n_cities = distance_matrix.shape[0]
        tour = []
        visited = set()
        
        # Start from random city
        start_city = np.random.randint(n_cities)
        tour.append(start_city)
        visited.add(start_city)
        
        current_city = start_city
        for _ in range(n_cities - 1):
            next_city = self._select_next_city(
                current_city, visited, pheromone, heuristic, distance_matrix
            )
            tour.append(next_city)
            visited.add(next_city)
            current_city = next_city
        
        tour = np.array(tour)
        distance = self._calculate_distance(tour, distance_matrix)
        
        return tour, distance
    
    def _calculate_distance(self, tour: np.ndarray, distance_matrix: np.ndarray) -> float:
        """Calculate total tour distance"""
        total_distance = 0
        n_cities = len(tour)
        for i in range(n_cities):
            from_city = tour[i]
            to_city = tour[(i + 1) % n_cities]
            total_distance += distance_matrix[from_city, to_city]
        return total_distance
    
    def _update_pheromone(
        self, 
        pheromone: np.ndarray,
        tours: List[np.ndarray],
        distances: List[float],
        distance_matrix: np.ndarray
    ) -> np.ndarray:
        """Update pheromone levels"""
        n_cities = distance_matrix.shape[0]
        
        # Evaporation
        pheromone *= (1 - self.evaporation_rate)
        
        # Deposit
        for tour, distance in zip(tours, distances):
            if distance > 0:
                delta_pheromone = self.q / distance
                for i in range(n_cities):
                    from_city = tour[i]
                    to_city = tour[(i + 1) % n_cities]
                    pheromone[from_city, to_city] += delta_pheromone
                    pheromone[to_city, from_city] += delta_pheromone
        
        return pheromone
    
    def optimize(
        self, 
        distance_matrix: np.ndarray,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Run ACO optimization"""
        n_cities = distance_matrix.shape[0]
        
        # Initialize pheromone
        pheromone = np.ones((n_cities, n_cities))
        
        # Heuristic information (inverse of distance)
        with np.errstate(divide='ignore'):
            heuristic = 1.0 / distance_matrix
        heuristic[heuristic == np.inf] = 1.0
        
        best_tour = None
        best_distance = float('inf')
        history = []
        
        for gen in range(self.generations):
            tours = []
            distances = []
            
            # Each ant constructs a tour
            for _ in range(self.n_ants):
                tour, distance = self._construct_tour(
                    pheromone, heuristic, distance_matrix
                )
                tours.append(tour)
                distances.append(distance)
                
                if distance < best_distance:
                    best_distance = distance
                    best_tour = tour.copy()
            
            # Update pheromone
            pheromone = self._update_pheromone(
                pheromone, tours, distances, distance_matrix
            )
            
            if verbose and gen % 10 == 0:
                print(f"Gen {gen:4d} | Best Distance: {best_distance:.6f} | "
                      f"Mean Distance: {np.mean(distances):.6f}")
            
            history.append({
                'generation': gen,
                'best_distance': best_distance,
                'mean_distance': np.mean(distances)
            })
        
        return {
            'best_tour': best_tour,
            'best_distance': best_distance,
            'history': history,
            'algorithm': 'ACO'
        }


def run_baseline_comparison(
    distance_matrix: np.ndarray,
    algorithms: List[str] = None,
    common_params: Dict[str, Any] = None,
    verbose: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Run multiple baseline algorithms for comparison
    
    Parameters
    ----------
    distance_matrix : np.ndarray
        TSP distance matrix
    algorithms : List[str]
        List of algorithms to run ['GA', 'SA', 'PSO', 'ACO']
    common_params : Dict[str, Any]
        Common parameters for all algorithms
    verbose : bool
        Print progress
    
    Returns
    -------
    Dict[str, Dict[str, Any]]
        Results from each algorithm
    """
    if algorithms is None:
        algorithms = ['GA', 'SA', 'PSO', 'ACO']
    
    if common_params is None:
        common_params = {'seed': 42}
    
    results = {}
    
    for algo_name in algorithms:
        print(f"\n{'='*60}")
        print(f"Running {algo_name}...")
        print('='*60)
        
        if algo_name == 'GA':
            algo = GeneticAlgorithm(
                population_size=common_params.get('population_size', 50),
                generations=common_params.get('generations', 100),
                seed=common_params.get('seed')
            )
            result = algo.optimize(distance_matrix, verbose=verbose)
        
        elif algo_name == 'SA':
            algo = SimulatedAnnealing(
                initial_temp=common_params.get('initial_temp', 1000.0),
                cooling_rate=common_params.get('cooling_rate', 0.995),
                iterations_per_temp=common_params.get('iterations_per_temp', 100),
                seed=common_params.get('seed')
            )
            result = algo.optimize(distance_matrix, verbose=verbose)
        
        elif algo_name == 'PSO':
            algo = ParticleSwarmOptimization(
                swarm_size=common_params.get('swarm_size', 30),
                generations=common_params.get('generations', 100),
                seed=common_params.get('seed')
            )
            result = algo.optimize(distance_matrix, verbose=verbose)
        
        elif algo_name == 'ACO':
            algo = AntColonyOptimization(
                n_ants=common_params.get('n_ants', 20),
                generations=common_params.get('generations', 100),
                seed=common_params.get('seed')
            )
            result = algo.optimize(distance_matrix, verbose=verbose)
        
        else:
            raise ValueError(f"Unknown algorithm: {algo_name}")
        
        results[algo_name] = result
    
    return results


if __name__ == "__main__":
    # Simple test
    np.random.seed(42)
    n_cities = 10
    coords = np.random.rand(n_cities, 2)
    
    # Compute distance matrix
    distance_matrix = np.zeros((n_cities, n_cities))
    for i in range(n_cities):
        for j in range(n_cities):
            distance_matrix[i, j] = np.linalg.norm(coords[i] - coords[j])
    
    print("Testing baseline algorithms...")
    results = run_baseline_comparison(
        distance_matrix,
        algorithms=['GA', 'SA'],
        common_params={'generations': 50, 'seed': 42},
        verbose=True
    )
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for algo_name, result in results.items():
        print(f"{algo_name}: Best Distance = {result['best_distance']:.6f}")
