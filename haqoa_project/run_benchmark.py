#!/usr/bin/env python3
"""
HAQOA Benchmark Framework
==========================

Comprehensive benchmark comparing HAQOA against baseline algorithms:
- Genetic Algorithm (GA)
- Simulated Annealing (SA)
- Particle Swarm Optimization (PSO)
- Ant Colony Optimization (ACO)

Includes statistical validation with multiple runs.
"""

import numpy as np
import sys
import time
from typing import Dict, List, Any, Tuple
import argparse

# Add src to path
sys.path.insert(0, '/workspace/haqoa_project/src')

from haqoa_engine import HAQOAOptimizer
from tsp_solver import TSPSolver
from baselines import (
    GeneticAlgorithm,
    SimulatedAnnealing,
    ParticleSwarmOptimization,
    AntColonyOptimization
)


def generate_tsp_instance(n_cities: int, seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """Generate random TSP instance"""
    if seed is not None:
        np.random.seed(seed)
    
    coords = np.random.rand(n_cities, 2)
    
    # Compute Euclidean distance matrix
    distance_matrix = np.zeros((n_cities, n_cities))
    for i in range(n_cities):
        for j in range(n_cities):
            distance_matrix[i, j] = np.linalg.norm(coords[i] - coords[j])
    
    return distance_matrix, coords


def run_algorithm(algorithm, distance_matrix: np.ndarray, verbose: bool = False) -> Dict[str, Any]:
    """Run an algorithm and measure performance"""
    start_time = time.time()
    
    # For HAQOA, we need to provide TSP-specific functions
    if hasattr(algorithm, 'config'):  # It's HAQOA
        from tsp_solver import TSPInstance, TSPSolution
        
        n_cities = distance_matrix.shape[0]
        
        # Create dummy coordinates for TSP instance
        coords = np.random.rand(n_cities, 2)
        tsp_instance = TSPInstance(
            n_cities=n_cities,
            coordinates=coords,
            distance_matrix=distance_matrix
        )
        
        tsp_solver = TSPSolver(tsp_instance)
        
        # Set up the optimizer with TSP-specific functions
        algorithm.solution_generator = lambda: tsp_solver.generate_random_tour().tour
        algorithm.solution_mutator = lambda tour: tsp_solver.mutate_swap(tour.copy())
        algorithm.solution_recombiner = lambda t1, t2: tsp_solver.recombine_ox1(t1.copy(), t2.copy())
        algorithm.quality_evaluator = lambda tour: -tsp_solver.evaluate_solution(
            TSPSolution(tour=list(tour) if not isinstance(tour, list) else tour)
        ).total_distance  # Negative for maximization
        
        # Initialize population
        algorithm.initialize_population()
        
        # Run optimization
        opt_result = algorithm.optimize(verbose=verbose)
        
        # Convert OptimizationResult to dict
        result = {
            'best_tour': opt_result.best_solution,
            'best_distance': -opt_result.best_quality,  # Convert back to positive
            'best_quality': opt_result.best_quality,
            'history': [vars(hr) if hasattr(hr, '__dict__') else hr for hr in opt_result.generation_results],
            'algorithm': 'HAQOA',
            'runtime': 0  # Will be set below
        }
    else:
        result = algorithm.optimize(distance_matrix, verbose=verbose)
    
    end_time = time.time()
    runtime = end_time - start_time
    
    result['runtime'] = runtime
    
    return result


def statistical_analysis(results: List[float]) -> Dict[str, float]:
    """Compute statistical metrics"""
    results = np.array(results)
    
    return {
        'mean': float(np.mean(results)),
        'std': float(np.std(results)),
        'min': float(np.min(results)),
        'max': float(np.max(results)),
        'median': float(np.median(results)),
        'variance': float(np.var(results))
    }


def run_benchmark(
    n_cities: int = 15,
    n_runs: int = 10,
    generations: int = 100,
    algorithms: List[str] = None,
    seed_base: int = 42,
    verbose: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Run comprehensive benchmark across multiple algorithms and runs
    
    Parameters
    ----------
    n_cities : int
        Number of cities in TSP
    n_runs : int
        Number of independent runs for statistical validity
    generations : int
        Maximum generations/iterations
    algorithms : List[str]
        Algorithms to compare
    seed_base : int
        Base random seed
    verbose : bool
        Print progress
    
    Returns
    -------
    Dict[str, Dict[str, Any]]
        Benchmark results with statistics
    """
    if algorithms is None:
        algorithms = ['HAQOA', 'GA', 'SA', 'PSO', 'ACO']
    
    # Store all results
    all_results = {algo: {'distances': [], 'runtimes': [], 'full_results': []} 
                   for algo in algorithms}
    
    print("="*70)
    print("HAQOA BENCHMARK FRAMEWORK")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  - Cities: {n_cities}")
    print(f"  - Runs: {n_runs}")
    print(f"  - Generations: {generations}")
    print(f"  - Algorithms: {', '.join(algorithms)}")
    print(f"  - Seed base: {seed_base}")
    print("\n" + "="*70)
    
    for run_idx in range(n_runs):
        current_seed = seed_base + run_idx
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"RUN {run_idx + 1}/{n_runs} (seed={current_seed})")
            print('='*70)
        
        # Generate TSP instance for this run
        distance_matrix, coords = generate_tsp_instance(n_cities, seed=current_seed)
        
        # Run each algorithm
        for algo_name in algorithms:
            if verbose:
                print(f"\n  Running {algo_name}...")
            
            try:
                if algo_name == 'HAQOA':
                    from haqoa_engine import HAQOAConfig
                    
                    config = HAQOAConfig(
                        initial_population_size=30,
                        max_generations=generations,
                        base_amplification=2.0,
                        entropy_sensitivity=0.5,
                        target_entropy=0.7,
                        random_seed=current_seed
                    )
                    algo = HAQOAOptimizer(config=config)
                
                elif algo_name == 'GA':
                    algo = GeneticAlgorithm(
                        population_size=50,
                        generations=generations,
                        mutation_rate=0.1,
                        crossover_rate=0.8,
                        elitism_count=2,
                        seed=current_seed
                    )
                
                elif algo_name == 'SA':
                    # Adjust SA parameters to match computational budget
                    iterations_per_temp = max(10, (generations * 50) // 1000)
                    algo = SimulatedAnnealing(
                        initial_temp=1000.0,
                        cooling_rate=0.995,
                        min_temp=1e-8,
                        iterations_per_temp=iterations_per_temp,
                        seed=current_seed
                    )
                
                elif algo_name == 'PSO':
                    algo = ParticleSwarmOptimization(
                        swarm_size=30,
                        generations=generations,
                        inertia_weight=0.7,
                        cognitive_coefficient=1.5,
                        social_coefficient=1.5,
                        seed=current_seed
                    )
                
                elif algo_name == 'ACO':
                    algo = AntColonyOptimization(
                        n_ants=20,
                        generations=generations,
                        alpha=1.0,
                        beta=2.0,
                        evaporation_rate=0.5,
                        q=1.0,
                        seed=current_seed
                    )
                
                else:
                    raise ValueError(f"Unknown algorithm: {algo_name}")
                
                # Run optimization
                result = run_algorithm(algo, distance_matrix, verbose=False)
                
                # Store results
                all_results[algo_name]['distances'].append(result['best_distance'])
                all_results[algo_name]['runtimes'].append(result['runtime'])
                all_results[algo_name]['full_results'].append(result)
                
                if verbose:
                    print(f"    Distance: {result['best_distance']:.6f}, "
                          f"Time: {result['runtime']:.3f}s")
                
            except Exception as e:
                print(f"    ERROR: {str(e)}")
                import traceback
                traceback.print_exc()
    
    # Compute statistics
    benchmark_summary = {}
    
    print("\n" + "="*70)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*70)
    
    for algo_name in algorithms:
        distances = all_results[algo_name]['distances']
        runtimes = all_results[algo_name]['runtimes']
        
        if len(distances) > 0:
            distance_stats = statistical_analysis(distances)
            runtime_stats = statistical_analysis(runtimes)
            
            benchmark_summary[algo_name] = {
                'distance': distance_stats,
                'runtime': runtime_stats,
                'n_successful_runs': len(distances)
            }
            
            print(f"\n{algo_name}:")
            print(f"  Distance - Mean: {distance_stats['mean']:.6f}, "
                  f"Std: {distance_stats['std']:.6f}, "
                  f"Best: {distance_stats['min']:.6f}")
            print(f"  Runtime  - Mean: {runtime_stats['mean']:.3f}s, "
                  f"Std: {runtime_stats['std']:.3f}s")
    
    # Find best performer
    if len(algorithms) > 1:
        mean_distances = {algo: benchmark_summary[algo]['distance']['mean'] 
                         for algo in algorithms 
                         if algo in benchmark_summary}
        
        if mean_distances:
            best_algo = min(mean_distances, key=mean_distances.get)
            print(f"\n{'='*70}")
            print(f"BEST PERFORMER: {best_algo} "
                  f"(Mean Distance: {mean_distances[best_algo]:.6f})")
            print("="*70)
    
    return benchmark_summary


def main():
    parser = argparse.ArgumentParser(
        description='HAQOA Benchmark Framework - Compare optimization algorithms'
    )
    parser.add_argument('--cities', type=int, default=15,
                       help='Number of cities in TSP (default: 15)')
    parser.add_argument('--runs', type=int, default=10,
                       help='Number of independent runs (default: 10)')
    parser.add_argument('--generations', type=int, default=100,
                       help='Maximum generations (default: 100)')
    parser.add_argument('--algorithms', type=str, nargs='+',
                       default=['HAQOA', 'GA', 'SA', 'PSO', 'ACO'],
                       help='Algorithms to compare')
    parser.add_argument('--seed', type=int, default=42,
                       help='Base random seed (default: 42)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress detailed output')
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    # Run benchmark
    results = run_benchmark(
        n_cities=args.cities,
        n_runs=args.runs,
        generations=args.generations,
        algorithms=args.algorithms,
        seed_base=args.seed,
        verbose=verbose
    )
    
    # Save results to file
    import json
    output_file = f'/workspace/haqoa_project/results/benchmark_{args.cities}cities_{args.runs}runs.json'
    
    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        return obj
    
    serializable_results = convert_numpy(results)
    
    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    main()
