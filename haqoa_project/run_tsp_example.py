#!/usr/bin/env python3
"""
HAQOA - TSP Example Runner

Demonstrates the HAQOA framework solving a Traveling Salesman Problem.
This is the AQSE-v1 (Adaptive Quantum-Inspired State Evolution Engine) in action.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from tsp_solver import create_tsp_instance, TSPSolver, TSPSolution
from haqoa_engine import HAQOAOptimizer, HAQOAConfig


def run_haqoa_tsp_example(
    n_cities: int = 20,
    n_generations: int = 200,
    seed: int = 42,
    verbose: bool = True
):
    """
    Run HAQOA on a TSP problem.
    
    Args:
        n_cities: Number of cities in TSP
        n_generations: Maximum generations to run
        seed: Random seed for reproducibility
        verbose: Print progress
    """
    print("=" * 70)
    print("HAQOA - Hybrid AI-Assisted Quantum-Inspired Optimization Architecture")
    print("AQSE-v1: Adaptive Quantum-Inspired State Evolution Engine")
    print("=" * 70)
    print(f"\nProblem: TSP with {n_cities} cities")
    print(f"Max generations: {n_generations}")
    print(f"Random seed: {seed}")
    print()
    
    # Create TSP instance
    tsp_instance = create_tsp_instance(n_cities=n_cities, seed=seed)
    tsp_solver = TSPSolver(tsp_instance)
    
    print(f"Distance matrix computed: {tsp_instance.distance_matrix.shape}")
    print(f"City coordinates: {tsp_instance.coordinates.shape}")
    print()
    
    # Define problem-specific functions for HAQOA
    
    def solution_generator():
        """Generate a new random TSP solution."""
        return tsp_solver.generate_random_tour()
    
    def solution_mutator(solution):
        """Mutate a TSP solution."""
        method = np.random.choice(["2opt", "swap", "insertion"])
        return tsp_solver.mutate(solution, method)
    
    def solution_recombiner(parent1, parent2):
        """Recombine two TSP solutions."""
        return tsp_solver.recombine(parent1, parent2, method="ox1")
    
    def quality_evaluator(solution):
        """Evaluate TSP solution quality (higher is better)."""
        return tsp_solver.compute_quality(solution)
    
    # Configure HAQOA
    config = HAQOAConfig(
        initial_population_size=30,
        max_states=150,
        min_states_after_collapse=15,
        max_generations=n_generations,
        max_time_seconds=60.0,
        base_amplification=1.5,
        entropy_sensitivity=0.4,
        collapse_interval=15,
        mutation_rate=0.15,
        recombination_probability=0.4,
        random_seed=seed,
    )
    
    # Create optimizer
    optimizer = HAQOAOptimizer(config=config)
    
    # Set problem functions
    optimizer.set_problem_functions(
        solution_generator=solution_generator,
        solution_mutator=solution_mutator,
        solution_recombiner=solution_recombiner,
        quality_evaluator=quality_evaluator
    )
    
    # Generate some initial solutions including greedy one
    initial_solutions = []
    
    # Add greedy solution
    greedy_sol = tsp_solver.greedy_initial_solution()
    initial_solutions.append(greedy_sol)
    
    # Add random solutions
    for _ in range(min(5, config.initial_population_size - 1)):
        initial_solutions.append(tsp_solver.generate_random_tour())
    
    print("-" * 70)
    print("Starting HAQOA Optimization...")
    print("-" * 70)
    
    # Run optimization
    result = optimizer.optimize(
        initial_solutions=initial_solutions,
        verbose=verbose
    )
    
    # Report results
    print()
    print("=" * 70)
    print("OPTIMIZATION RESULTS")
    print("=" * 70)
    
    best_solution = result.best_solution
    best_distance = best_solution.total_distance if hasattr(best_solution, 'total_distance') else float('inf')
    
    print(f"\nBest Solution Found:")
    print(f"  Tour: {best_solution.tour[:10]}{'...' if len(best_solution.tour) > 10 else ''}")
    print(f"  Total Distance: {best_distance:.4f}")
    print(f"  Quality Score: {result.best_quality:.6f}")
    
    print(f"\nOptimization Statistics:")
    print(f"  Status: {result.status.value}")
    print(f"  Generations: {result.total_generations}")
    print(f"  Runtime: {result.total_runtime:.2f} seconds")
    print(f"  Final States: {result.final_num_states}")
    print(f"  Final Entropy: {result.final_entropy:.4f}")
    
    print(f"\nConvergence History:")
    print(f"  Initial Best: {result.convergence_history[0]:.6f}" if result.convergence_history else "  N/A")
    print(f"  Final Best: {result.best_quality:.6f}")
    print(f"  Improvement: {(result.best_quality - result.convergence_history[0]) / abs(result.convergence_history[0]) * 100:.2f}%" if result.convergence_history else "  N/A")
    
    # Get detailed statistics
    stats = optimizer.get_statistics()
    print(f"\nComponent Statistics:")
    print(f"  Amplifier - Mean β: {stats['amplifier']['mean_amplification']:.4f}")
    print(f"  Collapse - Total: {stats['collapse']['total_collapses']}")
    evolution_stats = stats.get('evolution', {})
    print(f"  Evolution - Mutations: {evolution_stats.get('total_mutations', 'N/A')}")
    print(f"  Evolution - Recombinations: {evolution_stats.get('total_recombinations', 'N/A')}")
    
    print()
    print("=" * 70)
    print("HAQOA optimization complete!")
    print("=" * 70)
    
    return result, tsp_solver


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run HAQOA on TSP problem")
    parser.add_argument("--cities", type=int, default=20, help="Number of cities")
    parser.add_argument("--generations", type=int, default=200, help="Max generations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    
    args = parser.parse_args()
    
    result, solver = run_haqoa_tsp_example(
        n_cities=args.cities,
        n_generations=args.generations,
        seed=args.seed,
        verbose=not args.quiet
    )
