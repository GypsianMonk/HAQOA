"""
Evolution Engine

Implements Section 10: Evolution Engine
Handles state regeneration, mutation, and probabilistic recombination.

Features:
- Probabilistic recombination
- Entropy-guided mutation
- Adaptive regeneration
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Callable
from dataclasses import dataclass

# Import without relative path for standalone usage
try:
    from .state_superposition import QuantumState, StateSuperpositionGenerator
    from .entropy_monitor import EntropyMetrics
except ImportError:
    from state_superposition import QuantumState, StateSuperpositionGenerator
    from entropy_monitor import EntropyMetrics


@dataclass
class MutationConfig:
    """Configuration for mutation operations."""
    
    base_mutation_rate: float = 0.1
    min_mutation_rate: float = 0.01
    max_mutation_rate: float = 0.5
    entropy_adaptive: bool = True
    
    # Mutation strength parameters
    mutation_strength: float = 0.2
    adaptive_strength: bool = True


@dataclass
class RecombinationConfig:
    """Configuration for recombination operations."""
    
    recombination_probability: float = 0.3
    parent_selection_method: str = "tournament"  # 'tournament', 'rank', 'probability'
    tournament_size: int = 3
    crossover_type: str = "uniform"  # 'uniform', 'single_point', 'two_point'


@dataclass
class EvolutionResult:
    """Result of an evolution operation."""
    
    new_states_created: int
    states_mutated: int
    states_recombined: int
    total_states_after: int
    
    # Statistics
    mean_quality_change: float = 0.0
    best_new_quality: float = 0.0
    diversity_change: float = 0.0
    
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class EvolutionEngine:
    """
    Implements evolutionary operations for HAQOA.
    
    Features:
    - State generation with quality estimation
    - Entropy-guided mutation rates
    - Probabilistic parent selection
    - Multiple recombination strategies
    """
    
    def __init__(
        self,
        mutation_config: Optional[MutationConfig] = None,
        recombination_config: Optional[RecombinationConfig] = None,
        solution_generator: Optional[Callable] = None,
        solution_mutator: Optional[Callable[[Any], Any]] = None,
        solution_recombiner: Optional[Callable[[Any, Any], Tuple[Any, Any]]] = None,
        quality_evaluator: Optional[Callable[[Any], float]] = None
    ):
        """
        Initialize the Evolution Engine.
        
        Args:
            mutation_config: Mutation configuration
            recombination_config: Recombination configuration
            solution_generator: Function to generate new solutions
            solution_mutator: Function to mutate a solution
            solution_recombiner: Function to recombine two solutions
            quality_evaluator: Function to evaluate solution quality
        """
        self.mutation_config = mutation_config or MutationConfig()
        self.recombination_config = recombination_config or RecombinationConfig()
        
        self.solution_generator = solution_generator
        self.solution_mutator = solution_mutator
        self.solution_recombiner = solution_recombiner
        self.quality_evaluator = quality_evaluator
        
        self.generation = 0
        self.evolution_history: List[EvolutionResult] = []
        
        # Adaptive mutation rate state
        self.current_mutation_rate = self.mutation_config.base_mutation_rate
    
    def set_problem_specific_functions(
        self,
        solution_generator: Callable,
        solution_mutator: Callable[[Any], Any],
        solution_recombiner: Callable[[Any, Any], Tuple[Any, Any]],
        quality_evaluator: Callable[[Any], float]
    ):
        """Set problem-specific functions for evolution."""
        self.solution_generator = solution_generator
        self.solution_mutator = solution_mutator
        self.solution_recombiner = solution_recombiner
        self.quality_evaluator = quality_evaluator
    
    def compute_adaptive_mutation_rate(
        self,
        entropy_metrics: Optional[EntropyMetrics]
    ) -> float:
        """
        Compute mutation rate based on entropy.
        
        High entropy → lower mutation (already diverse)
        Low entropy → higher mutation (need diversity)
        """
        base_rate = self.mutation_config.base_mutation_rate
        
        if not self.mutation_config.entropy_adaptive or entropy_metrics is None:
            return base_rate
        
        normalized_entropy = entropy_metrics.normalized_entropy
        
        # Inverse relationship: low entropy → high mutation
        entropy_factor = 1.0 - normalized_entropy
        
        # Adjust rate
        rate = base_rate * (1 + entropy_factor)
        
        # Apply bounds
        rate = np.clip(
            rate,
            self.mutation_config.min_mutation_rate,
            self.mutation_config.max_mutation_rate
        )
        
        self.current_mutation_rate = float(rate)
        return self.current_mutation_rate
    
    def generate_new_states(
        self,
        superposition_gen: StateSuperpositionGenerator,
        n_new: int,
        entropy_metrics: Optional[EntropyMetrics] = None
    ) -> List[QuantumState]:
        """
        Generate new states to add to the population.
        
        Args:
            superposition_gen: State superposition generator
            n_new: Number of new states to generate
            entropy_metrics: Current entropy metrics
            
        Returns:
            List of newly created states
        """
        if self.solution_generator is None:
            raise ValueError("solution_generator must be set")
        
        if self.quality_evaluator is None:
            raise ValueError("quality_evaluator must be set")
        
        new_states = []
        
        for _ in range(n_new):
            # Generate new solution
            solution = self.solution_generator()
            
            # Evaluate quality
            quality = self.quality_evaluator(solution)
            
            # Add to superposition
            state = superposition_gen.add_state(
                solution=solution,
                quality_score=quality
            )
            
            new_states.append(state)
        
        return new_states
    
    def mutate_states(
        self,
        states: List[QuantumState],
        entropy_metrics: Optional[EntropyMetrics] = None,
        mutation_rate_override: Optional[float] = None
    ) -> List[QuantumState]:
        """
        Apply mutation to selected states.
        
        Args:
            states: States to potentially mutate
            entropy_metrics: For adaptive mutation rate
            mutation_rate_override: Override computed mutation rate
            
        Returns:
            List of mutated states
        """
        if self.solution_mutator is None:
            return []
        
        # Compute mutation rate
        if mutation_rate_override is not None:
            rate = mutation_rate_override
        else:
            rate = self.compute_adaptive_mutation_rate(entropy_metrics)
        
        mutated_states = []
        
        for state in states:
            if np.random.random() < rate:
                # Apply mutation
                original_solution = state.solution
                mutated_solution = self.solution_mutator(original_solution)
                
                # Evaluate new quality
                if self.quality_evaluator is not None:
                    new_quality = self.quality_evaluator(mutated_solution)
                    
                    # Update state
                    state.solution = mutated_solution
                    state.quality_score = new_quality
                    
                    # Reset age on successful mutation
                    state.age = 0
                    
                    mutated_states.append(state)
        
        return mutated_states
    
    def recombine_states(
        self,
        states: List[QuantumState],
        entropy_metrics: Optional[EntropyMetrics] = None
    ) -> List[QuantumState]:
        """
        Perform recombination between selected states.
        
        Args:
            states: Population of states
            entropy_metrics: Current entropy metrics
            
        Returns:
            List of newly created offspring states
        """
        if self.solution_recombiner is None:
            return []
        
        if len(states) < 2:
            return []
        
        offspring = []
        prob = self.recombination_config.recombination_probability
        
        # Select parents based on configured method
        parents = self._select_parents(states)
        
        # Create pairs and recombine
        for i in range(0, len(parents) - 1, 2):
            if np.random.random() > prob:
                continue
            
            parent1 = parents[i]
            parent2 = parents[i + 1]
            
            # Recombine solutions
            child1_sol, child2_sol = self.solution_recombiner(
                parent1.solution,
                parent2.solution
            )
            
            # Evaluate children
            if self.quality_evaluator is not None:
                child1_quality = self.quality_evaluator(child1_sol)
                child2_quality = self.quality_evaluator(child2_sol)
                
                # Create amplitude from parent probabilities
                amp1 = (parent1.amplitude + parent2.amplitude) / 2
                amp2 = amp1.copy()
                
                # Create child states (not added to superposition yet)
                child1 = QuantumState(
                    state_id=-1,  # Temporary ID
                    solution=child1_sol,
                    amplitude=amp1,
                    quality_score=child1_quality
                )
                
                child2 = QuantumState(
                    state_id=-1,
                    solution=child2_sol,
                    amplitude=amp2,
                    quality_score=child2_quality
                )
                
                offspring.extend([child1, child2])
        
        return offspring
    
    def _select_parents(
        self,
        states: List[QuantumState]
    ) -> List[QuantumState]:
        """Select parents for recombination."""
        method = self.recombination_config.parent_selection_method
        
        if method == "tournament":
            return self._tournament_selection(states)
        elif method == "rank":
            return self._rank_selection(states)
        elif method == "probability":
            return self._probability_selection(states)
        else:
            return states[:min(len(states), 10)]
    
    def _tournament_selection(
        self,
        states: List[QuantumState],
        k: Optional[int] = None
    ) -> List[QuantumState]:
        """Tournament selection."""
        if k is None:
            k = self.recombination_config.tournament_size
        
        selected = []
        n_select = min(len(states), 10)
        
        for _ in range(n_select):
            # Random tournament
            tournament = np.random.choice(states, size=min(k, len(states)), replace=False)
            winner = max(tournament, key=lambda s: s.quality_score)
            selected.append(winner)
        
        return selected
    
    def _rank_selection(self, states: List[QuantumState]) -> List[QuantumState]:
        """Rank-based selection."""
        sorted_states = sorted(states, key=lambda s: s.quality_score, reverse=True)
        # Select top portion
        n_select = max(2, len(sorted_states) // 2)
        return sorted_states[:n_select]
    
    def _probability_selection(self, states: List[QuantumState]) -> List[QuantumState]:
        """Probability-weighted selection."""
        probs = np.array([s.probability for s in states])
        probs = probs / (probs.sum() + 1e-10)
        
        n_select = min(len(states), 10)
        indices = np.random.choice(
            len(states),
            size=n_select,
            replace=False,
            p=probs
        )
        
        return [states[i] for i in indices]
    
    def evolve_generation(
        self,
        superposition_gen: StateSuperpositionGenerator,
        removed_ids: List[int],
        entropy_metrics: Optional[EntropyMetrics] = None,
        target_population_size: Optional[int] = None
    ) -> EvolutionResult:
        """
        Execute full evolution cycle after collapse.
        
        Args:
            superposition_gen: State superposition generator
            removed_ids: IDs of states removed in collapse
            entropy_metrics: Current entropy metrics
            target_population_size: Desired population size
            
        Returns:
            EvolutionResult with statistics
        """
        n_removed = len(removed_ids)
        
        # Strategy: Replace removed states + add some for exploration
        n_replacement = n_removed
        n_exploration = max(1, n_removed // 2)
        
        # Get current states
        current_states = superposition_gen.get_all_states()
        
        # Track statistics
        initial_mean_quality = np.mean([s.quality_score for s in current_states]) if current_states else 0.0
        
        # Generate replacement states
        new_states = []
        
        # Mutate existing states for some replacements
        if self.solution_mutator is not None and n_replacement > 0:
            mutated = self.mutate_states(
                current_states,
                entropy_metrics,
                mutation_rate_override=min(0.5, n_replacement / len(current_states)) if current_states else 0.1
            )
            new_states.extend(mutated[:n_replacement // 2])
        
        # Generate completely new states for exploration
        if self.solution_generator is not None and n_exploration > 0:
            generated = self.generate_new_states(
                superposition_gen,
                n_exploration,
                entropy_metrics
            )
            new_states.extend(generated)
        
        # Perform recombination
        offspring = []
        if self.solution_recombiner is not None and len(current_states) >= 2:
            offspring = self.recombine_states(current_states, entropy_metrics)
            
            # Add best offspring to superposition
            for child in offspring[:2]:  # Limit additions
                if self.quality_evaluator is not None:
                    quality = self.quality_evaluator(child.solution)
                    superposition_gen.add_state(child.solution, quality)
        
        # Compute final statistics
        final_states = superposition_gen.get_all_states()
        final_mean_quality = np.mean([s.quality_score for s in final_states]) if final_states else 0.0
        
        if initial_mean_quality != 0:
            mean_quality_change = (final_mean_quality - initial_mean_quality) / abs(initial_mean_quality)
        else:
            mean_quality_change = 0.0
        
        best_new_quality = max([s.quality_score for s in final_states]) if final_states else 0.0
        
        result = EvolutionResult(
            new_states_created=len(new_states),
            states_mutated=len([s for s in new_states if s.age == 0]),
            states_recombined=len(offspring),
            total_states_after=len(final_states),
            mean_quality_change=float(mean_quality_change),
            best_new_quality=float(best_new_quality),
            diversity_change=0.0,  # Would need before/after diversity computation
            metadata={
                "generation": self.generation,
                "n_removed": n_removed,
                "n_replacement": n_replacement,
                "n_exploration": n_exploration,
            }
        )
        
        self.evolution_history.append(result)
        self.generation += 1
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistical summary of evolution history."""
        if not self.evolution_history:
            return {
                "total_generations": self.generation,
                "mean_quality_change": 0.0,
                "current_mutation_rate": self.current_mutation_rate,
            }
        
        quality_changes = [r.mean_quality_change for r in self.evolution_history]
        
        return {
            "total_generations": self.generation,
            "mean_quality_change": float(np.mean(quality_changes)),
            "best_quality_change": float(np.max(quality_changes)),
            "total_new_states": sum(r.new_states_created for r in self.evolution_history),
            "total_mutations": sum(r.states_mutated for r in self.evolution_history),
            "total_recombinations": sum(r.states_recombined for r in self.evolution_history),
            "current_mutation_rate": self.current_mutation_rate,
        }
    
    def reset(self):
        """Reset the evolution engine."""
        self.generation = 0
        self.evolution_history.clear()
        self.current_mutation_rate = self.mutation_config.base_mutation_rate
