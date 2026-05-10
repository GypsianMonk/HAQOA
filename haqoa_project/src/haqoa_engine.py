"""
HAQOA Main Optimizer Engine

Implements the complete HAQOA optimization framework integrating all components:
- State Superposition Generator
- Probabilistic State Encoder
- Entropy Monitoring System
- Adaptive Probability Amplifier
- Dynamic Collapse Controller
- AI Guidance Engine
- Evolution Engine

This is the AQSE-v1 (Adaptive Quantum-Inspired State Evolution Engine).
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import time

# Import without relative path for standalone usage
try:
    from .state_superposition import StateSuperpositionGenerator, QuantumState
    from .probabilistic_encoding import ProbabilisticStateEncoder
    from .entropy_monitor import EntropyMonitoringSystem, EntropyMetrics
    from .amplification import AdaptiveProbabilityAmplifier, AmplificationConfig
    from .collapse_controller import DynamicCollapseController, CollapseStrategy
    from .ai_guidance import AIGuidanceEngine, GuidanceMode, RewardConfig
    from .evolution import EvolutionEngine, MutationConfig, RecombinationConfig
except ImportError:
    from state_superposition import StateSuperpositionGenerator, QuantumState
    from probabilistic_encoding import ProbabilisticStateEncoder
    from entropy_monitor import EntropyMonitoringSystem, EntropyMetrics
    from amplification import AdaptiveProbabilityAmplifier, AmplificationConfig
    from collapse_controller import DynamicCollapseController, CollapseStrategy
    from ai_guidance import AIGuidanceEngine, GuidanceMode, RewardConfig
    from evolution import EvolutionEngine, MutationConfig, RecombinationConfig


class OptimizationStatus(Enum):
    """Status of the optimization run."""
    RUNNING = "running"
    CONVERGED = "converged"
    MAX_ITERATIONS = "max_iterations"
    TIME_LIMIT = "time_limit"
    STOPPED = "stopped"


@dataclass
class HAQOAConfig:
    """Configuration for HAQOA optimizer."""
    
    # Population parameters
    initial_population_size: int = 50
    max_states: int = 200
    min_states_after_collapse: int = 20
    
    # Entropy parameters
    high_entropy_threshold: float = 0.7
    low_entropy_threshold: float = 0.2
    target_entropy: float = 0.5
    
    # Amplification parameters
    base_amplification: float = 1.0
    entropy_sensitivity: float = 0.5
    
    # Collapse parameters
    collapse_strategy: str = "adaptive"
    collapse_interval: int = 10
    
    # Evolution parameters
    mutation_rate: float = 0.1
    recombination_probability: float = 0.3
    
    # AI guidance parameters
    guidance_mode: str = "hybrid"
    enable_ai_guidance: bool = True
    
    # Termination parameters
    max_generations: int = 500
    max_time_seconds: float = 300.0
    convergence_threshold: float = 1e-6
    patience_generations: int = 50
    
    # Random seed
    random_seed: Optional[int] = None


@dataclass
class GenerationResult:
    """Results from a single generation."""
    
    generation: int
    best_quality: float
    mean_quality: float
    worst_quality: float
    entropy: float
    num_states: int
    amplification: float
    status: str
    elapsed_time: float = 0.0


@dataclass
class OptimizationResult:
    """Final optimization result."""
    
    best_solution: Any
    best_quality: float
    best_state: Optional[QuantumState]
    
    final_generation: int
    total_generations: int
    status: OptimizationStatus
    
    # History
    generation_results: List[GenerationResult]
    convergence_history: List[float]
    
    # Statistics
    total_runtime: float
    mean_quality: float
    quality_std: float
    
    # Final state statistics
    final_num_states: int
    final_entropy: float
    final_amplification: float
    
    metadata: Dict[str, Any] = field(default_factory=dict)


class HAQOAOptimizer:
    """
    Main HAQOA Optimization Engine (AQSE-v1).
    
    Implements the complete Hybrid AI-Assisted Quantum-Inspired
    Optimization Architecture for solving complex combinatorial problems.
    """
    
    def __init__(
        self,
        config: Optional[HAQOAConfig] = None,
        solution_generator: Optional[Callable] = None,
        solution_mutator: Optional[Callable[[Any], Any]] = None,
        solution_recombiner: Optional[Callable[[Any, Any], Tuple[Any, Any]]] = None,
        quality_evaluator: Optional[Callable[[Any], float]] = None
    ):
        """
        Initialize the HAQOA Optimizer.
        
        Args:
            config: Optimizer configuration
            solution_generator: Function to generate new solutions
            solution_mutator: Function to mutate solutions
            solution_recombiner: Function to recombine solutions
            quality_evaluator: Function to evaluate solution quality
        """
        self.config = config or HAQOAConfig()
        
        # Set random seed if provided
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
        
        # Initialize all components
        self._initialize_components()
        
        # Store problem-specific functions
        self.solution_generator = solution_generator
        self.solution_mutator = solution_mutator
        self.solution_recombiner = solution_recombiner
        self.quality_evaluator = quality_evaluator
        
        # State tracking
        self.current_generation = 0
        self.best_solution_ever: Any = None
        self.best_quality_ever: float = -np.inf
        self.status = OptimizationStatus.STOPPED
        
        # History
        self.generation_results: List[GenerationResult] = []
        self.convergence_history: List[float] = []
        
        # Timing
        self.start_time: float = 0.0
    
    def _initialize_components(self):
        """Initialize all HAQOA components."""
        config = self.config
        
        # 1. State Superposition Generator
        self.superposition_gen = StateSuperpositionGenerator(
            initial_population_size=config.initial_population_size,
            max_states=config.max_states,
            random_seed=config.random_seed
        )
        
        # 2. Probabilistic State Encoder
        self.encoder = ProbabilisticStateEncoder(
            base_amplification=config.base_amplification,
            temperature_schedule="adaptive"
        )
        
        # 3. Entropy Monitoring System
        self.entropy_monitor = EntropyMonitoringSystem(
            high_entropy_threshold=config.high_entropy_threshold,
            low_entropy_threshold=config.low_entropy_threshold
        )
        
        # 4. Adaptive Probability Amplifier
        amp_config = AmplificationConfig(
            base_amplification=config.base_amplification,
            entropy_sensitivity=config.entropy_sensitivity
        )
        self.amplifier = AdaptiveProbabilityAmplifier(
            config=amp_config,
            entropy_monitor=self.entropy_monitor
        )
        
        # 5. Dynamic Collapse Controller
        strategy_map = {
            "adaptive": CollapseStrategy.ADAPTIVE,
            "threshold": CollapseStrategy.THRESHOLD,
            "percentile": CollapseStrategy.PERCENTILE,
            "fixed_count": CollapseStrategy.FIXED_COUNT,
            "age_based": CollapseStrategy.AGE_BASED,
        }
        self.collapse_controller = DynamicCollapseController(
            strategy=strategy_map.get(config.collapse_strategy, CollapseStrategy.ADAPTIVE),
            min_states_to_maintain=config.min_states_after_collapse
        )
        
        # 6. AI Guidance Engine
        mode_map = {
            "reward_based": GuidanceMode.REWARD_BASED,
            "prediction_based": GuidanceMode.PREDICTION_BASED,
            "hybrid": GuidanceMode.HYBRID,
            "passive": GuidanceMode.PASSIVE,
        }
        self.ai_guidance = AIGuidanceEngine(
            mode=mode_map.get(config.guidance_mode, GuidanceMode.HYBRID),
            enable_adaptive_weights=config.enable_ai_guidance
        )
        
        # 7. Evolution Engine
        mutation_config = MutationConfig(
            base_mutation_rate=config.mutation_rate,
            entropy_adaptive=True
        )
        recombination_config = RecombinationConfig(
            recombination_probability=config.recombination_probability
        )
        self.evolution_engine = EvolutionEngine(
            mutation_config=mutation_config,
            recombination_config=recombination_config
        )
    
    def set_problem_functions(
        self,
        solution_generator: Callable,
        solution_mutator: Callable[[Any], Any],
        solution_recombiner: Callable[[Any, Any], Tuple[Any, Any]],
        quality_evaluator: Callable[[Any], float]
    ):
        """
        Set problem-specific functions.
        
        Args:
            solution_generator: Generates new candidate solutions
            solution_mutator: Mutates an existing solution
            solution_recombiner: Recombines two solutions
            quality_evaluator: Evaluates solution quality (higher is better)
        """
        self.solution_generator = solution_generator
        self.solution_mutator = solution_mutator
        self.solution_recombiner = solution_recombiner
        self.quality_evaluator = quality_evaluator
        
        # Also set in evolution engine
        self.evolution_engine.set_problem_specific_functions(
            solution_generator,
            solution_mutator,
            solution_recombiner,
            quality_evaluator
        )
    
    def initialize_population(self, initial_solutions: Optional[List[Any]] = None):
        """
        Initialize the population with solutions.
        
        Args:
            initial_solutions: Optional list of initial solutions
        """
        if self.solution_generator is None and initial_solutions is None:
            raise ValueError(
                "Either provide initial_solutions or set solution_generator"
            )
        
        def gen_solution():
            if initial_solutions is not None and len(initial_solutions) > 0:
                return initial_solutions.pop(0)
            else:
                return self.solution_generator()
        
        # Generate initial states
        self.superposition_gen.generate_initial_states(
            solution_generator=gen_solution
        )
        
        # Evaluate initial qualities
        for state in self.superposition_gen.get_all_states():
            if state.quality_score == 0.0 and self.quality_evaluator is not None:
                state.quality_score = self.quality_evaluator(state.solution)
        
        # Initial encoding
        states = self.superposition_gen.get_all_states()
        self.encoder.encode_states(states)
        
        # Update best
        self._update_best_solution()
    
    def _update_best_solution(self):
        """Update the best solution found so far."""
        states = self.superposition_gen.get_all_states()
        
        for state in states:
            if state.quality_score > self.best_quality_ever:
                self.best_quality_ever = state.quality_score
                self.best_solution_ever = state.solution
                self.best_state_ever = state
    
    def _run_generation(self) -> GenerationResult:
        """Execute a single generation of HAQOA."""
        gen_start = time.time()
        
        # Get current states
        states = self.superposition_gen.get_all_states()
        
        # 1. Encode states with probabilities
        encoded_states = self.encoder.encode_states(
            states,
            self.amplifier.current_amplification
        )
        
        # 2. Compute entropy metrics
        entropy_metrics = self.entropy_monitor.compute_entropy_from_states(states)
        
        # 3. Update amplification based on entropy
        new_beta = self.amplifier.compute_amplification(entropy_metrics)
        self.encoder.update_amplification(new_beta)
        
        # 4. AI Guidance - compute rewards
        if self.config.enable_ai_guidance and self.quality_evaluator is not None:
            rewards = self.ai_guidance.compute_rewards(states)
            guidance = self.ai_guidance.get_guidance_signal(states, rewards, entropy_metrics)
            
            # Record improvement pattern
            improved = len(self.convergence_history) < 2 or \
                       self.best_quality_ever > self.convergence_history[-2] if len(self.convergence_history) >= 2 else False
            self.ai_guidance.record_improvement(improved)
        
        # 5. Check if collapse is needed
        generations_since_collapse = self.current_generation % self.config.collapse_interval
        should_collapse = self.collapse_controller.should_trigger_collapse(
            num_states=len(states),
            entropy_metrics=entropy_metrics,
            generation_since_last_collapse=generations_since_collapse,
            min_generation_interval=self.config.collapse_interval
        )
        
        removed_ids = []
        if should_collapse:
            # Execute collapse
            collapse_result = self.collapse_controller.execute_collapse(
                states,
                entropy_metrics
            )
            removed_ids = collapse_result.removed_state_ids
            
            # Remove states from superposition
            for state_id in removed_ids:
                self.superposition_gen.remove_state(state_id)
            
            # 6. Evolution - regenerate states
            if self.solution_generator is not None:
                evolution_result = self.evolution_engine.evolve_generation(
                    self.superposition_gen,
                    removed_ids,
                    entropy_metrics
                )
        
        # 7. Update generation counter
        self.superposition_gen.update_generation()
        
        # 8. Update best solution
        self._update_best_solution()
        
        # 9. Record convergence history
        self.convergence_history.append(self.best_quality_ever)
        
        # Compute generation statistics
        qualities = [s.quality_score for s in self.superposition_gen.get_all_states()]
        
        gen_result = GenerationResult(
            generation=self.current_generation,
            best_quality=float(np.max(qualities)) if qualities else 0.0,
            mean_quality=float(np.mean(qualities)) if qualities else 0.0,
            worst_quality=float(np.min(qualities)) if qualities else 0.0,
            entropy=float(entropy_metrics.total_entropy),
            num_states=len(states),
            amplification=float(new_beta),
            status="ok",
            elapsed_time=time.time() - gen_start
        )
        
        self.generation_results.append(gen_result)
        self.current_generation += 1
        
        return gen_result
    
    def _check_termination(self) -> Tuple[bool, OptimizationStatus]:
        """Check if optimization should terminate."""
        # Time limit
        elapsed = time.time() - self.start_time
        if elapsed > self.config.max_time_seconds:
            return True, OptimizationStatus.TIME_LIMIT
        
        # Max generations
        if self.current_generation >= self.config.max_generations:
            return True, OptimizationStatus.MAX_ITERATIONS
        
        # Convergence check
        if len(self.convergence_history) >= self.config.patience_generations:
            recent = self.convergence_history[-self.config.patience_generations:]
            if max(recent) - min(recent) < self.config.convergence_threshold:
                return True, OptimizationStatus.CONVERGED
        
        return False, OptimizationStatus.RUNNING
    
    def optimize(
        self,
        initial_solutions: Optional[List[Any]] = None,
        callback: Optional[Callable[[int, GenerationResult], None]] = None,
        verbose: bool = True
    ) -> OptimizationResult:
        """
        Run the HAQOA optimization.
        
        Args:
            initial_solutions: Optional initial solutions
            callback: Optional callback function(generation, result)
            verbose: Whether to print progress
            
        Returns:
            OptimizationResult with final results
        """
        if self.solution_generator is None and initial_solutions is None:
            raise ValueError(
                "Must provide either solution_generator or initial_solutions"
            )
        
        # Initialize
        self.status = OptimizationStatus.RUNNING
        self.start_time = time.time()
        self.current_generation = 0
        self.best_quality_ever = -np.inf
        self.best_solution_ever = None
        self.generation_results.clear()
        self.convergence_history.clear()
        
        # Reset components
        self._reset_components()
        
        # Initialize population
        self.initialize_population(initial_solutions)
        
        if verbose:
            print(f"HAQOA Optimization Started")
            print(f"Initial population: {len(self.superposition_gen.get_all_states())} states")
            print(f"Max generations: {self.config.max_generations}")
            print("-" * 60)
        
        # Main optimization loop
        while True:
            # Run generation
            gen_result = self._run_generation()
            
            # Callback
            if callback is not None:
                callback(self.current_generation, gen_result)
            
            # Verbose output
            if verbose and self.current_generation % 10 == 0:
                print(f"Gen {self.current_generation:4d} | "
                      f"Best: {gen_result.best_quality:.6f} | "
                      f"Mean: {gen_result.mean_quality:.6f} | "
                      f"Entropy: {gen_result.entropy:.4f} | "
                      f"States: {gen_result.num_states}")
            
            # Check termination
            should_stop, status = self._check_termination()
            if should_stop:
                self.status = status
                break
        
        # Create final result
        result = self._create_optimization_result()
        
        if verbose:
            print("-" * 60)
            print(f"Optimization Complete: {self.status.value}")
            print(f"Total generations: {self.current_generation}")
            print(f"Best quality: {result.best_quality:.6f}")
            print(f"Total runtime: {result.total_runtime:.2f}s")
        
        return result
    
    def _reset_components(self):
        """Reset all components for new optimization run."""
        self.superposition_gen.reset()
        self.encoder.reset()
        self.entropy_monitor.reset()
        self.amplifier.reset()
        self.collapse_controller.reset()
        self.ai_guidance.reset()
        self.evolution_engine.reset()
    
    def _create_optimization_result(self) -> OptimizationResult:
        """Create the final optimization result."""
        qualities = [r.best_quality for r in self.generation_results]
        
        final_states = self.superposition_gen.get_all_states()
        final_entropy_metrics = self.entropy_monitor.compute_entropy_from_states(final_states)
        
        return OptimizationResult(
            best_solution=self.best_solution_ever,
            best_quality=float(self.best_quality_ever),
            best_state=getattr(self, 'best_state_ever', None),
            final_generation=self.current_generation,
            total_generations=self.current_generation,
            status=self.status,
            generation_results=self.generation_results,
            convergence_history=self.convergence_history,
            total_runtime=time.time() - self.start_time,
            mean_quality=float(np.mean(qualities)) if qualities else 0.0,
            quality_std=float(np.std(qualities)) if qualities else 0.0,
            final_num_states=len(final_states),
            final_entropy=float(final_entropy_metrics.total_entropy),
            final_amplification=float(self.amplifier.current_amplification),
            metadata={
                "config": str(self.config),
                "initial_states": self.config.initial_population_size,
            }
        )
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current optimizer state."""
        states = self.superposition_gen.get_all_states()
        entropy_metrics = self.entropy_monitor.compute_entropy_from_states(states)
        
        return {
            "generation": self.current_generation,
            "status": self.status.value,
            "best_quality": self.best_quality_ever,
            "num_states": len(states),
            "entropy": float(entropy_metrics.total_entropy),
            "amplification": float(self.amplifier.current_amplification),
            "mean_quality": float(np.mean([s.quality_score for s in states])) if states else 0.0,
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all components."""
        return {
            "optimizer": {
                "generation": self.current_generation,
                "best_quality": self.best_quality_ever,
                "status": self.status.value,
            },
            "superposition": self.superposition_gen.get_statistics(),
            "amplifier": self.amplifier.get_statistics(),
            "entropy": self.entropy_monitor.get_entropy_statistics(),
            "collapse": self.collapse_controller.get_statistics(),
            "ai_guidance": self.ai_guidance.get_statistics(),
            "evolution": self.evolution_engine.get_statistics(),
        }
