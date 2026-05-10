"""
State Superposition Generator

Implements Section 6.1: State Superposition
Maintains a probabilistic population of candidate states instead of single solutions.

Mathematical representation:
|Ψ_t⟩ = Σ α_i^(t) |s_i⟩

Where:
- s_i = candidate solution state
- α_i = probability amplitude
- N = number of active states
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field


@dataclass
class QuantumState:
    """Represents a single candidate state in superposition."""
    
    state_id: int
    solution: Any  # Problem-specific solution representation
    amplitude: complex  # Probability amplitude (α_i)
    quality_score: float  # State quality score (S_i)
    probability: float = 0.0  # Computed probability P_i (computed from amplitude)
    entropy_contribution: float = 0.0
    age: int = 0  # Number of generations since creation
    collapse_history: List[int] = field(default_factory=list)
    
    def __post_init__(self):
        """Ensure amplitude and probability are consistent."""
        if isinstance(self.amplitude, (int, float)):
            self.amplitude = complex(self.amplitude, 0)
        # Compute probability from amplitude if not provided or zero
        if self.probability == 0.0:
            self.probability = abs(self.amplitude) ** 2
    
    def update_amplitude(self, new_amplitude: complex):
        """Update amplitude and recalculate probability."""
        self.amplitude = new_amplitude
        self.probability = abs(self.amplitude) ** 2
    
    def normalize_amplitude(self, normalization_factor: float):
        """Normalize amplitude by a factor."""
        self.amplitude /= normalization_factor
        self.probability = abs(self.amplitude) ** 2


class StateSuperpositionGenerator:
    """
    Generates and manages quantum-inspired state superposition.
    
    Implements probabilistic population maintenance with:
    - Dynamic state creation
    - Amplitude management
    - Probability distribution tracking
    """
    
    def __init__(
        self,
        initial_population_size: int = 50,
        max_states: int = 500,
        amplitude_distribution: str = "uniform",
        random_seed: Optional[int] = None
    ):
        """
        Initialize the State Superposition Generator.
        
        Args:
            initial_population_size: Number of initial states to generate
            max_states: Maximum number of states to maintain
            amplitude_distribution: Distribution for initial amplitudes
                                   ('uniform', 'gaussian', 'exponential')
            random_seed: Random seed for reproducibility
        """
        self.initial_population_size = initial_population_size
        self.max_states = max_states
        self.amplitude_distribution = amplitude_distribution
        self.random_seed = random_seed
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        self.states: Dict[int, QuantumState] = {}
        self.state_counter = 0
        self.generation = 0
        self.total_probability = 1.0
    
    def generate_initial_states(
        self,
        solution_generator: callable,
        initial_quality_scores: Optional[List[float]] = None
    ) -> List[QuantumState]:
        """
        Generate initial superposition of states.
        
        Args:
            solution_generator: Function that generates candidate solutions
            initial_quality_scores: Optional pre-computed quality scores
            
        Returns:
            List of generated QuantumState objects
        """
        states = []
        
        for i in range(self.initial_population_size):
            solution = solution_generator()
            
            # Generate initial amplitude based on distribution
            amplitude = self._generate_initial_amplitude()
            
            # Use provided quality score or default
            if initial_quality_scores is not None and i < len(initial_quality_scores):
                quality_score = initial_quality_scores[i]
            else:
                quality_score = 0.0
            
            state = QuantumState(
                state_id=self.state_counter,
                solution=solution,
                amplitude=amplitude,
                quality_score=quality_score
            )
            
            states.append(state)
            self.states[self.state_counter] = state
            self.state_counter += 1
        
        # Normalize all amplitudes
        self._normalize_amplitudes()
        
        return states
    
    def _generate_initial_amplitude(self) -> complex:
        """Generate initial amplitude based on configured distribution."""
        if self.amplitude_distribution == "uniform":
            magnitude = np.random.uniform(0.1, 1.0)
        elif self.amplitude_distribution == "gaussian":
            magnitude = np.abs(np.random.normal(0.5, 0.2))
        elif self.amplitude_distribution == "exponential":
            magnitude = np.random.exponential(0.5)
        else:
            magnitude = np.random.uniform(0.1, 1.0)
        
        # Random phase
        phase = np.random.uniform(0, 2 * np.pi)
        
        return magnitude * np.exp(1j * phase)
    
    def _normalize_amplitudes(self):
        """Normalize all state amplitudes to ensure total probability = 1."""
        total_magnitude_squared = sum(
            abs(state.amplitude) ** 2 for state in self.states.values()
        )
        
        if total_magnitude_squared > 0:
            normalization_factor = np.sqrt(total_magnitude_squared)
            for state in self.states.values():
                state.normalize_amplitude(normalization_factor)
        
        self.total_probability = sum(
            state.probability for state in self.states.values()
        )
    
    def add_state(
        self,
        solution: Any,
        quality_score: float = 0.0,
        amplitude: Optional[complex] = None
    ) -> QuantumState:
        """
        Add a new state to the superposition.
        
        Args:
            solution: Candidate solution
            quality_score: Quality score of the solution
            amplitude: Optional amplitude (generated if not provided)
            
        Returns:
            The created QuantumState
        """
        if len(self.states) >= self.max_states:
            raise ValueError(
                f"Maximum state limit ({self.max_states}) reached. "
                "Perform collapse operation first."
            )
        
        if amplitude is None:
            amplitude = self._generate_initial_amplitude()
        
        state = QuantumState(
            state_id=self.state_counter,
            solution=solution,
            amplitude=amplitude,
            quality_score=quality_score
        )
        
        self.states[self.state_counter] = state
        self.state_counter += 1
        
        self._normalize_amplitudes()
        
        return state
    
    def remove_state(self, state_id: int) -> bool:
        """
        Remove a state from the superposition.
        
        Args:
            state_id: ID of state to remove
            
        Returns:
            True if removed, False if state not found
        """
        if state_id in self.states:
            del self.states[state_id]
            self._normalize_amplitudes()
            return True
        return False
    
    def get_state(self, state_id: int) -> Optional[QuantumState]:
        """Get a specific state by ID."""
        return self.states.get(state_id)
    
    def get_all_states(self) -> List[QuantumState]:
        """Get all states sorted by probability (descending)."""
        return sorted(
            self.states.values(),
            key=lambda s: s.probability,
            reverse=True
        )
    
    def get_probabilities(self) -> np.ndarray:
        """Get array of all state probabilities."""
        return np.array([state.probability for state in self.states.values()])
    
    def get_amplitudes(self) -> np.ndarray:
        """Get array of all state amplitudes."""
        return np.array([state.amplitude for state in self.states.values()])
    
    def sample_state(self, n_samples: int = 1) -> List[QuantumState]:
        """
        Sample states based on their probability distribution.
        
        Args:
            n_samples: Number of states to sample
            
        Returns:
            List of sampled states
        """
        states_list = list(self.states.values())
        probabilities = self.get_probabilities()
        
        # Normalize probabilities for sampling
        prob_sum = np.sum(probabilities)
        if prob_sum > 0:
            probabilities = probabilities / prob_sum
        
        indices = np.random.choice(
            len(states_list),
            size=min(n_samples, len(states_list)),
            replace=False,
            p=probabilities
        )
        
        return [states_list[i] for i in indices]
    
    def update_generation(self):
        """Increment generation counter and update state ages."""
        self.generation += 1
        for state in self.states.values():
            state.age += 1
    
    def get_statistics(self) -> Dict[str, float]:
        """Get statistical summary of current superposition."""
        if not self.states:
            return {
                "num_states": 0,
                "total_probability": 0.0,
                "mean_amplitude": 0.0,
                "max_probability": 0.0,
                "min_probability": 0.0,
                "mean_quality": 0.0,
            }
        
        probabilities = self.get_probabilities()
        qualities = np.array([s.quality_score for s in self.states.values()])
        
        return {
            "num_states": len(self.states),
            "total_probability": float(np.sum(probabilities)),
            "mean_amplitude": float(np.mean(np.abs(self.get_amplitudes()))),
            "max_probability": float(np.max(probabilities)),
            "min_probability": float(np.min(probabilities)),
            "mean_quality": float(np.mean(qualities)),
            "std_quality": float(np.std(qualities)),
        }
    
    def reset(self):
        """Reset the superposition generator to initial state."""
        self.states.clear()
        self.state_counter = 0
        self.generation = 0
        self.total_probability = 1.0
