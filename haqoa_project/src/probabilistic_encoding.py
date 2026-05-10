"""
Probabilistic State Encoder

Implements Section 6.2: State Probability
Encodes states with probability distributions based on quality scores.

Mathematical representation:
P_i = e^(β * S_i) / Σ_j e^(β * S_j)

Where:
- S_i = state quality score
- β = adaptive amplification factor
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Callable
from dataclasses import dataclass

# Import without relative path for standalone usage
try:
    from .state_superposition import QuantumState, StateSuperpositionGenerator
except ImportError:
    from state_superposition import QuantumState, StateSuperpositionGenerator


@dataclass
class EncodedState:
    """Represents a probabilistically encoded state."""
    
    state_id: int
    solution: Any
    quality_score: float
    raw_probability: float  # Before normalization
    normalized_probability: float  # After softmax normalization
    entropy_contribution: float
    rank: int  # Ranking by quality
    encoding_metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.encoding_metadata is None:
            self.encoding_metadata = {}


class ProbabilisticStateEncoder:
    """
    Encodes candidate states with probabilistic weights.
    
    Implements dynamic probabilistic reinforcement based on:
    - State quality scores
    - Adaptive amplification factor (β)
    - Boltzmann-style probability distribution
    """
    
    def __init__(
        self,
        base_amplification: float = 1.0,
        temperature_schedule: str = "adaptive",
        min_temperature: float = 0.01,
        max_temperature: float = 10.0,
        scaling_factor: float = 1.0
    ):
        """
        Initialize the Probabilistic State Encoder.
        
        Args:
            base_amplification: Base amplification factor (β_0)
            temperature_schedule: Temperature scheduling strategy
                                 ('constant', 'linear', 'exponential', 'adaptive')
            min_temperature: Minimum temperature value
            max_temperature: Maximum temperature value
            scaling_factor: Additional scaling for quality scores
        """
        self.base_amplification = base_amplification
        self.current_amplification = base_amplification
        self.temperature_schedule = temperature_schedule
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        self.scaling_factor = scaling_factor
        
        self.encoded_states: Dict[int, EncodedState] = {}
        self.generation = 0
        self.temperature_history: List[float] = []
    
    def compute_inverse_temperature(self, beta: float) -> float:
        """
        Compute effective temperature from amplification factor.
        
        Temperature T = 1/β represents exploration level.
        High temperature = more exploration
        Low temperature = more exploitation
        """
        if beta <= 0:
            return self.max_temperature
        
        temperature = 1.0 / beta
        return np.clip(temperature, self.min_temperature, self.max_temperature)
    
    def encode_states(
        self,
        states: List[QuantumState],
        amplification_factor: Optional[float] = None
    ) -> List[EncodedState]:
        """
        Encode a list of quantum states with probabilities.
        
        Args:
            states: List of QuantumState objects to encode
            amplification_factor: Optional override for current amplification
            
        Returns:
            List of EncodedState objects
        """
        if amplification_factor is not None:
            beta = amplification_factor
        else:
            beta = self.current_amplification
        
        if not states:
            return []
        
        # Extract quality scores
        quality_scores = np.array([s.quality_score * self.scaling_factor for s in states])
        
        # Compute raw probabilities using Boltzmann distribution
        # P_i ∝ e^(β * S_i)
        exp_values = np.exp(beta * (quality_scores - np.max(quality_scores)))  # Numerical stability
        raw_probabilities = exp_values
        
        # Normalize probabilities
        prob_sum = np.sum(raw_probabilities)
        if prob_sum > 0:
            normalized_probabilities = raw_probabilities / prob_sum
        else:
            normalized_probabilities = np.ones(len(states)) / len(states)
        
        # Compute entropy contributions for each state
        # H_i = -P_i * log(P_i)
        entropy_contributions = []
        for p in normalized_probabilities:
            if p > 0:
                entropy_contributions.append(-p * np.log(p))
            else:
                entropy_contributions.append(0.0)
        
        # Rank states by quality
        sorted_indices = np.argsort(-quality_scores)  # Descending order
        ranks = np.zeros(len(states), dtype=int)
        for rank, idx in enumerate(sorted_indices):
            ranks[idx] = rank + 1
        
        # Create encoded states
        encoded_list = []
        for i, state in enumerate(states):
            encoded_state = EncodedState(
                state_id=state.state_id,
                solution=state.solution,
                quality_score=state.quality_score,
                raw_probability=float(raw_probabilities[i]),
                normalized_probability=float(normalized_probabilities[i]),
                entropy_contribution=float(entropy_contributions[i]),
                rank=int(ranks[i]),
                encoding_metadata={
                    "generation": self.generation,
                    "amplification_factor": beta,
                    "temperature": self.compute_inverse_temperature(beta),
                }
            )
            
            encoded_list.append(encoded_state)
            self.encoded_states[state.state_id] = encoded_state
        
        # Update quantum state probabilities
        for i, state in enumerate(states):
            state.probability = float(normalized_probabilities[i])
            state.entropy_contribution = float(entropy_contributions[i])
        
        return encoded_list
    
    def update_amplification(
        self,
        new_amplification: float,
        schedule_step: Optional[float] = None
    ):
        """
        Update the amplification factor.
        
        Args:
            new_amplification: New amplification value
            schedule_step: Optional step for scheduled updates
        """
        self.current_amplification = np.clip(
            new_amplification,
            self.min_temperature,
            self.max_temperature * 10  # Allow higher beta values
        )
    
    def apply_temperature_schedule(
        self,
        generation: int,
        total_generations: int,
        current_entropy: Optional[float] = None
    ):
        """
        Apply temperature scheduling based on generation or entropy.
        
        Args:
            generation: Current generation number
            total_generations: Total expected generations
            current_entropy: Current system entropy (for adaptive scheduling)
        """
        self.generation = generation
        
        if self.temperature_schedule == "constant":
            beta = self.base_amplification
        
        elif self.temperature_schedule == "linear":
            # Linear cooling schedule
            progress = generation / max(total_generations, 1)
            beta_max = self.base_amplification * 2
            beta_min = self.base_amplification * 0.5
            beta = beta_max - progress * (beta_max - beta_min)
        
        elif self.temperature_schedule == "exponential":
            # Exponential cooling
            decay_rate = 0.95
            beta = self.base_amplification * (decay_rate ** generation)
            beta = max(beta, self.base_amplification * 0.1)
        
        elif self.temperature_schedule == "adaptive":
            # Entropy-based adaptive scheduling
            if current_entropy is not None:
                # Higher entropy → lower amplification (more exploration)
                # Lower entropy → higher amplification (more exploitation)
                entropy_factor = np.exp(-current_entropy)
                beta = self.base_amplification * (1 + entropy_factor)
            else:
                # Default to gradual increase
                progress = generation / max(total_generations, 1)
                beta = self.base_amplification * (1 + progress)
        
        else:
            beta = self.base_amplification
        
        self.update_amplification(beta)
        self.temperature_history.append(beta)
        
        return beta
    
    def get_encoded_state(self, state_id: int) -> Optional[EncodedState]:
        """Get an encoded state by ID."""
        return self.encoded_states.get(state_id)
    
    def get_all_encoded_states(self) -> List[EncodedState]:
        """Get all encoded states sorted by probability."""
        return sorted(
            self.encoded_states.values(),
            key=lambda s: s.normalized_probability,
            reverse=True
        )
    
    def select_top_states(
        self,
        n_states: int,
        threshold_probability: Optional[float] = None
    ) -> List[EncodedState]:
        """
        Select top states by probability.
        
        Args:
            n_states: Number of states to select
            threshold_probability: Minimum probability threshold
            
        Returns:
            List of selected encoded states
        """
        sorted_states = self.get_all_encoded_states()
        
        selected = []
        for state in sorted_states:
            if len(selected) >= n_states:
                break
            if threshold_probability is None or state.normalized_probability >= threshold_probability:
                selected.append(state)
        
        return selected
    
    def compute_partition_function(
        self,
        states: List[QuantumState],
        beta: Optional[float] = None
    ) -> float:
        """
        Compute the partition function Z = Σ_j e^(β * S_j).
        
        This is the normalization constant for the probability distribution.
        
        Args:
            states: List of states
            beta: Amplification factor (uses current if not provided)
            
        Returns:
            Partition function value
        """
        if beta is None:
            beta = self.current_amplification
        
        if not states:
            return 0.0
        
        quality_scores = np.array([s.quality_score * self.scaling_factor for s in states])
        exp_values = np.exp(beta * (quality_scores - np.max(quality_scores)))
        
        # Adjust for the shift we applied for numerical stability
        z = np.sum(exp_values) * np.exp(beta * np.max(quality_scores))
        
        return float(z)
    
    def get_probability_distribution(self) -> np.ndarray:
        """Get array of all normalized probabilities."""
        return np.array([
            s.normalized_probability for s in self.encoded_states.values()
        ])
    
    def reset(self):
        """Reset the encoder."""
        self.encoded_states.clear()
        self.generation = 0
        self.temperature_history.clear()
        self.current_amplification = self.base_amplification
