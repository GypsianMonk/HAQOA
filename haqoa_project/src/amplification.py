"""
Adaptive Probability Amplifier

Implements Section 7.2: Adaptive Amplification
Dynamically adjusts probability amplification based on entropy and search progress.

Mathematical representation:
β_t = β_0 * (1 + κ * H_t)

Where:
- β_0 = base amplification
- κ = entropy sensitivity coefficient
- H_t = current entropy
"""

import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

# Import without relative path for standalone usage
try:
    from .entropy_monitor import EntropyMetrics, EntropyMonitoringSystem
    from .state_superposition import QuantumState
except ImportError:
    from entropy_monitor import EntropyMetrics, EntropyMonitoringSystem
    from state_superposition import QuantumState


@dataclass
class AmplificationConfig:
    """Configuration for adaptive amplification."""
    
    base_amplification: float = 1.0
    entropy_sensitivity: float = 0.5  # κ coefficient
    min_amplification: float = 0.1
    max_amplification: float = 10.0
    damping_factor: float = 0.9  # For smoothing updates
    adaptation_rate: float = 0.1  # Learning rate for adjustments
    
    # Mode settings
    use_entropy_feedback: bool = True
    use_convergence_feedback: bool = True
    use_generation_schedule: bool = False
    
    # Schedule parameters (if using generation schedule)
    schedule_type: str = "linear"  # 'linear', 'exponential', 'sigmoid'


class AdaptiveProbabilityAmplifier:
    """
    Implements adaptive probability amplification control.
    
    Features:
    - Entropy-regulated amplification
    - Convergence-aware adjustment
    - Damping for stability
    - Multiple adaptation modes
    """
    
    def __init__(
        self,
        config: Optional[AmplificationConfig] = None,
        entropy_monitor: Optional[EntropyMonitoringSystem] = None
    ):
        """
        Initialize the Adaptive Probability Amplifier.
        
        Args:
            config: Amplification configuration
            entropy_monitor: Optional entropy monitor for feedback
        """
        self.config = config or AmplificationConfig()
        self.entropy_monitor = entropy_monitor
        
        self.current_amplification = self.config.base_amplification
        self.amplification_history: List[float] = []
        self.generation = 0
        
        # Internal state
        self._smoothed_entropy = None
        self._momentum = 0.0
    
    def compute_amplification(
        self,
        entropy_metrics: Optional[EntropyMetrics] = None,
        override_entropy: Optional[float] = None
    ) -> float:
        """
        Compute the new amplification factor.
        
        Args:
            entropy_metrics: Current entropy metrics
            override_entropy: Optional override entropy value
            
        Returns:
            New amplification factor
        """
        beta_base = self.config.base_amplification
        
        if not self.config.use_entropy_feedback:
            # Simple scheduled amplification
            beta = self._compute_scheduled_amplification()
        else:
            # Entropy-based amplification
            if override_entropy is not None:
                entropy = override_entropy
            elif entropy_metrics is not None:
                entropy = entropy_metrics.normalized_entropy
            else:
                entropy = 0.5  # Default middle value
            
            # Apply smoothing to entropy
            if self._smoothed_entropy is None:
                self._smoothed_entropy = entropy
            else:
                self._smoothed_entropy = (
                    self.config.damping_factor * self._smoothed_entropy +
                    (1 - self.config.damping_factor) * entropy
                )
            
            # Core amplification formula: β_t = β_0 * (1 + κ * H_t)
            kappa = self.config.entropy_sensitivity
            beta = beta_base * (1 + kappa * self._smoothed_entropy)
            
            # Add convergence feedback if enabled
            if self.config.use_convergence_feedback and entropy_metrics is not None:
                beta = self._apply_convergence_adjustment(beta, entropy_metrics)
        
        # Apply bounds
        beta = np.clip(
            beta,
            self.config.min_amplification,
            self.config.max_amplification
        )
        
        # Apply momentum for smooth transitions
        delta = beta - self.current_amplification
        beta = self.current_amplification + self.config.adaptation_rate * delta
        
        # Update state
        self.current_amplification = float(beta)
        self.amplification_history.append(self.current_amplification)
        self.generation += 1
        
        return self.current_amplification
    
    def _compute_scheduled_amplification(self) -> float:
        """Compute amplification based on generation schedule."""
        if not self.config.use_generation_schedule:
            return self.config.base_amplification
        
        schedule_type = self.config.schedule_type
        progress = min(self.generation / 1000, 1.0)  # Assume 1000 gen max
        
        if schedule_type == "linear":
            # Linear increase over time
            beta = self.config.base_amplification * (1 + progress)
        
        elif schedule_type == "exponential":
            # Exponential increase
            beta = self.config.base_amplification * np.exp(2 * progress)
        
        elif schedule_type == "sigmoid":
            # Sigmoid curve: slow start, fast middle, slow end
            sigmoid = 1 / (1 + np.exp(-10 * (progress - 0.5)))
            beta = self.config.base_amplification * (1 + 2 * sigmoid)
        
        else:
            beta = self.config.base_amplification
        
        return float(beta)
    
    def _apply_convergence_adjustment(
        self,
        beta: float,
        metrics: EntropyMetrics
    ) -> float:
        """
        Adjust amplification based on convergence risk.
        
        High convergence risk → decrease amplification (more exploration)
        Low convergence risk → increase amplification (more exploitation)
        """
        risk = metrics.convergence_risk
        
        if risk > 0.5:
            # High risk: reduce amplification to encourage exploration
            adjustment = 1 - 0.3 * (risk - 0.5) * 2
        else:
            # Low risk: can increase amplification slightly
            adjustment = 1 + 0.1 * (0.5 - risk) * 2
        
        return beta * adjustment
    
    def amplify_probabilities(
        self,
        states: List[QuantumState],
        quality_scores: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Apply amplification to state probabilities.
        
        Args:
            states: List of quantum states
            quality_scores: Optional pre-computed quality scores
            
        Returns:
            Array of amplified probabilities
        """
        if not states:
            return np.array([])
        
        if quality_scores is None:
            quality_scores = np.array([s.quality_score for s in states])
        
        beta = self.current_amplification
        
        # Compute amplified probabilities using Boltzmann distribution
        # P_i ∝ e^(β * S_i)
        scaled_scores = beta * quality_scores
        
        # Numerical stability: subtract max
        scaled_scores = scaled_scores - np.max(scaled_scores)
        
        exp_values = np.exp(scaled_scores)
        probabilities = exp_values / np.sum(exp_values)
        
        return probabilities
    
    def get_amplification_gradient(
        self,
        states: List[QuantumState],
        target_quality: float
    ) -> float:
        """
        Compute gradient of amplification with respect to solution quality.
        
        Useful for understanding sensitivity and tuning.
        
        Args:
            states: List of states
            target_quality: Target quality level
            
        Returns:
            Gradient estimate
        """
        if not states:
            return 0.0
        
        beta = self.current_amplification
        
        # Compute probabilities at current beta
        quality_scores = np.array([s.quality_score for s in states])
        probs_current = self.amplify_probabilities(states, quality_scores)
        
        # Compute probabilities at slightly perturbed beta
        delta_beta = 0.01
        self.current_amplification = beta + delta_beta
        probs_perturbed = self.amplify_probabilities(states, quality_scores)
        
        # Restore beta
        self.current_amplification = beta
        
        # Compute gradient approximation
        gradient = np.mean((probs_perturbed - probs_current) / delta_beta)
        
        return float(gradient)
    
    def reset_momentum(self):
        """Reset internal momentum state."""
        self._smoothed_entropy = None
        self._momentum = 0.0
    
    def get_statistics(self) -> Dict[str, float]:
        """Get statistical summary of amplification history."""
        if not self.amplification_history:
            return {
                "current_amplification": self.current_amplification,
                "mean_amplification": 0.0,
                "std_amplification": 0.0,
                "min_amplification": 0.0,
                "max_amplification": 0.0,
            }
        
        history = np.array(self.amplification_history)
        
        return {
            "current_amplification": self.current_amplification,
            "mean_amplification": float(np.mean(history)),
            "std_amplification": float(np.std(history)),
            "min_amplification": float(np.min(history)),
            "max_amplification": float(np.max(history)),
            "trend": float(history[-1] - history[0]) if len(history) > 1 else 0.0,
        }
    
    def set_parameters(
        self,
        base_amplification: Optional[float] = None,
        entropy_sensitivity: Optional[float] = None,
        damping_factor: Optional[float] = None
    ):
        """
        Update amplifier parameters.
        
        Args:
            base_amplification: New base amplification
            entropy_sensitivity: New entropy sensitivity coefficient
            damping_factor: New damping factor
        """
        if base_amplification is not None:
            self.config.base_amplification = base_amplification
        
        if entropy_sensitivity is not None:
            self.config.entropy_sensitivity = entropy_sensitivity
        
        if damping_factor is not None:
            self.config.damping_factor = damping_factor
    
    def reset(self):
        """Reset the amplifier to initial state."""
        self.current_amplification = self.config.base_amplification
        self.amplification_history.clear()
        self.generation = 0
        self.reset_momentum()
