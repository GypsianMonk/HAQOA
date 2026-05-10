"""
Entropy Monitoring System

Implements Section 7: Entropy-Controlled Exploration
Monitors and regulates search entropy to balance exploration vs exploitation.

Mathematical representation:
H_t = -Σ P_i^(t) * log(P_i^(t))

Where:
- P_i = probability of state i
- H_t = system entropy at time t
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

# Import without relative path for standalone usage
try:
    from .state_superposition import QuantumState, StateSuperpositionGenerator
    from .probabilistic_encoding import EncodedState, ProbabilisticStateEncoder
except ImportError:
    from state_superposition import QuantumState, StateSuperpositionGenerator
    from probabilistic_encoding import EncodedState, ProbabilisticStateEncoder


@dataclass
class EntropyMetrics:
    """Container for entropy-related metrics."""
    
    total_entropy: float
    normalized_entropy: float  # Entropy / max_possible_entropy
    entropy_rate: float  # Rate of change
    diversity_index: float  # Alternative diversity measure
    effective_states: int  # Number of states contributing significantly
    entropy_variance: float
    generation: int
    timestamp: float = field(default_factory=lambda: 0.0)
    
    # Interpretation flags
    is_high_entropy: bool = False
    is_low_entropy: bool = False
    is_stable: bool = True
    convergence_risk: float = 0.0  # Risk of premature convergence (0-1)


class EntropyMonitoringSystem:
    """
    Monitors and controls search entropy dynamics.
    
    Implements:
    - Shannon entropy computation
    - Entropy rate monitoring
    - Diversity preservation metrics
    - Convergence risk assessment
    - Adaptive entropy regulation signals
    """
    
    def __init__(
        self,
        high_entropy_threshold: float = 0.7,
        low_entropy_threshold: float = 0.2,
        entropy_window_size: int = 10,
        stability_tolerance: float = 0.05,
        min_effective_states: int = 5
    ):
        """
        Initialize the Entropy Monitoring System.
        
        Args:
            high_entropy_threshold: Threshold for "high" entropy (normalized)
            low_entropy_threshold: Threshold for "low" entropy (normalized)
            entropy_window_size: Window size for rate computation
            stability_tolerance: Tolerance for considering entropy stable
            min_effective_states: Minimum number of effective states
        """
        self.high_entropy_threshold = high_entropy_threshold
        self.low_entropy_threshold = low_entropy_threshold
        self.entropy_window_size = entropy_window_size
        self.stability_tolerance = stability_tolerance
        self.min_effective_states = min_effective_states
        
        self.entropy_history: List[float] = []
        self.metrics_history: List[EntropyMetrics] = []
        self.generation = 0
        
        # Entropy regulation parameters
        self.entropy_target: Optional[float] = None
        self.regulation_active = False
    
    def compute_entropy(
        self,
        probabilities: np.ndarray,
        base: float = np.e
    ) -> float:
        """
        Compute Shannon entropy for a probability distribution.
        
        H = -Σ P_i * log(P_i)
        
        Args:
            probabilities: Array of probabilities (should sum to 1)
            base: Logarithm base (e for natural log, 2 for bits)
            
        Returns:
            Entropy value
        """
        # Filter out zero probabilities to avoid log(0)
        probs = probabilities[probabilities > 0]
        
        if len(probs) == 0:
            return 0.0
        
        entropy = -np.sum(probs * np.log(probs) / np.log(base))
        return float(entropy)
    
    def compute_normalized_entropy(
        self,
        probabilities: np.ndarray,
        base: float = np.e
    ) -> float:
        """
        Compute normalized entropy (0 to 1 scale).
        
        H_norm = H / H_max where H_max = log(N)
        
        Args:
            probabilities: Array of probabilities
            base: Logarithm base
            
        Returns:
            Normalized entropy in [0, 1]
        """
        n = len(probabilities)
        if n <= 1:
            return 0.0
        
        entropy = self.compute_entropy(probabilities, base)
        max_entropy = np.log(n) / np.log(base)
        
        if max_entropy > 0:
            return float(entropy / max_entropy)
        else:
            return 0.0
    
    def compute_entropy_from_states(
        self,
        states: List[QuantumState]
    ) -> EntropyMetrics:
        """
        Compute comprehensive entropy metrics from quantum states.
        
        Args:
            states: List of QuantumState objects
            
        Returns:
            EntropyMetrics object with all computed metrics
        """
        if not states:
            metrics = EntropyMetrics(
                total_entropy=0.0,
                normalized_entropy=0.0,
                entropy_rate=0.0,
                diversity_index=0.0,
                effective_states=0,
                entropy_variance=0.0,
                generation=self.generation
            )
            self._update_history(metrics)
            return metrics
        
        # Get probabilities
        probabilities = np.array([s.probability for s in states])
        
        # Compute basic entropy
        total_entropy = self.compute_entropy(probabilities)
        normalized_entropy = self.compute_normalized_entropy(probabilities)
        
        # Compute entropy rate of change
        self.entropy_history.append(total_entropy)
        if len(self.entropy_history) > self.entropy_window_size:
            self.entropy_history.pop(0)
        
        if len(self.entropy_history) >= 2:
            entropy_rate = (
                self.entropy_history[-1] - self.entropy_history[-2]
            )
        else:
            entropy_rate = 0.0
        
        # Compute variance in recent entropy values
        if len(self.entropy_history) >= 2:
            entropy_variance = float(np.var(self.entropy_history))
        else:
            entropy_variance = 0.0
        
        # Compute diversity index (Simpson's index alternative)
        # D = 1 - Σ P_i^2
        diversity_index = 1.0 - np.sum(probabilities ** 2)
        
        # Compute effective number of states
        # N_eff = e^H (exponential of entropy)
        effective_states = int(np.exp(total_entropy))
        
        # Determine entropy level flags
        is_high_entropy = normalized_entropy > self.high_entropy_threshold
        is_low_entropy = normalized_entropy < self.low_entropy_threshold
        
        # Assess stability
        is_stable = abs(entropy_rate) < self.stability_tolerance
        
        # Compute convergence risk
        convergence_risk = self._compute_convergence_risk(
            normalized_entropy,
            effective_states,
            entropy_rate
        )
        
        metrics = EntropyMetrics(
            total_entropy=total_entropy,
            normalized_entropy=normalized_entropy,
            entropy_rate=entropy_rate,
            diversity_index=diversity_index,
            effective_states=effective_states,
            entropy_variance=entropy_variance,
            generation=self.generation,
            is_high_entropy=is_high_entropy,
            is_low_entropy=is_low_entropy,
            is_stable=is_stable,
            convergence_risk=convergence_risk
        )
        
        self._update_history(metrics)
        
        return metrics
    
    def _compute_convergence_risk(
        self,
        normalized_entropy: float,
        effective_states: int,
        entropy_rate: float
    ) -> float:
        """
        Compute risk of premature convergence.
        
        Args:
            normalized_entropy: Current normalized entropy
            effective_states: Effective number of states
            entropy_rate: Rate of entropy change
            
        Returns:
            Convergence risk score (0-1)
        """
        risk_factors = []
        
        # Low entropy increases risk
        if normalized_entropy < self.low_entropy_threshold:
            risk_factors.append((self.low_entropy_threshold - normalized_entropy) * 2)
        
        # Few effective states increases risk
        if effective_states < self.min_effective_states:
            risk_factors.append(
                (self.min_effective_states - effective_states) / self.min_effective_states
            )
        
        # Rapidly decreasing entropy increases risk
        if entropy_rate < -0.1:
            risk_factors.append(min(abs(entropy_rate), 1.0))
        
        if not risk_factors:
            return 0.0
        
        return float(np.mean(risk_factors))
    
    def compute_entropy_contributions(
        self,
        states: List[QuantumState]
    ) -> Dict[int, float]:
        """
        Compute individual entropy contribution of each state.
        
        H_i = -P_i * log(P_i)
        
        Args:
            states: List of states
            
        Returns:
            Dictionary mapping state_id to entropy contribution
        """
        contributions = {}
        
        for state in states:
            if state.probability > 0:
                contribution = -state.probability * np.log(state.probability)
            else:
                contribution = 0.0
            
            contributions[state.state_id] = float(contribution)
            state.entropy_contribution = contribution
        
        return contributions
    
    def set_entropy_target(self, target: float):
        """
        Set a target entropy level for regulation.
        
        Args:
            target: Target normalized entropy (0-1)
        """
        self.entropy_target = np.clip(target, 0.0, 1.0)
        self.regulation_active = True
    
    def get_entropy_regulation_signal(self, current_metrics: EntropyMetrics) -> float:
        """
        Compute regulation signal to adjust entropy toward target.
        
        Args:
            current_metrics: Current entropy metrics
            
        Returns:
            Regulation signal (-1 to decrease, +1 to increase)
        """
        if not self.regulation_active or self.entropy_target is None:
            return 0.0
        
        error = self.entropy_target - current_metrics.normalized_entropy
        
        # Normalize signal to [-1, 1]
        signal = np.tanh(error * 2)
        
        return float(signal)
    
    def detect_entropy_oscillation(
        self,
        window_size: Optional[int] = None
    ) -> Tuple[bool, float]:
        """
        Detect if entropy is oscillating unstably.
        
        Args:
            window_size: Window for detection (uses default if not provided)
            
        Returns:
            Tuple of (is_oscillating, oscillation_magnitude)
        """
        if window_size is None:
            window_size = self.entropy_window_size
        
        if len(self.entropy_history) < window_size:
            return False, 0.0
        
        recent = self.entropy_history[-window_size:]
        
        # Count sign changes in rate
        sign_changes = 0
        for i in range(1, len(recent)):
            prev_diff = recent[i-1] - (recent[i-2] if i > 1 else recent[0])
            curr_diff = recent[i] - recent[i-1]
            
            if prev_diff * curr_diff < 0:  # Sign change
                sign_changes += 1
        
        # Oscillation if many sign changes
        oscillation_threshold = window_size * 0.6
        is_oscillating = sign_changes > oscillation_threshold
        
        # Magnitude based on variance
        magnitude = float(np.std(recent) / (np.mean(recent) + 1e-10))
        
        return is_oscillating, magnitude
    
    def get_diversity_preservation_score(
        self,
        states: List[QuantumState]
    ) -> float:
        """
        Compute a score indicating how well diversity is preserved.
        
        Combines multiple metrics into a single score.
        
        Args:
            states: List of states
            
        Returns:
            Diversity preservation score (0-1, higher is better)
        """
        if not states:
            return 0.0
        
        metrics = self.compute_entropy_from_states(states)
        
        # Weight different factors
        entropy_weight = 0.4
        diversity_weight = 0.3
        effective_states_weight = 0.3
        
        score = (
            entropy_weight * metrics.normalized_entropy +
            diversity_weight * metrics.diversity_index +
            effective_states_weight * min(
                metrics.effective_states / self.min_effective_states,
                1.0
            )
        )
        
        return float(np.clip(score, 0.0, 1.0))
    
    def _update_history(self, metrics: EntropyMetrics):
        """Update history with new metrics."""
        self.metrics_history.append(metrics)
        self.generation += 1
        
        # Limit history size
        max_history = 1000
        if len(self.metrics_history) > max_history:
            self.metrics_history = self.metrics_history[-max_history:]
    
    def get_entropy_statistics(self) -> Dict[str, float]:
        """Get statistical summary of entropy history."""
        if not self.entropy_history:
            return {
                "mean_entropy": 0.0,
                "std_entropy": 0.0,
                "min_entropy": 0.0,
                "max_entropy": 0.0,
                "trend": 0.0,
            }
        
        entropies = np.array(self.entropy_history)
        
        # Compute trend (linear regression slope)
        if len(entropies) >= 2:
            x = np.arange(len(entropies))
            trend = np.polyfit(x, entropies, 1)[0]
        else:
            trend = 0.0
        
        return {
            "mean_entropy": float(np.mean(entropies)),
            "std_entropy": float(np.std(entropies)),
            "min_entropy": float(np.min(entropies)),
            "max_entropy": float(np.max(entropies)),
            "trend": float(trend),
            "current_entropy": float(entropies[-1]),
        }
    
    def reset(self):
        """Reset the monitoring system."""
        self.entropy_history.clear()
        self.metrics_history.clear()
        self.generation = 0
        self.entropy_target = None
        self.regulation_active = False
