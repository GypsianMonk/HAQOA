"""
Dynamic Collapse Controller

Implements Section 9: Dynamic Collapse Mechanism
Periodically removes weak states to control combinatorial explosion.

Mathematical representation:
C(s_i) = 1 if P(s_i) > θ_t, else 0

Where:
- θ_t = adaptive collapse threshold
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum

# Import without relative path for standalone usage
try:
    from .state_superposition import QuantumState, StateSuperpositionGenerator
    from .entropy_monitor import EntropyMetrics
except ImportError:
    from state_superposition import QuantumState, StateSuperpositionGenerator
    from entropy_monitor import EntropyMetrics


class CollapseStrategy(Enum):
    """Strategies for state collapse."""
    THRESHOLD = "threshold"  # Remove states below probability threshold
    PERCENTILE = "percentile"  # Remove bottom X% of states
    FIXED_COUNT = "fixed_count"  # Remove N weakest states
    ADAPTIVE = "adaptive"  # Adaptively determine based on entropy
    AGE_BASED = "age_based"  # Consider state age in collapse decision


@dataclass
class CollapseResult:
    """Result of a collapse operation."""
    
    states_removed: int
    states_retained: int
    removed_state_ids: List[int]
    retained_probability_mass: float
    collapse_threshold_used: float
    strategy_applied: str
    generation: int
    
    # Statistics
    mean_removed_probability: float = 0.0
    max_removed_probability: float = 0.0
    diversity_impact: float = 0.0  # Estimated impact on diversity


class DynamicCollapseController:
    """
    Controls dynamic state collapse operations.
    
    Features:
    - Multiple collapse strategies
    - Adaptive threshold computation
    - Diversity preservation awareness
    - Configurable collapse triggers
    """
    
    def __init__(
        self,
        strategy: CollapseStrategy = CollapseStrategy.ADAPTIVE,
        base_threshold: float = 0.01,
        removal_percentile: float = 0.2,
        fixed_removal_count: int = 5,
        min_states_to_maintain: int = 10,
        max_collapse_ratio: float = 0.5,
        age_weight: float = 0.1,
        use_entropy_feedback: bool = True
    ):
        """
        Initialize the Dynamic Collapse Controller.
        
        Args:
            strategy: Collapse strategy to use
            base_threshold: Base probability threshold for collapse
            removal_percentile: Percentile for percentile-based removal (0-1)
            fixed_removal_count: Number of states to remove for fixed strategy
            min_states_to_maintain: Minimum states to keep after collapse
            max_collapse_ratio: Maximum fraction of states that can be removed
            age_weight: Weight for age factor in collapse decisions
            use_entropy_feedback: Whether to use entropy for adaptive threshold
        """
        self.strategy = strategy
        self.base_threshold = base_threshold
        self.removal_percentile = removal_percentile
        self.fixed_removal_count = fixed_removal_count
        self.min_states_to_maintain = min_states_to_maintain
        self.max_collapse_ratio = max_collapse_ratio
        self.age_weight = age_weight
        self.use_entropy_feedback = use_entropy_feedback
        
        self.collapse_history: List[CollapseResult] = []
        self.generation = 0
        self.total_collapses = 0
        
        # Adaptive threshold state
        self.current_threshold = base_threshold
        self.threshold_history: List[float] = []
    
    def compute_adaptive_threshold(
        self,
        entropy_metrics: Optional[EntropyMetrics] = None,
        num_states: int = 0,
        target_retention_ratio: Optional[float] = None
    ) -> float:
        """
        Compute adaptive collapse threshold.
        
        Args:
            entropy_metrics: Current entropy metrics
            num_states: Current number of states
            target_retention_ratio: Desired ratio of states to retain
            
        Returns:
            Computed threshold value
        """
        threshold = self.base_threshold
        
        if self.use_entropy_feedback and entropy_metrics is not None:
            # High entropy → higher threshold (more aggressive collapse)
            # Low entropy → lower threshold (preserve diversity)
            entropy_factor = entropy_metrics.normalized_entropy
            
            # Adjust threshold based on entropy
            if entropy_metrics.is_high_entropy:
                # Can afford to remove more states
                threshold *= (1 + 0.5 * entropy_factor)
            elif entropy_metrics.is_low_entropy:
                # Be conservative to preserve diversity
                threshold *= (1 - 0.3 * (1 - entropy_factor))
            
            # Consider convergence risk
            if entropy_metrics.convergence_risk > 0.5:
                threshold *= 0.8  # Reduce threshold to maintain diversity
        
        if target_retention_ratio is not None and num_states > 0:
            # Adjust threshold to achieve target retention
            target_remove = int(num_states * (1 - target_retention_ratio))
            # This is a simplified adjustment; real implementation would
            # iterate to find optimal threshold
            threshold *= (target_remove / max(num_states * self.removal_percentile, 1))
        
        # Ensure threshold is reasonable
        threshold = np.clip(threshold, 0.001, 0.5)
        
        self.current_threshold = float(threshold)
        self.threshold_history.append(self.current_threshold)
        
        return self.current_threshold
    
    def execute_collapse(
        self,
        states: List[QuantumState],
        entropy_metrics: Optional[EntropyMetrics] = None,
        custom_strategy: Optional[CollapseStrategy] = None,
        additional_criteria: Optional[Callable[[QuantumState], bool]] = None
    ) -> CollapseResult:
        """
        Execute collapse operation on states.
        
        Args:
            states: List of quantum states to evaluate
            entropy_metrics: Current entropy metrics for adaptive threshold
            custom_strategy: Override default strategy
            additional_criteria: Optional function that returns True if state should be kept
            
        Returns:
            CollapseResult with details of the operation
        """
        if not states:
            return CollapseResult(
                states_removed=0,
                states_retained=0,
                removed_state_ids=[],
                retained_probability_mass=0.0,
                collapse_threshold_used=0.0,
                strategy_applied="none",
                generation=self.generation
            )
        
        strategy = custom_strategy or self.strategy
        
        # Compute threshold if using adaptive strategy
        if strategy == CollapseStrategy.ADAPTIVE:
            threshold = self.compute_adaptive_threshold(
                entropy_metrics,
                len(states)
            )
        else:
            threshold = self.base_threshold
        
        # Determine which states to remove based on strategy
        if strategy == CollapseStrategy.THRESHOLD:
            to_remove = self._threshold_collapse(states, threshold)
        
        elif strategy == CollapseStrategy.PERCENTILE:
            to_remove = self._percentile_collapse(states, self.removal_percentile)
        
        elif strategy == CollapseStrategy.FIXED_COUNT:
            to_remove = self._fixed_count_collapse(states, self.fixed_removal_count)
        
        elif strategy == CollapseStrategy.ADAPTIVE:
            to_remove = self._adaptive_collapse(states, entropy_metrics)
        
        elif strategy == CollapseStrategy.AGE_BASED:
            to_remove = self._age_based_collapse(states)
        
        else:
            to_remove = []
        
        # Apply additional criteria filter
        if additional_criteria is not None:
            to_remove = [
                s for s in to_remove
                if additional_criteria(s)
            ]
        
        # Enforce minimum states constraint
        max_removable = len(states) - self.min_states_to_maintain
        if len(to_remove) > max_removable:
            # Keep the best ones among those marked for removal
            to_remove_sorted = sorted(to_remove, key=lambda s: s.probability)
            to_remove = to_remove_sorted[:max_removable]
        
        # Enforce maximum collapse ratio
        max_by_ratio = int(len(states) * self.max_collapse_ratio)
        if len(to_remove) > max_by_ratio:
            to_remove_sorted = sorted(to_remove, key=lambda s: s.probability)
            to_remove = to_remove_sorted[:max_by_ratio]
        
        # Extract IDs of removed states
        removed_ids = [s.state_id for s in to_remove]
        
        # Compute statistics
        retained_states = [s for s in states if s.state_id not in removed_ids]
        retained_prob_mass = sum(s.probability for s in retained_states)
        
        if to_remove:
            mean_removed_prob = np.mean([s.probability for s in to_remove])
            max_removed_prob = max(s.probability for s in to_remove)
        else:
            mean_removed_prob = 0.0
            max_removed_prob = 0.0
        
        # Estimate diversity impact
        diversity_impact = self._estimate_diversity_impact(states, to_remove)
        
        result = CollapseResult(
            states_removed=len(to_remove),
            states_retained=len(retained_states),
            removed_state_ids=removed_ids,
            retained_probability_mass=float(retained_prob_mass),
            collapse_threshold_used=threshold,
            strategy_applied=strategy.value,
            generation=self.generation,
            mean_removed_probability=float(mean_removed_prob),
            max_removed_probability=float(max_removed_prob),
            diversity_impact=float(diversity_impact)
        )
        
        self.collapse_history.append(result)
        self.total_collapses += 1
        
        return result
    
    def _threshold_collapse(
        self,
        states: List[QuantumState],
        threshold: float
    ) -> List[QuantumState]:
        """Remove states with probability below threshold."""
        return [s for s in states if s.probability < threshold]
    
    def _percentile_collapse(
        self,
        states: List[QuantumState],
        percentile: float
    ) -> List[QuantumState]:
        """Remove bottom percentile of states by probability."""
        sorted_states = sorted(states, key=lambda s: s.probability)
        n_remove = max(1, int(len(states) * percentile))
        return sorted_states[:n_remove]
    
    def _fixed_count_collapse(
        self,
        states: List[QuantumState],
        count: int
    ) -> List[QuantumState]:
        """Remove fixed number of weakest states."""
        sorted_states = sorted(states, key=lambda s: s.probability)
        return sorted_states[:min(count, len(states))]
    
    def _adaptive_collapse(
        self,
        states: List[QuantumState],
        entropy_metrics: Optional[EntropyMetrics]
    ) -> List[QuantumState]:
        """Adaptively determine which states to remove."""
        if entropy_metrics is None:
            return self._percentile_collapse(states, self.removal_percentile)
        
        # Adjust removal based on entropy state
        if entropy_metrics.is_high_entropy:
            # Can remove more states
            percentile = min(self.removal_percentile * 1.5, 0.4)
        elif entropy_metrics.is_low_entropy:
            # Remove fewer to preserve diversity
            percentile = max(self.removal_percentile * 0.5, 0.1)
        else:
            percentile = self.removal_percentile
        
        return self._percentile_collapse(states, percentile)
    
    def _age_based_collapse(
        self,
        states: List[QuantumState]
    ) -> List[QuantumState]:
        """Consider both probability and age in collapse decision."""
        # Compute combined score: low probability + old age = high collapse priority
        scored_states = []
        for s in states:
            # Normalize age contribution
            age_score = s.age * self.age_weight
            # Combined score (lower = more likely to collapse)
            combined_score = s.probability - age_score
            scored_states.append((s, combined_score))
        
        # Sort by combined score and remove worst
        scored_states.sort(key=lambda x: x[1])
        
        n_remove = max(1, int(len(states) * self.removal_percentile))
        return [s for s, _ in scored_states[:n_remove]]
    
    def _estimate_diversity_impact(
        self,
        all_states: List[QuantumState],
        removed_states: List[QuantumState]
    ) -> float:
        """
        Estimate the impact of removal on population diversity.
        
        Returns a score where higher means more diversity loss.
        """
        if not removed_states or len(all_states) <= 1:
            return 0.0
        
        # Simple metric: variance in probabilities of removed vs retained
        removed_probs = [s.probability for s in removed_states]
        retained_states = [s for s in all_states if s.state_id not in 
                          [r.state_id for r in removed_states]]
        retained_probs = [s.probability for s in retained_states]
        
        if not retained_probs:
            return 1.0
        
        # Compare distributions
        removed_mean = np.mean(removed_probs)
        retained_mean = np.mean(retained_probs)
        
        # Higher difference = more impact
        impact = abs(removed_mean - retained_mean) / (retained_mean + 1e-10)
        
        return float(min(impact, 1.0))
    
    def should_trigger_collapse(
        self,
        num_states: int,
        entropy_metrics: Optional[EntropyMetrics] = None,
        generation_since_last_collapse: int = 0,
        min_generation_interval: int = 5
    ) -> bool:
        """
        Determine if collapse should be triggered.
        
        Args:
            num_states: Current number of states
            entropy_metrics: Current entropy metrics
            generation_since_last_collapse: Generations since last collapse
            min_generation_interval: Minimum generations between collapses
            
        Returns:
            True if collapse should be triggered
        """
        # Check minimum interval
        if generation_since_last_collapse < min_generation_interval:
            return False
        
        # Trigger if too many states
        if num_states > 100:  # Configurable threshold
            return True
        
        # Trigger based on entropy signals
        if entropy_metrics is not None:
            # High convergence risk → trigger collapse to refresh
            if entropy_metrics.convergence_risk > 0.7:
                return True
            
            # Very low entropy → might need collapse + regeneration
            if entropy_metrics.is_low_entropy and entropy_metrics.effective_states < 5:
                return True
        
        return False
    
    def get_statistics(self) -> Dict[str, float]:
        """Get statistical summary of collapse history."""
        if not self.collapse_history:
            return {
                "total_collapses": 0,
                "mean_states_removed": 0.0,
                "mean_threshold": 0.0,
            }
        
        removed_counts = [c.states_removed for c in self.collapse_history]
        thresholds = [c.collapse_threshold_used for c in self.collapse_history]
        
        return {
            "total_collapses": self.total_collapses,
            "mean_states_removed": float(np.mean(removed_counts)),
            "max_states_removed": int(np.max(removed_counts)),
            "mean_threshold": float(np.mean(thresholds)),
            "current_threshold": self.current_threshold,
        }
    
    def reset(self):
        """Reset the collapse controller."""
        self.collapse_history.clear()
        self.generation = 0
        self.total_collapses = 0
        self.current_threshold = self.base_threshold
        self.threshold_history.clear()
