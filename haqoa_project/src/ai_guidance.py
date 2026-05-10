"""
AI Guidance Engine

Implements Section 8: AI Guidance Layer
Provides intelligent search direction through reward estimation and path prediction.

Reward function:
R(s_i) = w_1*Q_i - w_2*C_i - w_3*E_i

Where:
- Q_i = solution quality
- C_i = computational cost
- E_i = instability/error penalty
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

# Import without relative path for standalone usage
try:
    from .state_superposition import QuantumState
    from .entropy_monitor import EntropyMetrics
except ImportError:
    from state_superposition import QuantumState
    from entropy_monitor import EntropyMetrics


class GuidanceMode(Enum):
    """Modes for AI guidance."""
    REWARD_BASED = "reward_based"
    PREDICTION_BASED = "prediction_based"
    HYBRID = "hybrid"
    PASSIVE = "passive"  # No active guidance


@dataclass
class RewardConfig:
    """Configuration for reward computation."""
    
    quality_weight: float = 1.0  # w_1
    cost_weight: float = 0.1     # w_2
    instability_weight: float = 0.2  # w_3
    
    # Normalization parameters
    use_normalized_rewards: bool = True
    reward_baseline: Optional[float] = None
    
    # Discount factor for temporal aspects
    discount_factor: float = 0.95


@dataclass
class StateReward:
    """Computed reward for a state."""
    
    state_id: int
    total_reward: float
    quality_component: float
    cost_component: float
    instability_component: float
    normalized_reward: float = 0.0
    rank: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class AIGuidanceEngine:
    """
    AI-guided search direction engine.
    
    Features:
    - Multi-component reward computation
    - Adaptive weight adjustment
    - Search pattern learning
    - Exploration/exploitation balancing
    """
    
    def __init__(
        self,
        mode: GuidanceMode = GuidanceMode.HYBRID,
        reward_config: Optional[RewardConfig] = None,
        learning_rate: float = 0.01,
        enable_adaptive_weights: bool = True
    ):
        """
        Initialize the AI Guidance Engine.
        
        Args:
            mode: Guidance mode to use
            reward_config: Reward computation configuration
            learning_rate: Learning rate for adaptive components
            enable_adaptive_weights: Whether to adaptively adjust weights
        """
        self.mode = mode
        self.reward_config = reward_config or RewardConfig()
        self.learning_rate = learning_rate
        self.enable_adaptive_weights = enable_adaptive_weights
        
        self.generation = 0
        self.reward_history: List[Dict[int, float]] = []
        self.best_reward_seen: float = -np.inf
        self.mean_reward_baseline: Optional[float] = None
        
        # Adaptive weight state
        self.weight_adjustment_history: List[Tuple[float, float, float]] = []
        
        # Pattern learning (simplified)
        self.improvement_patterns: List[bool] = []
    
    def compute_rewards(
        self,
        states: List[QuantumState],
        quality_fn: Optional[Callable[[Any], float]] = None,
        cost_fn: Optional[Callable[[Any], float]] = None,
        instability_fn: Optional[Callable[[Any], float]] = None
    ) -> List[StateReward]:
        """
        Compute rewards for all states.
        
        Args:
            states: List of quantum states
            quality_fn: Function to compute solution quality
            cost_fn: Function to compute computational cost
            instability_fn: Function to compute instability measure
            
        Returns:
            List of StateReward objects
        """
        if not states:
            return []
        
        rewards = []
        
        for state in states:
            # Quality component (use pre-computed score or custom function)
            if quality_fn is not None:
                quality = quality_fn(state.solution)
            else:
                quality = state.quality_score
            
            # Cost component (default to age-based or uniform)
            if cost_fn is not None:
                cost = cost_fn(state.solution)
            else:
                cost = state.age * 0.01  # Simple age-based cost
            
            # Instability component (based on probability variance)
            if instability_fn is not None:
                instability = instability_fn(state.solution)
            else:
                # Use probability as inverse stability indicator
                instability = 1.0 - state.probability
            
            # Compute total reward: R = w1*Q - w2*C - w3*E
            total_reward = (
                self.reward_config.quality_weight * quality -
                self.reward_config.cost_weight * cost -
                self.reward_config.instability_weight * instability
            )
            
            state_reward = StateReward(
                state_id=state.state_id,
                total_reward=float(total_reward),
                quality_component=float(quality),
                cost_component=float(cost),
                instability_component=float(instability),
                metadata={
                    "generation": self.generation,
                    "probability": state.probability,
                }
            )
            
            rewards.append(state_reward)
            
            # Track best reward
            if total_reward > self.best_reward_seen:
                self.best_reward_seen = total_reward
        
        # Normalize rewards if configured
        if self.reward_config.use_normalized_rewards:
            self._normalize_rewards(rewards)
        
        # Rank states by reward
        sorted_rewards = sorted(rewards, key=lambda r: r.total_reward, reverse=True)
        for rank, reward in enumerate(sorted_rewards):
            reward.rank = rank + 1
        
        # Store history
        reward_dict = {r.state_id: r.total_reward for r in rewards}
        self.reward_history.append(reward_dict)
        
        # Update baseline
        self._update_baseline(rewards)
        
        return rewards
    
    def _normalize_rewards(self, rewards: List[StateReward]):
        """Normalize rewards to [0, 1] range."""
        if not rewards:
            return
        
        total_rewards = np.array([r.total_reward for r in rewards])
        
        min_r = np.min(total_rewards)
        max_r = np.max(total_rewards)
        
        range_r = max_r - min_r
        if range_r > 0:
            for reward in rewards:
                reward.normalized_reward = (
                    (reward.total_reward - min_r) / range_r
                )
        else:
            for reward in rewards:
                reward.normalized_reward = 0.5
    
    def _update_baseline(self, rewards: List[StateReward]):
        """Update reward baseline for normalization."""
        if not rewards:
            return
        
        mean_reward = np.mean([r.total_reward for r in rewards])
        
        if self.mean_reward_baseline is None:
            self.mean_reward_baseline = mean_reward
        else:
            # Exponential moving average
            self.mean_reward_baseline = (
                self.reward_config.discount_factor * self.mean_reward_baseline +
                (1 - self.reward_config.discount_factor) * mean_reward
            )
    
    def get_guidance_signal(
        self,
        states: List[QuantumState],
        rewards: List[StateReward],
        entropy_metrics: Optional[EntropyMetrics] = None
    ) -> Dict[str, Any]:
        """
        Generate guidance signals for search direction.
        
        Args:
            states: Current states
            rewards: Computed rewards
            entropy_metrics: Current entropy metrics
            
        Returns:
            Dictionary with guidance information
        """
        if not rewards:
            return {"action": "explore", "intensity": 0.5, "targets": []}
        
        # Determine exploration vs exploitation
        if entropy_metrics is not None:
            if entropy_metrics.is_low_entropy:
                base_action = "exploit"
            elif entropy_metrics.is_high_entropy:
                base_action = "explore"
            else:
                base_action = "balance"
        else:
            base_action = "balance"
        
        # Adjust based on reward trends
        if len(self.reward_history) >= 2:
            recent_mean = np.mean(list(self.reward_history[-1].values()))
            previous_mean = np.mean(list(self.reward_history[-2].values()))
            
            if recent_mean < previous_mean * 0.9:
                # Rewards decreasing → increase exploration
                if base_action == "exploit":
                    base_action = "balance"
                elif base_action == "balance":
                    base_action = "explore"
        
        # Identify target states for amplification
        top_states = sorted(rewards, key=lambda r: r.total_reward, reverse=True)
        top_count = max(1, int(len(rewards) * 0.2))  # Top 20%
        target_ids = [r.state_id for r in top_states[:top_count]]
        
        # Compute guidance intensity
        if entropy_metrics is not None:
            intensity = 1.0 - entropy_metrics.normalized_entropy
        else:
            intensity = 0.5
        
        guidance = {
            "action": base_action,
            "intensity": float(intensity),
            "targets": target_ids,
            "top_reward": float(top_states[0].total_reward) if top_states else 0.0,
            "mean_reward": float(np.mean([r.total_reward for r in rewards])),
            "recommendation": self._generate_recommendation(base_action, intensity, rewards),
        }
        
        return guidance
    
    def _generate_recommendation(
        self,
        action: str,
        intensity: float,
        rewards: List[StateReward]
    ) -> str:
        """Generate human-readable recommendation."""
        if action == "explore":
            return f"Increase exploration (intensity: {intensity:.2f}). Consider mutation or diversification."
        elif action == "exploit":
            return f"Focus on exploitation (intensity: {intensity:.2f}). Amplify top performers."
        else:
            return "Maintain balanced search strategy."
    
    def update_weights_from_feedback(
        self,
        success_signal: float,
        adjustment_magnitude: float = 0.05
    ):
        """
        Adaptively adjust reward weights based on feedback.
        
        Args:
            success_signal: Success metric (-1 to 1)
            adjustment_magnitude: Maximum adjustment per update
        """
        if not self.enable_adaptive_weights:
            return
        
        # Adjust quality weight based on success
        if success_signal > 0:
            # Success → increase quality weight slightly
            self.reward_config.quality_weight = min(
                self.reward_config.quality_weight * (1 + adjustment_magnitude),
                2.0
            )
        else:
            # Failure → might need to consider other factors more
            self.reward_config.quality_weight = max(
                self.reward_config.quality_weight * (1 - adjustment_magnitude),
                0.5
            )
        
        # Store adjustment history
        self.weight_adjustment_history.append((
            self.reward_config.quality_weight,
            self.reward_config.cost_weight,
            self.reward_config.instability_weight
        ))
    
    def predict_state_potential(
        self,
        state: QuantumState,
        historical_data: Optional[Dict] = None
    ) -> float:
        """
        Predict the potential of a state for future improvement.
        
        Simplified implementation using heuristic features.
        
        Args:
            state: State to evaluate
            historical_data: Optional historical data for prediction
            
        Returns:
            Predicted potential score (0-1)
        """
        # Features for prediction
        features = [
            state.probability,  # Current probability
            state.quality_score,  # Quality
            1.0 / (1.0 + state.age),  # Recency (newer = higher)
            abs(state.amplitude),  # Amplitude magnitude
        ]
        
        # Simple weighted combination (could be replaced with ML model)
        weights = [0.3, 0.4, 0.2, 0.1]
        
        potential = sum(f * w for f, w in zip(features, weights))
        
        return float(np.clip(potential, 0.0, 1.0))
    
    def record_improvement(self, improved: bool):
        """Record whether an improvement occurred this generation."""
        self.improvement_patterns.append(improved)
        
        # Limit history size
        if len(self.improvement_patterns) > 100:
            self.improvement_patterns = self.improvement_patterns[-100:]
    
    def get_improvement_rate(self, window_size: int = 10) -> float:
        """Get recent improvement rate."""
        if not self.improvement_patterns:
            return 0.0
        
        recent = self.improvement_patterns[-window_size:]
        return float(np.mean(recent))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistical summary of guidance performance."""
        if not self.reward_history:
            return {
                "total_generations": self.generation,
                "best_reward": self.best_reward_seen,
                "improvement_rate": 0.0,
            }
        
        recent_rewards = [
            np.mean(list(r.values()))
            for r in self.reward_history[-10:]
        ]
        
        return {
            "total_generations": self.generation,
            "best_reward": float(self.best_reward_seen),
            "mean_reward_baseline": self.mean_reward_baseline,
            "recent_mean_reward": float(np.mean(recent_rewards)) if recent_rewards else 0.0,
            "improvement_rate": self.get_improvement_rate(),
            "current_mode": self.mode.value,
        }
    
    def reset(self):
        """Reset the guidance engine."""
        self.generation = 0
        self.reward_history.clear()
        self.best_reward_seen = -np.inf
        self.mean_reward_baseline = None
        self.weight_adjustment_history.clear()
        self.improvement_patterns.clear()
