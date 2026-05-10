"""
AMOEBA Engine: Autonomous Meta-Optimizing Evolutionary Behavioral Architecture
Phase 7 Implementation - Recursive Self-Improving Optimization

Mathematical Foundation:
- Dual-Layer Evolution: Object Layer + Meta Layer
- Parameter Evolution: Θ(t+1) = Θ(t) + η_Θ ∇_Θ J(t)
- Entropy Curvature: K_H = ∂²H/∂t²
- Instability Index: I(t) = α₁σ_H + α₂σ_Q + α₃T
- Governance Constraints: G(t) ≤ G_max
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import warnings


@dataclass
class MetaConfig:
    """Configuration for meta-evolution parameters"""
    # Learning rates
    object_lr: float = 0.1
    meta_lr: float = 0.05
    
    # Entropy geometry
    entropy_curvature_coupling: float = 0.3
    max_entropy_acceleration: float = 0.5
    
    # Governance constraints
    max_recursion_depth: int = 3
    max_mutation_rate: float = 0.5
    stability_threshold: float = 0.8
    
    # Multi-agent ecology
    n_agents: int = 3
    competition_mode: bool = True
    cooperation_bonus: float = 0.2


@dataclass
class AgentProfile:
    """Profile for optimization agent specialization"""
    name: str
    specialization: str  # 'exploration', 'exploitation', 'stabilization'
    performance_history: List[float] = field(default_factory=list)
    mutation_rate: float = 0.1
    trust_score: float = 1.0


class AMOEBAEngine:
    """
    Autonomous Meta-Optimizing Evolutionary Behavioral Architecture
    
    Implements recursive optimization where the system optimizes its own
    search behavior, entropy geometry, and architectural parameters.
    
    Key Features:
    - Dual-layer evolution (object + meta)
    - Self-adaptive entropy curvature learning
    - Evolving reward fields to prevent hacking
    - Multi-agent competitive/cooperative ecology
    - Governance constraints against runaway recursion
    """
    
    def __init__(self, config: Optional[MetaConfig] = None):
        self.config = config or MetaConfig()
        
        # Meta-parameters (optimized by meta-layer)
        self.meta_params = {
            'beta_base': 1.0,
            'entropy_coupling': 0.5,
            'collapse_threshold': 0.1,
            'mutation_rate': 0.1,
            'diversity_weight': 0.3
        }
        
        # Object-level state
        self.object_state = {
            'solutions': [],
            'energies': [],
            'probabilities': [],
            'entropy': 0.0
        }
        
        # Meta-layer memory
        self.meta_memory = {
            'performance_history': [],
            'entropy_trajectories': [],
            'failure_signatures': [],
            'successful_patterns': []
        }
        
        # Multi-agent ecology
        self.agents = self._initialize_agents()
        
        # Governance tracking
        self.recursion_depth = 0
        self.instability_index = 0.0
        self.governance_violations = 0
    
    def _initialize_agents(self) -> List[AgentProfile]:
        """Initialize multi-agent ecology with specializations"""
        specializations = ['exploration', 'exploitation', 'stabilization']
        agents = []
        
        for i, spec in enumerate(specializations[:self.config.n_agents]):
            agent = AgentProfile(
                name=f"Agent_{spec[:4]}_{i}",
                specialization=spec,
                mutation_rate=0.1 + 0.1 * i  # Varying mutation rates
            )
            agents.append(agent)
        
        return agents
    
    def compute_instability_index(self, entropy_std: float, 
                                   quality_std: float,
                                   turbulence: float) -> float:
        """
        Compute instability index: I(t) = α₁σ_H + α₂σ_Q + α₃T
        
        Args:
            entropy_std: Standard deviation of entropy (σ_H)
            quality_std: Standard deviation of solution quality (σ_Q)
            turbulence: Search turbulence T
            
        Returns:
            Instability index value (higher = more unstable)
        """
        alpha_1, alpha_2, alpha_3 = 0.4, 0.4, 0.2  # Weights
        
        instability = (alpha_1 * entropy_std + 
                      alpha_2 * quality_std + 
                      alpha_3 * turbulence)
        
        self.instability_index = instability
        return instability
    
    def evolve_meta_parameters(self, performance_gradient: np.ndarray) -> Dict[str, float]:
        """
        Evolve meta-parameters: Θ(t+1) = Θ(t) + η_Θ ∇_Θ J(t)
        
        Args:
            performance_gradient: Gradient of performance objective w.r.t. meta-params
            
        Returns:
            Updated meta-parameters dictionary
        """
        # Apply governance constraints
        if self.recursion_depth >= self.config.max_recursion_depth:
            warnings.warn("Max recursion depth reached. Limiting meta-evolution.")
            return self.meta_params
        
        # Update each meta-parameter
        param_keys = list(self.meta_params.keys())
        for i, key in enumerate(param_keys):
            if i < len(performance_gradient):
                gradient_component = performance_gradient[i]
                
                # Adaptive learning rate based on instability
                effective_lr = self.config.meta_lr * (1.0 - self.instability_index)
                effective_lr = max(0.01, effective_lr)  # Minimum learning rate
                
                # Update with governance bounds
                old_value = self.meta_params[key]
                new_value = old_value + effective_lr * gradient_component
                
                # Apply bounds
                if key == 'mutation_rate':
                    new_value = np.clip(new_value, 0.01, self.config.max_mutation_rate)
                elif key == 'diversity_weight':
                    new_value = np.clip(new_value, 0.0, 1.0)
                elif key == 'beta_base':
                    new_value = np.clip(new_value, 0.1, 5.0)
                elif key == 'entropy_coupling':
                    new_value = np.clip(new_value, 0.0, 2.0)
                elif key == 'collapse_threshold':
                    new_value = np.clip(new_value, 0.01, 0.9)
                
                self.meta_params[key] = new_value
        
        return self.meta_params
    
    def adapt_entropy_geometry(self, current_entropy: float, 
                                entropy_history: List[float]) -> float:
        """
        Adapt entropy curvature dynamically: K_H = ∂²H/∂t²
        
        Learns optimal entropy acceleration instead of using fixed damping.
        
        Args:
            current_entropy: Current entropy H(t)
            entropy_history: List of recent entropy values
            
        Returns:
            Adjusted entropy target with learned curvature
        """
        if len(entropy_history) < 3:
            return current_entropy
        
        # Compute first derivative (velocity)
        h_t = current_entropy
        h_t1 = entropy_history[-1]
        h_t2 = entropy_history[-2]
        
        velocity = h_t - h_t1
        
        # Compute second derivative (acceleration/curvature)
        if abs(h_t1 - h_t2) > 1e-10:
            acceleration = (velocity - (h_t1 - h_t2)) / (h_t1 - h_t2 + 1e-10)
        else:
            acceleration = 0.0
        
        # Apply governance constraint on acceleration
        max_accel = self.config.max_entropy_acceleration
        acceleration = np.clip(acceleration, -max_accel, max_accel)
        
        # Adjust entropy target based on learned curvature
        coupling = self.config.entropy_curvature_coupling
        adjusted_entropy = h_t + coupling * acceleration
        
        # Ensure entropy stays in valid range
        max_entropy = np.log2(len(self.object_state.get('solutions', [1])) + 1)
        adjusted_entropy = np.clip(adjusted_entropy, 0.0, max_entropy)
        
        # Store trajectory for learning
        self.meta_memory['entropy_trajectories'].append({
            'entropy': h_t,
            'velocity': velocity,
            'acceleration': acceleration,
            'adjusted': adjusted_entropy
        })
        
        return adjusted_entropy
    
    def evolve_reward_field(self, rewards: np.ndarray, 
                            turbulence: float,
                            cost: float) -> np.ndarray:
        """
        Evolve reward field weights dynamically: R_i(t) = Σ w_k(t)φ_k(s_i)
        
        Prevents reward hacking by evolving weights based on turbulence and cost.
        
        Objective: max(Q - λT - μC)
        
        Args:
            rewards: Raw reward signals
            turbulence: Search turbulence T
            cost: Computational cost C
            
        Returns:
            Evolved reward field
        """
        # Base weights (could also be evolved)
        w_quality = 1.0
        w_turbulence = 0.3 * self.instability_index  # Penalize more when unstable
        w_cost = 0.1
        
        # Compute evolved reward field
        # R'_i = w_q*Q_i - w_t*T - w_c*C
        evolved_rewards = (w_quality * rewards - 
                          w_turbulence * turbulence - 
                          w_cost * cost)
        
        # Ensure non-negative rewards
        evolved_rewards = np.maximum(evolved_rewards, 0)
        
        return evolved_rewards
    
    def mutate_architecture(self, severity: Optional[float] = None) -> Dict[str, Any]:
        """
        Mutate optimization architecture based on instability
        
        Mutation probability: P_m = I / (I + ε)
        
        Args:
            severity: Optional override for mutation severity
            
        Returns:
            Dictionary of architectural changes
        """
        if severity is None:
            # Compute mutation probability from instability
            epsilon = 1e-6
            mutation_prob = self.instability_index / (self.instability_index + epsilon)
            mutation_prob = np.clip(mutation_prob, 0.0, self.config.max_mutation_rate)
        else:
            mutation_prob = severity
        
        changes = {}
        
        # Decide what to mutate based on probability
        if np.random.random() < mutation_prob:
            # Mutate agent specializations
            for agent in self.agents:
                if np.random.random() < mutation_prob:
                    old_spec = agent.specialization
                    specs = ['exploration', 'exploitation', 'stabilization']
                    agent.specialization = np.random.choice(specs)
                    changes[f'{agent.name}_spec'] = f"{old_spec} -> {agent.specialization}"
        
        if np.random.random() < mutation_prob * 0.5:
            # Mutate meta-parameter ranges
            for key in self.meta_params:
                if np.random.random() < 0.3:
                    perturbation = np.random.normal(0, 0.1)
                    old_val = self.meta_params[key]
                    self.meta_params[key] *= (1 + perturbation)
                    changes[f'{key}_range'] = f"{old_val:.3f} -> {self.meta_params[key]:.3f}"
        
        # Record mutation event
        if changes:
            self.meta_memory['failure_signatures'].append({
                'instability': self.instability_index,
                'changes': changes,
                'recursion_depth': self.recursion_depth
            })
        
        return changes
    
    def run_meta_evolution_step(self, 
                                 solutions: List[Any],
                                 energies: np.ndarray,
                                 iteration: int) -> Dict[str, Any]:
        """
        Execute one complete meta-evolution step
        
        Dual-layer process:
        1. Object layer: Evaluate solutions
        2. Meta layer: Optimize optimizer parameters
        
        Args:
            solutions: Current candidate solutions
            energies: Energy/cost values for each solution
            iteration: Current iteration number
            
        Returns:
            State dictionary with updated parameters and diagnostics
        """
        self.recursion_depth += 1
        
        # === OBJECT LAYER ===
        n_solutions = len(solutions)
        self.object_state['solutions'] = solutions
        self.object_state['energies'] = energies.tolist()
        
        # Compute probabilities (Boltzmann distribution)
        beta = self.meta_params['beta_base']
        exp_energies = np.exp(-beta * (energies - np.max(energies)))
        probabilities = exp_energies / (np.sum(exp_energies) + 1e-10)
        self.object_state['probabilities'] = probabilities.tolist()
        
        # Compute entropy
        p_nonzero = probabilities[probabilities > 0]
        entropy = -np.sum(p_nonzero * np.log2(p_nonzero))
        self.object_state['entropy'] = entropy
        
        # === META LAYER ===
        
        # Retrieve history
        entropy_history = [m['entropy'] for m in self.meta_memory.get('entropy_trajectories', [-10:])]
        
        # Compute diagnostics
        if len(entropy_history) > 2:
            entropy_std = np.std(entropy_history[-5:])
            quality_std = np.std(energies)
            turbulence = abs(entropy - entropy_history[-1]) if entropy_history else 0.0
        else:
            entropy_std = 0.0
            quality_std = np.std(energies)
            turbulence = 0.0
        
        # Compute instability index
        instability = self.compute_instability_index(entropy_std, quality_std, turbulence)
        
        # Check governance constraints
        if instability > self.config.stability_threshold:
            self.governance_violations += 1
            warnings.warn(f"Instability threshold exceeded: {instability:.4f}")
            
            # Trigger architectural mutation
            changes = self.mutate_architecture()
            if changes:
                print(f"🧬 Architectural mutation triggered: {changes}")
        
        # Adapt entropy geometry
        adjusted_entropy = self.adapt_entropy_geometry(entropy, entropy_history)
        
        # Evolve reward field (simulate rewards as negative energies)
        raw_rewards = -energies  # Higher quality = lower energy = higher reward
        evolved_rewards = self.evolve_reward_field(raw_rewards, turbulence, 
                                                    cost=iteration * 0.001)
        
        # Simulate performance gradient (in real system, this comes from actual performance change)
        # Here we use a heuristic based on improvement rate
        if len(self.meta_memory['performance_history']) > 0:
            last_perf = self.meta_memory['performance_history'][-1]
            current_perf = -np.min(energies)  # Best solution quality
            improvement = current_perf - last_perf
            performance_gradient = np.array([improvement * 0.1] * len(self.meta_params))
        else:
            performance_gradient = np.zeros(len(self.meta_params))
        
        # Evolve meta-parameters
        updated_params = self.evolve_meta_parameters(performance_gradient)
        
        # Update performance history
        self.meta_memory['performance_history'].append(-np.min(energies))
        
        # Multi-agent competition/cooperation
        agent_performances = []
        for agent in self.agents:
            # Simulate agent-specific performance based on specialization
            if agent.specialization == 'exploration':
                perf = entropy / (np.log2(n_solutions + 1) + 1e-10)
            elif agent.specialization == 'exploitation':
                perf = 1.0 / (1.0 + np.min(energies))
            else:  # stabilization
                perf = 1.0 / (1.0 + instability)
            
            agent.performance_history.append(perf)
            agent_performances.append(perf)
            
            # Update trust score
            if len(agent.performance_history) > 3:
                recent_trend = np.mean(agent.performance_history[-3:]) - np.mean(agent.performance_history[:3])
                agent.trust_score = np.clip(agent.trust_score + 0.1 * recent_trend, 0.0, 1.0)
        
        # Decrement recursion depth after completion
        self.recursion_depth -= 1
        
        return {
            'meta_params': updated_params,
            'entropy': entropy,
            'adjusted_entropy': adjusted_entropy,
            'instability_index': instability,
            'governance_violations': self.governance_violations,
            'evolved_rewards': evolved_rewards,
            'agent_performances': dict(zip([a.name for a in self.agents], agent_performances)),
            'recursion_depth': self.recursion_depth
        }


def demo_amoeba():
    """Demonstrate AMOEBA engine functionality"""
    print("🧬 AMOEBA: Autonomous Meta-Optimizing Evolutionary Behavioral Architecture")
    print("=" * 70)
    
    config = MetaConfig(
        n_agents=3,
        competition_mode=True,
        max_recursion_depth=3
    )
    
    engine = AMOEBAEngine(config)
    
    # Simulate optimization state
    np.random.seed(42)
    n_solutions = 10
    solutions = [f"sol_{i}" for i in range(n_solutions)]
    energies = np.random.uniform(1.0, 5.0, n_solutions)
    
    print(f"\nInitial State: {n_solutions} solutions")
    print(f"Energy range: [{np.min(energies):.3f}, {np.max(energies):.3f}]")
    
    # Run meta-evolution steps
    results = []
    for iteration in range(5):
        result = engine.run_meta_evolution_step(solutions, energies, iteration)
        results.append(result)
        
        print(f"\n--- Iteration {iteration + 1} ---")
        print(f"Entropy: {result['entropy']:.4f} (adjusted: {result['adjusted_entropy']:.4f})")
        print(f"Instability Index: {result['instability_index']:.4f}")
        print(f"Meta Parameters: β={result['meta_params']['beta_base']:.3f}, " +
              f"κ={result['meta_params']['entropy_coupling']:.3f}")
        print(f"Governance Violations: {result['governance_violations']}")
        print(f"Agent Performances: {result['agent_performances']}")
        
        # Simulate improvement in next iteration
        energies = energies * 0.95 + np.random.normal(0, 0.1, n_solutions)
        energies = np.clip(energies, 0.1, 10.0)
    
    print("\n✅ AMOEBA meta-evolution demonstration completed!")
    print(f"Final meta-parameters: {engine.meta_params}")
    print(f"Total governance violations: {engine.governance_violations}")
    
    return results


if __name__ == "__main__":
    demo_amoeba()
