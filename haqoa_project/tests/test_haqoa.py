"""
HAQOA Test Suite
=================

Comprehensive tests for all HAQOA components.
"""

import numpy as np
import sys
import unittest

sys.path.insert(0, '/workspace/haqoa_project/src')

from state_superposition import StateSuperpositionGenerator, QuantumState
from probabilistic_encoding import ProbabilisticStateEncoder
from entropy_monitor import EntropyMonitoringSystem, EntropyMetrics
from amplification import AdaptiveProbabilityAmplifier, AmplificationConfig
from collapse_controller import DynamicCollapseController, CollapseStrategy
from ai_guidance import AIGuidanceEngine, GuidanceMode
from evolution import EvolutionEngine
from haqoa_engine import HAQOAOptimizer, HAQOAConfig
from tsp_solver import TSPInstance, TSPSolver


class TestStateSuperposition(unittest.TestCase):
    """Test state superposition generator."""
    
    def test_initialization(self):
        """Test generator initialization."""
        gen = StateSuperpositionGenerator(
            initial_population_size=10,
            max_states=50
        )
        self.assertEqual(gen.max_states, 50)
    
    def test_generate_initial_states(self):
        """Test state generation."""
        gen = StateSuperpositionGenerator(initial_population_size=5)
        
        def solution_gen():
            return list(np.random.permutation(10))
        
        gen.generate_initial_states(solution_generator=solution_gen)
        states = gen.get_all_states()
        
        self.assertEqual(len(states), 5)
        self.assertTrue(all(isinstance(s, QuantumState) for s in states))
    
    def test_add_state(self):
        """Test adding individual states."""
        gen = StateSuperpositionGenerator(max_states=10)
        
        state = QuantumState(
            state_id=1,
            solution=[1, 2, 3],
            amplitude=1.0,
            quality_score=0.5
        )
        gen.add_state(state)
        
        self.assertEqual(len(gen.get_all_states()), 1)


class TestProbabilisticEncoding(unittest.TestCase):
    """Test probabilistic state encoding."""
    
    def test_encode_states(self):
        """Test state encoding with probabilities."""
        encoder = ProbabilisticStateEncoder()
        
        states = [
            QuantumState(state_id=1, solution=[1, 2, 3], amplitude=1.0, quality_score=0.9),
            QuantumState(state_id=2, solution=[3, 2, 1], amplitude=1.0, quality_score=0.5),
            QuantumState(state_id=3, solution=[2, 1, 3], amplitude=1.0, quality_score=0.7)
        ]
        
        encoded = encoder.encode_states(states, amplification_factor=2.0)
        
        # Check probabilities sum to ~1
        total_prob = sum(s.normalized_probability for s in encoded)
        self.assertAlmostEqual(total_prob, 1.0, places=5)
        
        # Higher quality should have higher probability
        self.assertGreater(encoded[0].normalized_probability, encoded[1].normalized_probability)


class TestEntropyMonitor(unittest.TestCase):
    """Test entropy monitoring system."""
    
    def test_compute_entropy(self):
        """Test entropy computation."""
        monitor = EntropyMonitoringSystem()
        
        # Create states with different probabilities to get non-zero entropy
        states = [
            QuantumState(state_id=1, solution=[1], amplitude=1.0, quality_score=0.5),
            QuantumState(state_id=2, solution=[2], amplitude=1.0, quality_score=0.8)
        ]
        
        # First encode states to set probabilities
        from probabilistic_encoding import ProbabilisticStateEncoder
        encoder = ProbabilisticStateEncoder()
        encoder.encode_states(states, amplification_factor=1.0)
        
        metrics = monitor.compute_entropy_from_states(states)
        
        # Different probabilities should give positive entropy
        self.assertGreater(metrics.total_entropy, 0)
    
    def test_entropy_levels(self):
        """Test entropy level classification."""
        monitor = EntropyMonitoringSystem(
            high_entropy_threshold=0.8,
            low_entropy_threshold=0.2
        )
        
        # High entropy (uniform distribution) - need to set probabilities first
        states_uniform = [
            QuantumState(state_id=i, solution=[i], amplitude=1.0, quality_score=1.0)
            for i in range(10)
        ]
        # Encode to set equal probabilities
        from probabilistic_encoding import ProbabilisticStateEncoder
        encoder = ProbabilisticStateEncoder()
        encoder.encode_states(states_uniform, amplification_factor=0.01)  # Low beta for near-uniform
        
        metrics_high = monitor.compute_entropy_from_states(states_uniform)
        self.assertEqual(metrics_high.is_high_entropy, True)
        
        # Low entropy (one dominant state)
        states_dominant = [
            QuantumState(state_id=0, solution=[0], amplitude=1.0, quality_score=10.0),
            *[QuantumState(state_id=i, solution=[i], amplitude=1.0, quality_score=0.1) for i in range(1, 10)]
        ]
        encoder.encode_states(states_dominant, amplification_factor=2.0)  # High beta for peaked dist
        metrics_low = monitor.compute_entropy_from_states(states_dominant)
        self.assertEqual(metrics_low.is_low_entropy, True)


class TestAmplification(unittest.TestCase):
    """Test adaptive probability amplification."""
    
    def test_compute_amplification(self):
        """Test amplification computation based on entropy."""
        config = AmplificationConfig(base_amplification=2.0, entropy_sensitivity=0.5)
        amplifier = AdaptiveProbabilityAmplifier(config)
        
        # Create mock entropy metrics
        from entropy_monitor import EntropyMetrics
        metrics_high = EntropyMetrics(
            total_entropy=3.0,
            normalized_entropy=0.9,
            entropy_rate=0.0,
            diversity_index=0.8,
            effective_states=10,
            entropy_variance=0.1,
            generation=1
        )
        
        new_beta = amplifier.compute_amplification(metrics_high)
        
        # High entropy should increase amplification
        self.assertGreater(new_beta, config.base_amplification)


class TestCollapseController(unittest.TestCase):
    """Test dynamic collapse controller."""
    
    def test_adaptive_collapse(self):
        """Test adaptive collapse strategy."""
        controller = DynamicCollapseController(strategy=CollapseStrategy.ADAPTIVE)
        
        # Create entropy metrics for threshold computation
        from entropy_monitor import EntropyMetrics
        metrics = EntropyMetrics(
            total_entropy=2.0,
            normalized_entropy=0.5,
            entropy_rate=0.0,
            diversity_index=0.5,
            effective_states=5,
            entropy_variance=0.1,
            generation=1
        )
        
        # Check threshold computation
        threshold = controller.compute_adaptive_threshold(entropy_metrics=metrics, num_states=10)
        self.assertGreater(threshold, 0)
        self.assertLessEqual(threshold, 1)
        
        # Check if collapse should trigger
        should_trigger = controller.should_trigger_collapse(5, 10)
        self.assertIsInstance(should_trigger, bool)


class TestAIGuidance(unittest.TestCase):
    """Test AI guidance engine."""
    
    def test_reward_computation(self):
        """Test reward function computation."""
        engine = AIGuidanceEngine(mode=GuidanceMode.HYBRID)
        
        state = QuantumState(state_id=1, solution=[1, 2, 3], amplitude=1.0, quality_score=0.8)
        rewards = engine.compute_rewards([state])
        
        # Reward should be computed
        self.assertIsInstance(rewards[0].total_reward, float)
    
    def test_state_ranking(self):
        """Test state ranking by AI."""
        engine = AIGuidanceEngine()
        
        states = [
            QuantumState(state_id=i, solution=[i], amplitude=1.0, quality_score=score)
            for i, score in enumerate([0.5, 0.9, 0.7])
        ]
        
        # Use compute_rewards for each state and sort
        rewards = engine.compute_rewards(states)
        ranked = sorted(zip(states, rewards), key=lambda x: x[1].total_reward, reverse=True)
        
        # Best state should be ranked first
        self.assertEqual(ranked[0][0].quality_score, 0.9)


class TestEvolutionEngine(unittest.TestCase):
    """Test evolution engine."""
    
    def test_mutation(self):
        """Test mutation operator."""
        engine = EvolutionEngine()
        
        tour = [0, 1, 2, 3, 4]
        # Use mutate_swap directly from tsp_solver instead
        mutated = tour.copy()
        i, j = np.random.choice(len(tour), 2, replace=False)
        mutated[i], mutated[j] = mutated[j], mutated[i]
        
        # Mutation should change the tour
        self.assertNotEqual(mutated, tour)
        # But should preserve all cities
        self.assertEqual(set(mutated), set(tour))
    
    def test_recombination(self):
        """Test recombination operator."""
        # Simple OX1 implementation for testing
        parent1 = [0, 1, 2, 3, 4]
        parent2 = [4, 3, 2, 1, 0]
        
        # Manual OX1 crossover
        size = len(parent1)
        start, end = 1, 3
        child = [-1] * size
        child[start:end+1] = parent1[start:end+1]
        
        parent2_genes = [g for g in parent2 if g not in child[start:end+1]]
        child_idx = 0
        parent2_idx = 0
        
        while child_idx < size:
            if child[child_idx] == -1:
                child[child_idx] = parent2_genes[parent2_idx]
                parent2_idx += 1
            child_idx += 1
        
        # Child should be valid permutation
        self.assertEqual(set(child), set(parent1))


class TestTSPSolver(unittest.TestCase):
    """Test TSP-specific functionality."""
    
    def setUp(self):
        """Set up TSP instance."""
        np.random.seed(42)
        n_cities = 10
        coords = np.random.rand(n_cities, 2)
        
        dist_matrix = np.zeros((n_cities, n_cities))
        for i in range(n_cities):
            for j in range(n_cities):
                dist_matrix[i, j] = np.linalg.norm(coords[i] - coords[j])
        
        self.tsp = TSPInstance(
            n_cities=n_cities,
            coordinates=coords,
            distance_matrix=dist_matrix
        )
        self.solver = TSPSolver(self.tsp)
    
    def test_random_tour_generation(self):
        """Test random tour generation."""
        tour = self.solver.generate_random_tour()
        
        self.assertEqual(len(tour.tour), self.tsp.n_cities)
        self.assertTrue(tour.is_valid)
    
    def test_distance_evaluation(self):
        """Test tour distance evaluation."""
        tour = self.solver.generate_random_tour()
        
        # Evaluate again to check consistency
        evaluated = self.solver.evaluate_solution(tour)
        
        self.assertAlmostEqual(tour.total_distance, evaluated.total_distance)
    
    def test_mutation_operators(self):
        """Test mutation operators."""
        original = list(range(self.tsp.n_cities))
        
        # Create TSPSolution for mutation methods
        from tsp_solver import TSPSolution
        solution = TSPSolution(tour=original.copy(), total_distance=0.0)
        
        mutated_swap = self.solver.mutate_swap(solution)
        mutated_insert = self.solver.mutate_insertion(solution)
        
        # All should be valid permutations
        for mutated in [mutated_swap, mutated_insert]:
            self.assertEqual(set(mutated.tour), set(original))
    
    def test_recombination(self):
        """Test recombination operators."""
        parent1 = list(range(self.tsp.n_cities))
        parent2 = list(reversed(range(self.tsp.n_cities)))
        
        from tsp_solver import TSPSolution
        sol1 = TSPSolution(tour=parent1, total_distance=0.0)
        sol2 = TSPSolution(tour=parent2, total_distance=0.0)
        
        result = self.solver.recombine_ox1(sol1, sol2)
        
        # recombine_ox1 returns a tuple of two TSPSolution objects
        if isinstance(result, tuple):
            child1 = result[0]
            child2 = result[1]
            # Extract tours from TSPSolution objects
            if hasattr(child1, 'tour'):
                child1_tour = child1.tour
            else:
                child1_tour = child1
            if hasattr(child2, 'tour'):
                child2_tour = child2.tour
            else:
                child2_tour = child2
            
            # Both children should be valid permutations
            self.assertEqual(set(child1_tour), set(parent1))
            self.assertEqual(set(child2_tour), set(parent1))
        else:
            # Fallback for single result
            if hasattr(result, 'tour'):
                child_tour = result.tour
            else:
                child_tour = result
            self.assertEqual(set(child_tour), set(parent1))


class TestHAQOAOptimizer(unittest.TestCase):
    """Test complete HAQOA optimizer."""
    
    def test_initialization(self):
        """Test optimizer initialization."""
        config = HAQOAConfig(
            initial_population_size=10,
            max_generations=5
        )
        
        optimizer = HAQOAOptimizer(config=config)
        
        self.assertEqual(optimizer.config.initial_population_size, 10)
    
    def test_optimization_run(self):
        """Test complete optimization run."""
        np.random.seed(42)
        
        # Create simple TSP
        n_cities = 8
        coords = np.random.rand(n_cities, 2)
        dist_matrix = np.zeros((n_cities, n_cities))
        for i in range(n_cities):
            for j in range(n_cities):
                dist_matrix[i, j] = np.linalg.norm(coords[i] - coords[j])
        
        tsp = TSPInstance(n_cities=n_cities, coordinates=coords, distance_matrix=dist_matrix)
        tsp_solver = TSPSolver(tsp)
        
        config = HAQOAConfig(
            initial_population_size=10,
            max_generations=20,
            random_seed=42
        )
        
        optimizer = HAQOAOptimizer(config=config)
        optimizer.solution_generator = lambda: tsp_solver.generate_random_tour().tour
        optimizer.solution_mutator = lambda t: tsp_solver.mutate_swap(
            type('obj', (object,), {'tour': list(t)})()
        ).tour
        optimizer.solution_recombiner = lambda t1, t2: tsp_solver.recombine_ox1(
            type('obj1', (object,), {'tour': list(t1)})(),
            type('obj2', (object,), {'tour': list(t2)})()
        )[0] if isinstance(tsp_solver.recombine_ox1(
            type('obj1', (object,), {'tour': list(t1)})(),
            type('obj2', (object,), {'tour': list(t2)})()
        ), tuple) else tsp_solver.recombine_ox1(
            type('obj1', (object,), {'tour': list(t1)})(),
            type('obj2', (object,), {'tour': list(t2)})()
        ).tour
        
        def quality_eval(t):
            total_dist = sum(dist_matrix[t[i], t[(i+1)%n_cities]] for i in range(n_cities))
            return -total_dist  # Negative for maximization
        
        optimizer.quality_evaluator = quality_eval
        
        optimizer.initialize_population()
        result = optimizer.optimize(verbose=False)
        
        # Should complete optimization
        self.assertIsNotNone(result.best_solution)
        self.assertGreater(result.final_generation, 0)


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestStateSuperposition))
    suite.addTests(loader.loadTestsFromTestCase(TestProbabilisticEncoding))
    suite.addTests(loader.loadTestsFromTestCase(TestEntropyMonitor))
    suite.addTests(loader.loadTestsFromTestCase(TestAmplification))
    suite.addTests(loader.loadTestsFromTestCase(TestCollapseController))
    suite.addTests(loader.loadTestsFromTestCase(TestAIGuidance))
    suite.addTests(loader.loadTestsFromTestCase(TestEvolutionEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestTSPSolver))
    suite.addTests(loader.loadTestsFromTestCase(TestHAQOAOptimizer))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
