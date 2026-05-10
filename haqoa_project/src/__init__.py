"""
HAQOA - Hybrid AI-Assisted Quantum-Inspired Optimization Architecture

Core module for adaptive quantum-inspired state evolution engine (AQSE-v1).
Implements entropy-regulated probabilistic optimization for combinatorial problems.
"""

__version__ = "1.0.0"
__author__ = "HAQOA Research Team"

from .state_superposition import StateSuperpositionGenerator
from .probabilistic_encoding import ProbabilisticStateEncoder
from .entropy_monitor import EntropyMonitoringSystem
from .amplification import AdaptiveProbabilityAmplifier
from .collapse_controller import DynamicCollapseController
from .ai_guidance import AIGuidanceEngine
from .evolution import EvolutionEngine
from .haqoa_engine import HAQOAOptimizer

__all__ = [
    "StateSuperpositionGenerator",
    "ProbabilisticStateEncoder", 
    "EntropyMonitoringSystem",
    "AdaptiveProbabilityAmplifier",
    "DynamicCollapseController",
    "AIGuidanceEngine",
    "EvolutionEngine",
    "HAQOAOptimizer",
]