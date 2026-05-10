"""
HAQOA — Hybrid AI-Assisted Quantum-Inspired Optimization Architecture
HAQOA-X — Hyper-Adaptive Extension (AQSE-v2)
"""
from .engine   import HAQOAEngine,  HAQOAConfig,  HAQOAResult,  QuantumState
from .engine_x import HAQOAXEngine, HAQOAXConfig, HAQOAXResult, QuantumStateX

__version__ = "0.3.0"
__all__ = [
    "HAQOAEngine",  "HAQOAConfig",  "HAQOAResult",  "QuantumState",
    "HAQOAXEngine", "HAQOAXConfig", "HAQOAXResult", "QuantumStateX",
]
