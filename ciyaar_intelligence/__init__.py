"""
Ciyaar Intelligence: Universal Quantum-Neural AI Platform
========================================================

A next-generation AI platform combining quantum computing and neural networks
for complex problem solving across industries.

Key Features:
- Quantum-enhanced neural architectures
- Industry-specific optimization modules
- High-performance distributed computing
- Scalable and modular design
"""

__version__ = "1.0.0"
__author__ = "Ciyaar Intelligence Team"
__email__ = "contact@ciyaar-intelligence.com"

from .core import QuantumNeuralNetwork, HybridOptimizer
from .quantum import QuantumLayer, QuantumCircuitBuilder
from .neural import AdaptiveArchitecture, IndustrySpecificModules
from .optimization import EvolutionaryOptimizer, QuantumAnnealing
from .platform import CiyaarPlatform, ModelRegistry

__all__ = [
    "QuantumNeuralNetwork",
    "HybridOptimizer", 
    "QuantumLayer",
    "QuantumCircuitBuilder",
    "AdaptiveArchitecture",
    "IndustrySpecificModules",
    "EvolutionaryOptimizer",
    "QuantumAnnealing",
    "CiyaarPlatform",
    "ModelRegistry",
]