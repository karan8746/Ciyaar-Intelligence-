"""
Ciyaar Intelligence Platform
===========================

Main platform class that orchestrates quantum-neural AI capabilities
across different industries and applications.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any, Union
from loguru import logger
from dataclasses import dataclass
import time
from pathlib import Path

from ..core import QuantumNeuralNetwork, HybridOptimizer, NetworkConfig, OptimizerConfig
from .model_registry import ModelRegistry
from .training_engine import TrainingEngine
from .inference_engine import InferenceEngine


@dataclass
class PlatformConfig:
    """Configuration for the Ciyaar Platform."""
    
    model_dir: str = "./models"
    cache_dir: str = "./cache"
    log_level: str = "INFO"
    auto_optimize: bool = True
    industry_mode: Optional[str] = None
    distributed: bool = False
    gpu_acceleration: bool = True


class CiyaarPlatform:
    """
    Universal Quantum-Neural AI Platform.
    
    Main interface for the Ciyaar Intelligence system that provides
    unified access to quantum-enhanced AI capabilities across industries.
    
    Features:
    - Automated model selection and optimization
    - Industry-specific adaptations
    - Quantum-classical hybrid processing
    - Distributed training and inference
    - Real-time performance monitoring
    """
    
    def __init__(self, config: Optional[PlatformConfig] = None):
        self.config = config or PlatformConfig()
        
        # Setup logging
        logger.remove()
        logger.add(
            sink=lambda msg: print(msg, end=""),
            level=self.config.log_level,
            format="<green>{time}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        )
        
        # Create directories
        Path(self.config.model_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize core components
        self.model_registry = ModelRegistry(self.config.model_dir)
        self.training_engine = TrainingEngine(self.config)
        self.inference_engine = InferenceEngine(self.config)
        
        # Platform state
        self.active_models = {}
        self.performance_history = []
        self.industry_adaptations = {}
        
        logger.info("Ciyaar Intelligence Platform initialized")
        logger.info(f"Platform configuration: {self.config}")
    
    def create_model(
        self,
        model_name: str,
        input_dim: int,
        output_dim: int,
        industry_mode: Optional[str] = None,
        network_config: Optional[NetworkConfig] = None
    ) -> QuantumNeuralNetwork:
        """
        Create a new quantum-neural model.
        
        Args:
            model_name: Unique identifier for the model
            input_dim: Input dimension
            output_dim: Output dimension
            industry_mode: Industry specialization mode
            network_config: Network configuration
            
        Returns:
            model: Created quantum-neural network
        """
        logger.info(f"Creating model '{model_name}' for industry: {industry_mode}")
        
        # Use platform industry mode if not specified
        if industry_mode is None:
            industry_mode = self.config.industry_mode
        
        # Create network configuration
        if network_config is None:
            network_config = NetworkConfig(
                use_gpu=self.config.gpu_acceleration,
                mixed_precision=self.config.gpu_acceleration
            )
        
        # Create model
        model = QuantumNeuralNetwork(
            input_dim=input_dim,
            output_dim=output_dim,
            config=network_config,
            industry_mode=industry_mode
        )
        
        # Register model
        self.model_registry.register_model(model_name, model, {
            'input_dim': input_dim,
            'output_dim': output_dim,
            'industry_mode': industry_mode,
            'created_at': time.time()
        })
        
        # Store in active models
        self.active_models[model_name] = model
        
        logger.info(f"Model '{model_name}' created successfully")
        return model
    
    def train_model(
        self,
        model_name: str,
        train_data: torch.Tensor,
        train_targets: torch.Tensor,
        validation_data: Optional[torch.Tensor] = None,
        validation_targets: Optional[torch.Tensor] = None,
        epochs: int = 100,
        optimizer_config: Optional[OptimizerConfig] = None
    ) -> Dict[str, Any]:
        """
        Train a quantum-neural model.
        
        Args:
            model_name: Name of the model to train
            train_data: Training input data
            train_targets: Training target data
            validation_data: Validation input data
            validation_targets: Validation target data
            epochs: Number of training epochs
            optimizer_config: Optimizer configuration
            
        Returns:
            training_results: Training metrics and statistics
        """
        if model_name not in self.active_models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model = self.active_models[model_name]
        
        logger.info(f"Starting training for model '{model_name}'")
        logger.info(f"Training data shape: {train_data.shape}")
        logger.info(f"Training targets shape: {train_targets.shape}")
        
        # Prepare validation data
        val_data = (validation_data, validation_targets) if validation_data is not None else None
        
        # Train model
        training_results = self.training_engine.train(
            model=model,
            train_data=(train_data, train_targets),
            validation_data=val_data,
            epochs=epochs,
            optimizer_config=optimizer_config
        )
        
        # Update model registry
        self.model_registry.update_model_metadata(model_name, {
            'last_trained': time.time(),
            'training_results': training_results,
            'performance_score': training_results.get('best_validation_loss', float('inf'))
        })
        
        # Track performance
        self.performance_history.append({
            'model_name': model_name,
            'timestamp': time.time(),
            'results': training_results
        })
        
        logger.info(f"Training completed for model '{model_name}'")
        return training_results
    
    def predict(
        self,
        model_name: str,
        input_data: torch.Tensor,
        return_uncertainty: bool = False,
        batch_size: Optional[int] = None
    ) -> Union[torch.Tensor, tuple]:
        """
        Make predictions using a trained model.
        
        Args:
            model_name: Name of the model to use
            input_data: Input data for prediction
            return_uncertainty: Whether to return uncertainty estimates
            batch_size: Batch size for inference
            
        Returns:
            predictions: Model predictions (and uncertainty if requested)
        """
        if model_name not in self.active_models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model = self.active_models[model_name]
        
        logger.info(f"Making predictions with model '{model_name}'")
        logger.info(f"Input data shape: {input_data.shape}")
        
        # Use inference engine
        predictions = self.inference_engine.predict(
            model=model,
            input_data=input_data,
            return_uncertainty=return_uncertainty,
            batch_size=batch_size
        )
        
        logger.info(f"Predictions completed for {input_data.shape[0]} samples")
        return predictions
    
    def optimize_for_industry(
        self,
        model_name: str,
        industry: str,
        sample_data: torch.Tensor,
        target_performance: float = 0.95
    ) -> Dict[str, Any]:
        """
        Optimize model for specific industry requirements.
        
        Args:
            model_name: Name of the model to optimize
            industry: Target industry ('finance', 'healthcare', 'manufacturing', 'research')
            sample_data: Representative data for the industry
            target_performance: Target performance threshold
            
        Returns:
            optimization_results: Results of the optimization
        """
        if model_name not in self.active_models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model = self.active_models[model_name]
        
        logger.info(f"Optimizing model '{model_name}' for industry: {industry}")
        
        # Industry-specific optimization
        optimization_results = {}
        
        # Optimize quantum depth
        if hasattr(model, 'optimize_quantum_depth'):
            depth_results = model.optimize_quantum_depth(sample_data, target_performance)
            optimization_results['quantum_depth'] = depth_results
        
        # Optimize adaptive architecture
        if hasattr(model, 'adaptive_arch'):
            arch_results = model.adaptive_arch.optimize_for_task(sample_data)
            optimization_results['adaptive_architecture'] = arch_results
        
        # Industry-specific module optimization
        if hasattr(model, 'industry_modules'):
            industry_results = model.industry_modules.optimize_for_industry(industry, sample_data)
            optimization_results['industry_modules'] = industry_results
        
        # Store optimization results
        self.industry_adaptations[model_name] = {
            'industry': industry,
            'optimization_results': optimization_results,
            'timestamp': time.time()
        }
        
        # Update model registry
        self.model_registry.update_model_metadata(model_name, {
            'industry_optimized': industry,
            'optimization_timestamp': time.time(),
            'optimization_results': optimization_results
        })
        
        logger.info(f"Industry optimization completed for model '{model_name}'")
        return optimization_results
    
    def get_model_performance(self, model_name: str) -> Dict[str, Any]:
        """Get comprehensive performance metrics for a model."""
        if model_name not in self.active_models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model = self.active_models[model_name]
        
        # Get model statistics
        performance = {
            'model_name': model_name,
            'architecture_summary': model.get_architecture_summary(),
            'training_stats': model.get_training_stats() if hasattr(model, 'get_training_stats') else {},
            'quantum_advantage_score': model.quantum_advantage_score() if hasattr(model, 'quantum_advantage_score') else 0.0,
        }
        
        # Get registry metadata
        metadata = self.model_registry.get_model_metadata(model_name)
        if metadata:
            performance['metadata'] = metadata
        
        # Get industry adaptation info
        if model_name in self.industry_adaptations:
            performance['industry_adaptation'] = self.industry_adaptations[model_name]
        
        return performance
    
    def list_models(self) -> List[str]:
        """List all registered models."""
        return list(self.model_registry.models.keys())
    
    def save_model(self, model_name: str, path: Optional[str] = None) -> str:
        """Save a model to disk."""
        if model_name not in self.active_models:
            raise ValueError(f"Model '{model_name}' not found")
        
        if path is None:
            path = f"{self.config.model_dir}/{model_name}.pt"
        
        model = self.active_models[model_name]
        model.save_checkpoint(path, include_stats=True)
        
        logger.info(f"Model '{model_name}' saved to {path}")
        return path
    
    def load_model(self, model_name: str, path: str) -> QuantumNeuralNetwork:
        """Load a model from disk."""
        logger.info(f"Loading model '{model_name}' from {path}")
        
        model = QuantumNeuralNetwork.load_checkpoint(path)
        
        # Register loaded model
        self.active_models[model_name] = model
        
        logger.info(f"Model '{model_name}' loaded successfully")
        return model
    
    def get_platform_stats(self) -> Dict[str, Any]:
        """Get comprehensive platform statistics."""
        stats = {
            'platform_config': self.config,
            'active_models': len(self.active_models),
            'total_registered_models': len(self.model_registry.models),
            'performance_history_length': len(self.performance_history),
            'industry_adaptations': len(self.industry_adaptations),
        }
        
        # Model-specific stats
        model_stats = {}
        for name, model in self.active_models.items():
            model_stats[name] = {
                'parameters': sum(p.numel() for p in model.parameters()),
                'device': str(next(model.parameters()).device),
                'industry_mode': getattr(model, 'industry_mode', None)
            }
        
        stats['model_stats'] = model_stats
        
        # Performance summary
        if self.performance_history:
            recent_performance = self.performance_history[-10:]  # Last 10 training runs
            avg_performance = np.mean([
                p['results'].get('best_validation_loss', float('inf')) 
                for p in recent_performance
            ])
            stats['avg_recent_performance'] = avg_performance
        
        return stats
    
    def benchmark_quantum_advantage(
        self,
        model_name: str,
        benchmark_data: torch.Tensor,
        classical_baseline: Optional[torch.nn.Module] = None
    ) -> Dict[str, float]:
        """
        Benchmark quantum advantage compared to classical approach.
        
        Args:
            model_name: Name of the quantum model
            benchmark_data: Data for benchmarking
            classical_baseline: Classical model for comparison
            
        Returns:
            benchmark_results: Quantum vs classical performance comparison
        """
        if model_name not in self.active_models:
            raise ValueError(f"Model '{model_name}' not found")
        
        quantum_model = self.active_models[model_name]
        
        logger.info(f"Benchmarking quantum advantage for model '{model_name}'")
        
        # Quantum model performance
        start_time = time.time()
        with torch.no_grad():
            quantum_output = quantum_model(benchmark_data)
            quantum_advantage_score = quantum_model.quantum_advantage_score() if hasattr(quantum_model, 'quantum_advantage_score') else 0.0
        quantum_time = time.time() - start_time
        
        # Classical baseline comparison
        classical_time = 0.0
        performance_ratio = 1.0
        
        if classical_baseline:
            start_time = time.time()
            with torch.no_grad():
                classical_output = classical_baseline(benchmark_data)
            classical_time = time.time() - start_time
            
            # Compare performance (simplified metric)
            quantum_loss = torch.nn.functional.mse_loss(quantum_output, benchmark_data[:, :quantum_output.size(1)])
            classical_loss = torch.nn.functional.mse_loss(classical_output, benchmark_data[:, :classical_output.size(1)])
            performance_ratio = classical_loss.item() / max(quantum_loss.item(), 1e-8)
        
        benchmark_results = {
            'quantum_advantage_score': quantum_advantage_score,
            'quantum_inference_time': quantum_time,
            'classical_inference_time': classical_time,
            'speed_ratio': classical_time / max(quantum_time, 1e-8),
            'performance_ratio': performance_ratio,
            'overall_advantage': (performance_ratio + quantum_advantage_score) / 2.0
        }
        
        logger.info(f"Quantum advantage benchmark completed: {benchmark_results['overall_advantage']:.3f}")
        return benchmark_results
    
    def shutdown(self) -> None:
        """Gracefully shutdown the platform."""
        logger.info("Shutting down Ciyaar Intelligence Platform")
        
        # Save all active models
        for model_name in self.active_models:
            try:
                self.save_model(model_name)
            except Exception as e:
                logger.warning(f"Failed to save model '{model_name}': {e}")
        
        # Clear active models
        self.active_models.clear()
        
        logger.info("Platform shutdown completed")