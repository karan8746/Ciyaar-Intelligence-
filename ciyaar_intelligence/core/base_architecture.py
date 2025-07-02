"""Base architecture for quantum-neural networks."""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np
from loguru import logger


class BaseArchitecture(nn.Module, ABC):
    """
    Abstract base class for quantum-neural architectures.
    
    Provides common functionality including:
    - Performance monitoring
    - Model persistence
    - Architecture validation
    - Memory optimization
    """
    
    def __init__(self):
        super().__init__()
        self.training_metrics = {}
        self.validation_metrics = {}
        self._memory_optimized = False
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        pass
    
    def count_parameters(self) -> Dict[str, int]:
        """Count different types of parameters in the model."""
        total_params = 0
        trainable_params = 0
        
        for param in self.parameters():
            param_count = param.numel()
            total_params += param_count
            if param.requires_grad:
                trainable_params += param_count
        
        return {
            "total": total_params,
            "trainable": trainable_params,
            "non_trainable": total_params - trainable_params
        }
    
    def get_model_size(self) -> Dict[str, float]:
        """Get model size information in MB."""
        param_size = 0
        buffer_size = 0
        
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        return {
            "parameters_mb": param_size / 1024 / 1024,
            "buffers_mb": buffer_size / 1024 / 1024,
            "total_mb": (param_size + buffer_size) / 1024 / 1024
        }
    
    def enable_memory_optimization(self) -> None:
        """Enable memory optimization techniques."""
        if not self._memory_optimized:
            # Enable gradient checkpointing for memory efficiency
            for module in self.modules():
                if hasattr(module, 'gradient_checkpointing'):
                    module.gradient_checkpointing = True
            
            self._memory_optimized = True
            logger.info("Memory optimization enabled")
    
    def validate_architecture(self) -> Dict[str, bool]:
        """Validate the architecture configuration."""
        checks = {
            "has_parameters": len(list(self.parameters())) > 0,
            "parameters_finite": all(torch.isfinite(p).all() for p in self.parameters()),
            "gradients_enabled": any(p.requires_grad for p in self.parameters()),
        }
        
        # Check for potential gradient flow issues
        try:
            dummy_input = torch.randn(2, getattr(self, 'input_dim', 10))
            if hasattr(self, 'device'):
                dummy_input = dummy_input.to(self.device)
            
            output = self(dummy_input)
            checks["forward_pass"] = True
            checks["output_finite"] = torch.isfinite(output).all().item()
        except Exception as e:
            logger.warning(f"Forward pass validation failed: {e}")
            checks["forward_pass"] = False
            checks["output_finite"] = False
        
        return checks
    
    def get_architecture_summary(self) -> str:
        """Get a detailed summary of the architecture."""
        param_info = self.count_parameters()
        size_info = self.get_model_size()
        validation = self.validate_architecture()
        
        summary = f"""
Architecture Summary:
===================
Parameters: {param_info['total']:,} ({param_info['trainable']:,} trainable)
Model Size: {size_info['total_mb']:.2f} MB
Memory Optimized: {self._memory_optimized}
Validation Passed: {all(validation.values())}

Validation Details:
{validation}
"""
        return summary