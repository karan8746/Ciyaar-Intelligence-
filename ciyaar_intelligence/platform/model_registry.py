"""Model registry for managing AI models."""

import torch
from typing import Dict, Any, Optional
from loguru import logger


class ModelRegistry:
    """Registry for managing quantum-neural models."""
    
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.models = {}
        self.metadata = {}
        
    def register_model(self, name: str, model: torch.nn.Module, metadata: Dict[str, Any]) -> None:
        """Register a new model."""
        self.models[name] = model
        self.metadata[name] = metadata
        logger.info(f"Model '{name}' registered")
        
    def get_model_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """Get model metadata."""
        return self.metadata.get(name)
        
    def update_model_metadata(self, name: str, updates: Dict[str, Any]) -> None:
        """Update model metadata."""
        if name in self.metadata:
            self.metadata[name].update(updates)