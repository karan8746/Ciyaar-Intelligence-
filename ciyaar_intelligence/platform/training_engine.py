"""Training engine for quantum-neural models."""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
from loguru import logger
from ..core import HybridOptimizer, OptimizerConfig


class TrainingEngine:
    """Engine for training quantum-neural models."""
    
    def __init__(self, config):
        self.config = config
        
    def train(
        self,
        model: torch.nn.Module,
        train_data: Tuple[torch.Tensor, torch.Tensor],
        validation_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        epochs: int = 100,
        optimizer_config: Optional[OptimizerConfig] = None
    ) -> Dict[str, Any]:
        """Train a model."""
        
        X_train, y_train = train_data
        optimizer = HybridOptimizer(model.parameters(), optimizer_config, model)
        
        best_val_loss = float('inf')
        training_history = []
        
        model.train()
        
        for epoch in range(epochs):
            # Training step
            optimizer.zero_grad()
            predictions = model(X_train)
            loss = F.mse_loss(predictions, y_train)
            loss.backward()
            
            step_metrics = optimizer.step(loss)
            
            # Validation
            if validation_data:
                model.eval()
                with torch.no_grad():
                    X_val, y_val = validation_data
                    val_predictions = model(X_val)
                    val_loss = F.mse_loss(val_predictions, y_val)
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss.item()
                
                model.train()
                step_metrics['validation_loss'] = val_loss.item()
            
            training_history.append(step_metrics)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Loss {loss.item():.6f}")
        
        return {
            'training_history': training_history,
            'best_validation_loss': best_val_loss,
            'final_loss': loss.item(),
            'epochs_completed': epochs
        }