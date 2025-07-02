"""
Hybrid Quantum-Classical Optimizer
==================================

Advanced optimizer combining classical gradient-based methods with 
quantum-inspired optimization techniques for enhanced convergence.
"""

import torch
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional, Callable, Any
from loguru import logger
import time
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class OptimizerConfig:
    """Configuration for hybrid optimizer."""
    
    base_lr: float = 0.001
    quantum_lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 1e-4
    adaptive_lr: bool = True
    quantum_advantage_threshold: float = 0.1
    patience: int = 10
    min_lr: float = 1e-6
    max_grad_norm: float = 1.0


class HybridOptimizer:
    """
    Hybrid optimizer combining classical and quantum optimization.
    
    Features:
    - Adaptive learning rate based on quantum advantage
    - Quantum-inspired parameter updates
    - Advanced momentum with quantum corrections
    - Automatic convergence detection
    - Multi-objective optimization support
    """
    
    def __init__(
        self,
        parameters,
        config: Optional[OptimizerConfig] = None,
        quantum_model: Optional[Any] = None
    ):
        self.config = config or OptimizerConfig()
        self.quantum_model = quantum_model
        
        # Initialize classical optimizer
        self.classical_optimizer = optim.AdamW(
            parameters,
            lr=self.config.base_lr,
            weight_decay=self.config.weight_decay
        )
        
        # Quantum-inspired state
        self.quantum_state = {}
        self.parameter_groups = list(parameters) if not hasattr(parameters, '__iter__') else list(parameters)
        
        # Performance tracking
        self.optimization_history = {
            'loss': [],
            'quantum_advantage': [],
            'learning_rates': [],
            'convergence_metrics': []
        }
        
        # Adaptive learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.classical_optimizer,
            mode='min',
            factor=0.7,
            patience=self.config.patience,
            min_lr=self.config.min_lr,
            verbose=True
        )
        
        # Convergence detection
        self.best_loss = float('inf')
        self.plateau_count = 0
        self.converged = False
        
        logger.info("Hybrid Quantum-Classical Optimizer initialized")
    
    def zero_grad(self) -> None:
        """Clear gradients."""
        self.classical_optimizer.zero_grad()
    
    def step(self, loss: torch.Tensor, quantum_metrics: Optional[Dict] = None) -> Dict[str, float]:
        """
        Perform optimization step with quantum-classical hybrid approach.
        
        Args:
            loss: Current loss value
            quantum_metrics: Additional quantum metrics for optimization
            
        Returns:
            step_metrics: Dictionary of optimization metrics
        """
        step_start_time = time.time()
        
        # Compute quantum advantage if available
        quantum_advantage = 0.0
        if self.quantum_model and hasattr(self.quantum_model, 'quantum_advantage_score'):
            quantum_advantage = self.quantum_model.quantum_advantage_score()
        
        # Apply gradient clipping
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.parameter_groups, 
                self.config.max_grad_norm
            )
        
        # Quantum-inspired parameter updates
        if quantum_advantage > self.config.quantum_advantage_threshold:
            self._apply_quantum_corrections(quantum_advantage)
        
        # Classical optimization step
        self.classical_optimizer.step()
        
        # Update learning rate based on quantum metrics
        if self.config.adaptive_lr:
            self._update_adaptive_lr(loss.item(), quantum_advantage)
        
        # Track optimization metrics
        step_metrics = {
            'loss': loss.item(),
            'quantum_advantage': quantum_advantage,
            'learning_rate': self.get_current_lr(),
            'step_time': time.time() - step_start_time
        }
        
        # Update history
        for key, value in step_metrics.items():
            if key in self.optimization_history:
                self.optimization_history[key].append(value)
        
        # Check convergence
        self._check_convergence(loss.item())
        
        return step_metrics
    
    def _apply_quantum_corrections(self, quantum_advantage: float) -> None:
        """Apply quantum-inspired corrections to parameter updates."""
        correction_factor = min(1.0, quantum_advantage * self.config.quantum_lr)
        
        for param_group in self.classical_optimizer.param_groups:
            for param in param_group['params']:
                if param.grad is not None:
                    # Apply quantum-inspired momentum correction
                    param_id = id(param)
                    
                    if param_id not in self.quantum_state:
                        self.quantum_state[param_id] = {
                            'momentum': torch.zeros_like(param.data),
                            'quantum_phase': 0.0
                        }
                    
                    state = self.quantum_state[param_id]
                    
                    # Update quantum phase based on gradient history
                    state['quantum_phase'] += 0.1 * torch.norm(param.grad).item()
                    
                    # Quantum-inspired momentum update
                    quantum_momentum = correction_factor * torch.sin(state['quantum_phase']) * param.grad
                    state['momentum'] = (
                        self.config.momentum * state['momentum'] + 
                        quantum_momentum
                    )
                    
                    # Apply correction
                    param.data -= self.config.quantum_lr * state['momentum']
    
    def _update_adaptive_lr(self, current_loss: float, quantum_advantage: float) -> None:
        """Update learning rate based on performance metrics."""
        # Standard scheduler step
        self.scheduler.step(current_loss)
        
        # Quantum-enhanced learning rate adaptation
        if quantum_advantage > self.config.quantum_advantage_threshold:
            # Increase learning rate when quantum advantage is high
            lr_boost = 1.0 + 0.1 * quantum_advantage
            for param_group in self.classical_optimizer.param_groups:
                param_group['lr'] = min(
                    param_group['lr'] * lr_boost,
                    self.config.base_lr * 2.0
                )
    
    def _check_convergence(self, current_loss: float) -> None:
        """Check if optimization has converged."""
        if current_loss < self.best_loss - 1e-6:
            self.best_loss = current_loss
            self.plateau_count = 0
        else:
            self.plateau_count += 1
        
        # Check for convergence
        if self.plateau_count > self.config.patience * 2:
            self.converged = True
            logger.info(f"Optimization converged at loss: {self.best_loss:.6f}")
    
    def get_current_lr(self) -> float:
        """Get current learning rate."""
        return self.classical_optimizer.param_groups[0]['lr']
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        if not self.optimization_history['loss']:
            return {}
        
        stats = {
            'total_steps': len(self.optimization_history['loss']),
            'best_loss': self.best_loss,
            'current_lr': self.get_current_lr(),
            'converged': self.converged,
            'avg_quantum_advantage': np.mean(self.optimization_history['quantum_advantage']),
            'avg_step_time': np.mean(self.optimization_history.get('step_time', [0])),
        }
        
        # Convergence analysis
        if len(self.optimization_history['loss']) > 10:
            recent_losses = self.optimization_history['loss'][-10:]
            stats['loss_std_recent'] = np.std(recent_losses)
            stats['loss_trend'] = np.polyfit(range(10), recent_losses, 1)[0]
        
        return stats
    
    def save_state(self, path: str) -> None:
        """Save optimizer state."""
        state = {
            'classical_optimizer': self.classical_optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'quantum_state': self.quantum_state,
            'optimization_history': self.optimization_history,
            'config': self.config,
            'best_loss': self.best_loss,
            'converged': self.converged
        }
        torch.save(state, path)
        logger.info(f"Optimizer state saved to {path}")
    
    def load_state(self, path: str) -> None:
        """Load optimizer state."""
        state = torch.load(path)
        
        self.classical_optimizer.load_state_dict(state['classical_optimizer'])
        self.scheduler.load_state_dict(state['scheduler'])
        self.quantum_state = state['quantum_state']
        self.optimization_history = state['optimization_history']
        self.best_loss = state['best_loss']
        self.converged = state['converged']
        
        logger.info(f"Optimizer state loaded from {path}")
    
    def reset(self) -> None:
        """Reset optimizer to initial state."""
        # Reset classical optimizer
        for group in self.classical_optimizer.param_groups:
            group['lr'] = self.config.base_lr
        
        # Clear state
        self.classical_optimizer.state.clear()
        self.quantum_state.clear()
        self.optimization_history = {
            'loss': [],
            'quantum_advantage': [],
            'learning_rates': [],
            'convergence_metrics': []
        }
        
        self.best_loss = float('inf')
        self.plateau_count = 0
        self.converged = False
        
        logger.info("Optimizer reset to initial state")
    
    def suggest_hyperparameters(self, loss_history: List[float]) -> Dict[str, float]:
        """Suggest optimal hyperparameters based on training history."""
        if len(loss_history) < 10:
            return {}
        
        # Analyze convergence patterns
        loss_trend = np.polyfit(range(len(loss_history)), loss_history, 1)[0]
        loss_variance = np.var(loss_history[-20:]) if len(loss_history) >= 20 else np.var(loss_history)
        
        suggestions = {}
        
        # Learning rate suggestions
        if loss_trend > 0:  # Loss increasing
            suggestions['base_lr'] = self.config.base_lr * 0.5
        elif loss_variance < 1e-6:  # Very stable
            suggestions['base_lr'] = self.config.base_lr * 1.2
        
        # Quantum learning rate
        avg_quantum_advantage = np.mean(self.optimization_history.get('quantum_advantage', [0]))
        if avg_quantum_advantage > 0.5:
            suggestions['quantum_lr'] = self.config.quantum_lr * 1.5
        elif avg_quantum_advantage < 0.1:
            suggestions['quantum_lr'] = self.config.quantum_lr * 0.7
        
        return suggestions