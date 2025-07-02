"""
Adaptive Neural Architecture
===========================

Self-modifying neural architecture that adapts to different problem domains
and industry requirements for optimal performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from loguru import logger
from dataclasses import dataclass

from .industry_modules import IndustrySpecificModules
from .attention_quantum import QuantumAttention


@dataclass
class ArchitectureConfig:
    """Configuration for adaptive architecture."""
    
    base_dim: int = 256
    max_depth: int = 8
    min_depth: int = 2
    expansion_factor: float = 2.0
    compression_ratio: float = 0.5
    adaptation_threshold: float = 0.1
    pruning_threshold: float = 0.01
    growth_rate: float = 1.5


class AdaptiveBlock(nn.Module):
    """Self-adapting neural block with dynamic capacity."""
    
    def __init__(self, input_dim: int, config: ArchitectureConfig):
        super().__init__()
        self.input_dim = input_dim
        self.config = config
        
        # Core transformation layers
        self.core_layers = nn.ModuleList([
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        ])
        
        # Adaptive expansion layers
        self.expansion_layers = nn.ModuleList()
        self.can_expand = True
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            input_dim, num_heads=max(1, input_dim // 64), batch_first=True
        )
        
        # Gating mechanism for adaptive routing
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, 2),
            nn.Softmax(dim=-1)
        )
        
        # Performance tracking
        self.utilization_history = []
        self.complexity_score = 0.0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with adaptive routing."""
        batch_size, seq_len, dim = x.shape
        residual = x
        
        # Core processing
        for layer in self.core_layers:
            if isinstance(layer, nn.Linear):
                x = layer(x)
            elif isinstance(layer, nn.LayerNorm):
                x = layer(x)
            elif isinstance(layer, nn.ReLU):
                x = layer(x)
            elif isinstance(layer, nn.Dropout):
                x = layer(x) if self.training else x
        
        # Self-attention
        attn_output, attn_weights = self.attention(x, x, x)
        x = x + attn_output
        
        # Adaptive routing through expansion layers
        if self.expansion_layers:
            # Compute gating weights
            gate_weights = self.gate(x.mean(dim=1))  # Global average pooling
            
            # Route through expansion layers based on complexity
            for i, layer in enumerate(self.expansion_layers):
                if i < gate_weights.size(-1):
                    weight = gate_weights[:, i].unsqueeze(1).unsqueeze(2)
                    expansion_output = layer(x)
                    x = x + weight * expansion_output
        
        # Residual connection
        if x.shape == residual.shape:
            x = x + residual
        
        # Track utilization
        self._track_utilization(attn_weights)
        
        return x
    
    def _track_utilization(self, attn_weights: torch.Tensor) -> None:
        """Track layer utilization for adaptation decisions."""
        # Compute attention entropy as utilization measure
        attn_entropy = -torch.sum(attn_weights * torch.log(attn_weights + 1e-8), dim=-1)
        avg_entropy = attn_entropy.mean().item()
        
        self.utilization_history.append(avg_entropy)
        
        # Keep only recent history
        if len(self.utilization_history) > 100:
            self.utilization_history = self.utilization_history[-100:]
    
    def should_expand(self) -> bool:
        """Determine if block should expand based on utilization."""
        if not self.can_expand or len(self.utilization_history) < 10:
            return False
        
        recent_utilization = np.mean(self.utilization_history[-10:])
        return recent_utilization > self.config.adaptation_threshold
    
    def expand_capacity(self) -> None:
        """Add expansion layer to increase capacity."""
        if self.can_expand and len(self.expansion_layers) < 3:
            expansion_dim = int(self.input_dim * self.config.expansion_factor)
            
            expansion_layer = nn.Sequential(
                nn.Linear(self.input_dim, expansion_dim),
                nn.ReLU(),
                nn.Linear(expansion_dim, self.input_dim),
                nn.Dropout(0.1)
            )
            
            self.expansion_layers.append(expansion_layer)
            logger.info(f"Expanded block capacity: {len(self.expansion_layers)} expansion layers")
    
    def prune_parameters(self) -> int:
        """Prune low-importance parameters."""
        pruned_count = 0
        
        for layer in self.core_layers:
            if isinstance(layer, nn.Linear):
                with torch.no_grad():
                    # Magnitude-based pruning
                    weight_magnitude = torch.abs(layer.weight)
                    threshold = torch.quantile(weight_magnitude, self.config.pruning_threshold)
                    
                    mask = weight_magnitude > threshold
                    layer.weight.data *= mask.float()
                    
                    pruned_count += (~mask).sum().item()
        
        return pruned_count


class AdaptiveArchitecture(nn.Module):
    """
    Adaptive neural architecture that modifies itself based on task requirements.
    
    Features:
    - Dynamic depth adjustment
    - Capacity expansion/compression
    - Industry-specific specialization
    - Quantum-enhanced attention
    - Automatic optimization
    """
    
    def __init__(
        self,
        base_dim: int,
        industry_mode: Optional[str] = None,
        config: Optional[ArchitectureConfig] = None
    ):
        super().__init__()
        
        self.base_dim = base_dim
        self.industry_mode = industry_mode
        self.config = config or ArchitectureConfig(base_dim=base_dim)
        
        # Initialize with minimal architecture
        self.blocks = nn.ModuleList([
            AdaptiveBlock(base_dim, self.config) for _ in range(self.config.min_depth)
        ])
        
        # Industry-specific modules
        self.industry_modules = IndustrySpecificModules(
            base_dim=base_dim,
            industry_mode=industry_mode
        )
        
        # Quantum-enhanced attention
        self.quantum_attention = QuantumAttention(
            embed_dim=base_dim,
            num_heads=max(1, base_dim // 64)
        )
        
        # Adaptive output projection
        self.output_projection = nn.Sequential(
            nn.Linear(base_dim, base_dim),
            nn.LayerNorm(base_dim),
            nn.ReLU()
        )
        
        # Architecture controller
        self.controller = nn.LSTM(
            input_size=base_dim + 16,  # Features + metadata
            hidden_size=128,
            num_layers=2,
            batch_first=True
        )
        
        # Adaptation tracking
        self.adaptation_history = []
        self.performance_metrics = {}
        
        logger.info(f"Adaptive architecture initialized: {len(self.blocks)} blocks, industry_mode={industry_mode}")
    
    @property
    def output_dim(self) -> int:
        """Get output dimension after industry-specific processing."""
        return self.industry_modules.output_dim
    
    def forward(self, x: torch.Tensor, adapt: bool = True) -> torch.Tensor:
        """
        Forward pass with optional adaptation.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)
            adapt: Whether to perform architecture adaptation
            
        Returns:
            output: Processed tensor
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
        
        # Process through adaptive blocks
        for i, block in enumerate(self.blocks):
            x = block(x)
            
            # Adaptive growth during training
            if adapt and self.training and block.should_expand():
                block.expand_capacity()
        
        # Quantum-enhanced attention
        x = self.quantum_attention(x)
        
        # Industry-specific processing
        x = self.industry_modules(x)
        
        # Output projection
        x = self.output_projection(x)
        
        # Architecture adaptation
        if adapt and self.training:
            self._adapt_architecture(x)
        
        return x.squeeze(1) if x.size(1) == 1 else x
    
    def _adapt_architecture(self, x: torch.Tensor) -> None:
        """Adapt architecture based on current performance."""
        # Collect performance features
        features = self._extract_performance_features(x)
        
        # Run architecture controller
        controller_input = torch.cat([
            x.mean(dim=1),  # Global average pooling
            features.unsqueeze(1).expand(x.size(0), -1)
        ], dim=-1)
        
        controller_output, _ = self.controller(controller_input.unsqueeze(1))
        adaptation_signal = torch.sigmoid(controller_output.squeeze(1))
        
        # Make adaptation decisions
        avg_signal = adaptation_signal.mean(dim=0)
        
        # Depth adaptation
        if avg_signal[0] > 0.7 and len(self.blocks) < self.config.max_depth:
            self._add_block()
        elif avg_signal[0] < 0.3 and len(self.blocks) > self.config.min_depth:
            self._remove_block()
        
        # Capacity adaptation
        if avg_signal[1] > 0.6:
            self._expand_capacity()
        elif avg_signal[1] < 0.4:
            self._compress_capacity()
    
    def _extract_performance_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features for architecture adaptation."""
        features = []
        
        # Statistical features
        features.extend([
            x.mean().item(),
            x.std().item(),
            x.min().item(),
            x.max().item()
        ])
        
        # Gradient-based features
        if x.requires_grad and x.grad is not None:
            features.extend([
                x.grad.norm().item(),
                x.grad.mean().item(),
                x.grad.std().item()
            ])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # Block utilization features
        block_utilizations = []
        for block in self.blocks:
            if block.utilization_history:
                block_utilizations.append(np.mean(block.utilization_history[-5:]))
            else:
                block_utilizations.append(0.0)
        
        # Pad to fixed size
        while len(block_utilizations) < 8:
            block_utilizations.append(0.0)
        features.extend(block_utilizations[:8])
        
        # Industry-specific features
        if self.industry_mode:
            features.append(1.0)  # Industry mode active
        else:
            features.append(0.0)
        
        # Ensure exactly 16 features
        while len(features) < 16:
            features.append(0.0)
        
        return torch.tensor(features[:16], dtype=torch.float32, device=x.device)
    
    def _add_block(self) -> None:
        """Add a new adaptive block."""
        new_block = AdaptiveBlock(self.base_dim, self.config)
        self.blocks.append(new_block)
        
        # Move to same device as existing blocks
        if self.blocks:
            device = next(self.blocks[0].parameters()).device
            new_block.to(device)
        
        self.adaptation_history.append(('add_block', len(self.blocks)))
        logger.info(f"Added block: {len(self.blocks)} total blocks")
    
    def _remove_block(self) -> None:
        """Remove the least utilized block."""
        if len(self.blocks) <= self.config.min_depth:
            return
        
        # Find least utilized block
        min_utilization = float('inf')
        min_idx = 0
        
        for i, block in enumerate(self.blocks):
            if block.utilization_history:
                avg_utilization = np.mean(block.utilization_history[-10:])
                if avg_utilization < min_utilization:
                    min_utilization = avg_utilization
                    min_idx = i
        
        # Remove block
        del self.blocks[min_idx]
        self.adaptation_history.append(('remove_block', len(self.blocks)))
        logger.info(f"Removed block: {len(self.blocks)} total blocks")
    
    def _expand_capacity(self) -> None:
        """Expand capacity of existing blocks."""
        for block in self.blocks:
            if block.should_expand():
                block.expand_capacity()
        
        self.adaptation_history.append(('expand_capacity', len(self.blocks)))
    
    def _compress_capacity(self) -> None:
        """Compress capacity by pruning parameters."""
        total_pruned = 0
        for block in self.blocks:
            total_pruned += block.prune_parameters()
        
        if total_pruned > 0:
            self.adaptation_history.append(('compress_capacity', total_pruned))
            logger.info(f"Pruned {total_pruned} parameters")
    
    def get_adaptation_stats(self) -> Dict[str, Any]:
        """Get comprehensive adaptation statistics."""
        stats = {
            'num_blocks': len(self.blocks),
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'adaptation_history': self.adaptation_history[-20:],  # Recent history
        }
        
        # Block-level statistics
        block_stats = []
        for i, block in enumerate(self.blocks):
            block_stat = {
                'index': i,
                'expansion_layers': len(block.expansion_layers),
                'avg_utilization': np.mean(block.utilization_history[-10:]) if block.utilization_history else 0.0,
                'complexity_score': block.complexity_score
            }
            block_stats.append(block_stat)
        
        stats['block_stats'] = block_stats
        
        # Industry module stats
        stats['industry_mode'] = self.industry_mode
        stats['industry_output_dim'] = self.output_dim
        
        return stats
    
    def optimize_for_task(self, task_data: torch.Tensor, target_complexity: float = 0.5) -> Dict[str, Any]:
        """
        Optimize architecture for specific task.
        
        Args:
            task_data: Representative data for the task
            target_complexity: Target complexity score (0-1)
            
        Returns:
            optimization_results: Results of the optimization
        """
        original_blocks = len(self.blocks)
        
        # Analyze task characteristics
        with torch.no_grad():
            task_output = self(task_data, adapt=False)
            task_complexity = self._compute_task_complexity(task_data, task_output)
        
        # Adjust architecture based on task complexity
        if task_complexity > target_complexity:
            # Task is complex, may need more capacity
            while len(self.blocks) < self.config.max_depth and task_complexity > target_complexity * 1.2:
                self._add_block()
                with torch.no_grad():
                    task_output = self(task_data, adapt=False)
                    task_complexity = self._compute_task_complexity(task_data, task_output)
        else:
            # Task is simple, can reduce capacity
            while len(self.blocks) > self.config.min_depth and task_complexity < target_complexity * 0.8:
                self._remove_block()
                with torch.no_grad():
                    task_output = self(task_data, adapt=False)
                    task_complexity = self._compute_task_complexity(task_data, task_output)
        
        results = {
            'original_blocks': original_blocks,
            'optimized_blocks': len(self.blocks),
            'task_complexity': task_complexity,
            'target_complexity': target_complexity,
            'optimization_ratio': len(self.blocks) / original_blocks
        }
        
        logger.info(f"Task optimization: {original_blocks} -> {len(self.blocks)} blocks")
        return results
    
    def _compute_task_complexity(self, inputs: torch.Tensor, outputs: torch.Tensor) -> float:
        """Compute complexity score for the current task."""
        # Input complexity
        input_entropy = -torch.sum(F.softmax(inputs.flatten(), dim=0) * F.log_softmax(inputs.flatten(), dim=0))
        
        # Output complexity
        output_entropy = -torch.sum(F.softmax(outputs.flatten(), dim=0) * F.log_softmax(outputs.flatten(), dim=0))
        
        # Normalize to 0-1 range
        max_entropy = torch.log(torch.tensor(float(inputs.numel())))
        complexity = (input_entropy + output_entropy) / (2 * max_entropy)
        
        return complexity.item()
    
    def reset_adaptation(self) -> None:
        """Reset architecture to initial state."""
        # Reset to minimum depth
        while len(self.blocks) > self.config.min_depth:
            self.blocks.pop()
        
        # Reset blocks
        for block in self.blocks:
            block.expansion_layers.clear()
            block.utilization_history.clear()
            block.complexity_score = 0.0
        
        # Clear history
        self.adaptation_history.clear()
        
        logger.info("Architecture reset to initial state")