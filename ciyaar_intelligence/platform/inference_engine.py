"""Inference engine for quantum-neural models."""

import torch
from typing import Optional, Union, Tuple


class InferenceEngine:
    """Engine for running inference with quantum-neural models."""
    
    def __init__(self, config):
        self.config = config
        
    def predict(
        self,
        model: torch.nn.Module,
        input_data: torch.Tensor,
        return_uncertainty: bool = False,
        batch_size: Optional[int] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Run inference with a model."""
        
        model.eval()
        
        with torch.no_grad():
            if hasattr(model, 'forward') and return_uncertainty:
                try:
                    predictions, uncertainty = model(input_data, return_uncertainty=True)
                    return predictions, uncertainty
                except:
                    # Fallback if uncertainty not supported
                    predictions = model(input_data)
                    uncertainty = torch.zeros_like(predictions)
                    return predictions, uncertainty
            else:
                predictions = model(input_data)
                if return_uncertainty:
                    uncertainty = torch.zeros_like(predictions)
                    return predictions, uncertainty
                return predictions