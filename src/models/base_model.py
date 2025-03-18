import torch.nn as nn

class BaseModel(nn.Module):
    """Base class for all models in the framework."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """Forward pass of the model."""
        raise NotImplementedError("Subclasses must implement this method")

    def get_params(self):
        """Get model parameters for optimization."""
        return self.parameters()