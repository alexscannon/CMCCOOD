import torch
import torch.nn as nn
from timm import create_model
import logging

from ..base_model import BaseModel

logger = logging.getLogger(__name__)

class VisionModelAdapter(BaseModel):
    """Adapter for vision foundation models with continual learning support."""

    def __init__(self, model_name, pretrained=True, feature_dim=None, freeze_backbone=True):
        super().__init__()
        self.model_name = model_name

        # Create the foundation model from timm package
        self.backbone = create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0  # Remove classification head, allowing model to be used as a feature extractor
        )

        # Freeze backbone if specified and only train the adapter
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Get feature dimension from the model
        if feature_dim is None:
            # Auto-detect backbonefeature dimension
            with torch.no_grad():
                dummy_input = torch.zeros(1, 3, 224, 224)
                features = self.backbone(dummy_input)
                self.feature_dim = features.shape[1] # NUm of features model produces for each input image
                logger.info(f"Feature dimension: {self.feature_dim}")

        else:
            self.feature_dim = feature_dim

    def forward(self, x):
        """Extract features using the foundation model."""
        return self.backbone(x)