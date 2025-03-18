import torch.nn as nn

class ClassificationHead(nn.Module):
    """Classification head for continual learning."""

    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, features):
        """Classify features into classes."""
        return self.classifier(features)