import torch.nn as nn

class CLVisionTransformerModel(nn.Module):
    """
    A general continual learning model that combines a feature extractor and a classification head.

    Attributes:
        feature_extractor: The backbone model used to extract features.
        classification_head: The classification head used to classify the features.
    """

    def __init__(self, feature_extractor, classification_head):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.classification_head = classification_head

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classification_head(features)