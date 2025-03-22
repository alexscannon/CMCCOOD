import importlib
import torch
from typing import Any, Dict, Optional

from src import constants

def load_model(model_config, num_classes, device):
    """
    Creates and loads a model dynamically based on the provided configuration.

    Args:
        model_config: The model configuration from the config file.
        num_classes: The number of output classes.
        device: The device to load the model onto.

    Returns:
        A model instance loaded on the specified device.
    """
    # Get model type from config
    model_type = model_config.get('type', None)

    if model_type == constants.VISION_TRANSFORMER_CLASSIFICATION:
        model = _build_vision_classification_model(model_config, num_classes)
    elif model_type == 'custom':
        model = _load_custom_model(model_config, num_classes)
    else:
        # Dynamically load model builder based on type
        try:
            module_path = f"src.models.builders.{model_type}_builder"
            builder_module = importlib.import_module(module_path)
            build_model_func = getattr(builder_module, f"build_{model_type}_model")
            model = build_model_func(model_config, num_classes)
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Unsupported model type: {model_type}. Error: {str(e)}")

    # Move model to device
    model = model.to(device)
    return model

def _build_vision_classification_model(model_config, num_classes):
    """Build a standard vision classification model with backbone + classification head"""
    from src.models.foundation_models.vision_adapter import VisionModelAdapter
    from src.models.classification_head import ClassificationHead
    from src.models.agents.continual_learning_model import CLVisionTransformerModel

    # Create feature extractor
    feature_extractor = VisionModelAdapter(
        model_name=model_config.model_name,
        pretrained=model_config.pretrained,
        feature_dim=model_config.feature_dim,
        freeze_backbone=model_config.freeze_backbone
    )

    # Create classification head
    classification_head = ClassificationHead(
        feature_dim=feature_extractor.feature_dim,
        num_classes=num_classes
    )

    # Create the full model
    model = CLVisionTransformerModel(feature_extractor, classification_head)

    return model

def _load_custom_model(model_config, num_classes):
    """Load a custom model from a specified module and class"""

    # Check if required attributes exist in the config
    if not hasattr(model_config, 'module_path') or not hasattr(model_config, 'class_name'):
        raise ValueError("Custom model configuration is missing 'module_path' or 'class_name' attribute")

    module_path = model_config.module_path
    class_name = model_config.class_name

    try:
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)

        # Get constructor args from config
        model_args = model_config.get('args', {})
        if 'num_classes' not in model_args and num_classes is not None:
            model_args['num_classes'] = num_classes

        # Instantiate the model
        model = model_class(**model_args)
        return model

    except (ImportError, AttributeError) as e:
        raise ValueError(f"Failed to load custom model {class_name} from {module_path}. Error: {str(e)}")