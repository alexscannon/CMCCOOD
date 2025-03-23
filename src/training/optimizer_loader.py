import torch
import logging
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

def load_optimizer(config: DictConfig, model):
    """
    Load and configure an optimizer based on the provided configuration.

    Args:
        config: The configuration containing optimizer settings
        model: The PyTorch model whose parameters will be optimized

    Returns:
        The configured optimizer
    """
    optimizer_config = config.training.optimizer
    logger.info(f"Creating optimizer: {optimizer_config.name}...")

    # Determine which parameters to optimize
    if optimizer_config.optimize_all_params:
        params_to_optimize = model.parameters()
        logger.info("Optimizing all model parameters")
    else:
        params_to_optimize = model.classification_head.parameters()
        logger.info("Optimizing only classification head parameters")

    # Create optimizer dynamically based on configuration
    if optimizer_config.name.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=optimizer_config.learning_rate,
            weight_decay=optimizer_config.weight_decay
        )
    elif optimizer_config.name.lower() == "sgd":
        optimizer = torch.optim.SGD(
            params_to_optimize,
            lr=optimizer_config.learning_rate,
            momentum=optimizer_config.momentum,
            weight_decay=optimizer_config.weight_decay
        )
    elif optimizer_config.name.lower() == "adam":
        optimizer = torch.optim.Adam(
            params_to_optimize,
            lr=optimizer_config.learning_rate,
            weight_decay=optimizer_config.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_config.name}")

    return optimizer