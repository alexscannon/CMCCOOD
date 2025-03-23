import torch
import logging
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

def load_scheduler(config: DictConfig, optimizer):
    """
    Load and configure a learning rate scheduler based on the provided configuration.

    Args:
        config: The configuration containing scheduler settings
        optimizer: The optimizer to which the scheduler will be applied

    Returns:
        The configured scheduler, or None if no scheduler is specified
    """
    scheduler_config = config.training.scheduler
    logger.info(f"Creating scheduler: {scheduler_config.name}...")

    if scheduler_config.name.lower() == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config.t_max
        )
    elif scheduler_config.name.lower() == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config.step_size,
            gamma=scheduler_config.gamma
        )
    elif scheduler_config.name.lower() == "none":
        scheduler = None
        logger.info("No scheduler will be used")
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_config.name}")

    return scheduler