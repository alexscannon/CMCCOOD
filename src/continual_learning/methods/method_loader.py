import logging
from omegaconf import DictConfig

from src.models.base_model import BaseModel
from src.continual_learning.methods.replay.er import ExperienceReplay
# Import other CL methods as needed

logger = logging.getLogger(__name__)

def load_cl_method(config: DictConfig, model: BaseModel, scenario):
    """
    Dynamically loads a continual learning method based on configuration.

    Args:
        cl_method_config: Configuration for the continual learning method
        model: The model to be used
        scenario: The continual learning scenario
        device: The device to run on

    Returns:
        A continual learning method instance
    """
    cl_method_config = config.continual_learning.method
    print(cl_method_config)
    method_name = cl_method_config.name.lower()

    if method_name == "experience_replay":
        return ExperienceReplay(
            model=model,
            scenario=scenario,
            memory_size=cl_method_config.memory_size,
            selection_strategy=cl_method_config.selection_strategy,
            batch_size_memory=cl_method_config.batch_size_memory
        )
    # Add more methods as needed
    # elif method_name == "ewc":
    #     return ElasticWeightConsolidation(...)
    # elif method_name == "gem":
    #     return GradientEpisodicMemory(...)
    else:
        raise ValueError(f"Unknown continual learning method: {method_name}")