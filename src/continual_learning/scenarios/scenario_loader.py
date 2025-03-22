from omegaconf import DictConfig
from src.datasets.continual_dataset import ContinualDataset
from src.continual_learning.scenarios.class_incremental import ClassIncrementalScenario
import logging

logger = logging.getLogger(__name__)

def load_scenario(config: DictConfig, dataset: ContinualDataset) -> ClassIncrementalScenario:
    """
    Load continual learning scenario based on configuration.

    Args:
        config: Configuration object
        dataset: Dataset to use for the scenario

    Returns:
        Configured continual learning scenario
    """
    logger.info("Creating continual learning scenario...")

    scenario_config = config.continual_learning.scenario

    if scenario_config.name == "class_incremental":
        return ClassIncrementalScenario(
            dataset=dataset,
            num_tasks=scenario_config.num_tasks,
            random_seed=config.seed,
            shuffle_classes=config.shuffle_classes
        )
    else:
        raise ValueError(f"Unknown scenario: {scenario_config.name}")