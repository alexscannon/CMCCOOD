import os
import sys
from dotenv import load_dotenv
import hydra
from omegaconf import OmegaConf
import logging

# Load environment variables from .env file
load_dotenv()

# Add the project root to the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Set environment variables for Hydra
os.environ['HYDRA_FULL_ERROR'] = '1'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def run_experiment(config):
    """
    Run a continual learning experiment with the specified configuration.

    This script is dataset-agnostic and relies on the configuration file to specify:
    - Dataset type and parameters
    - Model architecture and parameters
    - Continual learning scenario settings
    - Training parameters
    """
    logger.info("Starting experiment with configuration:")
    logger.info("\n" + OmegaConf.to_yaml(config))

    # Validate essential config parameters
    required_configs = [
        "dataset",
        "model",
        "continual_learning.scenario",
        "training"
    ]

    for cfg in required_configs:
        if not OmegaConf.select(config, cfg):
            raise ValueError(f"Missing required configuration: {cfg}")

    try:
        # Import and run main function
        from src.main import main
        results = main(config)

        logger.info("Experiment completed successfully!")
        logger.info("Final results:")
        for metric, value in results.items():
            logger.info(f"{metric}: {value}")

    except Exception as e:
        logger.error(f"Experiment failed with error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    run_experiment()