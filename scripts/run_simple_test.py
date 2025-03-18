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

# Add after the imports
logging.basicConfig(level=logging.INFO)

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def run_test(config):
    """Run a simple test of the continual learning framework."""
    print("Running simple test with configuration:")
    print(OmegaConf.to_yaml(config))

    # Override configuration for quicker testing
    config.model.pretrained = False  # Skip downloading weights for testing
    config.dataset.num_classes = 10  # Use fewer classes
    config.continual_learning.scenario.num_tasks = 2  # Just 2 tasks
    config.continual_learning.scenario.num_classes_per_task = 5  # 5 classes per task
    config.training.epochs_per_task = 1  # Just 1 epoch per task for quick testing

    # Import the main function and run
    from src.main import main
    main(config)

    print("Test completed successfully!")

if __name__ == "__main__":
    run_test()