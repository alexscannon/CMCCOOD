import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import os
import logging

from src.training.trainer import Trainer
from src.loggers.wandb_logger import WandbLogger
from src.training.device_utils import get_device
from src.models.model_loader import load_model
from src.continual_learning.methods.method_loader import load_cl_method
from src.continual_learning.ood_detection.ood_detector_loader import load_ood_detector
from src.datasets.dataset_loader import load_dataset
from src.continual_learning.scenarios.scenario_loader import load_scenario
from src.training.optimizer_loader import load_optimizer
from src.training.scheduler_loader import load_scheduler

# Configure logger
logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    """Main entry point for running the experiment."""

    # Suppress CUDA warning if CPU is intended
    if config.device == "cuda" and not torch.cuda.is_available():
        import warnings
        warnings.filterwarnings("ignore", message="User provided device_type of 'cuda' but cuda is not available")

    # Set device
    device = get_device(config.device)
    logger.info(f"Using device: {device}")

    # Print config
    logger.info(OmegaConf.to_yaml(config))

    # Set random seed
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)


    # Load dataset based on configuration
    continual_learning_dataset = load_dataset(config)
    # Create continual learning scenario
    scenario = load_scenario(config, continual_learning_dataset)
    # Create model dynamically based on configuration
    logger.info(f"Creating model with backbone: {config.model.model_name}...")
    model = load_model(
        model_config=config.model,
        num_classes=continual_learning_dataset.num_classes,
        device=device
    )

    # Create continual learning method
    logger.info(f"Creating continual learning method...")
    cl_method = load_cl_method(
        config=config,
        model=model,
        scenario=scenario
    )

    # Create OOD detector
    ood_detector = load_ood_detector(config)

    # Create optimizer and scheduler
    optimizer = load_optimizer(config, model)
    scheduler = load_scheduler(config, optimizer)

    # Create logger
    logger.info("Creating logger...")
    wandb_logger = WandbLogger(
        project_name=config.logging.wandb.project_name,
        experiment_name=config.experiment.name,
        config=OmegaConf.to_container(config, resolve=True),
        tags=config.logging.wandb.tags,
        api_key=config.logging.wandb.api_key,
        entity=config.logging.wandb.entity,
        save_code=config.logging.wandb.save_code,
        log_model=config.logging.wandb.log_model
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        cl_method=cl_method,
        scenario=scenario,
        ood_detector=ood_detector,
        config=config,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        wandb_logger=wandb_logger
    )

    # Train on all tasks
    logger.info("Starting training...")
    trainer.train_all_tasks()

    # Save model and log to W&B if configured
    if config.save_model or config.logging.wandb.log_model:
        output_dir = config.output_dir
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, "final_model.pt")

        if config.save_model:
            torch.save(model.state_dict(), model_path)
            logger.info(f"Model saved to {model_path}")

        if config.logging.wandb.log_model:
            wandb_logger.log_model(model_path, f"model_{config.experiment.name}")

    # Finish logging
    wandb_logger.finish()

if __name__ == "__main__":
    main()