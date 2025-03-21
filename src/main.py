import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import os
import logging

from src.models.foundation_models.vision_adapter import VisionModelAdapter
from src.models.classification_head import ClassificationHead
from src.datasets.continual_dataset import ContinualDataset
from src.continual_learning.scenarios.class_incremental import ClassIncrementalScenario
from src.continual_learning.methods.replay.er import ExperienceReplay
from src.continual_learning.ood_detection.energy import EnergyBasedOODDetector
from src.training.trainer import Trainer
from src.loggers.wandb_logger import WandbLogger
from src.training.device_utils import get_device
from src.datasets.dataset_loader import load_dataset

# Configure logger
logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    """Main entry point for running the experiment."""

    # Suppress CUDA warning if CPU is intended
    if config.device == "cuda" and not torch.cuda.is_available():
        import warnings
        warnings.filterwarnings("ignore", message="User provided device_type of 'cuda'")

    # Print config
    logger.info(OmegaConf.to_yaml(config))

    # Set random seed
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # Set device
    device = get_device(config.device)
    logger.info(f"Using device: {device}")

    # Load dataset based on configuration
    continual_learning_dataset = load_dataset(config)

    # Create continual learning scenario
    logger.info("Creating continual learning scenario...")
    scenario = ClassIncrementalScenario(
        dataset=continual_learning_dataset,
        num_tasks= 2,  # TODO: Make this dynamic based on a load_scenario function
        random_seed=config.seed,
        shuffle_classes=config.shuffle_classes
    )

    # Create foundation model
    logger.info(f"Creating foundation model: {config.model.model_name}...")
    feature_extractor = VisionModelAdapter(
        model_name=config.model.model_name,
        pretrained=config.model.pretrained,
        feature_dim=config.model.feature_dim,
        freeze_backbone=config.model.freeze_backbone
    )

    # Create classification head
    classification_head = ClassificationHead(
        feature_dim=feature_extractor.feature_dim,
        num_classes=continual_learning_dataset.num_classes
    )

    # Create full model
    # TODO: Create an agents dir to store all full models
    class ContinualLearningModel(torch.nn.Module):
        def __init__(self, feature_extractor, classification_head):
            super().__init__()
            self.feature_extractor = feature_extractor
            self.classification_head = classification_head

        def forward(self, x):
            features = self.feature_extractor(x)
            return self.classification_head(features)

    model = ContinualLearningModel(feature_extractor, classification_head)
    model = model.to(device)

    # Create continual learning method
    # TODO: Make this dynamic based on a load_method function for CL methods
    logger.info("Creating continual learning method...")
    cl_method = ExperienceReplay(
        model=model,
        scenario=scenario,
        memory_size=100,  # Reduced for testing
        selection_strategy='random',
        batch_size_memory=16
    )

    # Create OOD detector
    # TODO: Make this dynamic based on a load_ood_detector function for OOD detectors
    logger.info("Creating OOD detector...")
    ood_detector = EnergyBasedOODDetector(
        temperature=1.0,
        threshold=None
    )

    # TODO: Make the learning rate and weight decay dynamic based on the model size
    # Create optimizer with higher learning rate for classification head only
    optimizer = torch.optim.AdamW(
        model.classification_head.parameters(),  # Only optimize classification head
        lr=0.001,
        weight_decay=0.01
    )

    # Create scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=5  # Simplified for testing
    )

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
    logger.info("Creating trainer...")
    trainer = Trainer(
        model=model,
        cl_method=cl_method,
        scenario=scenario,
        ood_detector=ood_detector,
        epochs_per_task= 2,  # TODO: change this to be dynamic?
        optimizer=optimizer,
        scheduler=scheduler,
        device=device
    )

    # Train on all tasks
    logger.info("Starting training...")
    trainer.train_all_tasks()

    # Final evaluation
    logger.info("Final evaluation:")
    results = trainer.evaluate()

    # Log results
    wandb_logger.log_metrics(results)

    # Save model and log to W&B if configured
    output_dir = config.output_dir
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "final_model.pt")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")

    if config.logging.wandb.log_model:
        wandb_logger.log_model(model_path, f"model_{config.experiment.name}")

    # Finish logging
    wandb_logger.finish()

    return results

if __name__ == "__main__":
    main()