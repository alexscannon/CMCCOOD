import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import TensorDataset
import os
import logging
from torchvision import datasets, transforms

from src.models.foundation_models.vision_adapter import VisionModelAdapter
from src.models.classification_head import ClassificationHead
from src.datasets.continual_dataset import ContinualDataset
from src.continual_learning.scenarios.class_incremental import ClassIncrementalScenario
from src.continual_learning.methods.replay.er import ExperienceReplay
from src.continual_learning.ood_detection.energy import EnergyBasedOODDetector
from src.training.trainer import Trainer
from src.training.device_utils import get_device
from src.loggers.wandb_logger import WandbLogger

# Configure logger
logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    """Main entry point for the simple test run."""
    # Print config
    logger.info(OmegaConf.to_yaml(config))

    # Set random seed
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # Set device
    device = get_device(config.device)
    logger.info(f"Using device: {device}")

    # Create dataset transforms
    transform = transforms.Compose([
        transforms.Resize((config.dataset.image_size, config.dataset.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.dataset.mean, std=config.dataset.std)
    ])

    # Load CIFAR-100 dataset
    logger.info("Loading CIFAR-100 dataset...")
    train_dataset = datasets.CIFAR100(
        root=config.dataset.root,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.CIFAR100(
        root=config.dataset.root,
        train=False,
        download=True,
        transform=transform
    )

    # Create continual dataset
    dataset = ContinualDataset(
        train_dataset,
        test_dataset,
        num_classes=config.dataset.num_classes
    )

    # Create continual learning scenario
    logger.info("Creating continual learning scenario...")
    scenario = ClassIncrementalScenario(
        dataset=dataset,
        num_tasks=2,  # Simplified for testing
        num_classes_per_task=50,
        random_seed=config.seed,
        shuffle_classes=True
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
        num_classes=dataset.num_classes
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
    logger.info("Creating continual learning method...")
    cl_method = ExperienceReplay(
        model=model,
        scenario=scenario,
        memory_size=100,  # Reduced for testing
        selection_strategy='random',
        batch_size_memory=16
    )

    # Create OOD detector
    logger.info("Creating OOD detector...")
    ood_detector = EnergyBasedOODDetector(
        temperature=1.0,
        threshold=None
    )

    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0001,
        weight_decay=0.0001
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
        experiment_name=config.experiment.name,  # Using the experiment name from config
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
        epochs_per_task=2,  # Reduced for testing. TODO: change this to be dynamic?
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