import wandb
import logging

logger = logging.getLogger(__name__)

class WandbLogger:
    """Weights & Biases logger for continual learning experiments."""

    def __init__(self, project_name, experiment_name, config=None, tags=None, api_key=None, entity=None, save_code=True, log_model=True):
        """Initialize the Weights & Biases logger.

        Args:
            project_name (str): Name of the W&B project
            experiment_name (str): Name of this specific run
            config (dict, optional): Configuration to log
            tags (list, optional): Tags for the run
            api_key (str, optional): Weights & Biases API key
            entity (str, optional): W&B username/organization
            save_code (bool, optional): Whether to save code to W&B
            log_model (bool, optional): Whether to log model artifacts
        """
        try:
            if api_key:
                wandb.login(key=api_key)
                logger.info("Successfully logged in to W&B using API key")
            else:
                logger.warning("No API key provided for W&B login")

            self.run = wandb.init(
                project=project_name,
                name=experiment_name,
                entity=entity,
                config=config,
                tags=tags,
                reinit=True,  # Allow for multiple runs in the same process
                settings=wandb.Settings(code_dir=".") if save_code else None
            )
            self.should_log_model = log_model
            logger.info(f"Successfully initialized W&B run: {experiment_name}")
        except Exception as e:
            logger.error(f"Failed to initialize W&B: {str(e)}")
            self.run = None

    def log_metrics(self, metrics, step=None):
        """Log metrics to W&B.

        Args:
            metrics (dict): Dictionary of metrics to log
            step (int, optional): Step number for the metrics
        """
        if self.run is not None:
            self.run.log(metrics, step=step)

    def log_model(self, model_path, name):
        """Log model to W&B artifacts.

        Args:
            model_path (str): Path to the saved model
            name (str): Name for the artifact
        """
        if self.run is not None and self.should_log_model:
            artifact = wandb.Artifact(name=name, type='model')
            artifact.add_file(model_path)
            self.run.log_artifact(artifact)

    def finish(self):
        """End the W&B run."""
        if self.run is not None:
            self.run.finish()