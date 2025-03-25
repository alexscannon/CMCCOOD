import torch
from tqdm import tqdm
import numpy as np
import torch.optim as optim

class Trainer:
    """Trainer class for continual learning."""

    def __init__(
            self,
            model,
            cl_method,
            scenario,
            config,
            ood_detector=None,
            optimizer=None,
            scheduler=None,
            device=None,
            wandb_logger=None,
        ):

        self.training_config = config.training
        self.epochs_per_task = self.training_config.epochs_per_task

        self.model = model
        self.cl_method = cl_method
        self.scenario = scenario
        self.ood_detector = ood_detector
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.wandb_logger = wandb_logger  # Add WandB logger

        # Set device
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)


    def train_task(self, task_id):
        """Train the model on a specific task."""
        print(f"Training on task {task_id}")

        # Prepare for new task
        self.cl_method.before_task(task_id)

        # Get data loader for current task
        train_loader = self.scenario.get_task_dataloader(
            task_id,
            train=True,
            batch_size=self.training_config.batch_size,
            num_workers=self.training_config.num_workers
        )

        # Training loop
        self.model.train()
        for epoch in range(self.epochs_per_task):
            running_loss = 0.0
            correct = 0
            total = 0

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs_per_task}")
            for inputs, targets in progress_bar:
                # Move to device and convert to float32
                inputs = inputs.to(self.device, dtype=torch.float32, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                # Zero gradients
                self.optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()

                # Forward pass
                with torch.amp.autocast('cuda'):  # Mixed precision training
                    outputs = self.model(inputs)
                    loss = self.cl_method.compute_loss(outputs, targets, task_id)

                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()

                # Update statistics with reduced precision
                with torch.no_grad():
                    running_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

                # Update progress bar less frequently
                if progress_bar.n % 10 == 0:  # Update every 10 batches
                    progress_bar.set_postfix({
                        'loss': running_loss / (progress_bar.n + 1),
                        'acc': 100. * correct / total
                    })

            # Log epoch metrics
            if self.wandb_logger is not None:
                epoch_metrics = {
                    f'task_{task_id}/epoch/loss': running_loss / len(train_loader),
                    f'task_{task_id}/epoch/accuracy': 100. * correct / total,
                    'epoch': epoch + 1,
                }
                self.wandb_logger.log_metrics(epoch_metrics)

                # Evaluate and log validation metrics after each epoch
                val_metrics = self.evaluate(task_id)
                for key, value in val_metrics.items():
                    epoch_metrics[f'task_{task_id}/val/{key}'] = value
                self.wandb_logger.log_metrics(epoch_metrics)

            # Step scheduler if provided
            if self.scheduler is not None:
                self.scheduler.step()

        # After task operations
        self.cl_method.after_task(task_id)

        # Calibrate OOD detector if provided
        if self.ood_detector is not None:
            print("Calibrating OOD detector...")
            train_loader = self.scenario.get_task_dataloader(task_id, train=False)  # Use validation data
            self.ood_detector.calibrate(self.model, train_loader, self.device)

    def evaluate(self, task_id=None):
        """Evaluate the model on one or all tasks."""
        self.model.eval()

        if task_id is None:
            # Evaluate on all seen tasks
            task_ids = range(self.scenario.num_tasks)
        else:
            task_ids = [task_id]

        results = {}
        for task_id in task_ids:
            test_loader = self.scenario.get_task_dataloader(task_id, train=False)

            correct = 0
            total = 0

            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.model(inputs)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

            accuracy = 100. * correct / total if total > 0 else 0
            results[f"accuracy"] = accuracy
            print(f"Task {task_id} Accuracy: {accuracy:.2f}%")

        # Calculate average accuracy
        if task_id is None and len(task_ids) > 1:
            avg_accuracy = np.mean(list(results.values()))
            results["average_accuracy"] = avg_accuracy
            print(f"Average Accuracy: {avg_accuracy:.2f}%")

            # Log average accuracy across all tasks
            if self.wandb_logger is not None:
                self.wandb_logger.log_metrics({"average_accuracy": avg_accuracy})

        return results

    def train_all_tasks(self):
        """Train on all tasks sequentially."""
        for task_id in range(self.scenario.num_tasks):
            self.train_task(task_id)

            # Evaluate after each task
            print(f"Evaluation after Task {task_id}:")
            all_task_metrics = self.evaluate()

            # Log comprehensive evaluation metrics after completing each task
            if self.wandb_logger is not None:
                metrics = {}

                # Log individual task performances
                for eval_task_id in range(task_id + 1):
                    test_loader = self.scenario.get_task_dataloader(eval_task_id, train=False)
                    task_results = self.evaluate(eval_task_id)
                    metrics[f"task_{eval_task_id}/test_accuracy"] = task_results["accuracy"]

                # Add average metrics
                metrics["cumulative_average_accuracy"] = all_task_metrics.get("average_accuracy", 0)
                metrics["current_task"] = task_id

                self.wandb_logger.log_metrics(metrics)