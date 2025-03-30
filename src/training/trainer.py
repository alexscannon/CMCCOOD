import torch
from tqdm import tqdm
import numpy as np
import torch.optim as optim
import logging

logger = logging.getLogger(__name__)

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


    def evaluate(self, task_id=None):
        """
        Evaluate the model's accuracy on one or all tasks.

        Args:
            task_id: The task ID to evaluate on. If None, evaluate on all tasks.

        Returns:
            Dictionary of evaluation metrics.
        """

        self.model.eval() # Set the model to evaluation mode
        task_ids = range(self.scenario.num_tasks) if task_id is None else [task_id]

        results = {}
        for task_id in task_ids:
            task_test_loader = self.scenario.get_task_dataloader(task_id, train=False)

            correct, total = 0, 0

            with torch.no_grad():
                for inputs, targets in task_test_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    logits = self.model(inputs) # Forward pass to get the logits for each sample in the batch
                    _, predicted = logits.max(1) # Get the class with the highest logit score
                    total += targets.size(0) # Number of samples in the batch
                    correct += predicted.eq(targets).sum().item() # Number of correct predictions

            accuracy = 100. * correct / total if total > 0 else 0

            logger.info(f"Task {task_id} Accuracy: {accuracy:.2f}%")

        return results

    def train_task(self, task_id: int) -> None:
        """
        Train the model on a specific task.

        Args:
            task_id: The task ID to train on.
        """

        logger.info(f"Training on task {task_id}")

        # Prepare for new task if CL method is uses set up procedures
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
            running_loss, correct, total = 0.0, 0, 0

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs_per_task}")
            for inputs, targets in progress_bar:
                # Move to device and convert to float32
                inputs = inputs.to(self.device, dtype=torch.float32, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                # Zero gradients
                self.optimizer.zero_grad(set_to_none=True)

                # Forward pass
                with torch.amp.autocast('cuda'):  # Mixed precision training
                    outputs = self.model(inputs)
                    # Compute loss which may be a combination of a classification loss and a regularization loss
                    # for specific CL methods. Not just a classification loss.
                    loss = self.cl_method.compute_loss(outputs, targets)

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
                }
                self.wandb_logger.log_metrics(epoch_metrics)

            # Step scheduler if provided
            if self.scheduler is not None:
                self.scheduler.step()

        # After task CL method specific operations
        self.cl_method.after_task(task_id)

        # Calibrate OOD detector if provided
        if self.ood_detector is not None and hasattr(self.ood_detector, 'calibrate'):
            print("Calibrating OOD detector...")
            test_loader = self.scenario.get_task_dataloader(task_id, train=False)  # Use validation data
            self.ood_detector.calibrate(self.model, test_loader, self.device)


    def evaluate_ood_detection_by_task(self, current_task_id: int, ood_task_id: int) -> dict:
        """
        Evaluate OOD detection performance between tasks.

        Args:
            current_task_id: ID of the current task (in-distribution data)
            ood_task_id: ID of the task to use as OOD data

        Returns:
            Dictionary of OOD detection metrics (ind_accuracy, ood_accuracy, TPR, FPR)
        """

        logger.info(f"Evaluating OOD detection: Task {current_task_id} (in-distribution) vs Task {ood_task_id} (OOD)")

        if self.ood_detector is None:
            logger.warning("No OOD detector provided, skipping OOD evaluation")
            return {}

        # Get dataloaders
        ind_dataloader = self.scenario.get_task_dataloader(current_task_id, train=False)
        ood_dataloader = self.scenario.get_task_dataloader(ood_task_id, train=False)

        # Collect predictions
        ind_preds, ind_scores = [], []
        ood_preds, ood_scores = [], []

        self.model.eval()
        with torch.no_grad():
            # Process in-distribution data
            for ind_inputs, _ in tqdm(ind_dataloader, desc="Processing in-distribution data"):
                ind_inputs = ind_inputs.to(self.device, non_blocking=True)
                ind_logits = self.model(ind_inputs)

                # Use the detector's prediction method
                ind_is_ood_preds, ind_energy_scores = self.ood_detector.predict(ind_logits)

                ind_preds.append(ind_is_ood_preds)
                ind_scores.append(ind_energy_scores)

            # Process out-of-distribution data
            for ood_inputs, _ in tqdm(ood_dataloader, desc="Processing out-of-distribution data"):
                ood_inputs = ood_inputs.to(self.device, non_blocking=True)
                ood_logits = self.model(ood_inputs)

                # Use the detector's prediction method
                ood_is_ood_preds, ood_energy_scores = self.ood_detector.predict(ood_logits)

                ood_preds.append(ood_is_ood_preds)
                ood_scores.append(ood_energy_scores)

        # Concatenate results
        ind_preds = torch.cat(ind_preds, dim=0) if ind_preds else torch.tensor([])
        ind_scores = torch.cat(ind_scores, dim=0) if ind_scores else torch.tensor([])
        ood_preds = torch.cat(ood_preds, dim=0) if ood_preds else torch.tensor([])
        ood_scores = torch.cat(ood_scores, dim=0) if ood_scores else torch.tensor([])

        # Calculate labels
        ind_labels = torch.zeros_like(ind_preds, dtype=torch.float)
        ood_labels = torch.ones_like(ood_preds, dtype=torch.float)

        # Calculate accuracy metrics
        ind_accuracy = (ind_preds == ind_labels).float().mean().item() if len(ind_preds) > 0 else 0
        ood_accuracy = (ood_preds == ood_labels).float().mean().item() if len(ood_preds) > 0 else 0

        logger.info(f"ID Accuracy: {ind_accuracy * 100}%")
        logger.info(f"OOD Accuracy: {ood_accuracy * 100}%")

        # Calculate True Positive Rate (TPR) and False Positive Rate (FPR)
        tpr = ood_preds.float().mean().item() if len(ood_preds) > 0 else 0
        fpr = ind_preds.float().mean().item() if len(ind_preds) > 0 else 0

        metrics = {
            "ood_detection/ind_accuracy": ind_accuracy * 100,
            "ood_detection/ood_accuracy": ood_accuracy * 100,
            "ood_detection/detection_rate": tpr * 100,
            "ood_detection/false_alarm_rate": fpr * 100,
        }

        return metrics

    def train_all_tasks(self):
        """Train on all tasks sequentially."""
        cumulative_accuracies = []

        for task_id in range(self.scenario.num_tasks):
            self.train_task(task_id)

            # Log comprehensive evaluation metrics after completing each task
            if self.wandb_logger is not None:
                metrics = {}

                # Log individual task performances
                total_accuracy = 0
                for eval_task_id in range(task_id + 1):
                    task_results = self.evaluate(eval_task_id)
                    task_accuracy = task_results[f"accuracy_{eval_task_id}"]
                    metrics[f"task_{eval_task_id}/test_accuracy"] = task_accuracy
                    total_accuracy += task_accuracy

                # Calculate and log running average accuracy
                current_avg_accuracy = total_accuracy / (task_id + 1)
                cumulative_accuracies.append(current_avg_accuracy)
                metrics["metrics/running_avg_accuracy"] = current_avg_accuracy

                # Also log the average accuracy across all tasks completed so far
                metrics["current_task"] = task_id

                # Evaluate OOD detection on next task's data (if not the last task)
                if task_id < self.scenario.num_tasks - 1 and self.ood_detector is not None:
                    ood_task_id = task_id + 1
                    ood_metrics = self.evaluate_ood_detection_by_task(task_id, ood_task_id)
                    # ood_metrics are ind_accuracy, ood_accuracy, TPR, FPR
                    metrics.update({
                        f"ood_detection": v for _, v in ood_metrics.items()
                    })

                self.wandb_logger.log_metrics(metrics)