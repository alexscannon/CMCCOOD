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
        """Evaluate the model on one or all tasks."""

        self.model.eval()
        # Evaluate on all seen tasks if no task_id is given.
        task_ids = range(self.scenario.num_tasks) if task_id is None else task_ids = [task_id]

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
            results[f"accuracy_{task_id}"] = accuracy

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


    def evaluate_ood_detection_by_task(self, current_task_id, ood_task_id):
        """Evaluate OOD detection performance between tasks.

        Args:
            current_task_id: ID of the current task (in-distribution data)
            ood_task_id: ID of the task to use as OOD data

        Returns:
            Dictionary of OOD detection metrics
        """
        print(f"Evaluating OOD detection: Task {current_task_id} (in-distribution) vs Task {ood_task_id} (OOD)")

        if self.ood_detector is None:
            logger.warning("No OOD detector provided, skipping OOD evaluation")
            return {}

        self.model.eval()

        # Get dataloaders
        id_loader = self.scenario.get_task_dataloader(current_task_id, train=False)
        ood_loader = self.scenario.get_task_dataloader(ood_task_id, train=False)

        # Calibrate detector on current task if it supports calibration
        print(f"Calibrating OOD detector on Task {current_task_id}...")
        if hasattr(self.ood_detector, 'calibrate'):
            self.ood_detector.calibrate(self.model, id_loader, self.device)

        # Collect predictions
        id_scores = []  # Generic scores instead of energies
        id_preds = []
        ood_scores = []
        ood_preds = []

        with torch.no_grad():
            # Process in-distribution data
            for ind_inputs, _ in id_loader:
                ind_inputs = ind_inputs.to(self.device)
                ind_logits = self.model(ind_inputs)

                # Use the detector's prediction method
                is_ood, score = self.ood_detector.predict(ind_logits)

                id_preds.append(is_ood)
                id_scores.append(score)

            # Process out-of-distribution data
            for ood_inputs, _ in ood_loader:
                ood_inputs = ood_inputs.to(self.device)
                ood_logits = self.model(ood_inputs)

                # Use the detector's prediction method
                is_ood, score = self.ood_detector.predict(ood_logits)

                ood_preds.append(is_ood)
                ood_scores.append(score)

        # Concatenate results
        id_preds = torch.cat(id_preds, dim=0) if id_preds else torch.tensor([])
        id_scores = torch.cat(id_scores, dim=0) if id_scores else torch.tensor([])
        ood_preds = torch.cat(ood_preds, dim=0) if ood_preds else torch.tensor([])
        ood_scores = torch.cat(ood_scores, dim=0) if ood_scores else torch.tensor([])

        # Calculate metrics
        id_labels = torch.zeros_like(id_preds, dtype=torch.float)
        ood_labels = torch.ones_like(ood_preds, dtype=torch.float)

        # Calculate metrics
        id_accuracy = (id_preds == id_labels).float().mean().item() if len(id_preds) > 0 else 0
        ood_accuracy = (ood_preds == ood_labels).float().mean().item() if len(ood_preds) > 0 else 0

        # Calculate True Positive Rate and False Positive Rate
        tpr = ood_preds.float().mean().item() if len(ood_preds) > 0 else 0  # True Positive Rate (detection rate)
        fpr = id_preds.float().mean().item() if len(id_preds) > 0 else 0   # False Positive Rate

        metrics = {
            "ood_detection/id_accuracy": id_accuracy * 100,
            "ood_detection/ood_accuracy": ood_accuracy * 100,
            "ood_detection/detection_rate": tpr * 100,
            "ood_detection/false_alarm_rate": fpr * 100,
        }

        # Calculate AUROC if both score arrays are non-empty and sklearn is available
        if len(id_scores) > 0 and len(ood_scores) > 0:
            try:
                from sklearn.metrics import roc_auc_score
                all_scores = torch.cat([id_scores, ood_scores], dim=0).cpu().numpy()
                all_labels = torch.cat([id_labels, ood_labels], dim=0).cpu().numpy()

                # Higher score should indicate higher likelihood of being OOD
                # Check if detector needs score inversion (depends on detector implementation)
                score_needs_inversion = hasattr(self.ood_detector, 'score_needs_inversion') and self.ood_detector.score_needs_inversion

                if score_needs_inversion:
                    auroc = roc_auc_score(all_labels, -all_scores)
                else:
                    auroc = roc_auc_score(all_labels, all_scores)

                metrics["ood_detection/auroc"] = auroc * 100
            except (ImportError, ValueError) as e:
                print(f"AUROC calculation error: {str(e)}")

        return metrics

    def train_all_tasks(self):
        """Train on all tasks sequentially."""

        for task_id in range(self.scenario.num_tasks):
            self.train_task(task_id)

            # Evaluate after each task
            # print(f"Evaluation after Task {task_id}:")
            # all_task_metrics = self.evaluate()

            # Log comprehensive evaluation metrics after completing each task
            if self.wandb_logger is not None:
                metrics = {}

                # Log individual task performances
                for eval_task_id in range(task_id + 1):
                    task_results = self.evaluate(eval_task_id)
                    metrics[f"task_{eval_task_id}/test_accuracy"] = task_results[f"accuracy_{eval_task_id}"]

                # Add average metrics
                # metrics["cumulative_average_accuracy"] = all_task_metrics.get("average_accuracy", 0)
                metrics["current_task"] = task_id

                # Evaluate OOD detection on next task's data (if not the last task)
                if task_id < self.scenario.num_tasks - 1 and self.ood_detector is not None:
                    ood_task_id = task_id + 1
                    ood_metrics = self.evaluate_ood_detection_by_task(task_id, ood_task_id)
                    metrics.update({
                        f"task_{task_id}_vs_{ood_task_id}/{k}": v
                        for k, v in ood_metrics.items()
                    })

                self.wandb_logger.log_metrics(metrics)