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
            ood_detector=None,
            epochs_per_task=10,
            optimizer=None,
            scheduler=None,
            device=None
        ):
        self.model = model
        self.cl_method = cl_method
        self.scenario = scenario
        self.ood_detector = ood_detector
        self.epochs_per_task = epochs_per_task

        # Set device
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

        # Set optimizer and scheduler
        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        else:
            self.optimizer = optimizer

        self.scheduler = scheduler

    def train_task(self, task_id):
        """Train the model on a specific task."""
        print(f"Training on task {task_id}")

        # Prepare for new task
        self.cl_method.before_task(task_id)

        # Get data loader for current task
        train_loader = self.scenario.get_task_dataloader(task_id, train=True)

        # Training loop
        self.model.train()
        for epoch in range(self.epochs_per_task):
            running_loss = 0.0
            correct = 0
            total = 0

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs_per_task}")
            for inputs, targets in progress_bar:

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Zero gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)

                # Compute loss with CL method
                loss = self.cl_method.compute_loss(outputs, targets, task_id)

                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()

                # Update statistics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # Update progress bar
                progress_bar.set_postfix({
                    'loss': running_loss / (progress_bar.n + 1),
                    'acc': 100. * correct / total
                })

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
            results[f"Task {task_id} Accuracy"] = accuracy
            print(f"Task {task_id} Accuracy: {accuracy:.2f}%")

        # Calculate average accuracy
        if task_id is None:
            avg_accuracy = np.mean(list(results.values()))
            results["Average Accuracy"] = avg_accuracy
            print(f"Average Accuracy: {avg_accuracy:.2f}%")

        return results

    def train_all_tasks(self):
        """Train on all tasks sequentially."""
        for task_id in range(self.scenario.num_tasks):
            self.train_task(task_id)

            # Evaluate after each task
            print(f"Evaluation after Task {task_id}:")
            self.evaluate()