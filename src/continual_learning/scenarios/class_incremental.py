import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

class ClassIncrementalScenario:
    """Class-incremental learning scenario."""

    def __init__(
            self,
            dataset,
            num_tasks,
            num_classes_per_task=None,
            random_seed=42,
            shuffle_classes=True
        ):
        self.dataset = dataset
        self.num_tasks = num_tasks
        self.random_seed = random_seed
        self.shuffle_classes = shuffle_classes

        # Get dataset properties
        self.train_dataset = dataset.train_dataset
        self.test_dataset = dataset.test_dataset
        self.total_classes = dataset.num_classes

        # Calculate classes per task if not specified
        if num_classes_per_task is None:
            self.num_classes_per_task = self.total_classes // num_tasks
        else:
            self.num_classes_per_task = num_classes_per_task

        # Class order and task mapping
        self.class_order = self._get_class_order()
        self.task_data = self._prepare_task_data()

    def _get_class_order(self):
        """Create an order of classes for the incremental learning scenario."""
        # Set random seed for reproducibility
        np.random.seed(self.random_seed)

        # Create class order (shuffle if needed)
        class_order = np.arange(self.total_classes)
        if self.shuffle_classes:
            np.random.shuffle(class_order)

        return class_order

    def _prepare_task_data(self):
        """Prepare data for each task based on the class order."""
        task_data = {}

        for task_id in range(self.num_tasks):
            start_idx = task_id * self.num_classes_per_task
            end_idx = min((task_id + 1) * self.num_classes_per_task, self.total_classes)

            # Classes for this task
            task_classes = self.class_order[start_idx:end_idx]

            # For testing, we'll create empty lists as placeholders
            # In a real implementation, we would filter the dataset by class
            task_data[task_id] = {
                'classes': task_classes,
                'train_indices': [],
                'test_indices': []
            }

        return task_data

    def get_task_dataloader(self, task_id, train=True, batch_size=128, num_workers=4):
        """Get dataloader for a specific task."""
        if task_id not in self.task_data:
            raise ValueError(f"Task ID {task_id} not found")

        # In a minimal implementation, we'll simply create a small dummy dataset
        # In a real implementation, we would return a proper subset of the dataset

        # Create a small dummy dataset for testing
        x = torch.randn(100, 3, 224, 224)
        y = torch.randint(0, len(self.task_data[task_id]['classes']), (100,))

        dummy_dataset = TensorDataset(x, y)

        return DataLoader(
            dummy_dataset,
            batch_size=batch_size,
            shuffle=train,
            num_workers=num_workers,
            pin_memory=True
        )

    def get_task_classes(self, task_id):
        """Get the classes for a specific task."""
        if task_id not in self.task_data:
            raise ValueError(f"Task ID {task_id} not found")
        return self.task_data[task_id]['classes']