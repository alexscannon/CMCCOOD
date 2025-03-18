import torch.nn.functional as F
from ...base import BaseCLMethod
from .buffer import MemoryBuffer

class ExperienceReplay(BaseCLMethod):
    """Experience Replay method for continual learning."""

    def __init__(
            self,
            model,
            scenario,
            memory_size=2000,
            selection_strategy='random',
            batch_size_memory=32
        ):
        super().__init__(model, scenario)
        self.memory_size = memory_size
        self.selection_strategy = selection_strategy
        self.batch_size_memory = batch_size_memory

        # Create memory buffer
        self.buffer = MemoryBuffer(memory_size=memory_size)

    def after_task(self, task_id):
        """Update memory after learning a task."""
        # Get dataloader for the current task
        dataloader = self.scenario.get_task_dataloader(task_id, train=True)

        # Sample from the dataloader and add to buffer
        samples = []
        for inputs, targets in dataloader:
            for i in range(inputs.size(0)):
                samples.append((inputs[i], targets[i]))
            # For simplicity in this minimal example, we'll just use the first batch
            # TODO: enhance task sampling strategy
            break

        # Add samples to memory
        self.buffer.add_samples(samples)

    def compute_loss(self, outputs, targets, task_id):
        """Compute loss with replay regularization."""
        # Standard classification loss for current task
        loss = F.cross_entropy(outputs, targets)

        # If there's memory data, add replay loss
        memory_inputs, memory_targets = self.buffer.get_samples(self.batch_size_memory)

        if memory_inputs is not None and memory_targets is not None:
            # Move to same device as model
            device = next(self.model.parameters()).device
            memory_inputs = memory_inputs.to(device)
            memory_targets = memory_targets.to(device)

            # Forward pass on memory samples
            memory_outputs = self.model(memory_inputs)

            # Compute replay loss
            replay_loss = F.cross_entropy(memory_outputs, memory_targets)

            # Combine losses of current task and memory examples to maintain performance
            # both current and old
            loss = loss + replay_loss

        return loss