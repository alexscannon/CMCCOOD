from collections import deque
import random
import torch

class MemoryBuffer:
    """Base class for memory buffers used in replay-based methods."""

    def __init__(self, memory_size=2000):

        self.memory_size = memory_size
        self.memory = deque(maxlen=memory_size)

    def add_samples(self, samples):
        """Add samples to the memory buffer."""
        self.memory.extend(samples)

        # If memory is not full, we don't need to sample
        if len(self.memory) > self.memory_size:
            # Randomly sample to fit memory size
            self.memory = deque(random.sample(list(self.memory), self.memory_size), maxlen=self.memory_size)

    def get_samples(self, batch_size):
        """Get a batch of samples from the memory buffer."""
        if not self.memory:
            return None, None

        # Sample a batch from memory
        batch_size = min(batch_size, len(self.memory))
        batch = random.sample(list(self.memory), batch_size)

        # Separate inputs and targets
        inputs = torch.stack([item[0] for item in batch])
        targets = torch.tensor([item[1] for item in batch])

        return inputs, targets

    def __len__(self):
        """Get the number of samples in the memory buffer."""
        return len(self.memory)