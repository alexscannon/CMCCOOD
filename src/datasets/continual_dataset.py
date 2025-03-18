from torch.utils.data import DataLoader

class ContinualDataset:
    """Base class for datasets adapted for continual learning."""

    def __init__(self, train_dataset, test_dataset, num_classes):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.num_classes = num_classes

    def get_train_loader(self, batch_size, num_workers=4, shuffle=True):
        """Create a DataLoader for the training dataset."""

        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )

    def get_test_loader(self, batch_size, num_workers=4, shuffle=False):
        """Create a DataLoader for the test dataset."""

        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )