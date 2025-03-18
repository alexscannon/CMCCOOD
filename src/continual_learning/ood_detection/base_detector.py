# TODO: Keeping this file for now. Might use this structure for future custom detectors.

class BaseOODDetector:
    """Base class for out-of-distribution detectors."""

    def __init__(self):
        pass

    def calibrate(self, model, dataloader, device):
        """Calibrate the detector on in-distribution data."""
        raise NotImplementedError("Subclasses must implement this method")

    def predict(self, logits):
        """Predict if samples are OOD based on logits."""
        raise NotImplementedError("Subclasses must implement this method")