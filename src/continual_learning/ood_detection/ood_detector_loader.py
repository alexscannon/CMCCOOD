from omegaconf import DictConfig
import logging

from src.continual_learning.ood_detection.energy import EnergyBasedOODDetector
# Import any other OOD detector implementations here

logger = logging.getLogger(__name__)

def load_ood_detector(config: DictConfig):
    """
    Dynamically loads an OOD detector based on the provided configuration.

    Args:
        config: Configuration for the OOD detector

    Returns:
        An instantiated OOD detector
    """

    ood_detection_config = config.continual_learning.ood_detection
    detector_type = ood_detection_config.name.lower()

    logger.info(f"Loading OOD detector Method: {detector_type}")

    if detector_type == "energy":
        return EnergyBasedOODDetector(
            temperature=ood_detection_config.temperature,
            threshold=ood_detection_config.threshold
        )
    # Add other detector types here
    elif detector_type == "mahalanobis":
        # return MahalanobisOODDetector(config)
        raise NotImplementedError("Mahalanobis OOD detector not yet implemented")
    elif detector_type == "msp":  # Maximum Softmax Probability
        # return MSPOODDetector(config)
        raise NotImplementedError("MSP OOD detector not yet implemented")
    else:
        raise ValueError(f"Unknown OOD detector type: {detector_type}")