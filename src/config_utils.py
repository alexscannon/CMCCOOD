from omegaconf import OmegaConf
from dataclasses import dataclass

# Define your constants
@dataclass
class Constants:
    VISION_TRANSFORMER_CLASSIFICATION: str = "vision_transformer_classification"
    # Add other constants as needed

def initialize_config_system():
    """Initialize the OmegaConf configuration system with constants"""
    # Create structured config
    constants_conf = OmegaConf.structured(Constants())

    # Register resolver for constants
    OmegaConf.register_resolver("constants", lambda name: constants_conf[name])

    return constants_conf

# Call this at the start of your application
constants = initialize_config_system()

def load_config(config_path):
    """Load a configuration file with constants properly resolved"""
    return OmegaConf.load(config_path)