from dataclasses import dataclass
from omegaconf import OmegaConf

### Model types ###
VISION_TRANSFORMER_CLASSIFICATION: str = "vision_transformer_classification"

### CL methods ###
REPLAY: str = "replay"

### OOD detection methods ###
ENERGY: str = "energy"
