from enum import Enum

class NormalizationType(Enum):
    """Enum for normalization types."""
    NONE = 0
    MEAN = 1
    LOG = 2

class TrainingMode(Enum):
    """Enum for training modes."""
    ONLINE = 0
    OFFLINE = 1