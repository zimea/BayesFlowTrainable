from enum import Enum

class NormalizationType(Enum):
    """Enum for normalization types."""
    NONE = 0
    MEAN = 1
    LOG = 2