from dataclasses import dataclass, field
from typing import List


@dataclass
class FeatureConfig:
    categorical_features: List[str]
    continual_features: List[str]
    target_column: str
