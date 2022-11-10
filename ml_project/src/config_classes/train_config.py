from dataclasses import dataclass
from .feature_config import FeatureConfig
from .train_test_split_config import TrainTestSplitConfig
from .model_config import ModelConfig


@dataclass
class TrainConfig:
    model: ModelConfig
    feature_details: FeatureConfig
    train_test_split_params: TrainTestSplitConfig
    input_data_path: str
    input_test_data_path: str

