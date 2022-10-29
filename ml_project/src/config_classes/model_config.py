from dataclasses import dataclass, field


@dataclass
class PreprocessingConfig:
    process_categorical: bool = field(default=True)
    process_continual: bool = field(default=True)


@dataclass
class TrainingParams:
    model_type: str = field(default='LogisticRegression')
    random_state: int = field(default=18)
    grid_search: bool = field(default=True)


@dataclass
class ModelConfig:
    path_to_output_model: str
    path_to_model_metric: str

    path_to_processed_data: str
    path_to_transformer: str

    preprocessing_params: PreprocessingConfig
    train_params: TrainingParams
