from dataclasses import dataclass, field


@dataclass()
class TrainTestSplitConfig:
    test_size: float = field(default=0.25)
    random_state: int = field(default=18)
