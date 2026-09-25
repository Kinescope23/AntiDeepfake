from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    batch_size: int = 16
    num_workers: int = 4
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0

    lr_module1: float = 1e-5
    lr_module2: float = 1e-5
    lr_module3: float = 1e-4
    lr_module4: float = 1e-4

    lambda_physics: float = 0.1
    lambda_recon: float = 0.1

    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    device: str = "cuda"
    seed: int = 42

    dataset_real: str = "data/libritts"
    dataset_fake: str = "data/asvspoof2019"
    dataset_test: str = "data/asvspoof2021"

    sample_rate: int = 16000
    duration_seconds: float = 4.0