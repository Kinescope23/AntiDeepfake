from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class TrainingConfig:
    # Базовые параметры
    batch_size: int = 4
    accumulation_steps: int = 4  # Эмулирует batch_size=16 через накопление градиентов
    num_workers: int = 4
    prefetch_factor: int = 2
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0

    # Learning rates для разных модулей
    lr_module1: float = 1e-5
    lr_module2: float = 1e-5
    lr_module3: float = 1e-4
    lr_module4: float = 1e-4

    # Веса функций потерь
    lambda_physics: float = 0.1
    lambda_recon: float = 0.1

    # Директории
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    # Устройство
    device: str = "cuda"
    seed: int = 42

    # Датасеты
    dataset_real: str = "E:\\data\\ASVspoof2019LA"
    dataset_fake: str = "E:\\data\\ASVspoof2019LA"
    dataset_test: str = "E:\\data\\ASVspoof2019LA"

    # Аудио
    sample_rate: int = 16000
    duration_seconds: float = 4.0

    # === CUDA ОПТИМИЗАЦИИ ===

    # Mixed Precision (AMP)
    use_mixed_precision: bool = True
    precision_dtype: str = "float16"  # Оставьте float16, bfloat16 может вызывать проблемы с масками

    allow_tf32: bool = True
    cudnn_benchmark: bool = True
    cudnn_allow_tf32: bool = True

    use_torch_compile: bool = False

    compile_backend: str = "inductor"
    compile_mode: str = "max-autotune"

    # Gradient Checkpointing (экономия памяти, медленнее на 20%)
    use_gradient_checkpointing: bool = False

    # Разрешаем PyTorch самому выбирать бэкенд (Flash, Math или Mem-Efficient)
    use_flash_attention: bool = True

    use_cuda_graphs: bool = False

    pin_memory: bool = True
    non_blocking: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2

    empty_cache_every_n_epochs: int = 1
    gc_collect_every_n_steps: int = 100
    log_every_n_steps: int = 10
    validate_every_n_epochs: int = 1