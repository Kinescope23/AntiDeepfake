import os
import gc
import torch
import logging
from typing import Optional


def setup_cuda_optimizations(config) -> torch.device:
    """Настраивает все CUDA-оптимизации глобально."""
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        print(f"=== CUDA Optimization Setup ===")
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        # TF32 для Tensor Cores
        if config.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("TF32 enabled for matmul and cuDNN")

        # cuDNN benchmark
        if config.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True
            print("cuDNN benchmark enabled")

        # Проверка поддержки bfloat16
        if config.precision_dtype == "bfloat16":
            capability = torch.cuda.get_device_capability()
            if capability[0] < 8:
                print(f"bfloat16 requires compute capability 8.0+, falling back to float16")
                config.precision_dtype = "float16"
            else:
                print(f"bfloat16 enabled (compute capability {capability[0]}.{capability[1]})")

        # === ИСПРАВЛЕНИЕ SDPA ===
        # Мы включаем ВСЕ бэкенды. Если flash-attention не сможет обработать
        # attention_mask из Wav2Vec2, PyTorch автоматически и бесшовно
        # переключится на math_sdp или mem_efficient_sdp без ошибки "Invalid backend".
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        print("SDPA auto-selection enabled (PyTorch will safely fallback if needed)")

        print(f"=================================")

    return device


def setup_logging(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("deepfake_detector")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(os.path.join(log_dir, "training.log"))
        console_handler = logging.StreamHandler()

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


def save_checkpoint(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        loss: float,
        checkpoint_dir: str,
        filename: str = "checkpoint.pth"
):
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }

    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(checkpoint, filepath)


def load_checkpoint(
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        checkpoint_path: str,
        device: torch.device
) -> int:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint['epoch']


def memory_cleanup():
    """Агрессивная очистка памяти."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()