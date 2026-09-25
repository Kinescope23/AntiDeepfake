import os
import torch
import logging
from typing import Optional


def setup_logging(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("deepfake_detector")
    logger.setLevel(logging.INFO)

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
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint['epoch']