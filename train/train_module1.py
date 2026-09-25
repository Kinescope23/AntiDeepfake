import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from infrastructure import ParameterSchema
from module1 import Module1
from train.config import TrainingConfig
from train.dataset import AudioDataset
from train.losses import MultiScaleSpectralLoss
from train.utils import save_checkpoint, setup_logging


def train_module1(
        config: TrainingConfig,
        train_dataset: AudioDataset,
        val_dataset: AudioDataset
):
    device = torch.device(config.device)
    schema = ParameterSchema(
        sample_rate=config.sample_rate,
        duration_seconds=config.duration_seconds
    )

    logger = setup_logging(config.log_dir)
    logger.info("Starting Module 1 training (Reconstruction)")

    model = Module1(schema).to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )

    optimizer = optim.AdamW(model.parameters(), lr=config.lr_module1, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)
    criterion = MultiScaleSpectralLoss()

    best_val_loss = float('inf')

    for epoch in range(config.num_epochs):
        model.train()
        train_loss = 0.0

        for batch_audio, _ in train_loader:
            batch_audio = batch_audio.to(device)

            optimizer.zero_grad()

            parameters = model(batch_audio)

            f0 = parameters['f0']
            harmonics = parameters['harmonic_amplitudes']
            noise = parameters['noise_magnitude']

            reconstructed = synthesize_from_parameters(f0, harmonics, noise, schema)

            loss = criterion(batch_audio, reconstructed)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch_audio, _ in val_loader:
                batch_audio = batch_audio.to(device)

                parameters = model(batch_audio)

                f0 = parameters['f0']
                harmonics = parameters['harmonic_amplitudes']
                noise = parameters['noise_magnitude']

                reconstructed = synthesize_from_parameters(f0, harmonics, noise, schema)

                loss = criterion(batch_audio, reconstructed)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step()

        logger.info(f"Epoch {epoch + 1}/{config.num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, config.checkpoint_dir, "module1_best.pth")
            logger.info(f"Saved best model with val loss: {val_loss:.4f}")


def synthesize_from_parameters(f0, harmonics, noise, schema):
    batch_size, num_frames, _ = f0.shape

    harmonic_signal = torch.zeros(batch_size, schema.num_samples, device=f0.device)

    for k in range(min(10, schema.num_harmonics)):
        freq = (k + 1) * f0.squeeze(-1)
        phase = 2 * torch.pi * torch.cumsum(freq, dim=1) / schema.sample_rate
        harmonic_signal += harmonics[:, :, k].unsqueeze(-1) * torch.sin(phase)

    return harmonic_signal


if __name__ == "__main__":
    config = TrainingConfig()

    train_dataset = AudioDataset(
        audio_files=["data/libritts/train/*.wav"],
        labels=[0] * 1000,
        sample_rate=config.sample_rate,
        duration_seconds=config.duration_seconds
    )

    val_dataset = AudioDataset(
        audio_files=["data/libritts/val/*.wav"],
        labels=[0] * 100,
        sample_rate=config.sample_rate,
        duration_seconds=config.duration_seconds
    )

    train_module1(config, train_dataset, val_dataset)