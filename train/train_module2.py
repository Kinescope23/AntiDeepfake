import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from infrastructure import ParameterSchema
from module1 import Module1
from module2 import Module2
from train.config import TrainingConfig
from train.dataset import AudioDataset
from train.utils import save_checkpoint, load_checkpoint, setup_logging


def train_module2(
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
    logger.info("Starting Module 2 training (Physics Manifold Learning)")

    module1 = Module1(schema).to(device)
    module1_checkpoint = os.path.join(config.checkpoint_dir, "module1_best.pth")

    if os.path.exists(module1_checkpoint):
        load_checkpoint(module1, None, module1_checkpoint, device)
        logger.info("Loaded pretrained Module 1")

    for param in module1.parameters():
        param.requires_grad = False

    module2 = Module2(schema).to(device)

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

    optimizer = optim.AdamW(module2.parameters(), lr=config.lr_module2, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')

    for epoch in range(config.num_epochs):
        module2.train()
        train_loss = 0.0

        for batch_audio, _ in train_loader:
            batch_audio = batch_audio.to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                parameters = module1(batch_audio)

            output = module2(parameters)

            anomalies = output['anomalies']

            target_anomalies = torch.zeros_like(anomalies['source'])

            loss = (
                           criterion(anomalies['source'], target_anomalies) +
                           criterion(anomalies['filter'], target_anomalies) +
                           criterion(anomalies['resonator'], target_anomalies)
                   ) / 3.0

            loss.backward()
            torch.nn.utils.clip_grad_norm_(module2.parameters(), config.gradient_clip)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        module2.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch_audio, _ in val_loader:
                batch_audio = batch_audio.to(device)

                parameters = module1(batch_audio)
                output = module2(parameters)

                anomalies = output['anomalies']
                target_anomalies = torch.zeros_like(anomalies['source'])

                loss = (
                               criterion(anomalies['source'], target_anomalies) +
                               criterion(anomalies['filter'], target_anomalies) +
                               criterion(anomalies['resonator'], target_anomalies)
                       ) / 3.0

                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step()

        logger.info(f"Epoch {epoch + 1}/{config.num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(module2, optimizer, epoch, val_loss, config.checkpoint_dir, "module2_best.pth")
            logger.info(f"Saved best Module 2 with val loss: {val_loss:.4f}")


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

    train_module2(config, train_dataset, val_dataset)