import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from infrastructure import ParameterSchema
from pipeline import DeepfakeDetectorPipeline
from train.config import TrainingConfig
from train.dataset import AudioDataset, load_asvspoof_dataset
from train.losses import MultiScaleSpectralLoss, PhysicsRegularizationLoss
from train.metrics import calculate_eer, calculate_min_tdcf, calculate_auc
from train.utils import save_checkpoint, load_checkpoint, setup_logging


def evaluate(model, dataloader, device):
    model.eval()

    all_scores = []
    all_labels = []

    with torch.no_grad():
        for batch_audio, batch_labels in dataloader:
            batch_audio = batch_audio.to(device)
            batch_labels = batch_labels.numpy()

            output = model(batch_audio)
            p_fake = output['p_fake'].cpu().numpy()

            all_scores.extend(p_fake.flatten())
            all_labels.extend(batch_labels)

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    eer = calculate_eer(all_scores, all_labels)
    min_tdcf = calculate_min_tdcf(all_scores, all_labels)
    auc_score = calculate_auc(all_scores, all_labels)

    return eer, min_tdcf, auc_score


def train_pipeline(
        config: TrainingConfig,
        train_dataset: AudioDataset,
        val_dataset: AudioDataset,
        test_dataset: AudioDataset
):
    device = torch.device(config.device)
    schema = ParameterSchema(
        sample_rate=config.sample_rate,
        duration_seconds=config.duration_seconds
    )

    logger = setup_logging(config.log_dir)
    logger.info("Starting Pipeline training (End-to-End)")

    model = DeepfakeDetectorPipeline(schema).to(device)

    module1_checkpoint = os.path.join(config.checkpoint_dir, "module1_best.pth")
    module2_checkpoint = os.path.join(config.checkpoint_dir, "module2_best.pth")

    if os.path.exists(module1_checkpoint):
        load_checkpoint(model.module1, None, module1_checkpoint, device)
        logger.info("Loaded pretrained Module 1")

    if os.path.exists(module2_checkpoint):
        load_checkpoint(model.module2, None, module2_checkpoint, device)
        logger.info("Loaded pretrained Module 2")

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

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )

    optimizer = optim.AdamW([
        {'params': model.module1.parameters(), 'lr': config.lr_module1},
        {'params': model.module2.parameters(), 'lr': config.lr_module2},
        {'params': model.module3.parameters(), 'lr': config.lr_module3},
        {'params': model.module4.parameters(), 'lr': config.lr_module4}
    ], weight_decay=config.weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)

    criterion_bce = nn.BCELoss()
    criterion_physics = PhysicsRegularizationLoss()
    criterion_recon = MultiScaleSpectralLoss()

    best_val_eer = float('inf')

    for epoch in range(config.num_epochs):
        model.train()
        train_loss = 0.0

        for batch_audio, batch_labels in train_loader:
            batch_audio = batch_audio.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()

            output = model(batch_audio)

            p_fake = output['p_fake'].squeeze()
            anomalies = output['module2_anomalies']
            parameters = output['parameters']

            loss_bce = criterion_bce(p_fake, batch_labels)

            real_mask = (batch_labels == 0.0)
            loss_physics = criterion_physics(anomalies, real_mask)

            f0 = parameters['f0']
            harmonics = parameters['harmonic_amplitudes']
            noise = parameters['noise_magnitude']

            reconstructed = synthesize_from_parameters(f0, harmonics, noise, schema)
            loss_recon = criterion_recon(batch_audio, reconstructed)

            loss = (
                    loss_bce +
                    config.lambda_physics * loss_physics +
                    config.lambda_recon * loss_recon
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        val_eer, val_tdcf, val_auc = evaluate(model, val_loader, device)

        scheduler.step()

        logger.info(
            f"Epoch {epoch + 1}/{config.num_epochs}, "
            f"Train Loss: {train_loss:.4f}, "
            f"Val EER: {val_eer:.4f}, "
            f"Val t-DCF: {val_tdcf:.4f}, "
            f"Val AUC: {val_auc:.4f}"
        )

        if val_eer < best_val_eer:
            best_val_eer = val_eer
            save_checkpoint(model, optimizer, epoch, val_eer, config.checkpoint_dir, "pipeline_best.pth")
            logger.info(f"Saved best pipeline with EER: {val_eer:.4f}")

            test_eer, test_tdcf, test_auc = evaluate(model, test_loader, device)
            logger.info(
                f"Test Results - EER: {test_eer:.4f}, "
                f"t-DCF: {test_tdcf:.4f}, "
                f"AUC: {test_auc:.4f}"
            )


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

    train_dataset = load_asvspoof_dataset(
        protocol_file="E:\\data\\ASVspoof2019LA\\CM_protocol\\CM_train.trn",
        audio_dir="E:\\data\\ASVspoof2019LA\\WAV\\train",
        sample_rate=config.sample_rate,
        duration_seconds=config.duration_seconds
    )

    val_dataset = load_asvspoof_dataset(
        protocol_file="E:\\data\\ASVspoof2019LA\\CM_protocol\\CM_dev.trl",
        audio_dir="E:\\data\\ASVspoof2019LA\\WAV\\dev",
        sample_rate=config.sample_rate,
        duration_seconds=config.duration_seconds
    )

    test_dataset = load_asvspoof_dataset(
        protocol_file="E:\\data\\ASVspoof2019LA\\CM_protocol\\CM_eval.trl",
        audio_dir="E:\\data\\ASVspoof2019LA\\WAV\\eval",
        sample_rate=config.sample_rate,
        duration_seconds=config.duration_seconds
    )

    train_pipeline(config, train_dataset, val_dataset, test_dataset)