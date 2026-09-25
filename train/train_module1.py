import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from infrastructure import ParameterSchema
from module1 import Module1
from train.config import TrainingConfig
from train.dataset import AudioDataset, load_asvspoof_dataset
from train.losses import MultiScaleSpectralLoss
from train.utils import save_checkpoint, setup_logging


def synthesize_from_parameters(f0, harmonics, noise, schema):
    """
    Корректный синтез сигнала из кадровых параметров.
    Сначала интерполирует кадровые признаки до частоты дискретизации,
    затем накапливает фазу для каждой гармоники.
    """
    batch_size, num_frames, _ = f0.shape

    # 1. Интерполяция кадровых признаков до уровня сэмплов (linear interpolation)
    f0_up = torch.nn.functional.interpolate(
        f0.transpose(1, 2), size=schema.num_samples, mode='linear', align_corners=True
    ).transpose(1, 2)  # Форма: (batch_size, num_samples, 1)

    harm_up = torch.nn.functional.interpolate(
        harmonics.transpose(1, 2), size=schema.num_samples, mode='linear', align_corners=True
    ).transpose(1, 2)  # Форма: (batch_size, num_samples, num_harmonics)

    harmonic_signal = torch.zeros(batch_size, schema.num_samples, device=f0.device)

    # 2. Аддитивный синтез первых 10 гармоник (для ускорения обучения на этапе реконструкции)
    num_harmonics_to_synth = min(10, schema.num_harmonics)

    for k in range(num_harmonics_to_synth):
        # Частота k-й гармоники
        freq = (k + 1) * f0_up  # Форма: (batch_size, num_samples, 1)

        # Накопление фазы: d(phase)/dt = 2*pi*f. Интегрируем по времени (dim=1)
        phase = 2 * torch.pi * torch.cumsum(freq, dim=1) / schema.sample_rate  # Форма: (batch_size, num_samples, 1)

        # Умножаем амплитуду на синус фазы и добавляем к сигналу
        amp = harm_up[:, :, k].unsqueeze(-1)  # Форма: (batch_size, num_samples, 1)
        harmonic_signal += (amp * torch.sin(phase)).squeeze(-1)

    return harmonic_signal


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

    # Очистка кэша CUDA перед началом для освобождения фрагментированной памяти
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger = setup_logging(config.log_dir)
    logger.info("Starting Module 1 training (Reconstruction)")

    model = Module1(schema).to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    optimizer = optim.AdamW(model.parameters(), lr=config.lr_module1, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)
    criterion = MultiScaleSpectralLoss()

    # Инициализация скалера для смешанной точности (AMP)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_val_loss = float('inf')

    for epoch in range(config.num_epochs):
        model.train()
        train_loss = 0.0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.num_epochs} [Train]", leave=False)

        for batch_audio, _ in train_pbar:
            batch_audio = batch_audio.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Включаем смешанную точность для прямого прохода
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                parameters = model(batch_audio)

                f0 = parameters['f0']
                harmonics = parameters['harmonic_amplitudes']
                noise = parameters['noise_magnitude']

                reconstructed = synthesize_from_parameters(f0, harmonics, noise, schema)
                loss = criterion(batch_audio, reconstructed)

            # Масштабируем потерю и делаем backward pass
            scaler.scale(loss).backward()

            # Анскейлим градиенты перед клиппингом
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)

            # Делаем шаг оптимизатора и обновляем скалер
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss /= len(train_loader)

        # Валидация
        model.eval()
        val_loss = 0.0

        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{config.num_epochs} [Val]  ", leave=False)

        with torch.no_grad():
            for batch_audio, _ in val_pbar:
                batch_audio = batch_audio.to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    parameters = model(batch_audio)

                    f0 = parameters['f0']
                    harmonics = parameters['harmonic_amplitudes']
                    noise = parameters['noise_magnitude']

                    reconstructed = synthesize_from_parameters(f0, harmonics, noise, schema)
                    loss = criterion(batch_audio, reconstructed)

                val_loss += loss.item()
                val_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        val_loss /= len(val_loader)
        scheduler.step()

        logger.info(f"Epoch {epoch + 1}/{config.num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, config.checkpoint_dir, "module1_best.pth")
            logger.info(f"Saved best model with val loss: {val_loss:.4f}")

        # Очистка кэша в конце эпохи для предотвращения фрагментации памяти
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


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

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")

    if len(train_dataset) == 0:
        print("ERROR: Train dataset is empty! Check paths and file extensions.")
        exit(1)

    train_module1(config, train_dataset, val_dataset)
