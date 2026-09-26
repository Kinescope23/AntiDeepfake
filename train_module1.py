import os
import gc
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
from train.utils import (
    save_checkpoint, setup_logging, setup_cuda_optimizations, memory_cleanup
)


def synthesize_from_parameters(f0, harmonics, noise, schema):
    """Оптимизированный синтез с векторизацией."""
    batch_size, num_frames, _ = f0.shape

    f0_up = torch.nn.functional.interpolate(
        f0.transpose(1, 2), size=schema.num_samples, mode='linear', align_corners=True
    ).transpose(1, 2)

    harm_up = torch.nn.functional.interpolate(
        harmonics.transpose(1, 2), size=schema.num_samples, mode='linear', align_corners=True
    ).transpose(1, 2)

    harmonic_signal = torch.zeros(batch_size, schema.num_samples, device=f0.device)

    num_harmonics_to_synth = min(10, schema.num_harmonics)

    # Векторизованное вычисление всех гармоник сразу (быстрее цикла)
    k_tensor = torch.arange(1, num_harmonics_to_synth + 1, device=f0.device).view(1, 1, -1)
    freqs = k_tensor * f0_up  # (batch, samples, K)

    phase = 2 * torch.pi * torch.cumsum(freqs, dim=1) / schema.sample_rate

    amps = harm_up[:, :, :num_harmonics_to_synth]

    harmonic_signal = (amps * torch.sin(phase)).sum(dim=-1)

    return harmonic_signal


def train_module1(
        config: TrainingConfig,
        train_dataset: AudioDataset,
        val_dataset: AudioDataset
):
    # Настройка CUDA оптимизаций
    device = setup_cuda_optimizations(config)

    schema = ParameterSchema(
        sample_rate=config.sample_rate,
        duration_seconds=config.duration_seconds
    )

    logger = setup_logging(config.log_dir)
    logger.info("Starting Module 1 training (Reconstruction) with CUDA optimizations")
    logger.info(f"Effective batch size: {config.batch_size * config.accumulation_steps}")
    logger.info(f"Mixed precision: {config.use_mixed_precision} ({config.precision_dtype})")
    logger.info(f"torch.compile: {config.use_torch_compile} (backend={config.compile_backend})")

    model = Module1(schema).to(device)

    # torch.compile для JIT-ускорения
    if config.use_torch_compile and device.type == 'cuda':
        logger.info(f"Compiling model with {config.compile_backend}...")
        try:
            model = torch.compile(
                model,
                backend=config.compile_backend,
                mode=config.compile_mode,
                fullgraph=False  # False для совместимости с dynamic shapes
            )
            logger.info("Model compiled successfully")
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}. Falling back to eager mode.")

    # DataLoader с оптимизациями
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers and config.num_workers > 0,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers and config.num_workers > 0,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None
    )

    # Fused AdamW (быстрее на CUDA)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.lr_module1,
        weight_decay=config.weight_decay,
        fused=True if device.type == 'cuda' else False
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)
    criterion = MultiScaleSpectralLoss()

    # GradScaler для mixed precision
    dtype = torch.bfloat16 if config.precision_dtype == "bfloat16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=config.use_mixed_precision and device.type == 'cuda')

    best_val_loss = float('inf')
    global_step = 0

    for epoch in range(config.num_epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        train_pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{config.num_epochs} [Train]",
            leave=False,
            dynamic_ncols=True
        )

        for batch_idx, (batch_audio, _) in enumerate(train_pbar):
            batch_audio = batch_audio.to(device, non_blocking=config.non_blocking)

            # Mixed precision forward pass
            with torch.cuda.amp.autocast(enabled=config.use_mixed_precision, dtype=dtype):
                parameters = model(batch_audio)

                f0 = parameters['f0']
                harmonics = parameters['harmonic_amplitudes']
                noise = parameters['noise_magnitude']

                reconstructed = synthesize_from_parameters(f0, harmonics, noise, schema)
                loss = criterion(batch_audio, reconstructed)

                # Нормализация по accumulation steps
                loss = loss / config.accumulation_steps

            # Backward с масштабированием
            scaler.scale(loss).backward()

            # Шаг оптимизатора только после накопления нужного количества батчей
            if (batch_idx + 1) % config.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                # Периодическая очистка памяти
                if global_step % config.gc_collect_every_n_steps == 0:
                    memory_cleanup()

            train_loss += loss.item() * config.accumulation_steps
            global_step += 1

            if batch_idx % config.log_every_n_steps == 0:
                train_pbar.set_postfix({
                    "loss": f"{loss.item() * config.accumulation_steps:.4f}",
                    "step": global_step
                })

        # Если в конце эпохи остались накопленные градиенты — делаем шаг
        if (batch_idx + 1) % config.accumulation_steps != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        train_loss /= len(train_loader)

        # Валидация
        if (epoch + 1) % config.validate_every_n_epochs == 0:
            model.eval()
            val_loss = 0.0

            val_pbar = tqdm(
                val_loader,
                desc=f"Epoch {epoch+1}/{config.num_epochs} [Val]  ",
                leave=False,
                dynamic_ncols=True
            )

            with torch.no_grad():
                for batch_audio, _ in val_pbar:
                    batch_audio = batch_audio.to(device, non_blocking=config.non_blocking)

                    with torch.cuda.amp.autocast(enabled=config.use_mixed_precision, dtype=dtype):
                        parameters = model(batch_audio)

                        f0 = parameters['f0']
                        harmonics = parameters['harmonic_amplitudes']
                        noise = parameters['noise_magnitude']

                        reconstructed = synthesize_from_parameters(f0, harmonics,noise, schema)
                        loss = criterion(batch_audio, reconstructed)

                    val_loss += loss.item()
                    val_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            val_loss /= len(val_loader)

            logger.info(
                f"Epoch {epoch+1}/{config.num_epochs}, "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, epoch, val_loss, config.checkpoint_dir, "module1_best.pth")
                logger.info(f"Saved best model with val loss: {val_loss:.4f}")
        else:
            logger.info(f"Epoch {epoch+1}/{config.num_epochs}, Train Loss: {train_loss:.4f}")

        scheduler.step()

        # Периодическая очистка кэша
        if (epoch + 1) % config.empty_cache_every_n_epochs == 0:
            memory_cleanup()

if __name__ == "__main__":
    config = TrainingConfig()

    train_dataset = load_asvspoof_dataset(
        protocol_file="D:\\data\\ASVspoof2019LA\\CM_protocol\\CM_train.trn",
        audio_dir="D:\\data\\ASVspoof2019LA\\WAV\\train",
        sample_rate=config.sample_rate,
        duration_seconds=config.duration_seconds
    )

    val_dataset = load_asvspoof_dataset(
        protocol_file="D:\\data\\ASVspoof2019LA\\CM_protocol\\CM_dev.trl",
        audio_dir="D:\\data\\ASVspoof2019LA\\WAV\\dev",
        sample_rate=config.sample_rate,
        duration_seconds=config.duration_seconds
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")

    if len(train_dataset) == 0:
        print("ERROR: Train dataset is empty! Check paths and file extensions.")
        exit(1)

    train_module1(config, train_dataset, val_dataset)