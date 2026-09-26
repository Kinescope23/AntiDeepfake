import os
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from infrastructure import ParameterSchema
from module1 import Module1
from module2 import Module2
from train.config import TrainingConfig
from train.dataset import AudioDataset, load_asvspoof_dataset
from train.utils import (
    save_checkpoint, load_checkpoint, setup_logging, setup_cuda_optimizations, memory_cleanup
)


def train_module2(
        config: TrainingConfig,
        train_dataset: AudioDataset,
        val_dataset: AudioDataset
):
    device = setup_cuda_optimizations(config)

    schema = ParameterSchema(
        sample_rate=config.sample_rate,
        duration_seconds=config.duration_seconds
    )

    logger = setup_logging(config.log_dir)
    logger.info("Starting Module 2 training (Physics Manifold Learning)")

    # Загрузка предобученного Module 1
    module1 = Module1(schema).to(device)
    module1_checkpoint = os.path.join(config.checkpoint_dir, "module1_best.pth")

    if os.path.exists(module1_checkpoint):
        load_checkpoint(module1, None, module1_checkpoint, device)
        logger.info("Loaded pretrained Module 1")
    else:
        logger.warning("No pretrained Module 1 found. Training from scratch.")

    # Заморозка Module 1
    for param in module1.parameters():
        param.requires_grad = False
    module1.eval()

    # Компиляция Module 2
    module2 = Module2(schema).to(device)

    if config.use_torch_compile and device.type == 'cuda':
        logger.info(f"Compiling Module 2 with {config.compile_backend}...")
        try:
            module2 = torch.compile(
                module2,
                backend=config.compile_backend,
                mode=config.compile_mode,
                fullgraph=False
            )
            logger.info("✓ Module 2 compiled")
        except Exception as e:
            logger.warning(f"torch.compile failed for Module 2: {e}")

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

    optimizer = optim.AdamW(
        module2.parameters(),
        lr=config.lr_module2,
        weight_decay=config.weight_decay,
        fused=True if device.type == 'cuda' else False
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)
    criterion = nn.MSELoss()

    dtype = torch.bfloat16 if config.precision_dtype == "bfloat16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=config.use_mixed_precision and device.type == 'cuda')

    best_val_loss = float('inf')

    for epoch in range(config.num_epochs):
        module2.train()
        train_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        train_pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{config.num_epochs} [Train]",
            leave=False,
            dynamic_ncols=True
        )

        for batch_idx, (batch_audio, _) in enumerate(train_pbar):
            batch_audio = batch_audio.to(device, non_blocking=config.non_blocking)

            with torch.cuda.amp.autocast(enabled=config.use_mixed_precision, dtype=dtype):
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

                loss = loss / config.accumulation_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % config.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(module2.parameters(), config.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            train_loss += loss.item() * config.accumulation_steps
            train_pbar.set_postfix({"loss": f"{loss.item() * config.accumulation_steps:.4f}"})

        if (batch_idx + 1) % config.accumulation_steps != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(module2.parameters(), config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        train_loss /= len(train_loader)

        module2.eval()
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
                val_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        val_loss /= len(val_loader)
        scheduler.step()

        logger.info(f"Epoch {epoch+1}/{config.num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(module2, optimizer, epoch, val_loss, config.checkpoint_dir, "module2_best.pth")
            logger.info(f"Saved best Module 2 with val loss: {val_loss:.4f}")

        if (epoch + 1) % config.empty_cache_every_n_epochs == 0:
            memory_cleanup()

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

    train_module2(config, train_dataset, val_dataset)