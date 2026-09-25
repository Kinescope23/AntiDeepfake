import os
import torch
import torchaudio
from torch.utils.data import Dataset
from typing import Tuple, List


class AudioDataset(Dataset):
    def __init__(
            self,
            audio_files: List[str],
            labels: List[int],
            sample_rate: int = 16000,
            duration_seconds: float = 4.0
    ):
        self.audio_files = audio_files
        self.labels = labels
        self.sample_rate = sample_rate
        self.num_samples = int(sample_rate * duration_seconds)

    def __len__(self) -> int:
        return len(self.audio_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        import soundfile as sf
        import numpy as np

        audio_path = self.audio_files[idx]
        label = self.labels[idx]

        # Читаем аудио напрямую через soundfile, обходя torchaudio.load
        waveform_np, sr = sf.read(audio_path, dtype='float32')
        waveform = torch.from_numpy(waveform_np)

        # soundfile возвращает форму (samples,) для моно или (samples, channels) для стерео
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.transpose(0, 1)  # Приводим к (channels, samples)

        # Ресемплинг, если частота не 16000 Гц
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Преобразуем в моно
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        waveform = waveform.squeeze()

        # Обрезка или паддинг до нужной длины (4 секунды = 64000 сэмплов)
        if waveform.shape[0] < self.num_samples:
            pad_size = self.num_samples - waveform.shape[0]
            waveform = torch.nn.functional.pad(waveform, (0, pad_size))
        elif waveform.shape[0] > self.num_samples:
            start = torch.randint(0, waveform.shape[0] - self.num_samples, (1,)).item()
            waveform = waveform[start:start + self.num_samples]

        return waveform, torch.tensor(label, dtype=torch.float32)


def load_asvspoof_dataset(
        protocol_file: str,
        audio_dir: str,
        sample_rate: int = 16000,
        duration_seconds: float = 4.0
) -> AudioDataset:
    audio_files = []
    labels = []

    print(f"Loading dataset from: {protocol_file}")
    print(f"Audio directory: {audio_dir}")

    if not os.path.exists(protocol_file):
        print(f"ERROR: Protocol file not found: {protocol_file}")
        return AudioDataset([], [], sample_rate, duration_seconds)

    if not os.path.exists(audio_dir):
        print(f"ERROR: Audio directory not found: {audio_dir}")
        return AudioDataset([], [], sample_rate, duration_seconds)

    with open(protocol_file, 'r') as f:
        for line_num, line in enumerate(f):
            parts = line.strip().split()

            if len(parts) < 4:
                continue

            # Формат ASVspoof 2019 LA:
            # speaker_id file_id system_id label
            # Пример: LA_0001 LA_0001_00001 bonafide -
            # или: LA_0001 LA_0001_00002 A01 - spoof

            file_id = parts[1]

            # Label может быть в позиции 3 или 4 в зависимости от формата
            if len(parts) >= 5:
                label_str = parts[4]
            else:
                label_str = parts[3]

            # Проверяем разные расширения
            audio_path = None
            for ext in ['.wav', '.flac']:
                candidate = os.path.join(audio_dir, f"{file_id}{ext}")
                if os.path.exists(candidate):
                    audio_path = candidate
                    break

            if audio_path is None:
                if line_num < 5:
                    print(f"WARNING: Audio file not found for {file_id}")
                continue

            audio_files.append(audio_path)
            label = 1.0 if label_str == "spoof" else 0.0
            labels.append(label)

    print(f"Loaded {len(audio_files)} files")

    return AudioDataset(audio_files, labels, sample_rate, duration_seconds)