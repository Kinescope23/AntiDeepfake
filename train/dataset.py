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
        audio_path = self.audio_files[idx]
        label = self.labels[idx]

        waveform, sr = torchaudio.load(audio_path)

        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        waveform = waveform.squeeze()

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

    with open(protocol_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue

            file_id = parts[1]
            label_str = parts[4] if len(parts) > 4 else parts[3]

            audio_path = os.path.join(audio_dir, f"{file_id}.flac")
            if not os.path.exists(audio_path):
                audio_path = os.path.join(audio_dir, f"{file_id}.wav")

            if os.path.exists(audio_path):
                audio_files.append(audio_path)
                label = 1.0 if label_str == "spoof" else 0.0
                labels.append(label)

    return AudioDataset(audio_files, labels, sample_rate, duration_seconds)