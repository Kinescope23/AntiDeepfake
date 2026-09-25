from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from einops import rearrange


@dataclass(frozen=True)
class ParameterSchema:
    sample_rate: int = 16000
    duration_seconds: float = 1.0
    frame_hop_size: int = 256
    stft_window_size: int = 1024
    num_harmonics: int = 64
    num_formants: int = 4
    num_noise_bands: int = 65
    phoneme_embedding_dim: int = 256

    @property
    def num_samples(self) -> int:
        return int(self.sample_rate * self.duration_seconds)

    @property
    def num_frames(self) -> int:
        return (self.num_samples // self.frame_hop_size) + 1

    @property
    def num_stft_bins(self) -> int:
        return self.stft_window_size // 2 + 1

    @property
    def num_physical_params(self) -> int:
        return (
            1
            + self.num_harmonics
            + 1
            + self.num_formants
            + self.num_noise_bands
        )

    @property
    def total_input_channels(self) -> int:
        return 1 + self.num_physical_params


class BaseModule(ABC, nn.Module):
    def __init__(self, schema: ParameterSchema):
        super().__init__()
        self.schema = schema

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def get_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        raise NotImplementedError


class AudioProcessor(nn.Module):
    def __init__(self, schema: ParameterSchema):
        super().__init__()
        self.schema = schema
        self.stft_transform = torchaudio.transforms.Spectrogram(
            n_fft=schema.stft_window_size,
            win_length=schema.stft_window_size,
            hop_length=schema.frame_hop_size,
            power=1.0,
            normalized=True,
            center=True,
        )
        self.register_buffer(
            "hann_window",
            torch.hann_window(schema.stft_window_size),
            persistent=False,
        )

    def normalize_audio(self, audio: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(audio ** 2, dim=-1, keepdim=True) + 1e-8)
        return audio / rms

    def compute_log_spectrogram(
        self, audio: torch.Tensor, eps: float = 1e-8
    ) -> torch.Tensor:
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        magnitude = self.stft_transform(audio)
        log_magnitude = torch.log(magnitude + eps)
        return log_magnitude

    def resample(
        self, audio: torch.Tensor, original_sr: int
    ) -> torch.Tensor:
        if original_sr == self.schema.sample_rate:
            return audio
        resampler = torchaudio.transforms.Resample(
            orig_freq=original_sr,
            new_freq=self.schema.sample_rate,
        ).to(audio.device)
        return resampler(audio)

    def crop_or_pad(
        self, audio: torch.Tensor, target_length: int
    ) -> torch.Tensor:
        current_length = audio.shape[-1]
        if current_length < target_length:
            pad_size = target_length - current_length
            return torch.nn.functional.pad(audio, (0, pad_size))
        elif current_length > target_length:
            return audio[..., :target_length]
        return audio

    def forward(
        self, audio: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        audio = self.normalize_audio(audio)
        spectrogram = self.compute_log_spectrogram(audio)
        return {
            "audio_normalized": audio,
            "log_spectrogram": spectrogram,
        }


