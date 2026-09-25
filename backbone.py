import torch
import torch.nn as nn


class CRNNBackbone(nn.Module):
    def __init__(self, schema):
        super().__init__()
        self.schema = schema
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, 9, padding=12, dilation=3),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, 27, padding=117, dilation=9),
            nn.ReLU(),
            nn.BatchNorm1d(256)
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(256, 512, 81, padding=1080, dilation=27),
            nn.ReLU(),
            nn.BatchNorm1d(512)
        )
        self.pool = nn.AdaptiveAvgPool1d(output_size=schema.num_frames)
        self.lstm = nn.LSTM(512, 256, batch_first=True, bidirectional=True)
        self.layer_norm = nn.LayerNorm(512)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool(x)
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x = self.layer_norm(x)
        return x


class ParameterHeads(nn.Module):
    def __init__(self, schema):
        super().__init__()
        self.f0_head = nn.Sequential(nn.Linear(512, 1), nn.Softplus())
        self.harmonic_head = nn.Sequential(
            nn.Linear(512, schema.num_harmonics),
            nn.Softmax(dim=-1)
        )
        self.tilt_head = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())
        self.formant_head = nn.Sequential(
            nn.Linear(512, schema.num_formants),
            nn.Softplus()
        )
        self.noise_head = nn.Sequential(
            nn.Linear(512, schema.num_noise_bands),
            nn.Softplus()
        )

    def forward(self, x):
        return {
            "f0": self.f0_head(x),
            "harmonic_amplitudes": self.harmonic_head(x),
            "glottal_tilt": self.tilt_head(x),
            "formant_bandwidths": self.formant_head(x),
            "noise_magnitude": self.noise_head(x)
        }