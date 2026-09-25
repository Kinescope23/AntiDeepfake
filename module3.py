import torch
import torch.nn as nn
from infrastructure import BaseModule, ParameterSchema


class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.relu(out)


class Module3(BaseModule):
    def __init__(self, schema: ParameterSchema):
        super().__init__(schema)

        in_channels = 1 + schema.num_physical_params

        self.conv_init = nn.Conv2d(in_channels, 64, kernel_size=1)

        self.block1 = ResidualBlock2D(64, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 4))

        self.block2 = ResidualBlock2D(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 4))

        self.block3 = ResidualBlock2D(128, 128)

    def forward(self, spectrogram: torch.Tensor, parameters: dict) -> torch.Tensor:
        batch_size, num_frames_spec, num_freq_bins = spectrogram.shape

        param_tensors = [
            parameters['f0'],
            parameters['harmonic_amplitudes'],
            parameters['glottal_tilt'],
            parameters['formant_bandwidths'],
            parameters['noise_magnitude']
        ]

        P = torch.cat(param_tensors, dim=-1)

        if P.shape[1] != num_frames_spec:
            P = torch.nn.functional.interpolate(
                P.transpose(1, 2),
                size=num_frames_spec,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)

        S_x = spectrogram.unsqueeze(1)

        P_reshaped = P.permute(0, 2, 1).unsqueeze(-1)
        P_expanded = P_reshaped.expand(-1, -1, -1, num_freq_bins)

        X_in = torch.cat([S_x, P_expanded], dim=1)

        x = self.conv_init(X_in)
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.pool2(x)
        x = self.block3(x)

        x = torch.mean(x, dim=-1)
        x = x.permute(0, 2, 1)

        return x

    def get_output_shape(self, input_shape: tuple) -> tuple:
        batch_size = input_shape[0]
        num_frames = input_shape[1]
        return (batch_size, num_frames, 128)