import torch
import torch.nn as nn
import torchaudio


class MultiScaleSpectralLoss(nn.Module):
    def __init__(self, fft_sizes: list = [2048, 1024, 512, 256, 128, 64]):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.l1_loss = nn.L1Loss()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0

        for fft_size in self.fft_sizes:
            hop_size = fft_size // 4
            window = torch.hann_window(fft_size, device=x.device)

            X = torch.stft(x, fft_size, hop_size, window=window, return_complex=True)
            Y = torch.stft(y, fft_size, hop_size, window=window, return_complex=True)

            X_mag = torch.abs(X)
            Y_mag = torch.abs(Y)

            log_X = torch.log(X_mag + 1e-8)
            log_Y = torch.log(Y_mag + 1e-8)

            loss = self.l1_loss(X_mag, Y_mag) + self.l1_loss(log_X, log_Y)
            total_loss = total_loss + loss

        return total_loss / len(self.fft_sizes)


class PhysicsRegularizationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
            self,
            anomalies: dict,
            real_mask: torch.Tensor
    ) -> torch.Tensor:
        if not real_mask.any():
            return torch.tensor(0.0, device=real_mask.device)

        source_anomalies = anomalies['source'][real_mask]
        filter_anomalies = anomalies['filter'][real_mask]
        resonator_anomalies = anomalies['resonator'][real_mask]

        loss = (
                       torch.mean(source_anomalies) +
                       torch.mean(filter_anomalies) +
                       torch.mean(resonator_anomalies)
               ) / 3.0

        return loss