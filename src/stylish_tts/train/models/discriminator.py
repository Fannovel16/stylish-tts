from typing import List, Optional, Tuple
import torch
from torch import nn
from torch.nn import functional as F
from torchaudio.transforms import Spectrogram
from torch.nn.utils.parametrizations import weight_norm
from einops import rearrange


LRELU_SLOPE = 0.1


def stft(x, fft_size, hop_size, win_length, window):
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.
    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """
    x_stft = torch.stft(x, fft_size, hop_size, win_length, window, return_complex=True)
    real = x_stft[..., 0]
    imag = x_stft[..., 1]

    return torch.abs(x_stft).transpose(2, 1)


class SpecDiscriminator(nn.Module):
    """docstring for Discriminator."""

    def __init__(
        self,
        use_spectral_norm=False,
    ):
        super(SpecDiscriminator, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.discriminators = nn.ModuleList(
            [
                norm_f(nn.Conv2d(1, 8, kernel_size=(3, 9), padding=(1, 4))),
                norm_f(
                    nn.Conv2d(8, 8, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4))
                ),
                norm_f(
                    nn.Conv2d(8, 8, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4))
                ),
                norm_f(
                    nn.Conv2d(8, 8, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4))
                ),
                norm_f(
                    nn.Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                ),
            ]
        )

        self.out = norm_f(nn.Conv2d(8, 1, 3, 1, 1))

    def forward(self, y):

        fmap = []
        for i, d in enumerate(self.discriminators):
            y = d(y)
            y = F.leaky_relu(y, LRELU_SLOPE)
            fmap.append(y)

        y = self.out(y)
        fmap.append(y)

        return torch.flatten(y, 1, -1), fmap


class MultiResolutionDiscriminator(torch.nn.Module):

    def __init__(
        self,
        discriminator_count,
    ):

        super(MultiResolutionDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList(
            [SpecDiscriminator() for _ in range(discriminator_count)]
        )

    def forward(self, *, target_list, pred_list):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for target, pred, disc in zip(target_list, pred_list, self.discriminators):
            y_d_r, fmap_r = disc(target)
            y_d_g, fmap_g = disc(pred)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
