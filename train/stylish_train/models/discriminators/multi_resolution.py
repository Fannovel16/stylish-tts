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
        fft_size=1024,
        shift_size=120,
        win_length=600,
        window="hann_window",
        use_spectral_norm=False,
    ):
        super(SpecDiscriminator, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.window = getattr(torch, window)(win_length)
        self.discriminators = nn.ModuleList(
            [
                norm_f(nn.Conv2d(1, 32, kernel_size=(3, 9), padding=(1, 4))),
                norm_f(
                    nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4))
                ),
                norm_f(
                    nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4))
                ),
                norm_f(
                    nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4))
                ),
                norm_f(
                    nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                ),
            ]
        )

        self.out = norm_f(nn.Conv2d(32, 1, 3, 1, 1))

    def forward(self, y):

        fmap = []
        y = y.squeeze(1)
        y = stft(
            y,
            self.fft_size,
            self.shift_size,
            self.win_length,
            self.window.to(y.get_device()),
        )
        y = y.unsqueeze(1)
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
        fft_sizes=[1024, 2048, 512],
        hop_sizes=[120, 240, 50],
        win_lengths=[600, 1200, 240],
        window="hann_window",
    ):

        super(MultiResolutionDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList(
            [
                SpecDiscriminator(fft_sizes[0], hop_sizes[0], win_lengths[0], window),
                SpecDiscriminator(fft_sizes[1], hop_sizes[1], win_lengths[1], window),
                SpecDiscriminator(fft_sizes[2], hop_sizes[2], win_lengths[2], window),
            ]
        )

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


# class MultiResolutionDiscriminator(torch.nn.Module):
#     def __init__(
#         self,
#         fft_sizes: Tuple[int, ...] = (2048, 1024, 512),
#         num_embeddings: Optional[int] = None,
#     ):
#         """
#         Multi-Resolution Discriminator module adapted from https://github.com/descriptinc/descript-audio-codec.
#         Additionally, it allows incorporating conditional information with a learned embeddings table.
#
#         Args:
#             fft_sizes (tuple[int]): Tuple of window lengths for FFT. Defaults to (2048, 1024, 512).
#             num_embeddings (int, optional): Number of embeddings. None means non-conditional discriminator.
#                 Defaults to None.
#         """
#
#         super().__init__()
#         self.discriminators = torch.nn.ModuleList(
#             [
#                 DiscriminatorR(window_length=w, num_embeddings=num_embeddings)
#                 for w in fft_sizes
#             ]
#         )
#
#     def forward(
#         self, y: torch.Tensor, y_hat: torch.Tensor, bandwidth_id: torch.Tensor = None
#     ) -> Tuple[
#         List[torch.Tensor],
#         List[torch.Tensor],
#         List[List[torch.Tensor]],
#         List[List[torch.Tensor]],
#     ]:
#         y_d_rs = []
#         y_d_gs = []
#         fmap_rs = []
#         fmap_gs = []
#
#         for d in self.discriminators:
#             y_d_r, fmap_r = d(x=y, cond_embedding_id=bandwidth_id)
#             y_d_g, fmap_g = d(x=y_hat, cond_embedding_id=bandwidth_id)
#             y_d_rs.append(y_d_r)
#             fmap_rs.append(fmap_r)
#             y_d_gs.append(y_d_g)
#             fmap_gs.append(fmap_g)
#
#         return y_d_rs, y_d_gs, fmap_rs, fmap_gs
#
#
# class DiscriminatorR(torch.nn.Module):
#     def __init__(
#         self,
#         window_length: int,
#         num_embeddings: Optional[int] = None,
#         channels: int = 32,
#         hop_factor: float = 0.25,
#         bands: Tuple[Tuple[float, float], ...] = (
#             (0.0, 0.1),
#             (0.1, 0.25),
#             (0.25, 0.5),
#             (0.5, 0.75),
#             (0.75, 1.0),
#         ),
#     ):
#         super().__init__()
#         self.window_length = window_length
#         self.hop_factor = hop_factor
#         self.spec_fn = Spectrogram(
#             n_fft=window_length,
#             hop_length=int(window_length * hop_factor),
#             win_length=window_length,
#             power=None,
#         )
#         n_fft = window_length // 2 + 1
#         bands = [(int(b[0] * n_fft), int(b[1] * n_fft)) for b in bands]
#         self.bands = bands
#         convs = lambda: torch.nn.ModuleList(
#             [
#                 weight_norm(
#                     torch.nn.Conv2d(2, channels, (3, 9), (1, 1), padding=(1, 4))
#                 ),
#                 weight_norm(
#                     torch.nn.Conv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4))
#                 ),
#                 weight_norm(
#                     torch.nn.Conv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4))
#                 ),
#                 weight_norm(
#                     torch.nn.Conv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4))
#                 ),
#                 weight_norm(
#                     torch.nn.Conv2d(channels, channels, (3, 3), (1, 1), padding=(1, 1))
#                 ),
#             ]
#         )
#         self.band_convs = torch.nn.ModuleList([convs() for _ in range(len(self.bands))])
#
#         if num_embeddings is not None:
#             self.emb = torch.nn.Embedding(
#                 num_embeddings=num_embeddings, embedding_dim=channels
#             )
#             torch.nn.init.zeros_(self.emb.weight)
#
#         self.conv_post = weight_norm(
#             torch.nn.Conv2d(channels, 1, (3, 3), (1, 1), padding=(1, 1))
#         )
#
#     def spectrogram(self, x):
#         # Remove DC offset
#         x = x - x.mean(dim=-1, keepdims=True)
#         # Peak normalize the volume of input audio
#         x = 0.8 * x / (x.abs().max(dim=-1, keepdim=True)[0] + 1e-9)
#         x = self.spec_fn(x)
#         x = torch.view_as_real(x)
#         x = rearrange(x, "b f t c -> b c t f")
#         # Split into bands
#         x_bands = [x[..., b[0] : b[1]] for b in self.bands]
#         return x_bands
#
#     def forward(self, x: torch.Tensor, cond_embedding_id: torch.Tensor = None):
#         x = x.squeeze(1)
#         x_bands = self.spectrogram(x)
#         fmap = []
#         x = []
#         for band, stack in zip(x_bands, self.band_convs):
#             for i, layer in enumerate(stack):
#                 band = layer(band)
#                 band = torch.nn.functional.leaky_relu(band, 0.1)
#                 if i > 0:
#                     fmap.append(band)
#             x.append(band)
#         x = torch.cat(x, dim=-1)
#         if cond_embedding_id is not None:
#             emb = self.emb(cond_embedding_id)
#             h = (emb.view(1, -1, 1, 1) * x).sum(dim=1, keepdims=True)
#         else:
#             h = 0
#         x = self.conv_post(x)
#         fmap.append(x)
#         x += h
#
#         return x, fmap
