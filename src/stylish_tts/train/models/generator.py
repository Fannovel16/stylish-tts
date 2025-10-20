import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils.parametrizations import weight_norm

from .conformer import Conformer
from .common import init_weights

from .stft import STFT
from einops import rearrange

import math
from stylish_tts.train.utils import DecoderPrediction
from .ada_norm import AdaptiveGeneratorBlock
from .ada_norm import AdaptiveLayerNorm
from .common import get_padding

import numpy as np
from functools import partial


class TorchSTFT(torch.nn.Module):
    def __init__(self, filter_length=800, hop_length=200, win_length=800):
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        window = torch.hann_window(win_length)
        self.register_buffer("window", window, persistent=False)

    def transform(self, input_data):
        forward_transform = torch.stft(
            input_data,
            self.filter_length,
            self.hop_length,
            self.win_length,
            window=self.window,
            return_complex=True,
        )
        mag = torch.abs(forward_transform) + 1e-9
        x = torch.real(forward_transform) / mag
        y = torch.imag(forward_transform) / mag
        return torch.abs(forward_transform), x, y

    def inverse(self, magnitude, x, y):
        inverse_transform = torch.istft(
            magnitude * (x + y * 1j),
            self.filter_length,
            self.hop_length,
            self.win_length,
            window=self.window,
        )

        # unsqueeze to stay consistent with conv_transpose1d implementation
        return inverse_transform.unsqueeze(-2)


def padDiff(x):
    return F.pad(
        F.pad(x, (0, 0, -1, 1), "constant", 0) - x, (0, 0, 0, -1), "constant", 0
    )


class UpsampleGenerator(torch.nn.Module):
    def __init__(
        self,
        style_dim,
        resblock_kernel_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_last_channel,
        resblock_dilation_sizes,
        upsample_kernel_sizes,
        gen_istft_n_fft,
        gen_istft_hop_size,
        sample_rate,
    ):
        super(UpsampleGenerator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.upsample_rates = upsample_rates
        self.gen_istft_n_fft = gen_istft_n_fft
        self.gen_istft_hop_size = gen_istft_hop_size

        self.m_source = SourceModuleHnNSF(
            sampling_rate=sample_rate,
            upsample_scale=math.prod(upsample_rates) * gen_istft_hop_size,
            harmonic_num=8,
            voiced_threshod=10,
        )
        self.f0_upsamp = torch.nn.Upsample(
            scale_factor=math.prod(upsample_rates) * gen_istft_hop_size, mode="linear"
        )
        self.noise_convs = nn.ModuleList()
        self.noise_res = nn.ModuleList()

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.alphas = nn.ParameterList()
        self.alphas.append(nn.Parameter(torch.ones(1, upsample_initial_channel, 1)))
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            self.alphas.append(nn.Parameter(torch.ones(1, ch, 1)))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(
                    AdaptiveGeneratorBlock(
                        channels=ch,
                        style_dim=style_dim,
                        kernel_size=k,
                        dilation=d,
                    )
                )
            c_cur = upsample_initial_channel // (2 ** (i + 1))

            if i + 1 < len(upsample_rates):  #
                stride_f0 = math.prod(upsample_rates[i + 1 :])
                self.noise_convs.append(
                    weight_norm(
                        Conv1d(
                            gen_istft_n_fft + 2,
                            c_cur,
                            kernel_size=stride_f0 * 2,
                            stride=stride_f0,
                            padding=(stride_f0 + 1) // 2,
                        )
                    )
                )
                self.noise_res.append(
                    AdaptiveGeneratorBlock(
                        channels=c_cur,
                        style_dim=style_dim,
                        kernel_size=7,
                        dilation=[1, 3, 5],
                    )
                )
            else:
                self.noise_convs.append(
                    weight_norm(Conv1d(gen_istft_n_fft + 2, c_cur, kernel_size=1))
                )
                self.noise_res.append(
                    AdaptiveGeneratorBlock(
                        channels=c_cur,
                        style_dim=style_dim,
                        kernel_size=11,
                        dilation=[1, 3, 5],
                    )
                )

        self.conformers = nn.ModuleList()
        self.post_n_fft = self.gen_istft_n_fft
        self.conv_post = weight_norm(
            Conv1d(upsample_last_channel, self.post_n_fft + 2, 7, 1, padding=3)
        )
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2**i)
            self.conformers.append(
                Conformer(
                    dim=ch,
                    depth=2,
                    dim_head=64,
                    heads=8,
                    ff_mult=4,
                    conv_expansion_factor=2,
                    conv_kernel_size=31,
                    attn_dropout=0.1,
                    ff_dropout=0.1,
                    conv_dropout=0.1,
                )
            )

        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.reflection_pad = torch.nn.ReflectionPad1d((1, 0))
        self.stft = TorchSTFT(
            filter_length=self.gen_istft_n_fft,
            hop_length=self.gen_istft_hop_size,
            win_length=self.gen_istft_n_fft,
        )

    def forward(self, mel, style, pitch, energy):
        # x: [b,d,t]
        x = mel
        f0 = pitch
        s = style
        with torch.no_grad():
            f0_len = f0.shape[1]
            f0 = self.f0_upsamp(f0[:, None]).transpose(1, 2)  # bs,n,t

            har_source, noi_source, uv = self.m_source(f0, f0_len)
            har_source = har_source.transpose(1, 2).squeeze(1)
            har_spec, har_x, har_y = self.stft.transform(har_source)
            har_phase = torch.atan2(har_y, har_x)
            har = torch.cat([har_spec, har_phase], dim=1)

        for i in range(self.num_upsamples):
            x = x + (1 / self.alphas[i]) * (torch.sin(self.alphas[i] * x) ** 2)
            x = rearrange(x, "b f t -> b t f")
            x = self.conformers[i](x)
            x = rearrange(x, "b t f -> b f t")

            x = self.ups[i](x)
            x_source = self.noise_convs[i](har)
            # if i == self.num_upsamples - 1:
            #     x = self.reflection_pad(x)
            #     x_source = self.reflection_pad(x_source)

            x_source = self.noise_res[i](x_source, s)

            x = x + x_source

            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x, s)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x, s)
            x = xs / self.num_kernels

        x = x + (1 / self.alphas[i + 1]) * (torch.sin(self.alphas[i + 1] * x) ** 2)
        x = self.conv_post(x)

        logamp = x[:, : self.post_n_fft // 2 + 1, :]
        spec = torch.exp(logamp)
        phase = x[:, self.post_n_fft // 2 + 1 :, :]
        x_phase = torch.cos(phase)
        y_phase = torch.sin(phase)
        out = self.stft.inverse(spec, x_phase, y_phase).to(x.device)
        return DecoderPrediction(audio=out, magnitude=logamp, phase=phase)


def generate_pcph(
    f0,
    voiced,
    hop_length: int,
    sample_rate: int,
    noise_amplitude=0.01,
    random_init_phase=True,
    power_factor=0.1,
    max_frequency=None,
    *args,
    **kwargs,
):
    """
    Generate pseudo-constant-power harmonic waveforms based on input F0 sequences.
    The spectral envelope of harmonics is designed to have flat spectral envelopes.

    Args:
        f0 (Tensor): F0 sequences with shape (batch, 1, frames).
        hop_length (int): Hop length of the F0 sequence.
        sample_rate (int): Sampling frequency of the waveform in Hz.
        noise_amplitude (float, optional): Amplitude of the noise component (default: 0.01).
        random_init_phase (bool, optional): Whether to initialize phases randomly (default: True).
        power_factor (float, optional): Factor to control the power of harmonics (default: 0.1).
        max_frequency (float, optional): Maximum frequency to define the number of harmonics (default: None).

    Returns:
        Tensor: Generated harmonic waveform with shape (batch, 1, frames * hop_length).
    """
    batch, _, frames = f0.size()
    device = f0.device
    noise = noise_amplitude * torch.randn(
        (batch, 1, frames * hop_length), device=device
    )
    if torch.all(voiced.round() <= 0.1):
        return noise

    # vuv = f0 > 10.0
    vuv = voiced.round().bool()
    min_f0_value = torch.min(f0[f0 > 20]).item()
    max_frequency = max_frequency if max_frequency is not None else sample_rate / 2
    max_n_harmonics = min(16, int(max_frequency / min_f0_value))
    n_harmonics = torch.ones_like(f0, dtype=torch.float)
    n_harmonics[vuv] = sample_rate / 2.0 / f0[vuv]

    indices = torch.arange(1, max_n_harmonics + 1, device=device).reshape(1, -1, 1)
    harmonic_f0 = f0 * indices

    # Compute harmonic mask
    harmonic_mask = harmonic_f0 <= (sample_rate / 2.0)
    harmonic_mask = torch.repeat_interleave(harmonic_mask, hop_length, dim=2)

    # Compute harmonic amplitude
    harmonic_amplitude = vuv * power_factor * torch.sqrt(2.0 / n_harmonics)
    harmocic_amplitude = torch.repeat_interleave(harmonic_amplitude, hop_length, dim=2)

    # Generate sinusoids
    f0 = torch.repeat_interleave(f0, hop_length, dim=2)
    radious = f0.to(torch.float64) / sample_rate
    if random_init_phase:
        radious[..., 0] += torch.rand((1, 1), device=device)
    radious = torch.cumsum(radious, dim=2)
    harmonic_phase = 2.0 * torch.pi * radious * indices
    harmonics = torch.sin(harmonic_phase).to(torch.float32)

    # Multiply coefficients to the harmonic signal
    harmonics = harmonic_mask * harmonics
    harmonics = harmocic_amplitude * torch.sum(harmonics, dim=1, keepdim=True)

    return harmonics + noise


class DownsampleConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__(in_channels, out_channels, downsample, downsample, 0)

    def forward(self, input):
        x = input
        B, C, T = x.shape
        k = self.kernel_size[0]
        s = self.stride[0]

        target_T = ((T + s - 1) // s) * s
        pad = target_T - T

        if pad > 0:
            last = x[:, :, -1:].expand(B, C, pad)
            x = torch.cat([x, last], dim=2)

        x = x[:, :, :-1]

        return super().forward(x)


class Generator(torch.nn.Module):
    def __init__(self, *, style_dim, n_fft, win_length, hop_length, config):
        super(Generator, self).__init__()

        self.up_factors = [1, 2, 5]
        self.last_hidden_dim = max(
            config.hidden_dim // (2 ** len(self.up_factors)), 128
        )
        self.prod_up_factors = math.prod(self.up_factors)
        self.amp_output_conv = Conv1d(
            self.last_hidden_dim,
            n_fft // 2 + 1,
            config.io_conv_kernel_size,
            1,
            padding=get_padding(config.io_conv_kernel_size, 1),
        )

        self.phase_output_real_conv = Conv1d(
            self.last_hidden_dim,
            n_fft // 2 + 1,
            config.io_conv_kernel_size,
            1,
            padding=get_padding(config.io_conv_kernel_size, 1),
        )
        self.phase_output_imag_conv = Conv1d(
            self.last_hidden_dim,
            n_fft // 2 + 1,
            config.io_conv_kernel_size,
            1,
            padding=get_padding(config.io_conv_kernel_size, 1),
        )

        self.amp_final_layer_norm = nn.LayerNorm(self.last_hidden_dim, eps=1e-6)
        self.phase_final_layer_norm = nn.LayerNorm(self.last_hidden_dim, eps=1e-6)
        self.apply(self._init_weights)

        self.stft = TorchSTFT(
            filter_length=n_fft,
            hop_length=hop_length // self.prod_up_factors,
            win_length=win_length,
        )
        self.prior_generator = partial(
            generate_pcph,
            hop_length=hop_length // self.prod_up_factors,
            sample_rate=24000,
        )

        self.conformers = nn.ModuleList([])
        self.upsamplers = nn.ModuleList([])
        self.amp_prior_convs = nn.ModuleList([])
        self.phase_prior_convs = nn.ModuleList([])
        self.projectors = nn.ModuleList([])
        self.convnext = nn.ModuleList([])
        for i, up_factor in enumerate(self.up_factors):
            current_dim = max(config.hidden_dim // (2**i), 128)
            next_dim = max(config.hidden_dim // (2 ** (i + 1)), 128)
            down_factor = self.prod_up_factors // math.prod(self.up_factors[: i + 1])
            self.conformers.append(
                Conformer(dim=current_dim, style_dim=style_dim, depth=1)
            )
            self.upsamplers.append(
                nn.Upsample(scale_factor=up_factor, mode="linear", align_corners=False)
            )
            self.amp_prior_convs.append(
                DownsampleConv1d(n_fft // 2 + 1, next_dim, down_factor)
            )
            self.phase_prior_convs.append(
                DownsampleConv1d(n_fft // 2 + 1, next_dim, down_factor)
            )
            self.projectors.append(
                nn.Conv1d(
                    current_dim + (next_dim * 2), next_dim, kernel_size=7, padding=3
                )
            )
            self.convnext.append(
                ConvNeXtBlock(next_dim, config.conv_intermediate_dim, style_dim)
            )

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):  # (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, *, mel, style, pitch, energy):
        with torch.no_grad():
            pitch = F.interpolate(
                pitch.unsqueeze(1),
                scale_factor=self.prod_up_factors,
                mode="linear",
                align_corners=False,
            )
            prior = self.prior_generator(pitch, (pitch > 10.0).float())
            prior = prior.squeeze(1)
            har_spec, har_x, har_y = self.stft.transform(prior)
            har_phase = torch.atan2(har_y, har_x)

        x = mel.transpose(1, 2)
        for conformer, upsample, amp_prior_conv, phase_prior_conv, project, conv in zip(
            self.conformers,
            self.upsamplers,
            self.amp_prior_convs,
            self.phase_prior_convs,
            self.projectors,
            self.convnext,
        ):
            x = conformer(x, style).transpose(1, 2)
            x = upsample(x)
            logamp_prior = amp_prior_conv(har_spec)
            phase_prior = phase_prior_conv(har_phase)
            x = project(torch.cat([x, logamp_prior, phase_prior], dim=1))
            x = conv(x, style).transpose(1, 2)

        logamp = self.amp_final_layer_norm(x)
        logamp = logamp.transpose(1, 2)
        logamp = self.amp_output_conv(logamp)

        phase = self.phase_final_layer_norm(x)
        phase = phase.transpose(1, 2)
        real = self.phase_output_real_conv(phase)
        imag = self.phase_output_imag_conv(phase)

        phase = torch.atan2(imag, real)

        logamp = F.pad(logamp, pad=(0, 1), mode="replicate")
        phase = F.pad(phase, pad=(0, 1), mode="replicate")

        spec = torch.exp(logamp)
        x = torch.cos(phase)
        y = torch.sin(phase)

        audio = self.stft.inverse(spec, x, y).to(x.device)
        audio = torch.tanh(audio)
        return DecoderPrediction(
            audio=audio,
            magnitude=logamp,
            phase=phase,
        )


class ConvNeXtBlock(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        style_dim,
    ):
        super().__init__()
        self.dwconv = torch.nn.Conv1d(
            dim, dim, kernel_size=7, padding=3, groups=dim
        )  # depthwise conv
        self.style_dim = style_dim
        if style_dim == 0:
            self.norm = torch.nn.LayerNorm(dim, eps=1e-6)
        else:
            self.norm = AdaptiveLayerNorm(style_dim, dim, eps=1e-6)
        self.pwconv1 = torch.nn.Linear(
            dim, intermediate_dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = torch.nn.SiLU()
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = torch.nn.Linear(intermediate_dim, dim)

    def forward(self, x, style=None):
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        if self.style_dim:
            x = self.norm(x, style)
        else:
            x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)

        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)

        x = residual + x
        return x


class GRN(torch.nn.Module):
    """GRN (Global Response Normalization) layer"""

    def __init__(self, dim):
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = torch.nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x
