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
from .zip_modules import Bypass, SimpleDownsample, SimpleUpsample

import numpy as np


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


# The following code was adapted from: https://github.com/nii-yamagishilab/project-CURRENNT-scripts/tree/master/waveform-modeling/project-NSF-v2-pretrained

# BSD 3-Clause License
#
# Copyright (c) 2018, Yamagishi Laboratory, National Institute of Informatics
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


class SineGenerator(nn.Module):
    """
    Definition of sine generator

    Generates sine waveforms with optional harmonics and additive noise.
    Can be used to create harmonic noise source for neural vocoders.

    Args:
        samp_rate (int): Sampling rate in Hz.
        harmonic_num (int): Number of harmonic overtones (default 0).
        sine_amp (float): Amplitude of sine-waveform (default 0.1).
        noise_std (float): Standard deviation of Gaussian noise (default 0.003).
        voiced_threshold (float): F0 threshold for voiced/unvoiced classification (default 0).
    """

    def __init__(
        self,
        samp_rate,
        harmonic_num=0,
        sine_amp=0.1,
        noise_std=0.003,
        voiced_threshold=0,
    ):
        super(SineGenerator, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold

        self.merge = nn.Sequential(
            nn.Linear(self.dim + 1, 1, bias=False),
            nn.Tanh(),
        )

    def _f02uv(self, f0):
        # generate uv signal
        uv = torch.ones_like(f0)
        uv = uv * (f0 > self.voiced_threshold)
        return uv

    def _f02sine(self, f0_values):
        """f0_values: (batchsize, length, dim)
        where dim indicates fundamental tone and overtones
        """
        # convert to F0 in rad. The integer part n can be ignored
        # because 2 * np.pi * n doesn't affect phase
        rad_values = (f0_values / self.sampling_rate) % 1

        # initial phase noise (no noise for fundamental component)
        rand_ini = torch.rand(
            f0_values.shape[0], f0_values.shape[2], device=f0_values.device
        )
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini

        # instantanouse phase sine[t] = sin(2*pi \sum_i=1 ^{t} rad)
        tmp_over_one = torch.cumsum(rad_values, 1) % 1
        tmp_over_one_idx = (tmp_over_one[:, 1:, :] - tmp_over_one[:, :-1, :]) < 0
        cumsum_shift = torch.zeros_like(rad_values)
        cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0

        sines = torch.sin(torch.cumsum(rad_values + cumsum_shift, dim=1) * 2 * np.pi)

        return sines

    def forward(self, f0, energy):
        with torch.no_grad():
            f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim, device=f0.device)
            # fundamental component
            f0_buf[:, :, 0] = f0[:, :, 0]
            for idx in np.arange(self.harmonic_num):
                f0_buf[:, :, idx + 1] = f0_buf[:, :, 0] * (idx + 2)

            sine_waves = self._f02sine(f0_buf) * self.sine_amp

            uv = self._f02uv(f0)

            noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
            noise = noise_amp * torch.randn_like(sine_waves)

            sine_waves = sine_waves * uv + noise

        # merge with grad
        return self.merge(torch.cat([sine_waves, energy], dim=-1))


class SimpleDownConv1d(nn.Conv1d):
    def forward(self, input):
        assert (self.kernel_size[0] == self.stride[0]) and self.padding[
            0
        ] == 0, (
            "To mimic SimpleDownsample, use kernel_size == stride and padding must be 0"
        )
        x = input
        B, C, T = x.shape
        k = self.kernel_size[0]
        s = self.stride[0]

        target_T = ((T + s - 1) // s) * s
        pad = target_T - T

        if pad > 0:
            # Replicate-pad by repeating last element on the **right**
            last = x[:, :, -1:].expand(B, C, pad)
            x = torch.cat([x, last], dim=2)

        return super().forward(x)


class Generator(torch.nn.Module):
    def __init__(self, *, style_dim, n_fft, win_length, hop_length, config):
        super(Generator, self).__init__()

        self.up_factors = [1, 1, 2, 1]
        self.amp_input_conv = Conv1d(
            config.input_dim,
            config.hidden_dim,
            config.io_conv_kernel_size,
            1,
            padding=get_padding(config.io_conv_kernel_size, 1),
        )

        self.amp_output_conv = Conv1d(
            config.hidden_dim,
            n_fft // 2 + 1,
            config.io_conv_kernel_size,
            1,
            padding=get_padding(config.io_conv_kernel_size, 1),
        )

        self.phase_output_real_conv = Conv1d(
            config.hidden_dim,
            n_fft // 2 + 1,
            config.io_conv_kernel_size,
            1,
            padding=get_padding(config.io_conv_kernel_size, 1),
        )
        self.phase_output_imag_conv = Conv1d(
            config.hidden_dim,
            n_fft // 2 + 1,
            config.io_conv_kernel_size,
            1,
            padding=get_padding(config.io_conv_kernel_size, 1),
        )

        self.amp_norm = nn.LayerNorm(config.hidden_dim, eps=1e-6)
        self.phase_norm = nn.LayerNorm(config.hidden_dim, eps=1e-6)
        self.phase_convnext = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim=config.hidden_dim,
                    intermediate_dim=config.conv_intermediate_dim,
                    style_dim=style_dim,
                )
                for _ in range(len(self.up_factors))
            ]
        )

        self.amp_pre_conformer = Conformer(
            dim=config.hidden_dim,
            style_dim=style_dim,
            depth=1,
            conv_kernel_size=31,
        )
        self.amp_post_conformer = Conformer(
            dim=config.hidden_dim,
            style_dim=style_dim,
            depth=1,
            conv_kernel_size=31,
        )
        self.amp_convnext = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim=config.hidden_dim,
                    intermediate_dim=config.conv_intermediate_dim,
                    style_dim=style_dim,
                )
                for _ in range(len(self.up_factors))
            ]
        )
        self.amp_final_layer_norm = nn.LayerNorm(config.hidden_dim, eps=1e-6)
        self.phase_final_layer_norm = nn.LayerNorm(config.hidden_dim, eps=1e-6)
        self.apply(self._init_weights)

        self.prod_up_factors = math.prod(self.up_factors)
        self.stft = TorchSTFT(
            filter_length=n_fft,
            hop_length=hop_length // self.prod_up_factors,
            win_length=win_length,
        )
        self.m_source = SineGenerator(24000)
        self.pre_upsampler = SimpleUpsample(self.prod_up_factors)
        self.prior_downsamplers = nn.ModuleList(
            [
                SimpleDownConv1d(
                    n_fft + 2,
                    config.hidden_dim,
                    self.prod_up_factors // ds,
                    self.prod_up_factors // ds,
                    0,
                )
                for ds in self.up_factors
            ]
        )
        self.downsamplers = nn.ModuleList(
            [SimpleDownsample(self.prod_up_factors // ds) for ds in self.up_factors]
        )
        self.upsamplers = nn.ModuleList(
            [SimpleUpsample(self.prod_up_factors // ds) for ds in self.up_factors]
        )
        self.amp_bypass = nn.ModuleList(
            [Bypass(f"amp_bypass_{ds}x", config.hidden_dim) for ds in self.up_factors]
        )
        # self.phase_bypass = nn.ModuleList(
        #     [Bypass(f"phase_bypass_{ds}x", config.hidden_dim) for ds in self.ds_factors]
        # )

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):  # (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, *, mel, style, pitch, energy):
        with torch.no_grad():
            audio_len = mel.shape[-1] * self.prod_up_factors * self.stft.hop_length
            pitch = F.interpolate(
                pitch.unsqueeze(1), size=audio_len, mode="linear", align_corners=False
            )
            energy = F.interpolate(
                energy.unsqueeze(1), size=audio_len, mode="linear", align_corners=False
            )
            har_source = self.m_source(
                pitch.transpose(1, 2), energy.transpose(1, 2)
            ).squeeze(-1)
            har_spec, har_x, har_y = self.stft.transform(har_source)
            har_phase = torch.atan2(har_y, har_x)
            har = torch.cat([har_spec, har_phase], dim=1)

        logamp = self.amp_input_conv(mel)
        logamp = logamp.transpose(1, 2)
        logamp = self.amp_norm(logamp)
        logamp = self.amp_pre_conformer(logamp, style)
        logamp = self.pre_upsampler(logamp.transpose(1, 2))

        for conv_block, down, prior_down, up, bypass in zip(
            self.amp_convnext,
            self.downsamplers,
            self.prior_downsamplers,
            self.upsamplers,
            self.amp_bypass,
        ):
            logamp_orig = logamp
            logamp = down(logamp)
            logamp_prior = prior_down(har)
            logamp_prior = logamp_prior[:, :, : logamp.shape[2]]

            logamp = conv_block(logamp + logamp_prior, style)
            logamp = up(logamp)
            logamp = logamp[:, :, : logamp_orig.shape[2]]
            logamp = bypass(
                logamp_orig.transpose(1, 2), logamp.transpose(1, 2)
            ).transpose(1, 2)

        logamp = logamp.transpose(1, 2)
        logamp = self.amp_post_conformer(logamp, style)
        phase = logamp
        logamp = self.amp_final_layer_norm(logamp)
        logamp = logamp.transpose(1, 2)
        logamp = self.amp_output_conv(logamp)

        phase = self.phase_norm(phase)
        phase = phase.transpose(1, 2)
        for conv_block in self.phase_convnext:
            phase = conv_block(phase, style)
        phase = phase.transpose(1, 2)
        phase = self.phase_final_layer_norm(phase)
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

        self.norm = AdaptiveLayerNorm(style_dim, dim, eps=1e-6)
        self.pwconv1 = torch.nn.Linear(
            dim, intermediate_dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.snake = torch.nn.Parameter(torch.ones(1, 1, intermediate_dim))
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = torch.nn.Linear(intermediate_dim, dim)

    def act(self, x):
        return x + (1 / self.snake) * (torch.sin(self.snake * x) ** 2)

    def forward(self, x, style):
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        x = self.norm(x, style)
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
