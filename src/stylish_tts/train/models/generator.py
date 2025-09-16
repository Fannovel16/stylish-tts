from functools import partial
from typing import Optional, Tuple
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
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
        return forward_transform
        # mag = torch.abs(forward_transform) + 1e-9
        # x = torch.real(forward_transform) / mag
        # y = torch.imag(forward_transform) / mag
        # return torch.abs(forward_transform), x, y

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


# class SineGen(torch.nn.Module):
#     """Definition of sine generator
#     SineGen(samp_rate, harmonic_num = 0,
#             sine_amp = 0.1, noise_std = 0.003,
#             voiced_threshold = 0,
#             flag_for_pulse=False)
#     samp_rate: sampling rate in Hz
#     harmonic_num: number of harmonic overtones (default 0)
#     sine_amp: amplitude of sine-wavefrom (default 0.1)
#     noise_std: std of Gaussian noise (default 0.003)
#     voiced_thoreshold: F0 threshold for U/V classification (default 0)
#     flag_for_pulse: this SinGen is used inside PulseGen (default False)
#     Note: when flag_for_pulse is True, the first time step of a voiced
#         segment is always sin(np.pi) or cos(0)
#     """

#     def __init__(
#         self,
#         samp_rate,
#         upsample_scale,
#         harmonic_num=0,
#         sine_amp=0.1,
#         noise_std=0.003,
#         voiced_threshold=0,
#         flag_for_pulse=False,
#     ):
#         super(SineGen, self).__init__()
#         self.sine_amp = sine_amp
#         self.noise_std = noise_std
#         self.harmonic_num = harmonic_num
#         self.dim = self.harmonic_num + 1
#         self.sampling_rate = samp_rate
#         self.voiced_threshold = voiced_threshold
#         self.flag_for_pulse = flag_for_pulse
#         self.upsample_scale = upsample_scale

#     def _f02uv(self, f0):
#         # generate uv signal
#         uv = (f0 > self.voiced_threshold).type(torch.float32)
#         return uv

#     def _f02sine(self, f0_values, source_len):
#         """f0_values: (batchsize, length, dim)
#         where dim indicates fundamental tone and overtones
#         """
#         # convert to F0 in rad. The interger part n can be ignored
#         # because 2 * np.pi * n doesn't affect phase
#         rad_values = (f0_values / self.sampling_rate) % 1

#         # initial phase noise (no noise for fundamental component)
#         rand_ini = torch.rand(
#             f0_values.shape[0], f0_values.shape[2], device=f0_values.device
#         )
#         rand_ini[:, 0] = 0
#         rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini

#         # instantanouse phase sine[t] = sin(2*pi \sum_i=1 ^{t} rad)
#         if not self.flag_for_pulse:
#             #             # for normal case

#             #             # To prevent torch.cumsum numerical overflow,
#             #             # it is necessary to add -1 whenever \sum_k=1^n rad_value_k > 1.
#             #             # Buffer tmp_over_one_idx indicates the time step to add -1.
#             #             # This will not change F0 of sine because (x-1) * 2*pi = x * 2*pi
#             #             tmp_over_one = torch.cumsum(rad_values, 1) % 1
#             #             tmp_over_one_idx = (padDiff(tmp_over_one)) < 0
#             #             cumsum_shift = torch.zeros_like(rad_values)
#             #             cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0

#             #             phase = torch.cumsum(rad_values, dim=1) * 2 * np.pi
#             rad_values = torch.nn.functional.interpolate(
#                 rad_values.transpose(1, 2),
#                 size=source_len,
#                 # scale_factor=1 / self.upsample_scale,
#                 mode="linear",
#             ).transpose(1, 2)

#             #             tmp_over_one = torch.cumsum(rad_values, 1) % 1
#             #             tmp_over_one_idx = (padDiff(tmp_over_one)) < 0
#             #             cumsum_shift = torch.zeros_like(rad_values)
#             #             cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0

#             phase = torch.cumsum(rad_values, dim=1) * 2 * torch.pi
#             phase = torch.nn.functional.interpolate(
#                 phase.transpose(1, 2) * self.upsample_scale,
#                 scale_factor=self.upsample_scale,
#                 mode="linear",
#             ).transpose(1, 2)
#             sines = torch.sin(phase)

#         else:
#             # If necessary, make sure that the first time step of every
#             # voiced segments is sin(pi) or cos(0)
#             # This is used for pulse-train generation

#             # identify the last time step in unvoiced segments
#             uv = self._f02uv(f0_values)
#             uv_1 = torch.roll(uv, shifts=-1, dims=1)
#             uv_1[:, -1, :] = 1
#             u_loc = (uv < 1) * (uv_1 > 0)

#             # get the instantanouse phase
#             tmp_cumsum = torch.cumsum(rad_values, dim=1)
#             # different batch needs to be processed differently
#             for idx in range(f0_values.shape[0]):
#                 temp_sum = tmp_cumsum[idx, u_loc[idx, :, 0], :]
#                 temp_sum[1:, :] = temp_sum[1:, :] - temp_sum[0:-1, :]
#                 # stores the accumulation of i.phase within
#                 # each voiced segments
#                 tmp_cumsum[idx, :, :] = 0
#                 tmp_cumsum[idx, u_loc[idx, :, 0], :] = temp_sum

#             # rad_values - tmp_cumsum: remove the accumulation of i.phase
#             # within the previous voiced segment.
#             i_phase = torch.cumsum(rad_values - tmp_cumsum, dim=1)

#             # get the sines
#             sines = torch.cos(i_phase * 2 * torch.pi)
#         return sines

#     def forward(self, f0, source_len):
#         """sine_tensor, uv = forward(f0)
#         input F0: tensor(batchsize=1, length, dim=1)
#                   f0 for unvoiced steps should be 0
#         output sine_tensor: tensor(batchsize=1, length, dim)
#         output uv: tensor(batchsize=1, length, 1)
#         """
#         f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim, device=f0.device)
#         # fundamental component
#         fn = torch.multiply(
#             f0, torch.FloatTensor([[range(1, self.harmonic_num + 2)]]).to(f0.device)
#         )

#         # generate sine waveforms
#         sine_waves = self._f02sine(fn, source_len) * self.sine_amp

#         # generate uv signal
#         # uv = torch.ones(f0.shape)
#         # uv = uv * (f0 > self.voiced_threshold)
#         uv = self._f02uv(f0)

#         # noise: for unvoiced should be similar to sine_amp
#         #        std = self.sine_amp/3 -> max value ~ self.sine_amp
#         # .       for voiced regions is self.noise_std
#         noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
#         noise = noise_amp * torch.randn(
#             sine_waves.size(), dtype=sine_waves.dtype, device=sine_waves.device
#         )

#         # first: set the unvoiced part to 0 by uv
#         # then: additive noise
#         sine_waves = sine_waves * uv + noise
#         return sine_waves, uv, noise


# class SourceModuleHnNSF(torch.nn.Module):
#     """SourceModule for hn-nsf
#     SourceModule(sampling_rate, harmonic_num=0, sine_amp=0.1,
#                  add_noise_std=0.003, voiced_threshod=0)
#     sampling_rate: sampling_rate in Hz
#     harmonic_num: number of harmonic above F0 (default: 0)
#     sine_amp: amplitude of sine source signal (default: 0.1)
#     add_noise_std: std of additive Gaussian noise (default: 0.003)
#         note that amplitude of noise in unvoiced is decided
#         by sine_amp
#     voiced_threshold: threhold to set U/V given F0 (default: 0)
#     Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
#     F0_sampled (batchsize, length, 1)
#     Sine_source (batchsize, length, 1)
#     noise_source (batchsize, length 1)
#     uv (batchsize, length, 1)
#     """

#     def __init__(
#         self,
#         sampling_rate,
#         upsample_scale,
#         harmonic_num=0,
#         sine_amp=0.1,
#         add_noise_std=0.003,
#         voiced_threshod=0,
#     ):
#         super(SourceModuleHnNSF, self).__init__()

#         self.sine_amp = sine_amp
#         self.noise_std = add_noise_std

#         # to produce sine waveforms
#         self.l_sin_gen = SineGen(
#             sampling_rate,
#             upsample_scale,
#             harmonic_num,
#             sine_amp,
#             add_noise_std,
#             voiced_threshod,
#         )

#         # to merge source harmonics into a single excitation
#         self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
#         self.l_tanh = torch.nn.Tanh()

#     def forward(self, x, source_len):
#         """
#         Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
#         F0_sampled (batchsize, length, 1)
#         Sine_source (batchsize, length, 1)
#         noise_source (batchsize, length 1)
#         """
#         # source for harmonic branch
#         with torch.no_grad():
#             sine_wavs, uv, _ = self.l_sin_gen(x, source_len)
#         sine_merge = self.l_tanh(self.l_linear(sine_wavs))

#         # source for noise branch, in the same shape as uv
#         noise = torch.randn_like(uv) * self.sine_amp / 3
#         return sine_merge, noise, uv


class SineGen(torch.nn.Module):
    """Definition of sine generator
    SineGen(samp_rate, harmonic_num = 0,
            sine_amp = 0.1, noise_std = 0.003,
            voiced_threshold = 0,
            flag_for_pulse=False)

    samp_rate: sampling rate in Hz
    harmonic_num: number of harmonic overtones (default 0)
    sine_amp: amplitude of sine-wavefrom (default 0.1)
    noise_std: std of Gaussian noise (default 0.003)
    voiced_thoreshold: F0 threshold for U/V classification (default 0)
    flag_for_pulse: this SinGen is used inside PulseGen (default False)

    Note: when flag_for_pulse is True, the first time step of a voiced
        segment is always sin(np.pi) or cos(0)
    """

    def __init__(
        self,
        samp_rate,
        harmonic_num=0,
        sine_amp=0.1,
        noise_std=0.003,
        voiced_threshold=0,
        flag_for_pulse=False,
    ):
        super(SineGen, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.flag_for_pulse = flag_for_pulse

    def _f02uv(self, f0):
        # generate uv signal
        uv = torch.ones_like(f0)
        uv = uv * (f0 > self.voiced_threshold)
        return uv

    def _f02sine(self, f0_values):
        """f0_values: (batchsize, length, dim)
        where dim indicates fundamental tone and overtones
        """
        # convert to F0 in rad. The interger part n can be ignored
        # because 2 * np.pi * n doesn't affect phase
        rad_values = (f0_values / self.sampling_rate) % 1

        # initial phase noise (no noise for fundamental component)
        rand_ini = torch.rand(
            f0_values.shape[0], f0_values.shape[2], device=f0_values.device
        )
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini

        # instantanouse phase sine[t] = sin(2*pi \sum_i=1 ^{t} rad)
        if not self.flag_for_pulse:
            # for normal case

            # To prevent torch.cumsum numerical overflow,
            # it is necessary to add -1 whenever \sum_k=1^n rad_value_k > 1.
            # Buffer tmp_over_one_idx indicates the time step to add -1.
            # This will not change F0 of sine because (x-1) * 2*pi = x *2*pi
            tmp_over_one = torch.cumsum(rad_values, 1) % 1
            tmp_over_one_idx = (tmp_over_one[:, 1:, :] - tmp_over_one[:, :-1, :]) < 0
            cumsum_shift = torch.zeros_like(rad_values)
            cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0

            sines = torch.sin(
                torch.cumsum(rad_values + cumsum_shift, dim=1) * 2 * np.pi
            )
        else:
            # If necessary, make sure that the first time step of every
            # voiced segments is sin(pi) or cos(0)
            # This is used for pulse-train generation

            # identify the last time step in unvoiced segments
            uv = self._f02uv(f0_values)
            uv_1 = torch.roll(uv, shifts=-1, dims=1)
            uv_1[:, -1, :] = 1
            u_loc = (uv < 1) * (uv_1 > 0)

            # get the instantanouse phase
            tmp_cumsum = torch.cumsum(rad_values, dim=1)
            # different batch needs to be processed differently
            for idx in range(f0_values.shape[0]):
                temp_sum = tmp_cumsum[idx, u_loc[idx, :, 0], :]
                temp_sum[1:, :] = temp_sum[1:, :] - temp_sum[0:-1, :]
                # stores the accumulation of i.phase within
                # each voiced segments
                tmp_cumsum[idx, :, :] = 0
                tmp_cumsum[idx, u_loc[idx, :, 0], :] = temp_sum

            # rad_values - tmp_cumsum: remove the accumulation of i.phase
            # within the previous voiced segment.
            i_phase = torch.cumsum(rad_values - tmp_cumsum, dim=1)

            # get the sines
            sines = torch.cos(i_phase * 2 * np.pi)
        return sines

    def forward(self, f0):
        """sine_tensor, uv = forward(f0)
        input F0: tensor(batchsize=1, length, dim=1)
                  f0 for unvoiced steps should be 0
        output sine_tensor: tensor(batchsize=1, length, dim)
        output uv: tensor(batchsize=1, length, 1)
        """
        with torch.no_grad():
            f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim, device=f0.device)
            # fundamental component
            f0_buf[:, :, 0] = f0[:, :, 0]
            for idx in np.arange(self.harmonic_num):
                # idx + 2: the (idx+1)-th overtone, (idx+2)-th harmonic
                f0_buf[:, :, idx + 1] = f0_buf[:, :, 0] * (idx + 2)

            # generate sine waveforms
            sine_waves = self._f02sine(f0_buf) * self.sine_amp

            # generate uv signal
            # uv = torch.ones(f0.shape)
            # uv = uv * (f0 > self.voiced_threshold)
            uv = self._f02uv(f0)

            # noise: for unvoiced should be similar to sine_amp
            #        std = self.sine_amp/3 -> max value ~ self.sine_amp
            # .       for voiced regions is self.noise_std
            noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
            noise = noise_amp * torch.randn_like(sine_waves)

            # first: set the unvoiced part to 0 by uv
            # then: additive noise
            sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise


class SourceModuleHnNSF(torch.nn.Module):
    """SourceModule for hn-nsf
    SourceModule(sampling_rate, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0)
    sampling_rate: sampling_rate in Hz
    harmonic_num: number of harmonic above F0 (default: 0)
    sine_amp: amplitude of sine source signal (default: 0.1)
    add_noise_std: std of additive Gaussian noise (default: 0.003)
        note that amplitude of noise in unvoiced is decided
        by sine_amp
    voiced_threshold: threhold to set U/V given F0 (default: 0)

    Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
    F0_sampled (batchsize, length, 1)
    Sine_source (batchsize, length, 1)
    noise_source (batchsize, length 1)
    uv (batchsize, length, 1)
    """

    def __init__(
        self,
        sampling_rate,
        harmonic_num=0,
        sine_amp=0.1,
        add_noise_std=0.003,
        voiced_threshod=0,
    ):
        super(SourceModuleHnNSF, self).__init__()

        self.sine_amp = sine_amp
        self.noise_std = add_noise_std

        # to produce sine waveforms
        self.l_sin_gen = SineGen(
            sampling_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshod
        )

        # to merge source harmonics into a single excitation
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()

    def forward(self, x):
        """
        Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
        F0_sampled (batchsize, length, 1)
        Sine_source (batchsize, length, 1)
        noise_source (batchsize, length 1)
        """
        # source for harmonic branch
        sine_wavs, uv, _ = self.l_sin_gen(x)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))

        # source for noise branch, in the same shape as uv
        noise = torch.randn_like(uv) * self.sine_amp / 3
        return sine_merge, noise, uv


class Generator(torch.nn.Module):
    def __init__(
        self, *, style_dim, n_fft, win_length, hop_length, sample_rate, config
    ):
        super(Generator, self).__init__()

        channels = 8
        n_bins = n_fft // 2 + 1
        mult_channels = 3
        kernel_size = [13, 7]

        # Prior waveform generator
        self.hop_length = hop_length
        # self.prior_generator = partial(
        #     generate_pcph,
        #     hop_length=hop_length,
        #     sample_rate=sample_rate,
        # )
        self.prior_generator = SourceModuleHnNSF(
            sampling_rate=sample_rate,
            harmonic_num=8,
            voiced_threshod=10,
        )

        self.prior_mag_proj = Conv1d(
            n_bins,
            n_bins,
            config.io_conv_kernel_size,
            1,
            padding=get_padding(config.io_conv_kernel_size, 1),
            padding_mode="reflect",
        )

        self.prior_phase_proj = Conv1d(
            n_bins,
            n_bins,
            config.io_conv_kernel_size,
            1,
            padding=get_padding(config.io_conv_kernel_size, 1),
            padding_mode="reflect",
        )

        self.amp_mel_proj = Conv1d(
            config.input_dim,
            n_bins,
            config.io_conv_kernel_size,
            1,
            padding=get_padding(config.io_conv_kernel_size, 1),
            padding_mode="reflect",
        )

        self.phase_mel_proj = Conv1d(
            config.input_dim,
            n_bins,
            config.io_conv_kernel_size,
            1,
            padding=get_padding(config.io_conv_kernel_size, 1),
            padding_mode="reflect",
        )

        self.amp_input_proj = nn.Conv2d(5, channels, 1, bias=False)
        self.amp_input_norm = LayerNorm2d(channels)
        self.phase_input_proj = nn.Conv2d(6, channels, 1, bias=False)
        self.phase_input_norm = LayerNorm2d(channels)

        self.amp_pre_norm = nn.LayerNorm(config.hidden_dim, eps=1e-6)
        self.phase_pre_norm = nn.LayerNorm(config.hidden_dim, eps=1e-6)
        self.phase_conformer = Conformer(
            dim=config.hidden_dim,
            style_dim=style_dim,
            depth=config.conformer_layers,
            attn_dropout=0.2,
            ff_dropout=0.2,
            conv_dropout=0.2,
        )
        self.phase_convnext = nn.ModuleList(
            [
                # ConvNeXtBlock(
                #     dim=config.hidden_dim,
                #     intermediate_dim=config.conv_intermediate_dim,
                #     style_dim=style_dim,
                # )
                ConvNeXtBlock2d(
                    channels,
                    mult_channels,
                    kernel_size,
                    drop_prob=0.2,
                    style_dim=style_dim,
                    layer_scale_init_value=1 / config.conv_layers,
                )
                for _ in range(config.conv_layers)
            ]
        )
        self.amp_conformer = Conformer(
            dim=config.hidden_dim,
            style_dim=style_dim,
            depth=config.conformer_layers,
            attn_dropout=0.2,
            ff_dropout=0.2,
            conv_dropout=0.2,
        )
        self.amp_convnext = nn.ModuleList(
            [
                # ConvNeXtBlock(
                #     dim=config.hidden_dim,
                #     intermediate_dim=config.conv_intermediate_dim,
                #     style_dim=style_dim,
                # )
                ConvNeXtBlock2d(
                    channels,
                    mult_channels,
                    kernel_size,
                    drop_prob=0.2,
                    style_dim=style_dim,
                    layer_scale_init_value=1 / config.conv_layers,
                )
                for _ in range(config.conv_layers)
            ]
        )

        self.amp_output_norm = LayerNorm2d(channels)
        self.phase_output_norm = LayerNorm2d(channels)
        self.amp_output_proj = nn.Conv2d(channels, 1, 1)
        self.phase_output_real_proj = nn.Conv2d(channels, 1, 1)
        self.phase_output_imag_proj = nn.Conv2d(channels, 1, 1)

        self.apply(self._init_weights)
        self.stft = TorchSTFT(
            filter_length=n_fft,
            hop_length=hop_length,
            win_length=win_length,
        )

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Conv2d)):  # (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, *, mel, style, pitch, energy):
        # Generate prior waveform and compute spectrogram
        # with torch.no_grad():
        #     prior = self.prior_generator(pitch)
        #     prior = prior.squeeze(1)
        #     prior_spec = self.stft.transform(prior)
        #     prior_spec = prior_spec[:, :, :-1]
        #     prior_mag = torch.log(torch.abs(prior_spec) + 1e-9)
        #     prior_phase = torch.angle(prior_spec)

        with torch.no_grad():
            prior, _, _ = self.prior_generator(pitch)
            prior = prior.transpose(1, 2).squeeze(1)
            prior_spec = self.stft.transform(prior)
            prior_spec = prior_spec[:, :, :-1]
            prior_mag = torch.log(torch.abs(prior_spec) + 1e-9)
            prior_phase = torch.angle(prior_spec)

        # Apply input projection
        prior_mag_proj = self.prior_mag_proj(prior_mag)
        prior_phase_proj = self.prior_phase_proj(prior_phase)

        logamp = self.amp_mel_proj(mel)
        logamp = logamp.transpose(1, 2)
        logamp = self.amp_pre_norm(logamp)
        logamp = self.amp_conformer(logamp, style)
        logamp = logamp.transpose(1, 2)

        logamp = torch.stack(
            [prior_mag, prior_phase, prior_mag_proj, prior_phase_proj, logamp], dim=1
        )
        logamp = self.amp_input_proj(logamp)
        logamp = self.amp_input_norm(logamp)

        for conv_block in self.amp_convnext:
            logamp = conv_block(logamp, style)

        logamp = self.amp_output_norm(logamp)
        logamp = self.amp_output_proj(logamp)
        logamp = logamp.squeeze(1)

        phase = self.phase_mel_proj(mel)
        phase = phase.transpose(1, 2)
        phase = self.phase_pre_norm(phase)
        phase = self.phase_conformer(phase, style)
        phase = phase.transpose(1, 2)

        phase = torch.stack(
            [prior_mag, prior_phase, prior_mag_proj, prior_phase_proj, logamp, phase],
            dim=1,
        )
        phase = self.phase_input_proj(phase)
        phase = self.phase_input_norm(phase)

        for conv_block in self.phase_convnext:
            phase = conv_block(phase, style)

        phase = self.phase_output_norm(phase)
        real = self.phase_output_real_proj(phase)
        imag = self.phase_output_imag_proj(phase)
        phase = torch.atan2(imag, real)
        phase = phase.squeeze(1)

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


class AdaptiveLayerNorm2d(nn.Module):
    def __init__(self, style_dim, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.fc = nn.Linear(style_dim, channels * 2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)

        x = x.transpose(1, 3)
        x = F.layer_norm(x, (self.channels,), eps=self.eps)
        x = x.transpose(1, 3)
        x = (1 + gamma) * x + beta
        return x


class ConvNeXtBlock2d(nn.Module):
    """
    A 2D residual block module based on ConvNeXt architecture.

    Reference:
        - https://github.com/facebookresearch/ConvNeXt
    """

    def __init__(
        self,
        channels: int,
        mult_channels: int,
        kernel_size: int,
        drop_prob: float = 0.0,
        style_dim: int = 64,
        layer_scale_init_value: float = None,
    ) -> None:
        """
        Initialize the ConvNeXtBlock2d module.

        Args:
            channels (int): Number of input and output channels for the block.
            mult_channels (int): Channel expansion factor used in pointwise convolutions.
            kernel_size (int): Size of the depthwise convolution kernel.
            drop_prob (float, optional): Probability of dropping paths for stochastic depth (default: 0.0).
            use_layer_norm (bool, optional): If True, layer normalization is used; otherwise,
                batch normalization is applied (default: True).
            layer_scale_init_value (float, optional): Initial value for the learnable layer scale parameter.
                If None, no scaling is applied (default: None).
        """
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        assert kernel_size[0] % 2 == 1, "Kernel size must be odd number."
        assert kernel_size[1] % 2 == 1, "Kernel size must be odd number."
        self.dwconv = nn.Conv2d(
            channels,
            channels,
            kernel_size,
            padding=(kernel_size[0] // 2, kernel_size[1] // 2),
            groups=channels,
            bias=False,
            padding_mode="reflect",
        )
        self.norm = AdaptiveLayerNorm2d(style_dim, channels)
        self.pwconv1 = nn.Conv2d(channels, channels * mult_channels, 1)
        self.nonlinear = nn.GELU()
        self.pwconv2 = nn.Conv2d(channels * mult_channels, channels, 1)
        self.gamma = (
            nn.Parameter(
                layer_scale_init_value * torch.ones(1, channels, 1, 1),
                requires_grad=True,
            )
            if layer_scale_init_value is not None
            else None
        )
        self.drop_path = torch.nn.Dropout(drop_prob)

    def forward(self, x: Tensor, style) -> Tensor:
        """
        Calculate forward propagation.

        Args:
            x (Tensor): Input tensor with shape (batch, channels, height, width).

        Returns:
            Tensor: Output tensor of the same shape (batch, channels, height, width).
        """
        residual = x
        x = self.dwconv(x)
        x = self.norm(x, style)
        x = self.pwconv1(x)
        x = self.nonlinear(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = residual + self.drop_path(x)
        return x


def generate_pcph(
    f0: Tensor,
    hop_length: int,
    sample_rate: int,
    noise_amplitude: Optional[float] = 0.01,
    random_init_phase: Optional[bool] = True,
    power_factor: Optional[float] = 0.1,
    max_frequency: Optional[float] = None,
    *args,
    **kwargs,
) -> Tensor:
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
    f0 = f0.unsqueeze(1)
    batch, _, frames = f0.size()
    device = f0.device
    noise = noise_amplitude * torch.randn(
        (batch, 1, frames * hop_length), device=device
    )
    if torch.all(f0 <= 10.0):
        return noise

    vuv = f0 > 10.0
    min_f0_value = torch.min(f0[f0 > 0]).item()
    max_frequency = max_frequency if max_frequency is not None else sample_rate / 2
    max_n_harmonics = int(max_frequency / min_f0_value)
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


class NormLayer(nn.Module):
    def __init__(
        self, channels: int, eps: Optional[float] = 1e-6, affine: Optional[bool] = True
    ) -> None:
        """
        Initialize the NormLayer module.

        Args:
            channels (int): Number of input features.
            eps (float, optional): A small constant added to the denominator for numerical stability (default: 1e-6).
            affine (bool, optional): If True, this module has learnable affine parameters (default: True).
        """
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(channels))
            self.beta = nn.Parameter(torch.zeros(channels))

    def normalize(
        self,
        x: Tensor,
        dim: int,
        mean: Optional[Tensor] = None,
        var: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Apply normalization to the input tensor.

        Args:
            x (Tensor): Input tensor with shape (batch, channels, ...).
            dim (int): Dimensions along which statistics are calculated.
            mean (Tensor, optional): Mean tensor (default: None).
            var (Tensor, optional): Variance tensor (default: None).

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Normalized tensor and statistics.
        """
        # Calculate the mean along dimensions to be reduced
        if mean is None:
            mean = x.mean(dim, keepdim=True)

        # Centerize the input tensor
        x = x - mean

        # Calculate the variance
        if var is None:
            var = (x**2).mean(dim=dim, keepdim=True)

        # Normalize
        x = x / torch.sqrt(var + self.eps)

        if self.affine:
            shape = [1, self.channels] + [1] * (x.ndim - 2)
            x = self.gamma.view(*shape) * x + self.beta.view(*shape)

        return x, mean, var


class LayerNorm2d(NormLayer):
    def __init__(
        self, channels: int, eps: Optional[float] = 1e-6, affine: Optional[bool] = True
    ) -> None:
        """
        Initialize the LayerNorm2d module.

        Args:
            channels (int): Number of input features.
            eps (float, optional): A small constant added to the denominator for numerical stability (default: 1e-6).
            affine (bool, optional): If True, this module has learnable affine parameters (default: True).
        """
        super().__init__(channels, eps, affine)
        self.reduced_dim = [1, 2, 3]

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply normalization to the input tensor.

        Args:
            x (Tensor): Input tensor with shape (batch, channels, height, width).

        Returns:
            Tensor: Normalized tensor.
        """
        x, *_ = self.normalize(x, dim=self.reduced_dim)
        return x
