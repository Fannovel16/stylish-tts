import torch

import torch
from torch.nn.utils.parametrizations import weight_norm
from .ada_norm import AdaptiveDecoderBlock
from .generator import ConvNeXtBlock
import torch.nn.functional as F
import math
import torch.nn as nn
import numpy as np
from einops.layers.torch import Rearrange
from einops import repeat


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        assert self.dim % 2 == 0, "SinusoidalPosEmb requires dim to be even"

    def forward(self, x, scale=1000):
        if x.ndim < 1:
            x = x.unsqueeze(0)
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


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
            nn.Linear(self.dim, 1, bias=False),
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

    def forward(self, f0):
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
        return self.merge(sine_waves)


class CfmMelDecoder(nn.Module):
    def __init__(
        self,
        *,
        feat_dim,
        asr_dim,
        spk_dim,
        style_dim,
        hidden_dim,
        residual_dim,
    ):
        super().__init__()

        self.feat_dim = feat_dim
        self.encode = AdaptiveDecoderBlock(
            dim_in=feat_dim + asr_dim + style_dim + 2,
            dim_out=hidden_dim,
            style_dim=style_dim,
        )
        self.F0_conv = nn.Sequential(
            Rearrange("b c t -> b t c"),
            SineGenerator(24000),
            Rearrange("b t c -> b c t"),
            weight_norm(nn.Conv1d(1, 1, kernel_size=7, padding=3)),
        )
        self.N_conv = weight_norm(nn.Conv1d(1, 1, kernel_size=7, padding=3))
        self.asr_res = nn.Sequential(
            weight_norm(nn.Conv1d(asr_dim + style_dim, residual_dim, kernel_size=1))
        )
        self.decode = nn.ModuleList(
            [
                ConvNeXtBlock(hidden_dim + residual_dim + 2, hidden_dim * 4, style_dim)
                for _ in range(8)
            ]
        )
        self.decode_proj = nn.ModuleList(
            [nn.Conv1d(hidden_dim + residual_dim + 2, hidden_dim, 1) for _ in range(8)]
        )
        self.output_proj = nn.Conv1d(hidden_dim, feat_dim, 1)
        self.sampler = CfmSampler(self._forward)
        self.spk_emb = nn.Sequential(
            nn.Linear(spk_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, style_dim),
        )
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, style_dim),
        )

    def concat(self, *xs):
        return torch.cat(xs, 1)

    def _forward(self, x, asr, F0_curve, N, spk_emb, t, mask=None):
        F0 = self.F0_conv(F.interpolate(F0_curve.unsqueeze(1), x.shape[-1]))
        N = self.N_conv(F.interpolate(N.unsqueeze(1), x.shape[-1]))
        spk_emb = repeat(self.spk_emb(spk_emb), "b c -> b c t", t=x.shape[-1])
        s = self.time_emb(t)

        x = self.encode(self.concat(x, asr, F0, N, spk_emb), s)
        asr_res = self.asr_res(self.concat(asr, spk_emb))
        for block, proj in zip(self.decode, self.decode_proj):
            x = block(self.concat(x, asr_res, F0, N), s)
            x = proj(x)

        return self.output_proj(x)

    @torch.inference_mode()
    def forward(self, asr, F0_curve, N, spk_emb, n_timesteps, temperature):
        b, _, t = asr.shape
        z = torch.rand((b, self.feat_dim, t), device=asr.device)
        return self.sampler(
            z,
            None,
            n_timesteps,
            temperature,
            asr=asr,
            F0_curve=F0_curve,
            N=N,
            spk_emb=spk_emb,
        )

    def compute_pred_target(self, asr, F0_curve, N, spk_emb, x1):
        return self.sampler.compute_pred_target(
            x1, None, asr=asr, F0_curve=F0_curve, N=N, spk_emb=spk_emb
        )


class CfmSampler(nn.Module):
    def __init__(
        self,
        estimator,
        guidance_w=0.7,
        cond_drop_prob=0.2,
        non_drop_conds=[],
        sigma_min=1e-4,
    ):
        """
        A versatile implementation of Model-Guidance Conditional Flow Matching based on https://arxiv.org/pdf/2504.20334
        The estimator's forward must have keyword arguments t (timestep) and mask
        """
        super().__init__()
        self.estimator = estimator
        self.guidance_w = guidance_w
        self.cond_drop_prob = cond_drop_prob
        self.non_drop_conds = non_drop_conds
        self.sigma_min = sigma_min

    @torch.inference_mode()
    def forward(self, z, mask, n_timesteps, temperature=1.0, **estimator_args):
        """Forward diffusion

        Args:
            z (torch.Tensor): Pure noise
                shape: batch_size, n_feats, mel_timesteps
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            **estimator_args (keyword arguments): Argument passing to the estimator.

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """
        z = z * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=z.device)
        return self.solve_euler(z, t_span=t_span, mask=mask, **estimator_args)

    def solve_euler(self, x, t_span, mask, **estimator_args):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            **estimator_args (keyword arguments): Argument passing to the estimator.
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        for step in range(1, len(t_span)):
            dphi_dt = self.estimator(x, t=t, mask=mask, **estimator_args)
            x = x + dt * dphi_dt
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
        return x

    def prepare_cond_uncond(self, x1, **estimator_args):
        cond_args, uncond_args = {}, {}
        for k, arg in estimator_args.items():
            cond, uncond = arg, arg
            if isinstance(arg, torch.Tensor):
                if k not in self.non_drop_conds:
                    drop_mask = (
                        torch.rand([x1.shape[0]] + [1] * (cond.ndim - 1)).to(x1.device)
                        > self.cond_drop_prob
                    )
                    cond = cond * drop_mask
                    uncond = torch.zeros_like(cond)
            cond_args[k] = cond
            uncond_args[k] = uncond
        return cond_args, uncond_args

    def compute_pred_target(self, x1, mask, **estimator_args):
        """Computes prediction and target.

        Args:
            x1 (torch.Tensor): Target
                shape: (batch_size, n_feats, mel_timesteps)
            **estimator_args (keyword arguments): Argument passing to the estimator.
        Returns:
            pred: (torch.Tensor): Prediction
                shape: (batch_size, n_feats, mel_timesteps)
            target: (torch.Tensor): Target of Flow Matching
                shape: (batch_size, n_feats, mel_timesteps)
        """
        b, _, t = x1.shape

        # random timestep
        t = torch.rand([b, 1, 1], device=x1.device, dtype=x1.dtype)
        # sample noise p(x_0)
        z = torch.randn_like(x1)

        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z

        cond_args, uncond_args = self.prepare_cond_uncond(x1, **estimator_args)
        v_cond = self.estimator(y, t=t.squeeze(), mask=mask, **cond_args)
        v_uncond = self.estimator(y, t=t.squeeze(), mask=mask, **uncond_args)
        delta_stop_grad = torch.detach(v_cond - v_uncond)
        v_cfg = v_cond + self.guidance_w * delta_stop_grad
        pred, target = v_cfg, u
        return pred, target
