import torch

import torch
from torch.nn.utils.parametrizations import weight_norm
from .ada_norm import AdaptiveDecoderBlock
from .generator import ConvNeXtBlock
import torch.nn.functional as F
import math
import torch.nn as nn


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
            dim_in=feat_dim + asr_dim + 2,
            dim_out=hidden_dim,
            style_dim=style_dim,
        )
        self.F0_conv = weight_norm(
            nn.Conv1d(1, 1, kernel_size=3, stride=1, groups=1, padding=1)
        )
        self.N_conv = weight_norm(
            nn.Conv1d(1, 1, kernel_size=3, stride=1, groups=1, padding=1)
        )
        self.asr_res = nn.Sequential(
            weight_norm(nn.Conv1d(asr_dim, residual_dim, kernel_size=1))
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

    def _forward(self, x, asr, F0_curve, N, spk_emb, t, mask=None):
        F0 = self.F0_conv(F.interpolate(F0_curve.unsqueeze(1), x.shape[-1]))
        N = self.N_conv(F.interpolate(N.unsqueeze(1), x.shape[-1]))
        s = self.spk_emb(spk_emb) + self.time_emb(t)
        x = self.encode(torch.cat([x, asr, F0, N], axis=1), s)

        asr_res = self.asr_res(asr)

        for block, proj in zip(self.decode, self.decode_proj):
            x = torch.cat([x, asr_res, F0, N], axis=1)
            x = block(x, s)
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
        return self.sampler.compute_loss(
            x1, None, asr=asr, F0_curve=F0_curve, N=N, spk_emb=spk_emb
        )


class CfmSampler(nn.Module):
    def __init__(
        self,
        estimator,
        sigma_min=1e-4,
    ):
        """
        The estimator's forward must have keyword arguments t (timestep) and mask
        """
        super().__init__()
        self.estimator = estimator
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

    def compute_pred_target(self, x1, mask, **estimator_args):
        """Computes prediction and target

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

        pred = self.estimator(y, t=t.squeeze(), mask=mask, **estimator_args)
        target = u
        return pred, target
