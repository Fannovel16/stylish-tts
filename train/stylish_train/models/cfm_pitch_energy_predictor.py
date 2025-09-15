from abc import ABC

import torch
import torch.nn.functional as F

import math
import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm
from .text_encoder import MultiHeadAttention
from .prosody_encoder import ProsodyEncoder
from utils import length_to_mask
from .ada_norm import AdaptiveLayerNorm, AdaptiveDecoderBlock
from .text_style_encoder import TextStyleEncoder
from einops import repeat


class CFMPitchEnergyPredictor(torch.nn.Module):
    def __init__(
        self,
        hubert_dim,
        spk_dim,
        style_dim,
        inter_dim,
        style_config,
        pitch_energy_config,
    ):
        super().__init__()

        self.phone_quant = torch.nn.Conv1d(hubert_dim, inter_dim, 1)

        self.style_encoder = torch.nn.Linear(spk_dim, style_dim)

        self.prosody_encoder = ProsodyEncoder(
            sty_dim=style_dim,
            d_model=inter_dim,
            nlayers=3,
            dropout=0.2,
        )

        dropout = pitch_energy_config.dropout

        self.F0 = torch.nn.ModuleList(
            [
                AdaptiveDecoderBlock(
                    inter_dim + style_dim,
                    inter_dim + style_dim,
                    style_dim,
                    dropout_p=dropout,
                )
                for _ in range(3)
            ]
        )

        self.N = torch.nn.ModuleList(
            [
                AdaptiveDecoderBlock(
                    inter_dim + style_dim,
                    inter_dim + style_dim,
                    style_dim,
                    dropout_p=dropout,
                )
                for _ in range(3)
            ]
        )

        self.F0_proj = torch.nn.Conv1d(inter_dim + style_dim, 1, 1, 1, 0)
        self.N_proj = torch.nn.Conv1d(inter_dim + style_dim, 1, 1, 1, 0)

    def forward(self, phones, phone_lengths, spk_emb):
        phones, raw_style = self.phone_quant(phones), self.spk_quant(spk_emb)
        raw_style = torch.cat(
            [phones, repeat(raw_style, "b c -> b c t", t=phones.shape[-1])], dim=1
        )
        style = self.style_encoder(raw_style, phone_lengths)
        x = self.prosody_encoder(phones, style, phone_lengths)

        F0 = x.transpose(1, 2)
        for block in self.F0:
            F0 = block(F0, style)
        F0 = self.F0_proj(F0)

        N = x.transpose(1, 2)
        for block in self.N:
            N = block(N, style)
        N = self.N_proj(N)

        return F0.squeeze(1), N.squeeze(1)


class CFMSampler(torch.nn.Module):
    def __init__(
        self,
        cfm_params,
        estimator,
    ):
        """
        The estimator's forward must have keyword arguments t (timestep) and mask
        """
        super().__init__()
        self.solver = cfm_params.solver
        self.estimator = estimator
        if hasattr(cfm_params, "sigma_min"):
            self.sigma_min = cfm_params.sigma_min
        else:
            self.sigma_min = 1e-4

    @torch.inference_mode()
    def forward(
        self, noise_shape, mask, n_timesteps, temperature=1.0, **estimator_args
    ):
        """Forward diffusion

        Args:
            noise_shape (batch_size, n_feats, mel_timesteps): Shape of the noise
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            **estimator_args (keyword arguments): Argument passing to the estimator.

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """
        z = torch.rand(noise_shape, device=mask.device) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mask.device)
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

    def compute_loss(self, x1, mask, **estimator_args):
        """Computes diffusion loss

        Args:
            x1 (torch.Tensor): Target
                shape: (batch_size, n_feats, mel_timesteps)
            **estimator_args (keyword arguments): Argument passing to the estimator.
        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """
        b, _, t = x1.shape

        # random timestep
        t = torch.rand([b, 1, 1], device=x1.device, dtype=x1.dtype)
        # sample noise p(x_0)
        z = torch.randn_like(x1)

        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z

        loss = F.mse_loss(
            self.estimator(y, t=t.squeeze(), mask=mask, **estimator_args),
            u,
            reduction="sum",
        ) / (torch.sum(mask) * u.shape[1])
        return loss, y
