import torch

import torch
from torch.nn.utils.parametrizations import weight_norm
import torch.nn.functional as F
import math
import torch.nn as nn
import numpy as np
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
from stylish_tts.train.models.xut.xut import XUTBackBone, TBackBone
from stylish_tts.train.models.xut.time_emb import TimestepEmbedding
from .cfm import CfmSampler


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


"""class CfmMelDecoder(nn.Module):
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
        f0_conv = nn.Sequential(
            Rearrange("b c n -> b n c"),
            SineGenerator(24000),
            Rearrange("b n c -> b c n"),
            weight_norm(nn.Conv1d(1, 1, kernel_size=7, padding=3)),
        )
        self.N_conv = weight_norm(nn.Conv1d(1, 1, kernel_size=7, padding=3))
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
        self.sampler = CfmSampler(self._forward, non_drop_conds=["spk_emb"])
        self.spk_emb = nn.Sequential(
            nn.Linear(spk_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, style_dim // 2),
        )
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, style_dim // 2),
        )

    def concat(self, *xs):
        return torch.cat(xs, 1)

    def _forward(self, x, asr, F0_curve, N, spk_emb, t, mask=None):
        F0 = f0_conv(F.interpolate(F0_curve.unsqueeze(1), x.shape[-1]))
        N = self.N_conv(F.interpolate(N.unsqueeze(1), x.shape[-1]))
        s = self.concat(self.time_emb(t), self.spk_emb(spk_emb))

        x = self.encode(self.concat(x, asr, F0, N), s)
        asr_res = self.asr_res(asr)
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
        )"""


class CfmMelDecoder(nn.Module):
    def __init__(
        self,
        feat_dim=80,
        asr_dim=768,
        spk_dim=1024,
        hidden_dim=256,
        emb_dim=256,
        xut_depth=4,
        xut_heads=8,
        xut_enc_blocks=1,
        xut_dec_blocks=2,
        tread_config={"prev_trns_depth": 1, "post_trns_depth": 3, "dropout_ratio": 0.5},
    ):
        super().__init__()
        self.time_emb = TimestepEmbedding(hidden_dim)
        self.asr_emb = nn.Sequential(
            Rearrange("b c t -> b t c"),
            nn.Linear(asr_dim, emb_dim * 4),
            nn.Mish(),
            nn.Linear(emb_dim * 4, emb_dim),
        )
        self.spk_emb = nn.Sequential(
            nn.Linear(spk_dim, emb_dim * 4),
            nn.Mish(),
            nn.Linear(emb_dim * 4, emb_dim),
        )
        self.m_source = nn.Sequential(Rearrange("b 1 n -> b n 1"), SineGenerator(24000))
        self.N_source = Rearrange("b 1 n -> b n 1")
        self.prior_generator = nn.Sequential(
            Rearrange("b n c -> b c n"),
            nn.Conv1d(3, feat_dim, kernel_size=7, padding=3),
            Rearrange("b c n -> b n c"),
        )
        self.backbone = XUTBackBone(
            dim=hidden_dim,
            ctx_dim=None,
            heads=xut_heads,
            mlp_dim=hidden_dim * 4,
            pos_dim=1,
            depth=xut_depth,
            enc_blocks=xut_enc_blocks,
            dec_blocks=xut_dec_blocks,
            use_adaln=True,
            use_shared_adaln=True,
        )
        self.feat_dim = feat_dim
        self.in_proj = nn.Linear(feat_dim + emb_dim + emb_dim, hidden_dim)
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, feat_dim), Rearrange("b n c -> b c n")
        )
        self.hidden_dim = hidden_dim
        self.build_shared_adaln()
        self.build_tread(
            tread_config, dim=hidden_dim, heads=xut_heads, mlp_dim=hidden_dim * 4
        )
        self.sampler = CfmSampler(self._forward, non_drop_conds=["spk_emb"])

    def build_shared_adaln(self):
        dim = self.hidden_dim
        self.shared_adaln_attn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim * 3),
        )
        nn.init.constant_(self.shared_adaln_attn[-1].bias, 0)
        nn.init.constant_(self.shared_adaln_attn[-1].weight, 0)
        self.shared_adaln_xattn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim * 3),
        )
        nn.init.constant_(self.shared_adaln_xattn[-1].bias, 0)
        nn.init.constant_(self.shared_adaln_xattn[-1].weight, 0)
        self.shared_adaln_ffw = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim * 3),
        )
        nn.init.constant_(self.shared_adaln_ffw[-1].bias, 0)

    def build_tread(
        self,
        tread_config,
        dim=1024,
        heads=16,
        dim_head=64,
        mlp_dim=3072,
        concat_ctx=True,
        ctx_dim=None,
        shared_adaln=True,
        use_dyt=False,
    ):
        self.use_tread = False
        if tread_config is not None:
            self.use_tread = True
            self.dropout_ratio = tread_config["dropout_ratio"]
            self.prev_tread_trns = TBackBone(
                dim,
                None if concat_ctx else ctx_dim,
                heads,
                dim_head,
                mlp_dim,
                1,
                tread_config["prev_trns_depth"],
                use_adaln=True,
                use_shared_adaln=shared_adaln,
                use_dyt=use_dyt,
            )
            self.post_tread_trns = TBackBone(
                dim,
                None if concat_ctx else ctx_dim,
                heads,
                dim_head,
                mlp_dim,
                1,
                tread_config["post_trns_depth"],
                use_adaln=True,
                use_shared_adaln=shared_adaln,
                use_dyt=use_dyt,
            )

    def _forward(self, x, asr, F0, N, spk_emb, t, mask=None):
        x = rearrange(x, "b c n -> b n c")
        batch, length, _ = x.shape
        asr = self.asr_emb(asr)
        F0 = F.interpolate(F0.unsqueeze(1), length)
        N = F.interpolate(N.unsqueeze(1), length)
        spk_emb = repeat(self.spk_emb(spk_emb), "b c -> b n c", n=length)

        har_source = torch.cat(
            [self.m_source(F0), self.N_source(N), repeat(t, "b -> b n 1", n=length)],
            dim=-1,
        )
        x = x + self.prior_generator(har_source)
        x = self.in_proj(torch.cat([x, asr, spk_emb], dim=-1))

        t_emb = self.time_emb(rearrange(t, "b -> b 1"))
        shared_adaln_state = [
            self.shared_adaln_attn(t_emb).chunk(3, dim=-1),
            self.shared_adaln_xattn(t_emb).chunk(3, dim=-1),
            self.shared_adaln_ffw(t_emb).chunk(3, dim=-1),
        ]

        # https://github.com/KohakuBlueleaf/HDM/blob/0a3cf7e/src/xut/modules/axial_rope.py#L64
        pos_map = torch.linspace(-1.0, 1.0, length, dtype=x.dtype, device=x.device)
        pos_map = repeat(pos_map, "n -> b n 1", b=batch)

        if self.use_tread:
            x = self.prev_tread_trns(
                x,
                ctx=None,
                pos_map=pos_map,
                y=t_emb,
                shared_adaln=shared_adaln_state,
            )
            if self.training:
                selection_length = length - int(length * self.dropout_ratio)
                selection = torch.stack(
                    [
                        torch.randperm(length, device=x.device) < selection_length
                        for _ in range(batch)
                    ]
                )
                not_masked_part, masked_part = x[~selection, :], x[selection, :]
                masked_part = rearrange(
                    masked_part, "(b l) d -> b l d", b=batch, l=selection_length
                )
                x = masked_part
                raw_pos_map = pos_map
                pos_map = rearrange(
                    pos_map[selection, :],
                    "(b l) d -> b l d",
                    b=batch,
                    l=selection_length,
                )
        backbone_out = self.backbone(
            x,
            y=t_emb,
            pos_map=pos_map,
            shared_adaln=shared_adaln_state,
        )
        if self.use_tread:
            if self.training:
                out = torch.empty(
                    batch, length, x.size(2), device=x.device, dtype=x.dtype
                )
                out[~selection, :] = not_masked_part
                out[selection, :] = rearrange(backbone_out, "b l d -> (b l) d")
                pos_map = raw_pos_map
            else:
                out = backbone_out
            out = self.post_tread_trns(
                out,
                ctx=None,
                pos_map=pos_map,
                y=t_emb,
                shared_adaln=shared_adaln_state,
            )
        else:
            out = backbone_out
        out = self.out_proj(out)
        return out

    @torch.inference_mode()
    def forward(self, asr, F0, N, spk_emb, n_timesteps, temperature):
        b, _, t = asr.shape
        z = torch.rand((b, self.feat_dim, t), device=asr.device)
        return self.sampler(
            z,
            None,
            n_timesteps,
            temperature,
            asr=asr,
            F0=F0,
            N=N,
            spk_emb=spk_emb,
        )

    def compute_pred_target(self, asr, F0, N, spk_emb, x1):
        return self.sampler.compute_pred_target(
            x1, None, asr=asr, F0=F0, N=N, spk_emb=spk_emb
        )
