from stylish_tts.train.models.generator import ConvNeXtBlock
from stylish_tts.train.models.conv_next import BasicConvNeXtBlock
from .cfm import SinusoidalPosEmb
import torch.nn as nn
from einops import rearrange, repeat
import torch
from .cfm import CfmSampler


class CfmPitchPredictor(nn.Module):
    def __init__(self, asr_dim, spk_dim):
        super().__init__()
        hidden_dim = 256
        self.asr_emb = nn.Sequential(
            nn.Conv1d(asr_dim, hidden_dim * 4, 1),
            nn.Mish(),
            nn.Conv1d(hidden_dim * 4, hidden_dim, 1),
        )
        self.spk_emb = nn.Sequential(
            nn.Linear(spk_dim, hidden_dim * 4),
            nn.Mish(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        # self.time_emb = nn.Sequential(
        #     SinusoidalPosEmb(hidden_dim),
        #     nn.Linear(hidden_dim, hidden_dim * 4),
        #     nn.Mish(),
        #     nn.Linear(hidden_dim * 4, hidden_dim),
        # )
        self.blocks = nn.ModuleList(
            [ConvNeXtBlock(hidden_dim, hidden_dim * 4, hidden_dim) for _ in range(8)]
        )
        self.in_proj = nn.Conv1d(1, hidden_dim, 1)
        self.out_proj = nn.Conv1d(hidden_dim, 1, 1)
        # self.sampler = CfmSampler(
        #     self._forward, guidance_w=0, cond_drop_prob=0, non_drop_conds=["spk"]
        # )

    def forward(self, asr, spk):
        asr, spk = (
            self.asr_emb(asr),
            self.spk_emb(spk),
        )
        x = asr
        for layer in self.blocks:
            x = layer(x, spk)
        x = self.out_proj(x)
        return x

    # @torch.inference_mode()
    # def forward(self, asr, spk, n_timesteps, temperature):
    #     b, _, t = asr.shape
    #     z = torch.rand((b, 1, t), device=asr.device)
    #     return self.sampler(
    #         z,
    #         None,
    #         n_timesteps,
    #         temperature,
    #         asr=asr,
    #         spk=spk,
    #     )

    # def compute_pred_target(self, asr, spk, x1):
    #     return self.sampler.compute_pred_target(x1, None, asr=asr, spk=spk)
