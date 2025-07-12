import torch
import torch.nn as nn
from utils import sequence_mask
from .conformer import Conformer, Swish
from .text_encoder import TextEncoder


class CodePredictionHead(nn.Module):
    def __init__(self, hidden_dim, codebook_size):
        super().__init__()
        self.pre_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            Swish(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            Swish(),
            nn.Dropout(0.1),
        )
        self.local_refiner = Conformer(
            hidden_dim // 4,
            depth=1,
            heads=2,
            dim_head=hidden_dim // 4 // 2,
            ff_mult=4,
            conv_expansion_factor=2,
            conv_kernel_size=31,
        )
        self.post_proj = nn.Sequential(
            Swish(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 4, codebook_size),
        )

    def forward(self, x, mask):
        x = self.pre_proj(x)
        x = self.local_refiner(x, mask)
        x = self.post_proj(x)
        return x


class VQIndexer(nn.Module):
    def __init__(
        self,
        tokens,
        hidden_dim,
        codebook_size,
        heads,
        text_encoder_config,
    ):
        super().__init__()
        text_encoder_config.hidden_dim = hidden_dim
        self.text_encoder = TextEncoder(tokens, hidden_dim, text_encoder_config)
        self.refiner = Conformer(
            hidden_dim,
            depth=4,
            heads=8,
            dim_head=hidden_dim // 8,
            ff_mult=4,
            conv_expansion_factor=2,
            conv_kernel_size=31,
        )
        self.heads = nn.ModuleList(
            [CodePredictionHead(hidden_dim, codebook_size) for _ in range(heads)]
        )

    def forward(self, texts, text_lengths, mel_lengths, alignment):
        mel_mask = sequence_mask(mel_lengths, alignment.shape[2])
        x, _, _ = self.text_encoder(texts, text_lengths)
        x = self.refiner((x @ alignment).transpose(-1, -2), mel_mask)
        return torch.stack(
            [head(x, mel_mask) for head in self.heads], dim=-2
        )  # BxTxHxC
