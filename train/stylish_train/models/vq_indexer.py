import torch
import torch.nn as nn
from utils import sequence_mask
from .conformer import Conformer, Swish


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
        self.text_encoder = nn.Embedding(tokens, hidden_dim)
        self.refiner = Conformer(
            hidden_dim,
            depth=4,
            heads=4,
            dim_head=hidden_dim // 4,
            ff_mult=4,
            conv_expansion_factor=2,
            conv_kernel_size=7,
        )
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    Swish(),
                    nn.Linear(hidden_dim // 2, hidden_dim // 4),
                    Swish(),
                    nn.Linear(hidden_dim // 4, codebook_size),
                )
                for _ in range(heads)
            ]
        )

    def forward(self, texts, text_lengths, mel_lengths, alignment):
        mel_mask = sequence_mask(mel_lengths, alignment.shape[2])
        x = self.text_encoder(texts).transpose(-1, -2)
        x = self.refiner((x @ alignment).transpose(-1, -2), mel_mask)
        return torch.stack([head(x) for head in self.heads], dim=-2)  # BxTxHxC
