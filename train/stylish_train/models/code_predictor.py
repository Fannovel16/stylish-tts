import torch
import torch.nn as nn
from utils import sequence_mask
from .plbert import PLBERT
from .conformer import Conformer


class CodePredictor(nn.Module):
    def __init__(
        self,
        tokens,
        codebook_size,
        heads,
        text_encoder_config,
    ):
        super().__init__()
        hidden_dim = 256
        self.text_encoder = PLBERT(
            vocab_size=tokens,
            hidden_size=768,
            num_attention_heads=12,
            intermediate_size=2048,
            max_position_embeddings=512,
            num_hidden_layers=12,
            dropout=0.1,
        )
        self.project = nn.Linear(768, hidden_dim)
        self.refiner = Conformer(
            dim=hidden_dim,
            depth=4,
            heads=4,
            dim_head=hidden_dim // 4,
        )
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.GELU("tanh"),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim // 2, hidden_dim // 4),
                    nn.GELU("tanh"),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim // 4, codebook_size),
                )
                for _ in range(heads)
            ]
        )

    def forward(self, texts, text_lengths, mel_lengths, alignment):
        text_mask = sequence_mask(text_lengths, alignment.shape[1])
        mel_mask = sequence_mask(mel_lengths, alignment.shape[2])
        x = self.text_encoder(
            texts,
            attention_mask=text_mask.int(),
        ).transpose(-1, -2)
        x = (x @ alignment).transpose(-1, -2)
        x = self.refiner(self.project(x), mel_mask)
        return torch.stack([head(x) for head in self.heads], dim=-2)  # BxTxHxC
