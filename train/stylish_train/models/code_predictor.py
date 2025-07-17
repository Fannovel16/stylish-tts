import torch
import torch.nn as nn
from utils import length_to_mask
from .plbert import PLBERT


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
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 4),
                    nn.GELU("tanh"),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim // 4, hidden_dim // 8),
                    nn.GELU("tanh"),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim // 8, codebook_size),
                )
                for _ in range(heads)
            ]
        )

    def forward(self, texts, text_lengths, mel_lengths, alignment):
        x = self.text_encoder(
            texts,
            attention_mask=(~length_to_mask(text_lengths)).int(),
        ).transpose(-1, -2)
        x = (x @ alignment).transpose(-1, -2)
        x = self.project(x)
        return torch.stack([head(x) for head in self.heads], dim=-2)  # BxTxHxC
