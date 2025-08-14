import torch
import torch.nn as nn
from utils import sequence_mask
from .plbert import PLBERT
from .conformer import Conformer, Swish, calc_same_padding
from .text_encoder import TextEncoder
from .conv_next import GRN


class BasicConvNeXtBlock(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
    ):
        super().__init__()
        self.dwconv = torch.nn.Conv1d(
            dim, dim, kernel_size=21, padding=10, groups=dim
        )  # depthwise conv

        self.norm = torch.nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = torch.nn.Linear(
            dim, intermediate_dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = torch.nn.GELU()
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = torch.nn.Linear(intermediate_dim, dim)

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)

        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)

        x = residual + x
        return x


class CodePredictor(nn.Module):
    def __init__(
        self,
        tokens,
        codebook_size,
        num_codebooks,
        text_encoder_config,
    ):
        super().__init__()
        hidden_dim = 768
        """self.text_encoder = PLBERT(
            vocab_size=tokens,
            hidden_size=768,
            num_attention_heads=12,
            intermediate_size=2048,
            max_position_embeddings=512,
            num_hidden_layers=12,
            dropout=0.1,
        )
        self.project = nn.Linear(768, hidden_dim)"""
        text_encoder_config.hidden_dim = hidden_dim
        self.text_encoder = TextEncoder(tokens, hidden_dim, text_encoder_config)
        """self.heads = nn.ModuleList(
            [nn.Linear(hidden_dim, codebook_size) for _ in range(num_codebooks)]
        )"""
        self.refiner = nn.Sequential(
            *[BasicConvNeXtBlock(hidden_dim, hidden_dim * 4) for _ in range(4)]
        )

    def forward(self, texts, text_lengths, mel_lengths, alignment):
        text_mask = sequence_mask(text_lengths, alignment.shape[1])
        mel_mask = sequence_mask(mel_lengths, alignment.shape[2])
        """x = self.text_encoder(
            texts,
            attention_mask=text_mask.int(),
        ).transpose(-1, -2)"""
        x, _, _ = self.text_encoder(texts, text_lengths)
        # x = x.transpose(-1, -2)
        # return torch.stack([head(x) for head in self.heads], dim=-2)  # BxTxHxC
        x = self.refiner(x @ alignment).transpose(-1, -2)
        return x
