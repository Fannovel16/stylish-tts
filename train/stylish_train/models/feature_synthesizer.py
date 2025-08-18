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


class FeatureSynthesizer(nn.Module):
    def __init__(
        self,
        tokens,
        feature_dim,
    ):
        super().__init__()
        self.text_encoder = TextEncoder(
            tokens,
            inter_dim=feature_dim,
            hidden_dim=feature_dim,
            filter_channels=feature_dim * 4,
            heads=4,
            layers=8,
            kernel_size=3,
            dropout=0.1,
        )
        self.refiner = nn.Sequential(
            *[BasicConvNeXtBlock(feature_dim, feature_dim * 4) for _ in range(4)]
        )

    def forward(self, texts, text_lengths, alignment):
        x, _, _ = self.text_encoder(texts, text_lengths)
        x = self.refiner(x @ alignment).transpose(-1, -2)
        return x
