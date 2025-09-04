import torch
import torch.nn as nn
import torch.nn.functional as F
from .text_encoder import (
    TextEncoder,
    MultiHeadAttention,
    FFN,
    sequence_mask,
    HubertEncoder,
)
from .fine_style_encoder import FineStyleEncoder
from stylish_lib.config_loader import (
    TextEncoderConfig,
    StyleEncoderConfig,
    AdaptiveFeatureEncoderConfig,
)
from einops import rearrange
from .text_style_encoder import TextStyleEncoder


class TextFeatureExtractor(nn.Module):
    def __init__(
        self,
        tokens,
        inter_dim,
        style_dim,
        text_encoder_config: TextEncoderConfig,
        style_encoder_config: StyleEncoderConfig,
        feature_encoder_config: AdaptiveFeatureEncoderConfig,
        encode_feature=True,
    ):
        super().__init__()
        self.encode_feature = encode_feature
        self.text_encoder = TextEncoder(
            tokens, inter_dim, **text_encoder_config.model_dump()
        )
        self.style_encoder = FineStyleEncoder(
            inter_dim, style_dim, style_encoder_config.layers
        )
        if self.encode_feature:
            self.feature_encoder = AdaptiveFeatureEncoder(
                sty_dim=style_dim,
                d_model=inter_dim,
                layers=feature_encoder_config.layers,
                dropout=feature_encoder_config.dropout,
                heads=feature_encoder_config.heads,
                kernel_size=feature_encoder_config.kernel_size,
            )

    def forward(self, x, x_lengths):
        x, _, _ = self.text_encoder(x, x_lengths)
        style = self.style_encoder(x)
        if self.encode_feature:
            x = self.feature_encoder(x, x_lengths, style)
        return x, style


class HubertFeatureExtractor(nn.Module):
    def __init__(
        self,
        hubert_dim,
        spk_dim,
        style_dim,
        inter_dim,
        feature_encoder_config: AdaptiveFeatureEncoderConfig,
        final_style_concat=True,
        f0=False,
    ):
        super().__init__()
        self.style_encoder = TextStyleEncoder(hubert_dim + spk_dim, style_dim)
        self.feature_encoder = AdaptiveFeatureEncoder(
            sty_dim=style_dim,
            d_model=hubert_dim,
            layers=feature_encoder_config.layers,
            dropout=feature_encoder_config.dropout,
            heads=feature_encoder_config.heads,
            kernel_size=feature_encoder_config.kernel_size,
            final_style_concat=final_style_concat,
            final_d_model=inter_dim,
        )

    def forward(self, x, x_lengths, spk_emb, pitch=None):
        raw_style = torch.cat(
            [x, spk_emb.unsqueeze(-1).repeat(1, 1, x.shape[-1])], dim=1
        )
        style = self.style_encoder(raw_style, x_lengths)
        x = self.feature_encoder(x, x_lengths, style)
        return x, style


class AdaptiveFeatureEncoder(nn.Module):
    def __init__(
        self,
        sty_dim,
        d_model,
        layers,
        dropout=0.1,
        heads=2,
        kernel_size=1,
        final_style_concat=True,
        final_d_model=None,
        **kwargs,
    ):
        super().__init__()
        hidden_channels = d_model + sty_dim
        self.n_heads = heads
        self.n_layers = layers
        self.kernel_size = kernel_size
        self.final_style_concat = final_style_concat
        self.final_d_model = final_d_model or d_model

        self.drop = torch.nn.Dropout(dropout)
        self.attn_layers = torch.nn.ModuleList()
        self.norm_layers_1 = torch.nn.ModuleList()
        self.ffn_layers = torch.nn.ModuleList()
        self.norm_layers_2 = torch.nn.ModuleList()
        self.proj_layers = torch.nn.ModuleList()

        for i in range(self.n_layers):
            self.attn_layers.append(
                MultiHeadAttention(
                    hidden_channels, hidden_channels, heads, p_dropout=dropout
                )
            )
            self.norm_layers_1.append(AdaLayerNorm(sty_dim, hidden_channels))
            self.ffn_layers.append(
                FFN(
                    hidden_channels,
                    hidden_channels,
                    hidden_channels * 2,
                    kernel_size,
                    p_dropout=dropout,
                )
            )
            self.norm_layers_2.append(AdaLayerNorm(sty_dim, hidden_channels))
            self.proj_layers.append(
                nn.Conv1d(
                    hidden_channels,
                    d_model if i < self.n_layers - 1 else self.final_d_model,
                    1,
                )
            )

    def forward(self, x, x_lengths, style):
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = torch.cat([x, style], dim=1)
        for i in range(self.n_layers):
            x = x * x_mask
            y = self.attn_layers[i](x, x, attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](torch.transpose(x + y, -1, -2), style).transpose(
                -1, -2
            )

            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](torch.transpose(x + y, -1, -2), style).transpose(
                -1, -2
            )
            x = self.proj_layers[i](x)
            if i < self.n_layers - 1 or (
                i == self.n_layers - 1 and self.final_style_concat
            ):
                x = torch.cat([x, style], dim=1)
        x = x * x_mask
        return x.transpose(-1, -2)


class AdaLayerNorm(nn.Module):
    def __init__(self, style_dim, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.fc = nn.Linear(style_dim, channels * 2)

    def forward(self, x, s):
        x = x.transpose(-1, -2)
        x = x.transpose(1, -1)

        s = rearrange(s, "b s t -> b t s")
        h = self.fc(s)
        gamma = h[:, :, : self.channels]
        beta = h[:, :, self.channels :]

        x = F.layer_norm(x, (self.channels,), eps=self.eps)
        x = (1 + gamma) * x + beta
        return x
