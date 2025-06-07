import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm
from einops import rearrange
from .common import InstanceNorm1d
from .duration_predictor import AdaLNTransformer


class PitchEnergyPredictor(torch.nn.Module):
    def __init__(self, style_dim, d_hid, dropout=0.1):
        super().__init__()

        self.shared = nn.LSTM(
            d_hid + style_dim, d_hid // 2, 1, batch_first=True, bidirectional=True
        )
        self.F0 = nn.ModuleList()
        self.F0.append(AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout))
        self.F0.append(
            AdainResBlk1d(
                d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout
            )
        )
        self.F0.append(
            AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout)
        )

        self.N = nn.ModuleList()
        self.N.append(AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout))
        self.N.append(
            AdainResBlk1d(
                d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout
            )
        )
        self.N.append(
            AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout)
        )

        self.F0_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)
        self.N_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)

    def forward(self, prosody, style):
        upstyle = torch.nn.functional.interpolate(style, scale_factor=2, mode="nearest")
        # x = torch.cat([prosody, style], dim=1)
        x = prosody
        x, _ = self.shared(x.transpose(-1, -2))

        s = style
        F0 = x.transpose(-1, -2)
        for block in self.F0:
            F0 = block(F0, s)
            if block.upsample_type == True:
                s = upstyle
        F0 = self.F0_proj(F0)

        s = style
        N = x.transpose(-1, -2)
        for block in self.N:
            N = block(N, s)
            if block.upsample_type == True:
                s = upstyle
        N = self.N_proj(N)

        return F0.squeeze(1), N.squeeze(1)


class ProsodicFeaturePredictor(torch.nn.Module):
    def __init__(self, style_dim, d_hid, nlayers, dropout=0.1):
        super().__init__()
        self.encoder = AdaLNTransformer(style_dim, d_hid, nlayers, dropout)
        self.lstm = nn.LSTM(
            d_hid + style_dim, d_hid // 2, 1, batch_first=True, bidirectional=True
        )
        self.convs = nn.ModuleList()
        self.convs.append(AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout))
        self.convs.append(
            AdainResBlk1d(
                d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout
            )
        )
        self.convs.append(
            AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout)
        )

        self.proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)

    def forward(self, text, style, text_lengths, alignment):
        prosody = self.encoder(text, style, text_lengths).transpose(-1, -2) @ alignment
        style = style @ alignment
        upstyle = torch.nn.functional.interpolate(style, scale_factor=2, mode="nearest")
        # x = torch.cat([prosody, style], dim=1)
        x, _ = self.lstm(prosody.transpose(-1, -2))

        s = style
        x = x.transpose(-1, -2)
        for block in self.convs:
            x = block(x, s)
            if block.upsample_type == True:
                s = upstyle
        x = self.proj(x)
        return x.squeeze(1)


class AdainResBlk1d(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        style_dim=64,
        actv=nn.LeakyReLU(0.2),
        upsample="none",
        dropout_p=0.0,
    ):
        super().__init__()
        self.actv = actv
        self.upsample_type = upsample
        self.upsample = UpSample1d(upsample)
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)
        self.dropout = nn.Dropout(dropout_p)

        if upsample == "none":
            self.pool = nn.Identity()
        else:
            self.pool = weight_norm(
                nn.ConvTranspose1d(
                    dim_in,
                    dim_in,
                    kernel_size=3,
                    stride=2,
                    groups=dim_in,
                    padding=1,
                    output_padding=1,
                )
            )

    def _build_weights(self, dim_in, dim_out, style_dim):
        self.conv1 = weight_norm(nn.Conv1d(dim_in, dim_out, 3, 1, 1))
        self.conv2 = weight_norm(nn.Conv1d(dim_out, dim_out, 3, 1, 1))
        self.norm1 = AdaIN1d(style_dim, dim_in)
        self.norm2 = AdaIN1d(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = weight_norm(nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False))

    def _shortcut(self, x):
        x = self.upsample(x)
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        x = self.pool(x)
        if self.upsample_type == True:
            s = torch.nn.functional.interpolate(s, scale_factor=2, mode="nearest")
        x = self.conv1(self.dropout(x))
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(self.dropout(x))
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        out = (out + self._shortcut(x)) / math.sqrt(2)
        return out


class UpSample1d(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == "none":
            return x
        else:
            return F.interpolate(x, scale_factor=2, mode="nearest")


class AdaIN1d(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = InstanceNorm1d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)
        self.num_features = num_features

    def forward(self, x, s):
        s = rearrange(s, "b s t -> b t s")
        h = self.fc(s)
        h = rearrange(h, "b t s -> b s t")
        gamma = h[:, : self.num_features, :]
        beta = h[:, self.num_features :, :]
        return (1 + gamma) * self.norm(x) + beta
