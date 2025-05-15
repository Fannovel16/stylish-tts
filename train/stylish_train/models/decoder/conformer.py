import torch
from torch import nn
import torch.nn.functional as F

from einops.layers.torch import Rearrange
from einops import rearrange
from torch import nn, einsum

# from .ring_attention_pytorch import RingAttention
import logging

logger = logging.getLogger(__name__)
# helper functions


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)


# helper classes


class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()


class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()


class DepthWiseConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups=chan_in)

    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)


# attention, feedforward, and conv module


class Scale(nn.Module):
    def __init__(self, scale, fn):
        super().__init__()
        self.fn = fn
        self.scale = scale

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x.to(x.device))
        try:
            result = self.fn(x.to(x.device), **kwargs)
        except Exception as e:
            logger.error(e)
            exit(str(e))
        return result


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, use_sdpa=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.use_sdpa = use_sdpa
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None, mask=None, context_mask=None):
        h = self.heads
        context = context if context is not None else x

        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        if self.use_sdpa:
            q_ = rearrange(q, "b h n d -> b n h d")
            k_ = rearrange(k, "b h n d -> b n h d")
            v_ = rearrange(v, "b h n d -> b n h d")

            if mask is not None or context_mask is not None:
                attn_mask = self._get_combined_mask(mask, context_mask, x, context)
                attn_mask = attn_mask.to(torch.bool)
            else:
                attn_mask = None

            out = F.scaled_dot_product_attention(
                q_, k_, v_, attn_mask=attn_mask, dropout_p=0.0
            )
            out = rearrange(out, "b n h d -> b n (h d)")
        else:
            dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

            if mask is not None or context_mask is not None:
                attn_mask = self._get_combined_mask(mask, context_mask, x, context)
                mask_value = -torch.finfo(dots.dtype).max
                dots.masked_fill_(~attn_mask, mask_value)

            attn = dots.softmax(dim=-1)
            out = einsum("b h i j, b h j d -> b h i d", attn, v)
            out = rearrange(out, "b h n d -> b n (h d)")
        return self.dropout(self.to_out(out))

    def _get_combined_mask(self, mask, context_mask, x, context):
        b, n = x.shape[:2]
        _, m = context.shape[:2]
        mask = mask if mask is not None else torch.ones(b, n, device=x.device)
        context_mask = (
            context_mask
            if context_mask is not None
            else (mask if not (context is not x) else torch.ones(b, m, device=x.device))
        )
        return rearrange(mask, "b i -> b () i ()") * rearrange(
            context_mask, "b j -> b () () j"
        )


class ConformerConvModule(nn.Module):
    def __init__(
        self, dim, causal=False, expansion_factor=2, kernel_size=31, dropout=0.0
    ):
        super().__init__()

        inner_dim = dim * expansion_factor
        padding = calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange("b n c -> b c n"),
            nn.Conv1d(dim, inner_dim * 2, 1),
            GLU(dim=1),
            DepthWiseConv1d(
                inner_dim, inner_dim, kernel_size=kernel_size, padding=padding
            ),
            nn.BatchNorm1d(inner_dim) if not causal else nn.Identity(),
            Swish(),
            nn.Conv1d(inner_dim, dim, 1),
            Rearrange("b c n -> b n c"),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# Conformer Block


class ConformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head=64,
        heads=8,
        ff_mult=4,
        conv_expansion_factor=2,
        conv_kernel_size=31,
        attn_dropout=0.0,
        ff_dropout=0.0,
        conv_dropout=0.0,
        conv_causal=False,
    ):
        super().__init__()
        self.ff1 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
        self.attn = Attention(
            dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout
        )
        self.self_attn_dropout = torch.nn.Dropout(attn_dropout)
        self.conv = ConformerConvModule(
            dim=dim,
            causal=conv_causal,
            expansion_factor=conv_expansion_factor,
            kernel_size=conv_kernel_size,
            dropout=conv_dropout,
        )
        self.ff2 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)

        self.attn = PreNorm(dim, self.attn)
        self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))

        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        x_ff1 = self.ff1(x) + x
        x = self.attn(x, mask=mask)
        x = self.self_attn_dropout(x)
        x = x + x_ff1
        x = self.conv(x) + x
        x = self.ff2(x) + x
        x = self.post_norm(x)
        return x


# Conformer


class Conformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        dim_head=64,
        heads=8,
        ff_mult=4,
        conv_expansion_factor=2,
        conv_kernel_size=31,
        attn_dropout=0.0,
        ff_dropout=0.0,
        conv_dropout=0.0,
        conv_causal=False,
    ):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                ConformerBlock(
                    dim=dim,
                    dim_head=dim_head,
                    heads=heads,
                    ff_mult=ff_mult,
                    conv_expansion_factor=conv_expansion_factor,
                    conv_kernel_size=conv_kernel_size,
                    conv_causal=conv_causal,
                )
            )

    def forward(self, x):

        for block in self.layers:
            x = block(x)

        return x
