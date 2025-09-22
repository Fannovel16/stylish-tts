import torch.nn as nn

from .layers import SwiGLU
from .attention import SelfAttention, CrossAttention
from .norm import RMSNorm, DyT
from .adaln import AdaLN
from .transformer import TransformerBlock


def isiterable(obj):
    try:
        iter(obj)
    except TypeError:
        return False
    return True


class XUTBackBone(nn.Module):
    """
    Basic backbone of cross-U-transformer.
    """

    def __init__(
        self,
        dim=1024,
        ctx_dim=None,
        heads=16,
        dim_head=64,
        mlp_dim=3072,
        pos_dim=2,
        depth=8,
        enc_blocks=1,
        dec_blocks=2,
        dec_ctx=False,
        use_adaln=False,
        use_shared_adaln=False,
        use_dyt=False,
    ):
        super().__init__()
        if isiterable(enc_blocks):
            enc_blocks = list(enc_blocks)
            assert len(enc_blocks) == depth
        else:
            enc_blocks = [int(enc_blocks)] * depth
        if isiterable(dec_blocks):
            dec_blocks = list(dec_blocks)
            assert len(dec_blocks) == depth
        else:
            dec_blocks = [int(dec_blocks)] * depth

        self.enc_blocks = nn.ModuleList()
        for i in range(depth):
            blocks = [
                TransformerBlock(
                    dim,
                    ctx_dim,
                    heads,
                    dim_head,
                    mlp_dim,
                    pos_dim,
                    use_adaln,
                    use_shared_adaln,
                    norm_layer=DyT if use_dyt else RMSNorm,
                )
                for _ in range(enc_blocks[i])
            ]
            self.enc_blocks.append(nn.ModuleList(blocks))

        self.dec_ctx = dec_ctx
        self.dec_blocks = nn.ModuleList()
        for i in range(depth):
            blocks = [
                TransformerBlock(
                    dim,
                    dim if bid == 0 else ctx_dim if dec_ctx else None,
                    heads,
                    dim_head,
                    mlp_dim,
                    pos_dim,
                    use_adaln,
                    use_shared_adaln,
                    ctx_from_self=bid == 0,
                    norm_layer=DyT if use_dyt else RMSNorm,
                )
                for bid in range(dec_blocks[i])
            ]
            self.dec_blocks.append(nn.ModuleList(blocks))

        self.grad_ckpt = False

    def init_weight(self):
        for param in self.parameters():
            if param.ndim == 1:
                nn.init.normal_(param, mean=0.0, std=(1 / param.size(0)) ** 0.5)
            elif param.ndim == 2:
                fan_in = param.size(1)
                nn.init.normal_(param, mean=0.0, std=(1 / fan_in) ** 0.5)
            elif param.ndim >= 3:
                fan_out, *fan_ins = param.shape
                # cumprod
                fan_in = 1
                for f in fan_ins:
                    fan_in *= f
                nn.init.normal_(param, mean=0.0, std=(1 / fan_in) ** 0.5)

    def forward(
        self,
        x,
        ctx=None,
        x_mask=None,
        ctx_mask=None,
        pos_map=None,
        y=None,
        shared_adaln=None,
        return_enc_out=False,
    ):
        if pos_map is not None:
            assert pos_map.size(1) == x.size(1)

        self_ctx = []
        for blocks in self.enc_blocks:
            for block in blocks:
                x = block(x, ctx, pos_map, None, y, x_mask, ctx_mask, shared_adaln)
            self_ctx.append(x)
        enc_out = x

        for blocks in self.dec_blocks:
            first_block = blocks[0]
            x = first_block(
                x, self_ctx[-1], pos_map, pos_map, y, x_mask, ctx_mask, shared_adaln
            )

            for block in blocks[1:]:
                x = block(
                    x,
                    ctx if self.dec_ctx else None,
                    pos_map,
                    None,
                    y,
                    x_mask,
                    ctx_mask,
                    shared_adaln,
                )

        if return_enc_out:
            return x, enc_out
        return x
