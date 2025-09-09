from typing import List, Optional, Tuple
import torch
from torch.nn.utils.parametrizations import weight_norm
from torch.nn import Conv2d


class MultiPeriodDiscriminator(torch.nn.Module):
    """
    Multi-Period Discriminator module adapted from https://github.com/jik876/hifi-gan.
    Additionally, it allows incorporating conditional information with a learned embeddings table.

    Args:
        periods (tuple[int]): Tuple of periods for each discriminator.
        num_embeddings (int, optional): Number of embeddings. None means non-conditional discriminator.
            Defaults to None.
    """

    def __init__(
        self,
        periods: Tuple[int, ...] = (2, 3, 5, 7, 11),
        num_embeddings: Optional[int] = None,
    ):
        super().__init__()
        self.discriminators = torch.nn.ModuleList(
            [DiscriminatorP(period=p, num_embeddings=num_embeddings) for p in periods]
        )

    def forward(self, *, target_list, pred_list):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for target, pred, disc in zip(target_list, pred_list, self.discriminators):
            y_d_r, fmap_r = disc(target.unsqueeze(0).unsqueeze(1))
            y_d_g, fmap_g = disc(pred.unsqueeze(0).unsqueeze(1))
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorP(torch.nn.Module):
    def __init__(
        self,
        period: int,
        in_channels: int = 1,
        kernel_size: int = 5,
        stride: int = 3,
        lrelu_slope: float = 0.1,
        num_embeddings: Optional[int] = None,
    ):
        super().__init__()
        self.period = period
        self.convs = torch.nn.ModuleList(
            [
                weight_norm(
                    Conv2d(
                        in_channels,
                        32,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(kernel_size // 2, 0),
                    )
                ),
                weight_norm(
                    Conv2d(
                        32,
                        128,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(kernel_size // 2, 0),
                    )
                ),
                weight_norm(
                    Conv2d(
                        128,
                        512,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(kernel_size // 2, 0),
                    )
                ),
                weight_norm(
                    Conv2d(
                        512,
                        1024,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(kernel_size // 2, 0),
                    )
                ),
                weight_norm(
                    Conv2d(
                        1024,
                        1024,
                        (kernel_size, 1),
                        (1, 1),
                        padding=(kernel_size // 2, 0),
                    )
                ),
            ]
        )
        if num_embeddings is not None:
            self.emb = torch.nn.Embedding(
                num_embeddings=num_embeddings, embedding_dim=1024
            )
            torch.nn.init.zeros_(self.emb.weight)

        self.conv_post = weight_norm(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))
        self.lrelu_slope = lrelu_slope

    def forward(
        self, x: torch.Tensor, cond_embedding_id: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # x = x.unsqueeze(1)
        fmap = []
        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = torch.nn.functional.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for i, l in enumerate(self.convs):
            x = l(x)
            x = torch.nn.functional.leaky_relu(x, self.lrelu_slope)
            if i > 0:
                fmap.append(x)
        if cond_embedding_id is not None:
            emb = self.emb(cond_embedding_id)
            h = (emb.view(1, -1, 1, 1) * x).sum(dim=1, keepdims=True)
        else:
            h = 0
        x = self.conv_post(x)
        fmap.append(x)
        x += h
        x = torch.flatten(x, 1, -1)

        return x, fmap
