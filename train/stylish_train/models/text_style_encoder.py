import torch
from utils import sequence_mask
from .conv_next import BasicConvNeXtBlock


class TextStyleEncoder(torch.nn.Module):
    # TODO: Remvoe hard-coded values
    def __init__(self, inter_dim, style_dim, config):
        super().__init__()
        self.conv_in = torch.nn.Conv1d(inter_dim, style_dim, kernel_size=7, padding=3)
        self.blocks = torch.nn.ModuleList(
            [
                BasicConvNeXtBlock(
                    dim=style_dim,
                    intermediate_dim=style_dim * 4,
                )
                for _ in range(config.layers)
            ]
        )

    def forward(self, x, lengths):
        x = self.conv_in(x)
        for block in self.blocks:
            x = block(x)
        mask = sequence_mask(lengths, x.shape[2]).unsqueeze(1).to(x.dtype).to(x.device)
        s = (x * mask).sum(dim=2) / lengths.unsqueeze(1)
        # s = s.squeeze(1)
        return s
