import torch
import torch.nn as nn
import torch.nn.functional as F


class DyT(nn.Module):
    """
    Transformers without Normalization
    https://arxiv.org/abs/2503.10622
    """

    def __init__(self, hidden_size, init_alpha=1.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.in_weight = nn.Parameter(torch.ones(hidden_size) * init_alpha)

    def forward(self, hidden_states):
        hidden_states = torch.tanh(self.in_weight * hidden_states)
        return hidden_states, 1.0


class RMSNormTorch(nn.RMSNorm):
    def __init__(self, hidden_size, *args, eps=1e-6, offset=0.0, **kwargs):
        super().__init__((hidden_size,), *args, eps=eps, **kwargs)
        self.offset = offset

    def forward(self, hidden_states):
        return (
            F.rms_norm(
                hidden_states,
                self.normalized_shape,
                self.weight + self.offset,
                self.eps,
            ),
            1.0,
        )


RMSNorm = RMSNormTorch


class DyT(nn.Module):
    """
    Transformers without Normalization
    https://arxiv.org/abs/2503.10622
    """

    def __init__(self, hidden_size, init_alpha=1.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.in_weight = nn.Parameter(torch.ones(hidden_size) * init_alpha)

    def forward(self, hidden_states):
        hidden_states = torch.tanh(self.in_weight * hidden_states)
        return hidden_states, 1.0


def Norm(module: nn.Module):
    module.org_forward = module.forward
    module.forward = lambda *args, **kwargs: module.org_forward(*args, **kwargs)[0]
    return module


if __name__ == "__main__":
    if LigerRMSNorm is None:
        print("LigerRMSNorm is available")
        exit()

    hidden_size = 512
    hidden_states = torch.randn(2, hidden_size).cuda()

    norm1 = RMSNorm(hidden_size).cuda()
    norm2 = RMSNormTorch(hidden_size).cuda()

    nn.init.normal_(norm1.weight)
    norm2.load_state_dict(norm1.state_dict())

    print(F.mse_loss(norm1(hidden_states)[0], norm2(hidden_states)[0]))
