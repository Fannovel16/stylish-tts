import torch
import torch.nn as nn
from .text_encoder import Encoder as RoPETransformer
from .generator import TorchSTFT


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


class WN(nn.Module):
    def __init__(
        self,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0,
        p_dropout=0,
    ):
        super(WN, self).__init__()
        assert kernel_size % 2 == 1
        self.hidden_channels = hidden_channels
        self.kernel_size = (kernel_size,)
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        if gin_channels != 0:
            cond_layer = nn.Linear(gin_channels, 2 * hidden_channels * n_layers)
            self.cond_layer = nn.utils.weight_norm(cond_layer, name="weight")

        for i in range(n_layers):
            dilation = dilation_rate**i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = nn.Conv1d(
                hidden_channels,
                2 * hidden_channels,
                kernel_size,
                dilation=dilation,
                padding=padding,
            )
            in_layer = nn.utils.weight_norm(in_layer, name="weight")
            self.in_layers.append(in_layer)

            # last one is not necessary
            res_skip_channels = (
                2 * hidden_channels if i < n_layers - 1 else hidden_channels
            )
            res_skip_layer = nn.Linear(hidden_channels, res_skip_channels)
            res_skip_layer = nn.utils.weight_norm(res_skip_layer, name="weight")
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, x, x_mask, g=None, **kwargs):
        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_channels])

        if g is not None:
            g = self.cond_layer(g.mT).mT

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset : cond_offset + 2 * self.hidden_channels, :]
            else:
                g_l = torch.zeros_like(x_in)

            acts = fused_add_tanh_sigmoid_multiply(x_in, g_l, n_channels_tensor)
            acts = self.drop(acts)

            res_skip_acts = self.res_skip_layers[i](acts.mT).mT
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, : self.hidden_channels, :]
                x = (x + res_acts) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels :, :]
            else:
                output = output + res_skip_acts
        return output * x_mask

    def remove_weight_norm(self):
        if self.gin_channels != 0:
            nn.utils.remove_weight_norm(self.cond_layer)
        for l in self.in_layers:
            nn.utils.remove_weight_norm(l)
        for l in self.res_skip_layers:
            nn.utils.remove_weight_norm(l)


class ResidualCouplingBlock(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        n_flows=4,
        gin_channels=0,
        use_transformer_flow=True,
    ):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.half_channels = channels // 2
        self.flows = nn.ModuleList()
        for i in range(n_flows):
            use_transformer = (
                use_transformer_flow if (i == n_flows - 1) else False
            )  # TODO or (i == n_flows - 2)
            self.flows.append(
                ResidualCouplingLayer(
                    self.half_channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    cond_channels=gin_channels,
                    use_transformer_flow=use_transformer,
                )
            )
            self.flows.append(Flip())

    def forward(self, zs, means, logstds, z_mask, cond=None, reverse=False):
        zs = torch.split(zs, [self.half_channels] * 2, 1)
        means = torch.split(means, [self.half_channels] * 2, 1)
        logstds = torch.split(logstds, [self.half_channels] * 2, 1)

        if reverse:
            for flow in reversed(self.flows):
                zs, means, logstds = flow(
                    zs, means, logstds, z_mask, cond=cond, reverse=reverse
                )
        else:
            for flow in self.flows:
                zs, means, logstds = flow(
                    zs, means, logstds, z_mask, cond=cond, reverse=reverse
                )

        zs = torch.cat(zs, 1)
        means = torch.cat(means, 1)
        logstds = torch.cat(logstds, 1)
        return zs, means, logstds


class ResidualCouplingLayer(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        p_dropout=0,
        cond_channels=0,
        use_transformer_flow=True,
    ):
        super().__init__()

        self.channels = channels
        self.pre_transformer = (
            RoPETransformer(
                self.channels,
                self.channels * 3,
                n_heads=2,
                n_layers=1,
                kernel_size=3,
                p_dropout=0.1,
            )
            if use_transformer_flow
            else None
        )

        self.pre = nn.Linear(self.channels, hidden_channels)
        self.enc = WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            p_dropout=p_dropout,
            gin_channels=cond_channels,
        )
        self.proj_mean = nn.Linear(hidden_channels, self.channels)
        self.proj_logstd = nn.Linear(hidden_channels, self.channels)
        zero_init(self.proj_mean)
        zero_init(self.proj_logstd)

    def forward(self, zs, means, logstds, z_mask, cond=None, reverse=False):
        z0, z1 = zs
        mean0, mean1 = means
        logstd0, logstd1 = logstds
        z0_ = z0
        if self.pre_transformer is not None:
            z0_ = self.pre_transformer(z0, z_mask)
            z0_ = z0_ + z0  # residual connection
        h = self.pre(z0_.mT).mT * z_mask
        h = self.enc(h, z_mask, g=cond)
        mean_flow = self.proj_mean(h.mT).mT * z_mask
        logstd_flow = self.proj_logstd(h.mT).mT * z_mask

        if reverse:
            z1 = (z1 - mean_flow) * torch.exp(-logstd_flow) * z_mask
            mean1 = (mean1 - mean_flow) * torch.exp(-logstd_flow) * z_mask
            logstd1 = logstd1 - logstd_flow
        else:
            z1 = mean_flow + z1 * torch.exp(logstd_flow) * z_mask
            mean1 = mean_flow + mean1 * torch.exp(logstd_flow) * z_mask
            logstd1 = logstd1 + logstd_flow

        return (z0, z1), (mean0, mean1), (logstd0, logstd1)


class Flip(nn.Module):
    def forward(self, zs, means, logstds, *args, **kwargs):
        z0, z1 = zs
        mean0, mean1 = means
        logstd0, logstd1 = logstds
        return (z1, z0), (mean1, mean0), (logstd1, logstd0)


def zero_init(layer):
    layer.weight.data.zero_()
    layer.bias.data.zero_()


class PosteriorEncoder(nn.Module):
    def __init__(
        self,
        out_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        n_fft: int,
        hop_length: int,
        win_length: int,
        gin_channels=0,
    ):
        """Posterior Encoder of MagPhase.

        ::
            x (waveform) -> mag/phase -> ConvNext mag/phase -> concat() -> WaveNet (non-causal) -> conv1x1() -> split() -> [m, s] -> sample(m, s) -> z

        Args:
            out_channels (int): Number of output tensor channels.
            hidden_channels (int): Number of hidden channels.
            kernel_size (int): Kernel size of the WaveNet convolution layers.
            dilation_rate (int): Dilation rate of the WaveNet layers.
            num_layers (int): Number of the WaveNet layers.
            cond_channels (int, optional): Number of conditioning tensor channels. Defaults to 0.
        """
        super().__init__()
        self.out_channels = out_channels
        self.stft = TorchSTFT(
            filter_length=n_fft,
            hop_length=hop_length,
            win_length=win_length,
        )

        self.pre_spec = nn.Conv1d(n_fft // 2 + 1, hidden_channels // 2, 1)
        self.pre_phase = nn.Conv1d(n_fft // 2 + 1, hidden_channels // 2, 1)
        self.enc = WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        self.proj_mean = nn.Linear(hidden_channels, out_channels)
        self.proj_logstd = nn.Linear(hidden_channels, out_channels)
        zero_init(self.proj_mean)
        zero_init(self.proj_logstd)

    def forward(self, x: torch.Tensor, g=None):
        har_spec, har_x, har_y = self.stft.transform(x)
        har_phase = torch.atan2(har_y, har_x)
        har_spec = har_spec[:, :, :-1]
        har_phase = har_phase[:, :, :-1]

        x = torch.cat([self.pre_spec(har_spec), self.pre_phase(har_phase)], dim=1)
        x = self.enc(x, 1, g=g)
        mean = self.proj_mean(x.mT).mT
        logstd = self.proj_logstd(x.mT).mT
        z = mean + torch.randn_like(mean) * torch.exp(logstd)
        return z, mean, logstd


class PriorEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        """
        For existing encoder
        """
        super().__init__()
        self.proj_mean = nn.Linear(in_channels, out_channels)
        self.proj_logstd = nn.Linear(in_channels, out_channels)
        zero_init(self.proj_mean)
        zero_init(self.proj_logstd)

    def forward(self, x: torch.Tensor):
        mean = self.proj_mean(x.mT).mT
        logstd = self.proj_logstd(x.mT).mT
        z = mean + torch.randn_like(mean) * torch.exp(logstd)
        return z, mean, logstd


if __name__ == "__main__":
    flow = ResidualCouplingBlock(128, 512, 7, 1, 2, 4, 64)
    z0, z1 = torch.rand(1, 128, 100), torch.rand(1, 128, 100)
    mean0, mean1 = torch.rand(1, 128, 100), torch.rand(1, 128, 100)
    logstd0, logstd1 = torch.rand(1, 128, 100), torch.rand(1, 128, 100)
    flow(
        (z0, z1),
        (mean0, mean1),
        (logstd0, logstd1),
        z_mask=torch.ones(1, 1, 100),
        cond=torch.rand(1, 64, 1),
    )
