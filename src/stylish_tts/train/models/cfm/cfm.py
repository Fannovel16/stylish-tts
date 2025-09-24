import torch, math
import torch.nn as nn
from einops import rearrange, repeat


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        assert self.dim % 2 == 0, "SinusoidalPosEmb requires dim to be even"

    def forward(self, x, scale=1000):
        if x.ndim < 1:
            x = x.unsqueeze(0)
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class CfmSampler(nn.Module):
    def __init__(
        self,
        estimator,
        guidance_w=0.7,
        cond_drop_prob=0.0,
        non_drop_conds=[],
        sigma_min=1e-4,
    ):
        """
        A versatile implementation of Model-Guidance Conditional Flow Matching based on https://arxiv.org/pdf/2504.20334
        The estimator's forward must have keyword arguments t (timestep) of shape [b] and mask
        """
        super().__init__()
        self.estimator = estimator
        self.guidance_w = guidance_w
        self.cond_drop_prob = cond_drop_prob
        self.non_drop_conds = non_drop_conds
        self.sigma_min = sigma_min

    @torch.inference_mode()
    def forward(self, z, mask, n_timesteps, temperature=1.0, **estimator_args):
        """Forward diffusion

        Args:
            z (torch.Tensor): Pure noise
                shape: batch_size, n_feats, mel_timesteps
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            **estimator_args (keyword arguments): Argument passing to the estimator.

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """
        z = z * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=z.device)
        return self.solve_euler(z, t_span=t_span, mask=mask, **estimator_args)

    def solve_euler(self, x, t_span, mask, **estimator_args):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            **estimator_args (keyword arguments): Argument passing to the estimator.
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        for step in range(1, len(t_span)):
            _t = repeat(t.unsqueeze(0), "1 -> b", b=x.shape[0])
            dphi_dt = self.estimator(x, t=_t, mask=mask, **estimator_args)
            x = x + dt * dphi_dt
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
        return x

    def prepare_cond_uncond(self, x1, **estimator_args):
        cond_args, uncond_args = {}, {}
        for k, arg in estimator_args.items():
            cond, uncond = arg, arg
            if isinstance(arg, torch.Tensor):
                if k not in self.non_drop_conds:
                    drop_mask = (
                        torch.rand([x1.shape[0]] + [1] * (cond.ndim - 1)).to(x1.device)
                        > self.cond_drop_prob
                    )
                    cond = cond * drop_mask
                    uncond = torch.zeros_like(cond)
            cond_args[k] = cond
            uncond_args[k] = uncond
        return cond_args, uncond_args

    def compute_pred_target(self, x1, mask, **estimator_args):
        """Computes prediction and target.

        Args:
            x1 (torch.Tensor): Target
                shape: (batch_size, n_feats, mel_timesteps)
            **estimator_args (keyword arguments): Argument passing to the estimator.
        Returns:
            pred: (torch.Tensor): Prediction
                shape: (batch_size, n_feats, mel_timesteps)
            target: (torch.Tensor): Target of Flow Matching
                shape: (batch_size, n_feats, mel_timesteps)
        """
        b, _, t = x1.shape

        # random timestep
        t = torch.rand([b, 1, 1], device=x1.device, dtype=x1.dtype)
        # sample noise p(x_0)
        z = torch.randn_like(x1)

        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z
        t = rearrange(t, "b 1 1 -> b")

        if self.guidance_w == 0:
            pred = self.estimator(y, t=t, mask=mask, **estimator_args)
            target = u
            return pred, target

        cond_args, uncond_args = self.prepare_cond_uncond(x1, **estimator_args)
        v_cond = self.estimator(y, t=t, mask=mask, **cond_args)
        v_uncond = self.estimator(y, t=t, mask=mask, **uncond_args)
        delta_stop_grad = torch.detach(v_cond - v_uncond)
        v_cfg = v_cond + self.guidance_w * delta_stop_grad
        pred, target = v_cfg, u
        return pred, target
