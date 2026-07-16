import torch
import torch.nn as nn
from .conv_next import BasicConvNeXtBlock, GeneratorConvNeXtBlock
from kanade_tokenizer.model import GlobalEncoder
from dataclasses import dataclass


def freeze_modules(*modules: list[nn.Module]):
    for module in modules:
        if module is not None:
            for param in module.parameters():
                param.requires_grad = False


@dataclass
class SoftKanadeFeatures:
    content_latent: torch.Tensor = None
    global_style: torch.Tensor = None
    content_recon: torch.Tensor = None
    content_logit: torch.Tensor = None
    mel: torch.Tensor = None


class StylishSequential(nn.Sequential):
    def __init__(self, *args, proj=nn.Identity()):
        super().__init__(*args)
        self.proj = proj

    def forward(self, x, style):
        for name, module in self.named_children():
            if name != "proj":
                x = module(x, style)
            else:
                x = module(x)
        return x


class SoftKanade(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        latent_dim,
        style_dim,
        content_discrete_vocab,
        downsample_factor,
        mel_upsample_factor,
        n_mels,
    ):
        super().__init__()
        inter_dim = hidden_dim * 4

        self.content_encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 1, 1),
            *[BasicConvNeXtBlock(hidden_dim, inter_dim) for _ in range(4)],
            nn.Conv1d(hidden_dim, latent_dim, downsample_factor, downsample_factor),
            nn.GroupNorm(
                1, latent_dim, affine=False
            )  # Channel-first LayerNorm, mean 0 std unit
        )
        self.content_decoder = nn.Sequential(
            nn.ConvTranspose1d(
                latent_dim, hidden_dim, downsample_factor, downsample_factor
            ),
            *[BasicConvNeXtBlock(hidden_dim, inter_dim) for _ in range(4)]
        )
        self.content_recon_head = nn.Conv1d(hidden_dim // 2, input_dim, 1, 1)
        self.content_quant_head = nn.Conv1d(
            hidden_dim // 2, content_discrete_vocab, 1, 1
        )

        self.global_encoder = GlobalEncoder(
            input_channels=input_dim,
            output_channels=style_dim,
            num_layers=4,
            dim=hidden_dim,
            intermediate_dim=inter_dim,
        )
        self.mel_upsample = nn.ConvTranspose1d(
            latent_dim, hidden_dim, mel_upsample_factor, mel_upsample_factor
        )
        self.mel_decoder = StylishSequential(
            *[
                GeneratorConvNeXtBlock(hidden_dim, hidden_dim * 4, style_dim)
                for _ in range(4)
            ],
            proj=nn.Conv1d(hidden_dim, n_mels, 1, 1)
        )

    def forward(
        self,
        local_emb,
        global_emb,
        train_feature=True,
        output_mel=True,
        mel_length=None,
    ):
        """Inputs: BXTXC, outputs: BXTxC or BxC"""
        content_latent = self.content_encoder(local_emb.mT)
        global_style = self.global_encoder(global_emb)
        features = SoftKanadeFeatures(
            content_latent=content_latent.mT, global_style=global_style
        )
        if train_feature:
            decoded_latents = self.content_decoder(content_latent).chunk(2, dim=1)
            features.content_recon = self.content_recon_head(decoded_latents[0]).mT
            features.content_logit = self.content_quant_head(decoded_latents[1]).mT
        else:
            freeze_modules(
                self.content_encoder,
                self.content_decoder,
                self.content_recon_head,
                self.content_quant_head,
            )
        # https://github.com/frothywater/kanade-tokenizer/blob/main/src/kanade_tokenizer/model.py#L324-L331
        if output_mel:
            content_mel = self.mel_upsample(content_latent)
            if mel_length:
                content_mel = nn.functional.interpolate(
                    content_mel, size=mel_length, mode="linear"
                )
            features.mel = self.mel_decoder(content_mel, global_style).mT
        return features
