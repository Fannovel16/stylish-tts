import torch
import torch.nn as nn
from .conv_next import BasicConvNeXtBlock, GeneratorConvNeXtBlock
from kanade_tokenizer.model import GlobalEncoder
from dataclasses import dataclass
import torch.nn.functional as F
from einops.layers.torch import Rearrange


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
    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, x, style):
        for block in self:
            if type(block) in [nn.Conv1d, nn.ConvTranspose1d, Rearrange]:
                x = block(x)
            else:
                x = block(x, style)
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
        content_logit_temperature=0.1,
    ):
        super().__init__()
        inter_dim = hidden_dim * 3
        BCT_to_BTC = lambda: Rearrange("b c t -> b t c")
        BTC_to_BCT = lambda: Rearrange("b t c -> b c t")

        self.content_encoder = nn.Sequential(
            BTC_to_BCT(),
            nn.Conv1d(input_dim, hidden_dim, 1, 1),
            *[BasicConvNeXtBlock(hidden_dim, inter_dim) for _ in range(4)],
            nn.Conv1d(hidden_dim, latent_dim, downsample_factor, downsample_factor),
            BCT_to_BTC(),
        )

        self.content_proj = nn.Sequential(
            BTC_to_BCT(),
            nn.ConvTranspose1d(
                latent_dim, latent_dim, downsample_factor, downsample_factor
            ),
            BCT_to_BTC(),
        )
        self.centroids = nn.Embedding(content_discrete_vocab, latent_dim)

        self.global_encoder = GlobalEncoder(
            input_channels=input_dim,
            output_channels=style_dim,
            num_layers=4,
            dim=hidden_dim,
            intermediate_dim=inter_dim,
        )
        self.mel_upsample = nn.Sequential(
            BTC_to_BCT(),
            nn.ConvTranspose1d(
                latent_dim, hidden_dim, mel_upsample_factor, mel_upsample_factor
            ),
        )
        self.mel_decoder = StylishSequential(
            *[
                GeneratorConvNeXtBlock(hidden_dim, inter_dim, style_dim)
                for _ in range(4)
            ],
            nn.Conv1d(hidden_dim, n_mels, 1, 1),
            BCT_to_BTC(),
        )
        self.content_logit_temperature = content_logit_temperature

    def forward(
        self,
        local_emb,
        global_emb,
        train_feature=True,
        output_mel=True,
        mel_length=None,
    ):
        """Inputs: BXTXC, outputs: BXTxC or BxC"""
        content_latent = self.content_encoder(local_emb)
        global_style = self.global_encoder(global_emb)
        features = SoftKanadeFeatures(
            content_latent=content_latent, global_style=global_style
        )
        if train_feature:
            # Angular softmax: cosine with a set of centroid embeddings, then softmax in cross-entropy loss
            # https://github.com/bshall/hubert/blob/main/hubert/model.py#L50-L62
            # Linear transformation doesn't preserve angle so the latent is still applicable to Euclidean distance
            centroid_embedding = F.normalize(self.centroids.weight, dim=-1)
            content_logit = F.normalize(self.content_proj(content_latent), dim=-1)
            features.content_logit = (
                content_logit @ centroid_embedding.mT
            ) / self.content_logit_temperature
        else:
            freeze_modules(
                self.content_encoder,
                self.content_proj,
                self.centroids,
            )
        # https://github.com/frothywater/kanade-tokenizer/blob/main/src/kanade_tokenizer/model.py#L324-L331
        if output_mel:
            mel = self.mel_upsample(content_latent)
            if mel_length:
                mel = F.interpolate(mel, size=mel_length, mode="linear")
            mel = self.mel_decoder(mel, global_style)
            features.mel = mel
        return features
