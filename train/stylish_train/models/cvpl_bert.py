import torch
import torch.nn as nn
from utils import sequence_mask
from .conformer import Conformer
from transformers import AlbertConfig, AlbertModel
from vector_quantize_pytorch import ResidualVQ
from .text_encoder import TextEncoder


"""class CVPLBERT(nn.Module):
    def __init__(
        self,
        tokens,
        hubert_dim,
        hidden_dim,
        text_encoder_config,
        codebook_size=1024,
        rq_num_quantizers=4,
        rq_commitment_weight=1.0,
        rq_ema_decay=0.95,
        rq_quantize_dropout_multiple_of=1,
        rq_groups=2,
        rq_stochastic_sample_codes=False,
        rq_rotation_trick=True,
        quantize_dropout_cutoff_index=1,
    ):
        super().__init__()
        self.text_encoder = AlbertModel(
            AlbertConfig(
                vocab_size=tokens,
                embedding_size=128,
                hidden_size=hubert_dim,
                num_attention_heads=8,
                intermediate_size=hubert_dim * 4,
                num_hidden_layers=9,
                num_hidden_groups=1,
                inner_group_num=1,
                dropout=0.1,
            )
        )
        self.down = nn.Linear(hubert_dim, hidden_dim)
        self.encoder = Conformer(
            dim=hidden_dim,
            depth=4,
            heads=8,
            dim_head=hidden_dim // 8,
            conv_kernel_size=9,
            ff_mult=4,
        )
        self.quantizer = GroupedResidualVQ(
            dim=hidden_dim,
            num_quantizers=rq_num_quantizers,
            codebook_size=codebook_size,
            groups=rq_groups,
            decay=rq_ema_decay,
            commitment_weight=rq_commitment_weight,
            quantize_dropout_multiple_of=rq_quantize_dropout_multiple_of,
            kmeans_init=True,
            threshold_ema_dead_code=2,
            quantize_dropout=True,
            quantize_dropout_cutoff_index=quantize_dropout_cutoff_index,
            stochastic_sample_codes=rq_stochastic_sample_codes,
            rotation_trick=rq_rotation_trick,
            kmeans_iters=500,
        )
        self.up = nn.Linear(hidden_dim, hubert_dim)
        self.decoder = Conformer(
            dim=hubert_dim,
            depth=4,
            heads=8,
            dim_head=hubert_dim // 8,
            conv_kernel_size=9,
            ff_mult=4,
        )

    def forward(self, texts, text_lengths, mel_lengths, alignment, global_step=99999):
        text_mask = sequence_mask(text_lengths, texts.shape[1])
        mel_mask = sequence_mask(mel_lengths, alignment.shape[2])
        x = self.text_encoder(texts, attention_mask=text_mask.float()).last_hidden_state
        x = self.down(x)
        x = self.encoder((x.transpose(-1, -2) @ alignment).transpose(-1, -2), mel_mask)
        if global_step < 500:
            indices = torch.full_like(x[..., 0], -1, dtype=torch.long)  # dummy
            cmt_loss = torch.tensor(0.0)
        else:
            x, indices, cmt_loss = self.quantizer(x, mask=mel_mask)
        x = self.up(x)
        x = self.decoder(x, mel_mask)
        if self.training:
            return x, indices, cmt_loss.mean()
        else:
            return x, indices, 0
"""


class CVPLBERT(nn.Module):
    def __init__(
        self,
        tokens,
        hidden_dim,
        codebook_size,
        heads,
        text_encoder_config,
    ):
        super().__init__()
        self.text_encoder = TextEncoder(tokens, hidden_dim, text_encoder_config)
        self.refiner = Conformer(
            hidden_dim,
            depth=4,
            heads=8,
            dim_head=hidden_dim // 8,
            ff_mult=2,
            conv_expansion_factor=2,
            conv_kernel_size=9,
        )
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, codebook_size),
                )
                for _ in range(heads)
            ]
        )

    def forward(self, texts, text_lengths, mel_lengths, alignment):
        mel_mask = sequence_mask(mel_lengths, alignment.shape[2])
        x, _, _ = self.text_encoder(texts, text_lengths)
        x = self.refiner((x @ alignment).transpose(-1, -2), mel_mask)
        return torch.cat([head(x) for head in self.heads], dim=-1)  # BxTx(HxC)
