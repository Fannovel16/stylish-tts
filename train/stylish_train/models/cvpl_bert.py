import torch
import torch.nn as nn
from utils import sequence_mask
from .conformer import Conformer
from transformers import AlbertConfig, AlbertModel
from vector_quantize_pytorch import GroupedResidualVQ


class CVPLBERT(nn.Module):
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
                hidden_size=hubert_dim,
                num_attention_heads=12,
                intermediate_size=2048,
                max_position_embeddings=512,
                num_hidden_layers=12,
                dropout=0.1,
            )
        )
        self.encoder = Conformer(
            dim=hubert_dim, depth=4, heads=8, dim_head=hubert_dim // 8
        )
        self.down = nn.Linear(hubert_dim, hidden_dim)
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
        )
        self.up = nn.Linear(hidden_dim, hubert_dim)
        self.decoder = Conformer(
            dim=hubert_dim, depth=4, heads=8, dim_head=hubert_dim // 8
        )

    def forward(self, texts, text_lengths, alignment):
        x = self.text_encoder(
            texts, attention_mask=sequence_mask(text_lengths, texts.shape[1]).float()
        )
        x = self.encoder((x @ alignment).transpose(-1, -2))
        x = self.down(x)
        x, indices, cmt_loss = self.quantizer(x)
        x = self.up(x)
        x = self.decoder(x)
        if self.training:
            return x, indices, cmt_loss.mean()
        else:
            return x, indices
