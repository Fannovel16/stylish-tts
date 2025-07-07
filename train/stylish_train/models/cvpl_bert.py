import torch
import torch.nn as nn
from .conv_next import BasicConvNeXtBlock
from transformers import AlbertModel, AlbertConfig
from utils import sequence_mask


class CVPLBERT(nn.Module):
    def __init__(self, tokens, hidden_dim):
        super().__init__()
        self.encoder = AlbertModel(
            AlbertConfig(
                vocab_size=tokens,
                hidden_size=hidden_dim,
                num_attention_heads=12,
                intermediate_size=2048,
                max_position_embeddings=512,
                num_hidden_layers=12,
                dropout=0.1,
            )
        )
        self.refine = BasicConvNeXtBlock(hidden_dim, hidden_dim * 4)

    def forward(self, x, x_lengths, alignment):
        x = self.encoder(
            x, attention_mask=sequence_mask(x_lengths, x.shape[1]).float()
        ).last_hidden_state
        return self.refine(x.transpose(-1, -2) @ alignment).transpose(-1, -2)
