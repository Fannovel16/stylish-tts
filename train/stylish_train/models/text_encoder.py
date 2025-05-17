from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm
from einops import rearrange
from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
)
from .common import LinearNorm, get_padding
from .conv_next import BasicConvNeXtBlock

# from reformer_pytorch import Reformer, Autopadder
from torchaudio.models import Conformer


class TextEncoder(nn.Module):
    def __init__(self, channels, kernel_size, depth, n_symbols, actv=nn.LeakyReLU(0.2)):
        super().__init__()
        self.embedding = nn.Embedding(n_symbols, channels)

        padding = (kernel_size - 1) // 2
        self.cnn = nn.ModuleList()
        for _ in range(depth):
            self.cnn.append(
                nn.Sequential(
                    weight_norm(
                        nn.Conv1d(
                            channels, channels, kernel_size=kernel_size, padding=padding
                        )
                    ),
                    LayerNorm(channels),
                    actv,
                    nn.Dropout(0.2),
                )
            )
        # self.cnn = nn.Sequential(*self.cnn)

        self.lstm = nn.LSTM(
            channels, channels // 2, 1, batch_first=True, bidirectional=True
        )

        self.prosody_lstm = nn.LSTM(
            channels, channels // 2, 1, batch_first=True, bidirectional=True
        )

    def forward(self, x, input_lengths, m, is_prosody):
        x = self.embedding(x)  # [B, T, emb]
        x = x.transpose(1, 2)  # [B, emb, T]
        m = m.to(input_lengths.device).unsqueeze(1)
        x.masked_fill_(m, 0.0)

        for c in self.cnn:
            x = c(x)
            x.masked_fill_(m, 0.0)

        x = x.transpose(1, 2)  # [B, T, chn]

        input_lengths = input_lengths.cpu()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True, enforce_sorted=False
        )

        if is_prosody:
            self.prosody_lstm.flatten_parameters()
            x, _ = self.prosody_lstm(x)
        else:
            self.lstm.flatten_parameters()
            x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        x = x.transpose(-1, -2)
        x_pad = torch.zeros([x.shape[0], x.shape[1], m.shape[-1]])

        x_pad[:, :, : x.shape[-1]] = x
        x = x_pad.to(x.device)

        x.masked_fill_(m, 0.0)

        return x

    def load_state_dict(self, state_dict, strict=False, assign=False):
        strict = False
        return super().load_state_dict(state_dict, strict, assign)


# class TextEncoder(torch.nn.Module):
#     def __init__(
#         self, channels, kernel_size, depth, n_symbols, actv=torch.nn.LeakyReLU(0.2)
#     ):
#         super().__init__()
#         padding = get_padding(kernel_size)
#         self.embedding = torch.nn.Embedding(n_symbols, channels)
#
#         self.cnn = torch.nn.ModuleList()
#         for _ in range(depth):
#             self.cnn.append(
#                 BasicConvNeXtBlock(channels, channels * 2)
#                 # torch.nn.Sequential(
#                 #     weight_norm(
#                 #         torch.nn.Conv1d(
#                 #             channels, channels, kernel_size=kernel_size, padding=padding
#                 #         )
#                 #     ),
#                 #     LayerNorm(channels),
#                 #     actv,
#                 #     torch.nn.Dropout(0.2),
#                 # )
#             )
#
#         # self.prepare_projection = LinearNorm(channels, channels // 2)
#         # self.post_projection = LinearNorm(channels // 2, channels)
#
#         cfg = xLSTMBlockStackConfig(
#             mlstm_block=mLSTMBlockConfig(
#                 mlstm=mLSTMLayerConfig(
#                     conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4
#                 )
#             ),
#             context_length=channels,
#             num_blocks=8,
#             # embedding_dim=channels // 2,
#             embedding_dim=channels,
#         )
#
#         self.lstm = xLSTMBlockStack(cfg)
#
#     def forward(self, x, input_lengths, m):
#         x = self.embedding(x)  # [B, T, emb]
#         x = x.transpose(1, 2)  # [B, emb, T]
#         m = m.to(input_lengths.device).unsqueeze(1)
#         x.masked_fill_(m, 0.0)
#
#         for c in self.cnn:
#             x = c(x)
#             x.masked_fill_(m, 0.0)
#
#         x = x.transpose(1, 2)  # [B, T, chn]
#
#         # x = self.prepare_projection(x)
#         x = self.lstm(x)
#         # x = self.post_projection(x)
#
#         x = x.transpose(1, 2)
#
#         x.masked_fill_(m, 0.0)
#
#         return x
#
#     def inference(self, x):
#         x = self.embedding(x)
#         x = x.transpose(1, 2)
#         x = self.cnn(x)
#         x = x.transpose(1, 2)
#
#         x, _ = self.lstm(x)
#         return x
#
#     def length_to_mask(self, lengths):
#         mask = (
#             torch.arange(lengths.max())
#             .unsqueeze(0)
#             .expand(lengths.shape[0], -1)
#             .type_as(lengths)
#         )
#
#         mask = torch.gt(mask + 1, lengths.unsqueeze(1))
#         return mask


class LayerNorm(torch.nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = torch.nn.Parameter(torch.ones(channels))
        self.beta = torch.nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


# class TextEncoder(torch.nn.Module):
#     def __init__(self, *, num_tokens, inter_dim, num_heads, num_layers):
#         super().__init__()
#         self.embed = torch.nn.Embedding(num_tokens, inter_dim)
#         self.conformer = Conformer(
#             input_dim=inter_dim,
#             num_heads=num_heads,
#             ffn_dim=inter_dim * 2,
#             num_layers=num_layers,
#             depthwise_conv_kernel_size=15,
#             use_group_norm=True,
#         )
#
#     def forward(self, x):
#         lengths = torch.full(
#             [x.shape[0]], fill_value=x.shape[1], dtype=int, device=x.device
#         )
#         x = self.embed(x)
#         x, _ = self.conformer(x, lengths)
#         return x


class TextMelGenerator(torch.nn.Module):
    # def __init__(self, dim_in, dim, depth, max_seq_len, heads = 8, dim_head = 64, bucket_size = 64, n_hashes = 4, ff_chunks = 100, attn_chunks = 1, causal = False, weight_tie = False, lsh_dropout = 0., ff_dropout = 0., ff_mult = 4, ff_activation = None, ff_glu = False, post_attn_dropout = 0., layer_dropout = 0., random_rotations_per_head = False, use_scale_norm = False, use_rezero = False, use_full_attn = False, full_attn_thres = 0, reverse_thres = 0, num_mem_kv = 0, one_value_head = False, return_embeddings = False, weight_tie_embedding = False, fixed_position_emb = False, absolute_position_emb = False, axial_position_emb = False, axial_position_shape = None, n_local_attn_heads = 0, pkm_layers = tuple(), pkm_num_keys = 128, n_mels=80):
    def __init__(self, *, dim_in, hidden_dim, num_heads, num_layers):
        super().__init__()
        # self.max_seq_len = max_seq_len

        self.projection = torch.nn.Linear(dim_in, hidden_dim, bias=False)
        # reformer = Reformer(dim, depth, heads = heads, dim_head = dim_head, bucket_size = bucket_size, n_hashes = n_hashes, ff_chunks = ff_chunks, attn_chunks = attn_chunks, causal = causal, weight_tie = weight_tie, lsh_dropout = lsh_dropout, ff_mult = ff_mult, ff_activation = ff_activation, ff_glu = ff_glu, ff_dropout = ff_dropout, post_attn_dropout = 0., layer_dropout = layer_dropout, random_rotations_per_head = random_rotations_per_head, use_scale_norm = use_scale_norm, use_rezero = use_rezero, use_full_attn = use_full_attn, full_attn_thres = full_attn_thres, reverse_thres = reverse_thres, num_mem_kv = num_mem_kv, one_value_head = one_value_head, n_local_attn_heads = n_local_attn_heads, pkm_layers = pkm_layers, pkm_num_keys = pkm_num_keys)
        # self.reformer = Autopadder(reformer)
        # self.norm = torch.nn.LayerNorm(dim)
        # self.conv = torch.nn.Sequential(
        #     BasicConvNeXtBlock(dim, dim * 2),
        #     BasicConvNeXtBlock(dim, dim * 2),
        #     BasicConvNeXtBlock(dim, dim * 2),
        # )
        self.conformer = Conformer(
            input_dim=hidden_dim,
            num_heads=num_heads,
            ffn_dim=hidden_dim * 4,
            num_layers=num_layers,
            depthwise_conv_kernel_size=15,
            use_group_norm=True,
        )
        self.to_out = torch.nn.Linear(hidden_dim, dim_in, bias=False)

    def forward(self, x):
        lengths = torch.full(
            [x.shape[0]], fill_value=x.shape[2], dtype=int, device=x.device
        )
        x = rearrange(x, "b f t -> b t f")
        x = self.projection(x)
        x, _ = self.conformer(x, lengths)
        x = self.to_out(x)
        x = rearrange(x, "b t f -> b f t")
        return x


class TextMelClassifier(torch.nn.Module):
    def __init__(self, *, inter_dim, n_mels):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(inter_dim + n_mels, inter_dim, bias=False),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(inter_dim, inter_dim, bias=False),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(inter_dim, 1, bias=False),
        )

    def forward(self, corrupted, embedding):
        x = torch.cat([corrupted, embedding], dim=1)
        x = rearrange(x, "b c t -> b t c")
        x = self.net(x)
        x = rearrange(x, "b t c -> b c t")
        return x
