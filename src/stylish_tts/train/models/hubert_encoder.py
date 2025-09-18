import torch
from .text_encoder import Encoder
from einops import repeat
from stylish_tts.train.utils import sequence_mask


class HubertEncoder(torch.nn.Module):
    def __init__(self, model_config, input_cond_dim=None):
        super().__init__()
        self.phone_emb = torch.nn.Conv1d(
            model_config.hubert.hidden_dim, model_config.inter_dim, 1
        )
        if input_cond_dim:
            self.cond_proj = torch.nn.Linear(input_cond_dim, model_config.style_dim)
            hidden_dim = model_config.inter_dim + model_config.style_dim
        else:
            hidden_dim = model_config.inter_dim
        encoder_config = model_config.text_encoder
        self.encoder = Encoder(
            hidden_dim,
            encoder_config.filter_channels,
            encoder_config.heads,
            encoder_config.layers,
            encoder_config.kernel_size,
            encoder_config.dropout,
        )
        if hidden_dim != model_config.inter_dim:
            self.final_proj = torch.nn.Conv1d(
                model_config.inter_dim + model_config.style_dim,
                model_config.inter_dim,
                1,
            )
        else:
            self.final_proj = torch.nn.Identity()

    def forward(self, phones, phone_lengths, cond=None):
        phones = self.phone_emb(phones)
        if cond:
            cond = self.cond_proj(cond)
            phones = torch.cat(
                [phones, repeat(cond, "b c -> b c t", t=phones.shape[-1])], dim=1
            )
        phone_mask = (
            sequence_mask(phone_lengths, phones.size(2)).unsqueeze(1).to(phones.dtype)
        )
        phones = self.final_proj(self.encoder(phones, phone_mask))
        return phones
