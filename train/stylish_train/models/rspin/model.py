from typing import List, Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F

from .wavlm_config import wavlm_base_config
from .wavlm import WavLM, WavLMConfig


class RSpinWavlm(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.encoder_cfg = WavLMConfig(wavlm_base_config)
        self.encoder = WavLM(self.encoder_cfg)

        self.pred_head = nn.Linear(
            self.encoder_cfg.encoder_embed_dim, cfg["codebook_dim"]
        )
        self.codebook = nn.Linear(cfg["codebook_dim"], cfg["codebook_size"], bias=False)

    def forward(
        self,
        wavs: torch.Tensor,
        padding_mask: Optional[torch.BoolTensor] = None,
        mask: bool = False,
        get_code: bool = False,
    ) -> Tuple[List[torch.Tensor], torch.BoolTensor, Union[None, torch.Tensor]]:
        """Forward pass

        Args:
            wavs (torch.Tensor): shape (batch_size, seq_len)
            padding_mask (torch.BoolTensor): shape (batch_size, seq_len)
            mask (bool, optional): Masking. Defaults to False.
            get_code (bool, optional): Get Spin codes. Defaults to False.

        Returns:
            Tuple[List[torch.Tensor], torch.BoolTensor, Union[None, torch.Tensor]]:
                List[torch.Tensor]: List of features
                torch.BoolTensor: Padding mask
                Union[None, torch.Tensor]: Spin codes
        """

        feats, padding_mask = self.encoder.extract_features(
            wavs, padding_mask, mask=mask, ret_conv=True, ret_layer_results=True
        )
        feats = [feats[0]] + [f[0].transpose(0, 1) for f in feats[1]]
        if get_code:
            z = F.normalize(self.pred_head(feats[-1]), dim=-1)
            code = self.codebook(z).argmax(dim=-1)
            return feats, padding_mask, code

        return feats, padding_mask, None

    @classmethod
    def load_from_checkpoint(cls, path: str):
        ckpt = torch.load(path)
        cfg = ckpt["cfg"]
        state_dict = ckpt["state_dict"]
        model = cls(cfg)
        model.load_state_dict(state_dict)
        return model
