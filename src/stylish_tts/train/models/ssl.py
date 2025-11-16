import torch
import torch.nn as nn
import torchaudio
from transformers import HubertModel
from einops import rearrange
import random
from huggingface_hub import snapshot_download
import os
import yaml
from .samresnet import SimAM_ResNet34_ASP
import logging
import torchaudio.compliance.kaldi as kaldi


class HubertModelWithFinalProj(HubertModel):
    def __init__(self, config):
        super().__init__(config)
        self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)


class AdaptiveHubert(nn.Module):
    def __init__(self, hubert_path: str, model_sr: int, hubert_sr: int):
        super().__init__()
        self.model = HubertModelWithFinalProj.from_pretrained(hubert_path)
        self.sr = hubert_sr
        self.resample = torchaudio.transforms.Resample(model_sr, hubert_sr)

    def forward(self, wave, time_dim):
        wave = self.resample(wave)
        x = self.model(wave)["last_hidden_state"]
        x = torch.nn.functional.interpolate(
            x.transpose(-1, -2),
            size=time_dim,
            mode="nearest",
        ).transpose(-1, -2)
        return rearrange(x, "b t c -> b c t")

def load_checkpoint(model: torch.nn.Module, path: str):
    checkpoint = torch.load(path, map_location='cpu')
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint,
                                                          strict=False)
    for key in missing_keys:
        logging.warning('missing tensor: {}'.format(key))
    for key in unexpected_keys:
        logging.warning('unexpected tensor: {}'.format(key))

# https://github.com/wenet-e2e/wespeaker/blob/67f0f4a8d472e6e2203d7baca38daba818af17f3/wespeaker/cli/speaker.py#L306
def load_model_pt(model_name_or_path: str):
    """There are the following files in the `model_dir`:
       - config.yaml: the model config file
       - avg_model.pt: the pytorch model file
    """
    model_dir = snapshot_download(model_name_or_path)
    required_files = ['config.yaml', 'avg_model.pt']
    for file in required_files:
        if not os.path.exists(os.path.join(model_dir, file)):
            raise FileNotFoundError(f"{file} not found in {model_dir}")
    # Read config file
    with open(os.path.join(model_dir, 'config.yaml'), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    model = SimAM_ResNet34_ASP(**config['model_args'])
    load_checkpoint(model, os.path.join(model_dir, 'avg_model.pt'))
    model.eval()
    return model

class SpeakerEmbeddingModel(nn.Module):
    def __init__(self, model_sr: int):
        super().__init__()
        self.model = load_model_pt("gaunernst/wespeaker-voxblink2-samresnet34-ft")
        self.resample_rate = 16000
        self.window_type = 'hamming'
        self.resample = torchaudio.transforms.Resample(
            model_sr, self.resample_rate
        )
    
    def compute_fbank(self,
                      wavform,
                      sample_rate=16000,
                      num_mel_bins=80,
                      frame_length=25,
                      frame_shift=10,
                      cmn=True):
        feat = kaldi.fbank(wavform,
                           num_mel_bins=num_mel_bins,
                           frame_length=frame_length,
                           frame_shift=frame_shift,
                           sample_frequency=sample_rate,
                           window_type=self.window_type)
        if cmn:
            feat = feat - torch.mean(feat, 0)
        return feat

    def forward(self, wave):
        device = next(self.parameters()).device
        wave = self.resample(wave)
        num_batch, _ = wave.shape
        feats = []
        for i in range(num_batch):
            _feats = self.compute_fbank(
                wave[i : i + 1, :],
                sample_rate=self.resample_rate,
                cmn=True,
            )
            feats.append(_feats)
        feats = torch.stack(feats, 0).to(device)
        return self.model(feats)
