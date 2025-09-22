import torch
import torch.nn as nn
import torchaudio
from transformers import HubertModel
from einops import rearrange
import wespeaker
import random


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


class SpeakerEmbeddingModel(nn.Module):
    def __init__(self, model_sr: int, device: str):
        super().__init__()
        self.model = wespeaker.load_model("campplus")
        self.model.model.xvector.dense = nn.Identity()
        self.model.set_device(device)
        self.device = device
        self.max_half = 16000 * 2
        self.resample = torchaudio.transforms.Resample(
            model_sr, self.model.resample_rate
        )

    def forward(self, wave):
        feats = []
        wave = self.resample(wave)
        num_batch, num_frames = wave.shape
        middle = (
            random.randrange(0, num_frames - self.max_half)
            if num_frames > 2 * self.max_half
            else 0
        )
        start, end = max(middle - self.max_half, 0), middle + self.max_half + 1
        for i in range(num_batch):
            _feats = self.model.compute_fbank(
                wave[i : i + 1, start:end],
                sample_rate=self.model.resample_rate,
                cmn=True,
            )
            feats.append(_feats)
        feats = torch.stack(feats, 0).to(self.device)
        with torch.no_grad():
            outputs = self.model.model(feats)
        return outputs
