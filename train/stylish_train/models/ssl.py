import torch
import torch.nn as nn
import torchaudio
from transformers import HubertModel
from .campplus.DTDNN import CAMPPlus, CAMPPLUS_PRETRAINED_MODEL
from safetensors.torch import load_file
from einops import rearrange


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
    def __init__(self, model_sr: int):
        super().__init__()
        self.model = CAMPPlus(feat_dim=80, embedding_size=192)
        self.model.load_state_dict(load_file(CAMPPLUS_PRETRAINED_MODEL))
        self.resample = torchaudio.transforms.Resample(model_sr, 16000)

    def forward(self, wave):
        wave = self.resample(wave)
        feat_list = []
        for bib in range(wave.size(0)):
            feat = torchaudio.compliance.kaldi.fbank(
                wave[bib : bib + 1, :],
                num_mel_bins=80,
                dither=0,
                sample_frequency=16000,
            )
            feat = feat - feat.mean(dim=0, keepdim=True)
            feat_list.append(feat)
        feat = torch.stack(feat_list, dim=0)
        spk_emb = self.model(feat)
        spk_emb = torch.nn.functional.normalize(
            spk_emb, p=2, dim=1
        )  # As suggested by NANSY (https://arxiv.org/pdf/2110.14513)
        return spk_emb
