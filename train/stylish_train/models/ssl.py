import torch
import torch.nn as nn
import torchaudio
from transformers import HubertModel
from einops import rearrange
import wespeaker
from wespeaker.models.samresnet import SimAM_ResNet34_ASP, SimAM_ResNet100_ASP


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
        self.model = wespeaker.load_model("vblinkp")
        self.model.set_device(device)
        self.device = device
        # Remove the final projection layer for (theoritically) richer infomation for style encoding
        if type(self.model.model) not in [SimAM_ResNet34_ASP, SimAM_ResNet100_ASP]:
            raise NotImplementedError(
                "Any model arch rather than SimAM-ResNet are not supported."
            )
        self.model.model.bottleneck = nn.Identity()
        self.out_dim = self.model.model.pooling.out_dim
        self.global_sr = model_sr
        self.max_half = 16000 * 1

    def forward(self, wave):
        spk_embs = []
        wave = wave.cpu()
        middle = wave.shape[1] // 2
        for i in range(wave.shape[0]):
            spk_emb = self.model.extract_embedding_from_pcm(
                wave[i : i + 1, int(middle - self.max_half) : middle + self.max_half],
                self.global_sr,
            )
            spk_embs.append(spk_emb)
        return torch.stack(spk_embs, 0).to(self.device)
