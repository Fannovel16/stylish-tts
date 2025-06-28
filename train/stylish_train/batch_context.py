import random
from typing import Optional

import torch
from torch.nn import functional as F
import torchaudio
from einops import rearrange, reduce
import train_context
from stylish_lib.config_loader import Config
from transformers import HubertModel
from utils import length_to_mask, log_norm, print_gpu_vram
import torch.nn as nn


class HubertModelWithFinalProj(HubertModel):
    def __init__(self, config):
        super().__init__(config)
        self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)


class AdaptiveHubert(nn.Module):
    def __init__(self, hubert: HubertModelWithFinalProj, model_sr: int, hubert_sr: int):
        super().__init__()
        self.hubert = hubert
        self.resample = torchaudio.transforms.Resample(model_sr, hubert_sr)

    def forward(self, wave, mel):
        wave = self.resample(wave)
        x = self.hubert(wave)["last_hidden_state"].transpose(-1, -2)
        x = torch.nn.functional.interpolate(
            x,
            size=mel.shape[-1],
            mode="linear",
            align_corners=True,
        ).transpose(-1, -2)
        return x


class BatchContext:
    def __init__(
        self,
        *,
        train: train_context.TrainContext,
        model,
    ):
        self.train: train_context.TrainContext = train
        self.config: Config = train.config
        # This is a subset containing only those models used this batch
        self.model = model
        hubert_config = self.train.model_config.hubert
        self.hubert = (
            AdaptiveHubert(
                HubertModelWithFinalProj.from_pretrained(hubert_config.model),
                self.train.model_config.sample_rate,
                hubert_config.sr,
            )
            .to(self.config.training.device)
            .eval()
        )

        self.pitch_prediction = None
        self.energy_prediction = None
        self.duration_prediction = None

    def acoustic_energy(self, mels: torch.Tensor):
        with torch.no_grad():
            energy = log_norm(mels.unsqueeze(1)).squeeze(1)
        return energy

    def calculate_pitch(self, batch, prediction=None):
        if prediction is None:
            prediction = batch.pitch
        return prediction

    def acoustic_prediction_single(self, batch, use_random_mono=True):
        acoustic_features, acoustic_styles = self.model.text_acoustic_extractor(
            batch.text, batch.text_length
        )
        print_gpu_vram("style extractor")
        energy = self.acoustic_energy(batch.mel)
        pitch = self.calculate_pitch(batch).detach()
        prediction = self.model.generator(
            acoustic_features @ batch.alignment,
            acoustic_styles @ batch.alignment,
            pitch,
            energy,
        )
        return prediction

    def textual_prediction_single(self, batch):
        acoustic_features, acoustic_styles = self.model.text_acoustic_extractor(
            batch.text, batch.text_length
        )
        duration_features, _ = self.model.text_duration_extractor(
            batch.text, batch.text_length
        )
        spectral_features, spectral_styles = self.model.text_spectral_extractor(
            batch.text, batch.text_length
        )
        self.duration_prediction = self.model.duration_predictor(
            duration_features,
        )
        self.pitch_prediction, self.energy_prediction = (
            self.model.pitch_energy_predictor(
                spectral_features.transpose(-1, -2) @ batch.alignment,
                spectral_styles @ batch.alignment,
            )
        )
        pitch = self.calculate_pitch(batch, self.pitch_prediction)
        prediction = self.model.generator(
            acoustic_features @ batch.alignment,
            acoustic_styles @ batch.alignment,
            pitch,
            self.energy_prediction,
        )
        print_gpu_vram("generator")
        return prediction

    def vc_prediction_single(self, batch):
        phones = self.hubert(batch.audio_gt, batch.mel)
        acoustic_features, acoustic_styles = self.model.hubert_acoustic_extractor(
            phones, batch.mel_length
        )
        acoustic_features = torch.nn.functional.interpolate(
            acoustic_features,
            scale_factor=2,
            mode="linear",
            align_corners=True,
        )
        acoustic_styles = torch.nn.functional.interpolate(
            acoustic_styles.transpose(-1, -2),
            scale_factor=2,
            mode="linear",
            align_corners=True,
        ).transpose(-1, -2)
        spectral_features, spectral_styles = self.model.hubert_spectral_extractor(
            phones, batch.mel_length
        )
        self.pitch_prediction, self.energy_prediction = (
            self.model.pitch_energy_predictor(
                spectral_features.transpose(-1, -2),
                spectral_styles,
            )
        )
        pitch = self.calculate_pitch(batch, self.pitch_prediction)
        print(
            acoustic_features.shape,
            acoustic_styles.shape,
            pitch.shape,
            self.energy_prediction.shape,
        )
        prediction = self.model.generator(
            acoustic_features,
            acoustic_styles,
            pitch,
            self.energy_prediction,
        )
        print_gpu_vram("generator")
        return prediction
