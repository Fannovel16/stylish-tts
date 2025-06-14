import random
from typing import Optional

import torch
from torch.nn import functional as F
import torchaudio
from einops import rearrange, reduce
import train_context
from stylish_lib.config_loader import Config
from utils import length_to_mask, log_norm, print_gpu_vram


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
        pitch_features, pitch_styles = self.model.text_pitch_extractor(
            batch.text, batch.text_length
        )
        energy_features, energy_styles =  self.model.text_energy_extractor(
            batch.text, batch.text_length
        )
        self.duration_prediction = self.model.duration_predictor(
            duration_features,
        )
        self.pitch_prediction = self.model.pitch_predictor(
            pitch_features.transpose(-1, -2) @ batch.alignment,
            pitch_styles @ batch.alignment,
        )
        
        self.energy_prediction = self.model.energy_predictor(
            energy_features.transpose(-1, -2) @ batch.alignment,
            energy_styles @ batch.alignment,
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
