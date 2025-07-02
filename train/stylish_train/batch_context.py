import random
from typing import Optional

import torch
from torch.nn import functional as F
import torchaudio
from einops import rearrange, reduce
import train_context
from stylish_lib.config_loader import Config
from utils import length_to_mask, log_norm, print_gpu_vram
from models.feature_extractor import TextFeatureExtractor, HubertFeatureExtractor

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Any


class FeatureDistilLoss(nn.Module):
    """
    A comprehensive manager for layer-to-layer FEATURE-ONLY knowledge distillation.

    This class encapsulates the distillation of hidden states (feature maps):
    1. Attaches forward hooks to the output of each encoder block in the models.
    2. Manages the lifecycle of these hooks.
    3. Computes the layer-wise distillation loss for hidden states.
    4. Handles time-dimension alignment between student and teacher.
    """

    def __init__(
        self,
        student_model: TextFeatureExtractor,
        teacher_model: HubertFeatureExtractor,
        feature_loss_type: str = "l1",
    ):
        """
        Initializes the DistillationManager.

        Args:
            student_model (nn.Module): The student model to be trained.
            teacher_model (nn.Module): The teacher model (will be frozen).
            alpha (float): Weight for the hidden state (feature) distillation loss.
            feature_loss_type (str): Loss function for features ('mse' or 'l1').
        """
        super().__init__()

        self.student_model = student_model
        self.teacher_model = teacher_model

        self.feature_loss_fn = self._get_loss_fn(feature_loss_type)

        # --- Hook Setup (Simplified) ---
        self.student_handles = []
        self.teacher_handles = []
        self._student_hidden_states: List[torch.Tensor] = []
        self._teacher_hidden_states: List[torch.Tensor] = []
        self._student_features: List[torch.Tensor] = []
        self._teacher_features: List[torch.Tensor] = []

        self._attach_hooks()

    def _get_loss_fn(self, loss_type: str):
        if loss_type.lower() == "mse":
            return F.mse_loss
        elif loss_type.lower() == "l1":
            return F.l1_loss
        else:
            raise ValueError(
                f"Unsupported loss type: {loss_type}. Choose 'mse' or 'l1'."
            )

    def _attach_hooks(self):
        """Finds the encoder submodules and attaches hooks to the block outputs."""
        try:
            student_encoder = self.student_model.text_encoder.encoder
            teacher_encoder = self.teacher_model.text_encoder.encoder
        except AttributeError:
            raise AttributeError("Could not find '.text_encoder.encoder' submodule.")

        # Hook the final LayerNorm of each block to get hidden states
        for norm_layer in student_encoder.norm_layers_2:
            handle = norm_layer.register_forward_hook(self._save_student_hidden_state)
            self.student_handles.append(handle)

        for norm_layer in teacher_encoder.norm_layers_2:
            handle = norm_layer.register_forward_hook(self._save_teacher_hidden_state)
            self.teacher_handles.append(handle)

        # Hook the the encoder itself to get final feature output
        self.student_model.register_forward_hook(self._save_student_features)
        self.teacher_model.register_forward_hook(self._save_teacher_features)

    # --- Hook callback functions ---
    def _save_student_hidden_state(self, module, inp, out):
        self._student_hidden_states.append(out)

    def _save_teacher_hidden_state(self, module, inp, out):
        self._teacher_hidden_states.append(out)

    def _save_student_features(self, module, inp, out):
        self._student_features = out

    def _save_teacher_features(self, module, inp, out):
        self._teacher_features = out

    def _clear_buffers(self):
        """Clears the stored outputs from the previous forward pass."""
        self._student_hidden_states.clear()
        self._teacher_hidden_states.clear()
        self._student_features = []
        self._teacher_features = []

    def remove_hooks(self):
        """Removes all hooks from the models. Call this after training."""
        for handle in self.student_handles + self.teacher_handles:
            handle.remove()

    def forward(self, alignment: torch.Tensor):
        """
        Computes the distillation loss based on the captured hidden states.

        Args:
            alignment_matrix (torch.Tensor): The matrix to align student's time dimension.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
            - total_distill_loss: The final distillation loss (which is just the feature loss).
            - feature_loss_for_logging: The feature loss component (for logging).
        """
        """num_layers = len(self._student_hidden_states)
        if num_layers == 0:
            # This can happen on the first pass if hooks haven't run yet.
            return torch.tensor(0.0, device=alignment.device)

        hidden_states_loss = 0.0
        for s_h, t_h in zip(self._student_hidden_states, self._teacher_hidden_states):
            t_h = t_h.detach()
            hidden_states_loss += self.feature_loss_fn(s_h @ alignment, t_h)
        hidden_states_loss = hidden_states_loss / num_layers"""

        features_loss = 0.0
        for s_feat, t_feat in zip(self._student_features, self._teacher_features):
            t_feat = t_feat.detach()
            if s_feat.shape[1] == alignment.shape[1]:
                aligned_s_feat = s_feat.transpose(-1, -2) @ alignment
                try:
                    features_loss += self.feature_loss_fn(
                        aligned_s_feat, t_feat.transpose(-1, -2)
                    )
                except:
                    features_loss += self.feature_loss_fn(aligned_s_feat, t_feat)
            else:
                aligned_s_feat = s_feat @ alignment
                features_loss += self.feature_loss_fn(aligned_s_feat, t_feat)

        self._clear_buffers()
        self.remove_hooks()
        return features_loss


# def quantize_f0(f0, f0_bin=256, f0_min=50.0, f0_max=1100.0):
#     """
#     Quantize raw F0 into mel bins.
#     f0: Tensor (N,) or (B, T)
#     Returns: Tensor of same shape with int bins in [0, f0_bin - 1]
#     """

#     # Mel scale range
#     f0_mel_min = 1127.0 * torch.log1p(torch.tensor(f0_min / 700.0))
#     f0_mel_max = 1127.0 * torch.log1p(torch.tensor(f0_max / 700.0))

#     # Initialize output
#     f0 = F.interpolate(
#         rearrange(f0, "n t -> n 1 t"), scale_factor=0.5, mode="linear"
#     ).squeeze(1)
#     mel_bins = torch.zeros_like(f0, dtype=torch.float32)

#     # Voiced mask
#     voiced = f0 > 0

#     # Convert to mel scale
#     f0_voiced = f0[voiced]
#     f0_mel = 1127.0 * torch.log1p(f0_voiced / 700.0)

#     # Normalize and scale to [1, f0_bin - 1]
#     f0_mel = (f0_mel - f0_mel_min) * (f0_bin - 2) / (f0_mel_max - f0_mel_min) + 1
#     f0_mel = torch.clamp(f0_mel, 1, f0_bin - 1)

#     # Store result
#     mel_bins[voiced] = torch.round(f0_mel)

#     return mel_bins.to(dtype=torch.int)


class BatchContext:
    def __init__(
        self,
        *,
        train: train_context.TrainContext,
        model,
        distil=False,
    ):
        self.train: train_context.TrainContext = train
        self.config: Config = train.config
        # This is a subset containing only those models used this batch
        self.model = model

        self.pitch_prediction = None
        self.energy_prediction = None
        self.duration_prediction = None

        if distil:
            self.acoustic_feature_loss = FeatureDistilLoss(
                self.model.text_acoustic_extractor, self.model.hubert_acoustic_extractor
            )
            self.spectral_feature_loss = FeatureDistilLoss(
                self.model.text_spectral_extractor, self.model.hubert_spectral_extractor
            )

    def acoustic_energy(self, mels: torch.Tensor):
        with torch.no_grad():
            energy = log_norm(mels.unsqueeze(1)).squeeze(1)
        return energy

    """def acoustic_prediction_single(self, batch, use_random_mono=True):
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
        return prediction"""

    def acoustic_prediction_single(self, batch, use_random_mono=True):
        phones = self.train.hubert(batch.audio_gt, batch.alignment.shape[-1])
        acoustic_features, acoustic_styles = self.model.hubert_acoustic_extractor(
            phones, batch.mel_length // 2
        )
        energy = self.acoustic_energy(batch.mel)
        pitch = batch.pitch
        prediction = self.model.generator(
            acoustic_features,
            acoustic_styles,
            pitch,
            energy,
        )
        print_gpu_vram("generator")
        return prediction

    def spectral_prediction_single(self, batch, use_random_mono=True):
        phones = self.train.hubert(batch.audio_gt, batch.alignment.shape[-1])
        acoustic_features, acoustic_styles = self.model.hubert_acoustic_extractor(
            phones, batch.mel_length // 2
        )
        spectral_features, spectral_styles = self.model.hubert_spectral_extractor(
            phones, batch.mel_length // 2
        )
        self.pitch_prediction, self.energy_prediction = (
            self.model.pitch_energy_predictor(
                spectral_features.transpose(-1, -2),
                spectral_styles,
            )
        )
        prediction = self.model.generator(
            acoustic_features,
            acoustic_styles,
            self.pitch_prediction,
            self.energy_prediction,
        )
        print_gpu_vram("generator")
        return prediction

    """def textual_prediction_single(self, batch):
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
        return prediction"""

    """ def textual_prediction_single(self, batch):
        # Invoke to only get hidden features
        phones = self.train.hubert(batch.audio_gt, batch.alignment.shape[-1])
        self.model.hubert_acoustic_extractor(phones, batch.mel_length // 2)
        self.model.hubert_spectral_extractor(phones, batch.mel_length // 2)

        # Textual
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
        prediction = self.model.generator(
            acoustic_features @ batch.alignment,
            acoustic_styles @ batch.alignment,
            self.pitch_prediction,
            self.energy_prediction,
        )
        print_gpu_vram("generator")
        return prediction """

    def textual_prediction_single(self, batch):
        # Invoke to only get hidden features
        phones = self.train.hubert(batch.audio_gt, batch.alignment.shape[-1])
        self.model.hubert_acoustic_extractor(phones, batch.mel_length // 2)
        self.model.hubert_spectral_extractor(phones, batch.mel_length // 2)

        # Textual
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
        prediction = self.model.generator(
            acoustic_features @ batch.alignment,
            acoustic_styles @ batch.alignment,
            self.pitch_prediction,
            self.energy_prediction,
        )
        print_gpu_vram("generator")
        return prediction
