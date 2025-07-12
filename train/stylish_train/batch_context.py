import random
from typing import Optional

import torch
from torch.nn import functional as F
import torchaudio
from einops import rearrange, reduce
import train_context
from stylish_lib.config_loader import Config
from utils import length_to_mask, log_norm, print_gpu_vram, sequence_mask


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
        self.phones = None
        self.phones_prediction = None
        self.cmt_loss = None
        self.logits_gt = None
        self.logits_prediction = None

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
        phones = self.text_to_hubert(batch)
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
        phones = self.text_to_hubert(batch)
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
        self.phones = self.train.hubert(batch.audio_gt, batch.alignment.shape[-1])
        self.phones_prediction, _, _ = self.model.text_hubert_distiller(
            batch.text, batch.text_length
        )
        self.phones_prediction = self.phones_prediction @ batch.alignment
        self.phones_prediction = self.phones_prediction.transpose(-1, -2)

        acoustic_features, acoustic_styles = self.model.hubert_acoustic_extractor(
            self.phones, batch.mel_length // 2
        )
        duration_features, _ = self.model.text_duration_extractor(
            batch.text, batch.text_length
        )
        spectral_features, spectral_styles = self.model.hubert_spectral_extractor(
            self.phones_prediction, batch.mel_length // 2
        )

        self.duration_prediction = self.model.duration_predictor(
            duration_features,
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

    def quantize_hubert(self, batch, hubert_embedding):
        mel_mask = sequence_mask(batch.mel_length, batch.alignment.shape[2])
        x, indices, cmt_loss = self.model.hubert_quantizer(
            hubert_embedding, mask=mel_mask
        )
        return x, indices, cmt_loss

    def track_codebook_metrics(self, codebook_indices):
        """
        Tracks codebook usage stats.
        """
        codebook_size = self.train.model_config.hubert_quantizer.codebook_size

        codebook_indices = codebook_indices.cpu()
        flat_idx = codebook_indices.view(-1)
        total = flat_idx.numel()

        # Codebook usage
        unique_codes, counts = torch.unique(flat_idx, return_counts=True)
        num_used = unique_codes.numel()
        usage_ratio = num_used / codebook_size

        # Entropy
        probs = counts.float() / total
        entropy = -torch.sum(probs * torch.log2(probs + 1e-8))
        max_entropy = torch.log2(torch.tensor(codebook_size, dtype=torch.float))
        entropy_ratio = entropy / max_entropy

        # Dead entries
        dead_codes = codebook_size - num_used

        print(f"\n--- Codebook Stats ---")
        print(f"Used Codes: {num_used}/{codebook_size} ({usage_ratio:.2%})")
        print(f"Dead Codes: {dead_codes}")
        print(
            f"Entropy: {entropy.item():.4f} / {max_entropy.item():.4f} ({entropy_ratio.item():.2%})"
        )
        print("-" * 40)

        # Top-used codes
        topk = min(10, len(counts))
        top_counts, top_ids = counts.topk(topk)
        print("Top Used Codes:")
        for i in range(topk):
            print(
                f"  Code {unique_codes[top_ids[i]].item():4d}: {top_counts[i].item()}"
            )

    def pre_hubert_quantizer(self, batch):
        self.phones = self.train.hubert(
            batch.audio_gt,
            batch.alignment.shape[-1],
        )
        self.phones_prediction, _, self.cmt_loss = self.quantize_hubert(
            batch, self.phones
        )
        global_step = self.train.manifest.current_step
        print_every = self.train.config.training.log_interval
        in_val = not torch.is_grad_enabled()
        if in_val or (global_step >= print_every and global_step % print_every == 0):
            with torch.no_grad():
                self.model.hubert_quantizer.eval()
                self.phones_prediction, codebook_indices, _ = self.quantize_hubert(
                    batch, self.phones
                )
            if not in_val:
                self.track_codebook_metrics(codebook_indices)
            self.model.hubert_quantizer.train()

    def predict_vq_logits(self, batch):
        bert_encoding = self.model.bert(
            batch.text,
            attention_mask=sequence_mask(batch.text_length, batch.text.shape[1]).long(),
        )
        return self.model.vq_indexer(
            bert_encoding, batch.text_length, batch.mel_length // 2, batch.alignment
        )

    def pre_vq_indexer(self, batch):
        self.phones = self.train.hubert(
            batch.audio_gt,
            batch.alignment.shape[-1],
        )
        with torch.no_grad():
            self.model.hubert_quantizer.eval()
            _, codebook_indices, _ = self.quantize_hubert(batch, self.phones)
            self.logits_gt = rearrange(codebook_indices, "b t h -> (b h) t")
        self.logits_prediction = rearrange(
            self.predict_vq_logits(batch),
            "b t h c -> (b h) c t",
        )

    def text_to_hubert(self, batch):
        logits = self.predict_vq_logits(batch)
        indices = logits.argmax(dim=-1)
        return self.model.hubert_quantizer.get_output_from_indices(indices)
