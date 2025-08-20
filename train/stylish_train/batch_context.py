import random
from typing import Optional

import torch
from torch.nn import functional as F
import torchaudio
from einops import rearrange, repeat
import train_context
from stylish_lib.config_loader import Config
from utils import length_to_mask, log_norm, print_gpu_vram, sequence_mask
from stylish_train.models.vevo_token_predictor import (
    DataCollatorWithPadding,
    phoneme_idx_to_token,
)
from transformers import T5ForConditionalGeneration
import evaluate


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
        self.byt5_data_collator = DataCollatorWithPadding(
            self.train.vevo_token_predictor_tokenizer, self.train.config.training.device
        )
        self.cer_metric = evaluate.load("cer")

        self.pitch_prediction = None
        self.energy_prediction = None
        self.duration_prediction = None
        self.phones = None
        self.phones_prediction = None
        self.cmt_loss = None
        self.logits_gt = None
        self.logits_prediction = None
        self.byt5_ce_loss = None
        self.byt5_cer_loss = None

    def acoustic_energy(self, mels: torch.Tensor):
        with torch.no_grad():
            energy = log_norm(mels.unsqueeze(1)).squeeze(1)
        return energy

    def acoustic_prediction_single(self, batch, use_random_mono=True):
        phones, _ = self.text_to_hubert(batch)
        # phones = (phones.transpose(-1, -2) @ batch.alignment).transpose(-1, -2)
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
        phones, _ = self.text_to_hubert(batch)
        # phones = (phones.transpose(-1, -2) @ batch.alignment).transpose(-1, -2)
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
        duration_features, _ = self.model.text_duration_extractor(
            batch.text, batch.text_length
        )
        self.duration_prediction = self.model.duration_predictor(
            duration_features,
        )
        prediction = self.model.generator(
            acoustic_features,
            acoustic_styles,
            self.pitch_prediction,
            self.energy_prediction,
        )
        print_gpu_vram("generator")
        return prediction

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
        x, indices, cmt_loss = self.model.hubert_quantizer(hubert_embedding)
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

    def extract_phones_from_audio(self, batch):
        """phones = self.train.hubert(
            batch.audio_gt,
            batch.alignment.shape[-1],
        )
        pooled_phones = (
            batch.alignment.float()
            @ phones
            / (batch.alignment.sum(-1, keepdim=True) + 1e-8)
        )
        return pooled_phones.detach()"""
        with torch.no_grad():
            return self.train.hubert(
                batch.audio_gt,
                batch.alignment.shape[-1],
            )

    def pre_hubert_quantizer(self, batch):
        self.phones = self.extract_phones_from_audio(batch)
        self.phones_prediction, codebook_indices, self.cmt_loss = self.quantize_hubert(
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

    """def text_to_hubert(self, batch):
        logits = self.model.hubert_feature_synthesizer(
            batch.text, batch.text_length, batch.mel_length // 2, batch.alignment
        )
        indices = logits.detach().argmax(dim=-1)
        phones = self.model.hubert_quantizer.get_output_from_indices(indices)
        return phones, logits"""

    def text_to_hubert(self, batch):
        phones = self.model.hubert_feature_synthesizer(
            batch.text, batch.text_length, batch.alignment
        )
        return phones, None

    """def compute_feature_synthesizer_loss(self, batch):
        mel_mask = sequence_mask(batch.mel_length // 2, batch.alignment.shape[2])
        mask = repeat(
            mel_mask,
            "b t -> (b h) t",
            h=self.train.model_config.hubert_quantizer.num_quantizers,
        )

        loss = F.cross_entropy(
            self.logits_prediction, self.logits_gt, reduction="none"
        )  # (B*H, T)

        masked_loss = loss * mask.float()
        return masked_loss.sum() / mask.sum()"""

    def compute_feature_synthesizer_loss(self, batch):
        return F.cross_entropy(
            self.logits_prediction,
            self.logits_gt,
        )  # (B*H, T)

    """def pre_feature_synthesizer(self, batch):
        self.phones = self.extract_phones_from_audio(batch)
        with torch.no_grad():
            self.model.hubert_quantizer.eval()
            _, codebook_indices, _ = self.quantize_hubert(batch, self.phones)
            self.logits_gt = rearrange(codebook_indices, "b t h -> (b h) t")
        self.phones_prediction, self.logits_prediction = self.text_to_hubert(batch)
        self.logits_prediction = rearrange(
            self.logits_prediction,
            "b t h c -> (b h) c t",
        )"""

    def pre_feature_synthesizer(self, batch):
        print(batch.text.shape)
        self.phones, _ = self.extract_phones_from_audio(batch)
        self.phones_prediction = self.model.hubert_feature_synthesizer(
            batch.text, batch.text_length, batch.alignment
        )

    def compute_cer(self, pred_ids, labels_ids):
        tokenizer = self.train.vevo_token_predictor_tokenizer

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = tokenizer.pad_token_id
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
        cer = self.cer_metric.compute(predictions=pred_str, references=label_str)["cer"]
        return cer

    def pre_vevo_token_predictor(self, batch, training=True):
        def duration_reduction_func(token_seq, n_gram=1):
            """
            Args:
                token_seq: (T,)
            Returns:
                reduced_token_seq: (T')
                reduced_token_seq_len: T'
            """
            n_gram_seq = token_seq.unfold(0, n_gram, 1)
            mask = torch.all(n_gram_seq[1:] != n_gram_seq[:-1], dim=1)
            reduced_token_seq = torch.cat(
                (n_gram_seq[0, :n_gram], n_gram_seq[1:, -1][mask])
            )
            return reduced_token_seq, len(reduced_token_seq)

        _, all_codes = self.extract_phones_from_audio(batch)
        all_codes = [duration_reduction_func(codes) for codes in all_codes]
        byt5_batch = self.byt5_data_collator(
            [
                {
                    "input_ids": grapheme,
                    "labels": "".join([phoneme_idx_to_token(code) for code in codes]),
                }
                for grapheme, codes in zip(batch.grapheme, all_codes)
            ]
        )
        self.byt5_ce_loss = self.model.vevo_token_predictor(**byt5_batch).loss
        if not training:
            pred_ids = self.model.vevo_token_predictor.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=250,
            )
            self.byt5_cer_loss = self.compute_cer(pred_ids, byt5_batch["labels"])
