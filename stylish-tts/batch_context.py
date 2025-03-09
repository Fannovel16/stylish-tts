import random

import torch
import torchaudio

from monotonic_align import mask_from_lens
from train_context import TrainContext
from config_loader import Config
from utils import length_to_mask, log_norm, maximum_path


class BatchContext:
    def __init__(
        self,
        train: TrainContext,
        model,
        text_lengths: torch.Tensor = None,
    ):
        self.train: TrainContext = train
        self.config: Config = train.config
        # This is a subset containing only those models used this batch
        self.model = model

        self.text_mask = None
        if text_lengths is not None:
            self.text_mask: torch.Tensor = length_to_mask(text_lengths).to(
                self.config.training.device
            )
        self.duration_results = None
        self.resample = torchaudio.transforms.Resample(
            self.train.model_config.preprocess.sample_rate, 16000
        ).to(self.config.training.device)
        self.to_mel = torchaudio.transforms.MelSpectrogram(
            n_mels=80, n_fft=2048, win_length=1200, hop_length=300, sample_rate=24000
        ).to(self.config.training.device)
        self.pitch_prediction = None
        self.energy_prediction = None
        self.duration_prediction = None

    def text_encoding(self, texts: torch.Tensor, text_lengths: torch.Tensor):
        return self.model.text_encoder(texts, text_lengths, self.text_mask)

    def bert_encoding(self, texts: torch.Tensor):
        mask = (~self.text_mask).int()
        bert_encoding = self.model.bert(texts, attention_mask=mask)
        text_encoding = self.model.bert_encoder(bert_encoding)
        return text_encoding.transpose(-1, -2)

    def acoustic_duration(
        self,
        mels: torch.Tensor,
        mel_lengths: torch.Tensor,
        texts: torch.Tensor,
        text_lengths: torch.Tensor,
        apply_attention_mask: bool = False,
        use_random_choice: bool = False,
    ) -> torch.Tensor:
        """
        Computes the duration used for training using a text aligner on
        the combined ground truth audio and text.
        Returns:
          - duration: Duration attention vector
        """
        # Create masks.
        mask = length_to_mask(mel_lengths // (2**self.train.n_down)).to(
            self.config.training.device
        )

        # --- Text Aligner Forward Pass ---
        s2s_pred, s2s_attn = self.model.text_aligner(mels, mask, texts)
        # Remove the last token to make the shape match texts
        s2s_attn = s2s_attn.transpose(-1, -2)
        s2s_attn = s2s_attn[..., 1:]
        s2s_attn = s2s_attn.transpose(-1, -2)

        # Optionally apply extra attention mask.
        if apply_attention_mask:
            with torch.no_grad():
                attn_mask = (
                    (~mask)
                    .unsqueeze(-1)
                    .expand(mask.shape[0], mask.shape[1], self.text_mask.shape[-1])
                    .float()
                    .transpose(-1, -2)
                )
                attn_mask = (
                    attn_mask
                    * (~self.text_mask)
                    .unsqueeze(-1)
                    .expand(
                        self.text_mask.shape[0], self.text_mask.shape[1], mask.shape[-1]
                    )
                    .float()
                )
                attn_mask = attn_mask < 1
            s2s_attn.masked_fill_(attn_mask, 0.0)

        # --- Monotonic Attention Path ---
        with torch.no_grad():
            mask_ST = mask_from_lens(
                s2s_attn, text_lengths, mel_lengths // (2**self.train.n_down)
            )
            s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

        # --- Text Encoder Forward Pass ---
        if use_random_choice and bool(random.getrandbits(1)):
            duration = s2s_attn
        else:
            duration = s2s_attn_mono

        self.attention = s2s_attn[0]
        self.duration_results = (s2s_attn, s2s_attn_mono)
        self.s2s_pred = s2s_pred
        return duration

    def get_attention(self):
        return self.attention

    def acoustic_pitch(self, mels: torch.Tensor):
        pitch, _, _ = self.model.pitch_extractor(mels.unsqueeze(1))
        return pitch

    # def acoustic_pitch(self, audio_gt: torch.Tensor):
    #     with torch.no_grad():
    #
    #         c = self.config
    #         fmin = 50
    #         fmax = 550
    #         model = "full"
    #         audio = self.resample(audio_gt).to(c.training.device)
    #         pitch = torchcrepe.predict(
    #             audio,
    #             16000,
    #             200,  # c.preprocess.hop_length,
    #             fmin,
    #             fmax,
    #             model,
    #             batch_size=2048,
    #             device=c.training.device,
    #         )[:, :-1]
    #     return pitch

    def acoustic_energy(self, mels: torch.Tensor):
        with torch.no_grad():
            energy = log_norm(mels.unsqueeze(1)).squeeze(1)
        return energy

    # def acoustic_style_embedding(self, mels: torch.Tensor):
    #     return self.model.style_encoder(mels.unsqueeze(1))

    def style_embedding(self, sentence_embedding: torch.Tensor):
        return self.model.style_encoder(sentence_embedding)

    def prosodic_style_embedding(self, sentence_embedding: torch.Tensor):
        return self.model.prosodic_style_encoder(sentence_embedding)

    def decoding(
        self,
        text_encoding,
        duration,
        pitch,
        energy,
        style,
        audio_gt,
        split=1,
        probing=False,
    ):
        if split == 1 or text_encoding.shape[0] != 1:
            prediction = self.model.decoder(
                text_encoding @ duration, pitch, energy, style, probing=probing
            )
            yield (prediction, audio_gt, 0, energy.shape[-1])
        else:
            text_hop = text_encoding.shape[-1] // split
            text_start = 0
            text_end = text_hop + text_encoding.shape[-1] % split
            mel_start = 0
            mel_end = 0
            for i in range(split):
                mel_hop = int(duration[:, text_start:text_end, :].sum().item())
                mel_start = mel_end
                mel_end = mel_start + mel_hop

                text_slice = text_encoding[:, :, text_start:text_end]
                duration_slice = duration[:, text_start:text_end, mel_start:mel_end]
                pitch_slice = pitch[:, mel_start * 2 : mel_end * 2]
                energy_slice = energy[:, mel_start * 2 : mel_end * 2]
                audio_gt_slice = audio_gt[
                    :, mel_start * 300 * 2 : mel_end * 300 * 2
                ].detach()
                prediction = self.train.model.decoder(
                    text_slice @ duration_slice,
                    pitch_slice,
                    energy_slice,
                    style,
                    probing=probing,
                )
                yield (prediction, audio_gt_slice, mel_start, mel_end)
                text_start += text_hop
                text_end += text_hop

    def decoding_single(
        self,
        text_encoding,
        duration,
        pitch,
        energy,
        style,
        probing=False,
    ):
        return self.model.decoder(
            text_encoding @ duration, pitch, energy, style, probing=probing
        )

    def acoustic_prediction(self, batch, split=1):
        text_encoding = self.text_encoding(batch.text, batch.text_length)
        duration = self.acoustic_duration(
            batch.mel,
            batch.mel_length,
            batch.text,
            batch.text_length,
            apply_attention_mask=True,
            use_random_choice=True,
        )
        energy = self.acoustic_energy(batch.mel)
        # style_embedding = self.acoustic_style_embedding(batch.mel)
        style_embedding = self.style_embedding(batch.sentence_embedding)
        prediction = self.decoding(
            text_encoding,
            duration,
            batch.pitch,
            energy,
            style_embedding,
            batch.audio_gt,
            split=split,
        )
        return prediction

    def acoustic_prediction_single(self, batch, use_random_mono=True):
        text_encoding = self.text_encoding(batch.text, batch.text_length)
        duration = self.acoustic_duration(
            batch.mel,
            batch.mel_length,
            batch.text,
            batch.text_length,
            apply_attention_mask=True,
            use_random_choice=use_random_mono,
        )
        energy = self.acoustic_energy(batch.mel)
        pitch = self.acoustic_pitch(batch.mel)
        # style_embedding = self.acoustic_style_embedding(batch.mel)
        style_embedding = self.style_embedding(batch.sentence_embedding)
        prediction = self.decoding_single(
            text_encoding,
            duration,
            # batch.pitch,
            pitch,
            energy,
            style_embedding,
        )
        return prediction

    def textual_prediction_single(self, batch):
        text_encoding = self.text_encoding(batch.text, batch.text_length)
        duration = self.acoustic_duration(
            batch.mel,
            batch.mel_length,
            batch.text,
            batch.text_length,
            apply_attention_mask=False,
            use_random_choice=False,
        )
        style_embedding = self.style_embedding(batch.sentence_embedding)
        prosody_embedding = self.prosodic_style_embedding(batch.sentence_embedding)
        plbert_embedding = self.model.bert(
            batch.text, attention_mask=(~self.text_mask).int()
        )
        duration_encoding = self.model.bert_encoder(plbert_embedding).transpose(-1, -2)
        self.duration_prediction, prosody = self.model.duration_predictor(
            duration_encoding,
            prosody_embedding,
            batch.text_length,
            self.duration_results[1],
            self.text_mask,
        )
        self.pitch_prediction, self.energy_prediction = (
            self.model.pitch_energy_predictor(prosody, prosody_embedding)
        )
        prediction = self.decoding_single(
            text_encoding,
            duration,
            self.pitch_prediction,
            self.energy_prediction,
            style_embedding,
        )
        return prediction

    # def pretrain_decoding(self, pitch, style, audio_gt, probing=False):
    #    mels = self.to_mel(audio_gt)[:, :, :-1]
    #    return self.model.decoder(
    #        mels, pitch, None, style, pretrain=True, probing=probing
    #    )
