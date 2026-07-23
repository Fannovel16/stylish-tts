import torch
import torch.nn as nn
import torchaudio
from einops import rearrange
import random
from huggingface_hub import snapshot_download
import os
import yaml
from .samresnet import SimAM_ResNet34_ASP, SimAM_ResNet100_ASP
import logging
import torchaudio.compliance.kaldi as kaldi
import torch.nn.functional as F
from .emotion2vec import Emotion2Vec

# from focalcodec import FocalCodec
from pathlib import Path
from kanade_tokenizer import KanadeModel, load_vocoder, vocode
from miocodec import MioCodecModel


from transformers import HubertModel


class HubertModelWithFinalProj(HubertModel):
    def __init__(self, config):
        super().__init__(config)
        self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)


class AdaptiveHubert(nn.Module):
    def __init__(self, hubert_path: str, global_sr: int):
        super().__init__()
        self.model = HubertModelWithFinalProj.from_pretrained(hubert_path)
        self.resample = torchaudio.transforms.Resample(global_sr, 16000)

    def forward(self, waveform, center_pad=True):
        waveform = self.resample(waveform)
        xs = []
        for wave in waveform:
            x = self.model(wave.unsqueeze(0))["last_hidden_state"]
            pad = 1
            if center_pad:
                x = F.pad(x.mT, (pad // 2, pad // 2 + (pad % 2)), "reflect").mT
            xs.append(x)
            torch.cuda.empty_cache()
        xs = torch.cat(xs, 0)
        return xs


class AdaptiveFocalCodec(nn.Module):
    def __init__(
        self, global_sr: int, codec_path: str = "lucadellalib/focalcodec_50hz"
    ):
        super().__init__()
        self.codec = FocalCodec.from_pretrained(codec_path)
        self.resample = torchaudio.transforms.Resample(global_sr, 16000)

    def remove_encoder(self):
        if hasattr(self.codec, "encoder"):
            del self.codec.encoder
            del self.codec.compressor
            torch.cuda.empty_cache()

    def forward(self, wave, *scales, center_pad=True):
        wave = self.resample(wave)
        x = self.codec.sig_to_feats(wave)
        codes = []
        for scale in scales:
            _x = F.interpolate(
                x.mT,
                scale_factor=scale,
                mode="nearest",
            )
            if center_pad:
                # Padding due to center=True??
                pad = scale
                _x = F.pad(_x, (pad // 2, pad // 2 + (pad % 2)), "reflect")
            codes.append(self.codec.feats_to_toks(_x.mT))
        return codes

    def decode(self, codes):
        return self.codec.toks_to_sig(codes)


def load_checkpoint(model: torch.nn.Module, path: str):
    checkpoint = torch.load(path, map_location="cpu")
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
    for key in missing_keys:
        logging.warning("missing tensor: {}".format(key))
    for key in unexpected_keys:
        logging.warning("unexpected tensor: {}".format(key))


# https://github.com/wenet-e2e/wespeaker/blob/67f0f4a8d472e6e2203d7baca38daba818af17f3/wespeaker/cli/speaker.py#L306
def load_model_pt(model_name_or_path: str):
    """There are the following files in the `model_dir`:
    - config.yaml: the model config file
    - avg_model.pt: the pytorch model file
    """
    model_dir = snapshot_download(model_name_or_path)
    required_files = ["config.yaml", "avg_model.pt"]
    for file in required_files:
        if not os.path.exists(os.path.join(model_dir, file)):
            raise FileNotFoundError(f"{file} not found in {model_dir}")
    # Read config file
    with open(os.path.join(model_dir, "config.yaml"), "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if config["model"] == "SimAM_ResNet34_ASP":
        model = SimAM_ResNet34_ASP(**config["model_args"])
    elif config["model"] == "SimAM_ResNet100_ASP":
        model = SimAM_ResNet100_ASP(**config["model_args"])
    else:
        raise NotImplementedError(config["model"])
    load_checkpoint(model, os.path.join(model_dir, "avg_model.pt"))
    model.eval()
    return model


class SpeakerEmbeddingModel(nn.Module):
    def __init__(self, model_sr: int):
        super().__init__()
        self.model = load_model_pt("gaunernst/wespeaker-voxblink2-samresnet34")
        # self.model.pooling = nn.Identity()
        # self.model.bottleneck = nn.Identity()

        self.resample_rate = 16000
        self.window_type = "hamming"
        self.resample = torchaudio.transforms.Resample(model_sr, self.resample_rate)

    def compute_fbank(
        self,
        wavform,
        sample_rate=16000,
        num_mel_bins=80,
        frame_length=25,
        frame_shift=10,
        cmn=True,
    ):
        feat = kaldi.fbank(
            wavform,
            num_mel_bins=num_mel_bins,
            frame_length=frame_length,
            frame_shift=frame_shift,
            sample_frequency=sample_rate,
            window_type=self.window_type,
        )
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


class AdaptiveEmotion2Vec(torch.nn.Module):
    def __init__(self, global_sr):
        super().__init__()
        self.model = Emotion2Vec.from_pretrained()
        self.resample = torchaudio.transforms.Resample(global_sr, 16000)

    def forward(self, wave, *scales, center_pad=True):
        wave = self.resample(wave)
        if self.model.cfg.normalize:
            wave = F.layer_norm(wave, wave.shape)
        x = self.model.extract_features(wave)["x"]
        x = rearrange(x, "b t c -> b c t")
        xs = []
        for scale in scales:
            _x = F.interpolate(
                x,
                scale_factor=scale,
                mode="nearest",
            )
            if center_pad:
                # Padding due to center=True??
                pad = scale
                _x = F.pad(_x, (pad // 2, pad // 2 + (pad % 2)), "reflect")
            xs.append(_x)
        return xs


# class AdaptiveS3Codec(torch.nn.Module):
#     def __init__(self, global_sr: int):
#         super().__init__()
#         self.codec = s3tokenizer.S3Tokenizer("speech_tokenizer_v1_25hz")
#         self.codec.init_from_onnx("D:\\TTS\\speech_tokenizer_v1.onnx")
#         self.resample = torchaudio.transforms.Resample(global_sr, 16000)

#     def remove_encoder(self):
#         if hasattr(self, "codec"):
#             del self.codec
#             torch.cuda.empty_cache()

#     def forward(self, wave, *scales, center_pad=True):
#         wave = self.resample(wave)
#         x = s3tokenizer.log_mel_spectrogram(wave)
#         codes = []
#         for scale in scales:
#             _x = F.interpolate(
#                 x,
#                 scale_factor=scale,
#                 mode="nearest",
#             )
#             if center_pad:
#                 # Padding due to center=True??
#                 pad = scale
#                 _x = F.pad(_x, (pad // 2, pad // 2 + (pad % 2)), "reflect")
#             _codes, _ = self.codec(_x, torch.tensor([_x.shape[-1]], device=x.device, dtype=torch.long))
#             codes.append(_codes)
#         return codes


class AdaptiveKanadeCodec(nn.Module):
    def __init__(self, global_sr: int, codec_path: str = "frothywater/kanade-25hz"):
        super().__init__()
        self.model = KanadeModel.from_pretrained(codec_path)
        self.vocoder = load_vocoder(self.model.config.vocoder_name)
        for param in self.model.parameters():
            param.requires_grad = False

    def normalize(self, wave):
        max_val = torch.max(torch.abs(wave)) + 1e-8
        wave = wave / max_val  # Normalize to [-1, 1]
        return wave

    def forward(self, wave, *scales, center_pad=False):
        x = [
            self.model.encode(_wave, True, False).content_token_indices
            for _wave in self.normalize(wave)
        ]
        x = torch.stack(x, 0)
        codes = []
        for scale in scales:
            _x = x.repeat(1, scale)
            if center_pad:
                # Padding due to center=True??
                pad = scale
                _x = F.pad(_x, (pad // 2, pad // 2 + (pad % 2)), "reflect")
            codes.append(_x)
        return codes

    def encode(self, waveform):
        return self.model.encode(self.normalize(waveform)[0])

    def get_ssl_embeddings(self, waveform: torch.Tensor):
        waveform = self.normalize(waveform)
        audio_length = waveform.size(-1)
        padding = self.model._calculate_waveform_padding(audio_length)
        local_ssl_features, global_ssl_features = [], []
        for wave in waveform:
            local_emb, global_emb = self.model.forward_ssl_features(
                wave.unsqueeze(0), padding=padding
            )
            local_ssl_features.append(local_emb)
            global_ssl_features.append(global_emb)
            torch.cuda.empty_cache()
        local_ssl_features = torch.cat(local_ssl_features, 0)
        global_ssl_features = torch.cat(global_ssl_features, 0)
        return local_ssl_features, global_ssl_features

    def decode(self, content_tokens, global_embs):
        waves = []
        for content_token, global_emb in zip(content_tokens, global_embs):
            mel = self.model.decode(
                content_token_indices=content_token,
                global_embedding=global_emb,
            ).unsqueeze(0)
            wave = self.decode_mel(mel)
            waves.append(wave)
        waveform = torch.cat(waves, 0)
        return waveform

    def decode_continuous(self, content_embs, global_embs):
        waves = []
        for content_emb, global_emb in zip(content_embs, global_embs):
            mel = self.model.decode(
                content_embedding=content_emb,
                global_embedding=global_emb,
            ).unsqueeze(0)
            wave = self.decode_mel(mel)
            waves.append(wave)
        waveform = torch.cat(waves, 0)
        return waveform

    def decode_mel(self, mel):
        return torch.cat([vocode(self.vocoder, _mel.unsqueeze(0)) for _mel in mel], 0)

    def encode_latent_classes(self, content_tokens):
        codes = self.model.local_quantizer.fsq.indices_to_codes(content_tokens)
        classes = self.model.local_quantizer.fsq._scale_and_shift(codes)
        classes = classes.round().long()
        return classes

    def decode_latent_classes(self, classes):
        indices = (classes * self.model.local_quantizer.fsq._basis).sum(dim=-1)
        return indices

    def get_latent_codes(self, content_tokens):
        return self.model.local_quantizer.fsq.indices_to_codes(content_tokens)

    def flatten_latent_codes(self, latent_codes):
        z_q = self.model.local_quantizer.fsq.quantize(latent_codes)
        return self.model.local_quantizer.fsq.codes_to_indices(z_q)

    def decode_content_tokens(self, content_tokens):
        return self.model.decode_token_indices(content_tokens)

    def retrieve_nearest_contents(self, pred_content_embs):
        existing_contents = self.model.decode_token_indices(
            torch.arange(12800).to(pred_content_embs.device)
        )
        dists = torch.cdist(pred_content_embs, existing_contents)
        closest_idx = dists.argmin(-1)
        closest_contents = existing_contents[closest_idx]
        return closest_contents

    def get_global_ssl(self, waveform):
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            ssl_features = self.model.ssl_feature_extractor(waveform)
        return self.model._process_ssl_features(ssl_features, (1, 2)).float()

    # def mel_to_wave(self, mel):
    #     return vocode(self.vocoder, mel)

    # def continuos_decode(self, z, global_embedding):
    #     z_q, indices = self.model.local_quantizer.fsq.encode(z)
    #     z_q = self.model.local_quantizer.proj_out(z_q)
    #     content_embedding = z_q

    #     # Estimate original audio length from content token sequence length
    #     seq_len = content_embedding.size(1)
    #     target_audio_length = self.model._calculate_original_audio_length(seq_len)

    #     mel_length = self.model._calculate_target_mel_length(target_audio_length)
    #     mel_spectrogram = self.model.forward_mel(content_embedding, global_embedding, mel_length=mel_length)
    #     return mel_spectrogram

    # def discrete_decode(self, content_tokens, global_embedding):
    #     content_embedding = self.model.decode_token_indices(content_tokens)
    #     # Estimate original audio length from content token sequence length
    #     seq_len = content_embedding.size(1)
    #     target_audio_length = self.model._calculate_original_audio_length(seq_len)

    #     mel_length = self.model._calculate_target_mel_length(target_audio_length)
    #     mel_spectrogram = self.model.forward_mel(content_embedding, global_embedding, mel_length=mel_length)
    #     return mel_spectrogram

    # def mel_to_wave(self, mel):
    #     return vocode(self.vocoder, mel)


# class AdaptiveOrangeWavLM(nn.Module):
#     def __init__(self, global_sr: int, model_path="Orange/Speaker-wavLM-pro"):
#         super().__init__()
#         self.model = EmbeddingsModel.from_pretrained(model_path)
#         self.resample = torchaudio.transforms.Resample(global_sr, 16000)

#     def forward(self, wave):
#         wave = self.resample(wave)
#         x = self.model(wave)
#         return x


class AdaptiveMioCodec(nn.Module):
    def __init__(self, global_sr: int, codec_path: str = "Aratako/MioCodec-25Hz-24kHz"):
        super().__init__()
        self.model = MioCodecModel.from_pretrained(codec_path)
        for param in self.model.parameters():
            param.requires_grad = False

    def normalize(self, wave):
        max_val = torch.max(torch.abs(wave)) + 1e-8
        wave = wave / max_val  # Normalize to [-1, 1]
        return wave

    def encode(self, waveform):
        return self.model.encode(self.normalize(waveform)[0])

    def forward(self, wave, *scales, center_pad=False):
        x = [
            self.model.encode(_wave, True, False).content_token_indices
            for _wave in self.normalize(wave)
        ]
        x = torch.stack(x, 0)
        codes = []
        for scale in scales:
            _x = x.repeat(1, scale)
            if center_pad:
                # Padding due to center=True??
                pad = scale
                _x = F.pad(_x, (pad // 2, pad // 2 + (pad % 2)), "reflect")
            codes.append(_x)
        return codes

    def get_latent_codes(self, content_tokens):
        return self.model.local_quantizer.fsq.indices_to_codes(content_tokens)

    def flatten_latent_codes(self, latent_codes):
        return self.model.local_quantizer.fsq.codes_to_indices(latent_codes)

    def encode_latent_classes(self, content_tokens):
        codes = self.model.local_quantizer.fsq.indices_to_codes(content_tokens)
        classes = self.model.local_quantizer.fsq._scale_and_shift(codes)
        classes = classes.round().long()
        return classes

    def decode_latent_classes(self, classes):
        indices = (classes * self.model.local_quantizer.fsq._basis).sum(dim=-1)
        return indices

    def get_global_embeddings(self, wave):
        x = [
            self.model.encode(_wave, False, True).global_embedding
            for _wave in self.normalize(wave)
        ]
        x = torch.stack(x, 0)
        return x

    def decode(self, content_tokens, global_embs):
        wave = []
        for content_token, global_emb in zip(content_tokens, global_embs):
            _wave = self.model.decode(
                content_token_indices=content_token,
                global_embedding=global_emb,
            )
            wave.append(_wave)
        return torch.stack(wave, 0)

    def get_global_ssl(self, waveform):
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            ssl_features = self.model.ssl_feature_extractor(waveform)
        return self.model._process_ssl_features(ssl_features, (1, 2)).float()


from huggingface_hub import snapshot_download
from .vevo_repcodec import VevoRepCodec


class AdaptiveVevoCodec(torch.nn.Module):
    def __init__(self, global_sr: int):
        super().__init__()
        self.hubert = torchaudio.pipelines.HUBERT_LARGE.get_model()

        down_dir = snapshot_download(
            repo_id="amphion/Vevo",
            repo_type="model",
            allow_patterns=["tokenizer/vq32/*"],
        )
        down_dir = Path(down_dir, "tokenizer/vq32")
        with open(down_dir / "hubert_large_l18_c32.yaml") as fp:
            conf = yaml.load(fp, Loader=yaml.FullLoader)

        self.vqvae = VevoRepCodec(**conf)
        self.vqvae.quantizer.initial()
        self.vqvae.load_state_dict(
            torch.load(down_dir / "hubert_large_l18_c32.pkl", map_location="cpu")[
                "model"
            ]["repcodec"]
        )
        self.resample = torchaudio.transforms.Resample(global_sr, 16000)

    def remove_encoder(self):
        if hasattr(self, "hubert"):
            del self.hubert
            del self.vqvae
            torch.cuda.empty_cache()

    @torch.no_grad()
    def extract_hubert_feature(self, wavs, wav_lens=None, output_layer=18):
        """
        Args:
            wavs: [B, T]
            wav_lens: [B,]
        Returns:
            feats: [B, T, D]
            feat_lengths: [B]
        """
        if wav_lens is None:
            wav_lens = torch.tensor([wavs.shape[1]] * wavs.shape[0]).to(wavs).int()

        feats, feat_lengths = self.hubert.extract_features(
            wavs, lengths=wav_lens, num_layers=output_layer
        )
        feats = feats[-1]
        return feats, feat_lengths

    def forward(self, wave, *scales, center_pad=True):
        wave = self.resample(wave)
        feats, _ = self.extract_hubert_feature(wave)
        x = self.vqvae.encoder(feats.mT)
        x = self.vqvae.projector(x)
        codes = []
        for scale in scales:
            _x = F.interpolate(
                x,
                scale_factor=scale,
                mode="nearest",
            )
            if center_pad:
                # Padding due to center=True??
                pad = scale
                _x = F.pad(_x, (pad // 2, pad // 2 + (pad % 2)), "reflect")

            _, idx = self.vqvae.quantizer.codebook.forward_index(_x.mT)
            codes.append(idx[0])
        return codes


# Adapted from:
# Vocos: https://github.com/gemelo-ai/vocos/blob/main/vocos/feature_extractors.py
# BigVGAN: https://github.com/NVIDIA/BigVGAN/blob/main/meldataset.py (Also used by HiFT)

import torch
import torchaudio
from librosa.filters import mel as librosa_mel_fn
from torch import nn


def safe_log(x: torch.Tensor, clip_val: float = 1e-7) -> torch.Tensor:
    return torch.log(torch.clip(x, min=clip_val))


class MelSpectrogramFeature(nn.Module):
    def __init__(
        self,
        sample_rate: int = 24000,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 100,
        padding: str = "center",
        fmin: int = 0,
        fmax: int | None = None,
        bigvgan_style_mel: bool = False,
    ):
        super().__init__()

        self.bigvgan_style_mel = bigvgan_style_mel
        if bigvgan_style_mel:
            # BigVGAN style: same padding, Slaney mel scale, with normalization
            self.n_fft = n_fft
            self.win_size = n_fft
            self.hop_size = hop_length
            # (n_mels, n_fft // 2 + 1)
            mel_basis = librosa_mel_fn(
                sr=sample_rate,
                n_fft=n_fft,
                n_mels=n_mels,
                norm="slaney",
                htk=False,
                fmin=fmin,
                fmax=fmax,
            )
            mel_basis = torch.from_numpy(mel_basis).float()
            hann_window = torch.hann_window(n_fft)
            self.register_buffer("mel_basis", mel_basis)
            self.register_buffer("hann_window", hann_window)
        else:
            # Vocos style: center padding, HTK mel scale, without normalization
            if padding not in ["center", "same"]:
                raise ValueError("Padding must be 'center' or 'same'.")

            self.padding = padding
            self.mel_spec = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                center=padding == "center",
                power=1,
                f_min=fmin,
                f_max=fmax,
            )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            mel_specgram (Tensor): Mel spectrogram of the input audio. (B, C, L)
        """
        if self.bigvgan_style_mel:
            return self.bigvgan_mel(audio)
        else:
            return self.vocos_mel(audio)

    def vocos_mel(self, audio: torch.Tensor) -> torch.Tensor:
        if self.padding == "same":
            pad = self.mel_spec.win_length - self.mel_spec.hop_length
            audio = torch.nn.functional.pad(audio, (pad // 2, pad // 2), mode="reflect")

        specgram = self.mel_spec.spectrogram(audio)
        mel_specgram = self.mel_spec.mel_scale(specgram)

        # Convert to log scale
        mel_specgram = safe_log(mel_specgram)
        return mel_specgram

    def bigvgan_mel(self, audio: torch.Tensor) -> torch.Tensor:
        # Pad so that the output length T = L // hop_length
        padding = (self.n_fft - self.hop_size) // 2
        audio = torch.nn.functional.pad(audio, (padding, padding), mode="reflect")
        audio = audio.reshape(-1, audio.shape[-1])

        spec = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=self.hann_window,
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        spec = spec.reshape(audio.shape[:-1] + spec.shape[-2:])

        spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)
        mel_spec = torch.matmul(self.mel_basis, spec)
        mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))
        return mel_spec
