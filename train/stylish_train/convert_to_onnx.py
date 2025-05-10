import random
from typing import Optional

import torch
from torch.nn import functional as F
import torchaudio
from einops import rearrange, reduce

# import train_context
from config_loader import Config
from utils import length_to_mask, log_norm, maximum_path
from models.models import build_model
from config_loader import load_model_config_yaml
from text_utils import TextCleaner
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn


from attr import attr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomSTFT(nn.Module):
    """
    STFT/iSTFT without unfold/complex ops, using conv1d and conv_transpose1d.

    - forward STFT => Real-part conv1d + Imag-part conv1d
    - inverse STFT => Real-part conv_transpose1d + Imag-part conv_transpose1d + sum
    - avoids F.unfold, so easier to export to ONNX
    - uses replicate or constant padding for 'center=True' to approximate 'reflect'
      (reflect is not supported for dynamic shapes in ONNX)
    """

    def __init__(
        self,
        filter_length=800,
        hop_length=200,
        win_length=800,
        window="hann",
        center=True,
        pad_mode="replicate",  # or 'constant'
    ):
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_fft = filter_length
        self.center = center
        self.pad_mode = pad_mode

        # Number of frequency bins for real-valued STFT with onesided=True
        self.freq_bins = self.n_fft // 2 + 1

        # Build window
        assert window == "hann", window
        window_tensor = torch.hann_window(
            win_length, periodic=True, dtype=torch.float32
        )
        if self.win_length < self.n_fft:
            # Zero-pad up to n_fft
            extra = self.n_fft - self.win_length
            window_tensor = F.pad(window_tensor, (0, extra))
        elif self.win_length > self.n_fft:
            window_tensor = window_tensor[: self.n_fft]
        self.register_buffer("window", window_tensor)

        # Precompute forward DFT (real, imag)
        # PyTorch stft uses e^{-j 2 pi k n / N} => real=cos(...), imag=-sin(...)
        n = np.arange(self.n_fft)
        k = np.arange(self.freq_bins)
        angle = 2 * np.pi * np.outer(k, n) / self.n_fft  # shape (freq_bins, n_fft)
        dft_real = np.cos(angle)
        dft_imag = -np.sin(angle)  # note negative sign

        # Combine window and dft => shape (freq_bins, filter_length)
        # We'll make 2 conv weight tensors of shape (freq_bins, 1, filter_length).
        forward_window = window_tensor.numpy()  # shape (n_fft,)
        forward_real = dft_real * forward_window  # (freq_bins, n_fft)
        forward_imag = dft_imag * forward_window

        # Convert to PyTorch
        forward_real_torch = torch.from_numpy(forward_real).float()
        forward_imag_torch = torch.from_numpy(forward_imag).float()

        # Register as Conv1d weight => (out_channels, in_channels, kernel_size)
        # out_channels = freq_bins, in_channels=1, kernel_size=n_fft
        self.register_buffer("weight_forward_real", forward_real_torch.unsqueeze(1))
        self.register_buffer("weight_forward_imag", forward_imag_torch.unsqueeze(1))

        # Precompute inverse DFT
        # Real iFFT formula => scale = 1/n_fft, doubling for bins 1..freq_bins-2 if n_fft even, etc.
        # For simplicity, we won't do the "DC/nyquist not doubled" approach here.
        # If you want perfect real iSTFT, you can add that logic.
        # This version just yields good approximate reconstruction with Hann + typical overlap.
        inv_scale = 1.0 / self.n_fft
        n = np.arange(self.n_fft)
        angle_t = 2 * np.pi * np.outer(n, k) / self.n_fft  # shape (n_fft, freq_bins)
        idft_cos = np.cos(angle_t).T  # => (freq_bins, n_fft)
        idft_sin = np.sin(angle_t).T  # => (freq_bins, n_fft)

        # Multiply by window again for typical overlap-add
        # We also incorporate the scale factor 1/n_fft
        inv_window = window_tensor.numpy() * inv_scale
        backward_real = idft_cos * inv_window  # (freq_bins, n_fft)
        backward_imag = idft_sin * inv_window

        # We'll implement iSTFT as real+imag conv_transpose with stride=hop.
        self.register_buffer(
            "weight_backward_real", torch.from_numpy(backward_real).float().unsqueeze(1)
        )
        self.register_buffer(
            "weight_backward_imag", torch.from_numpy(backward_imag).float().unsqueeze(1)
        )

    def transform(self, waveform: torch.Tensor):
        """
        Forward STFT => returns magnitude, phase
        Output shape => (batch, freq_bins, frames)
        """
        # waveform shape => (B, T).  conv1d expects (B, 1, T).
        # Optional center pad
        if self.center:
            pad_len = self.n_fft // 2
            waveform = F.pad(waveform, (pad_len, pad_len), mode=self.pad_mode)

        x = waveform.unsqueeze(1)  # => (B, 1, T)
        # Convolution to get real part => shape (B, freq_bins, frames)
        real_out = F.conv1d(
            x,
            self.weight_forward_real,
            bias=None,
            stride=self.hop_length,
            padding=0,
        )
        # Imag part
        imag_out = F.conv1d(
            x,
            self.weight_forward_imag,
            bias=None,
            stride=self.hop_length,
            padding=0,
        )

        # magnitude, phase
        magnitude = torch.sqrt(real_out**2 + imag_out**2 + 1e-14)
        phase = torch.atan2(imag_out, real_out)
        # Handle the case where imag_out is 0 and real_out is negative to correct ONNX atan2 to match PyTorch
        # In this case, PyTorch returns pi, ONNX returns -pi
        correction_mask = (imag_out == 0) & (real_out < 0)
        phase[correction_mask] = torch.pi
        return magnitude, phase

    def inverse(self, magnitude: torch.Tensor, phase: torch.Tensor, length=None):
        """
        Inverse STFT => returns waveform shape (B, T).
        """
        # magnitude, phase => (B, freq_bins, frames)
        # Re-create real/imag => shape (B, freq_bins, frames)
        real_part = magnitude * torch.cos(phase)
        imag_part = magnitude * torch.sin(phase)

        # conv_transpose wants shape (B, freq_bins, frames). We'll treat "frames" as time dimension
        # so we do (B, freq_bins, frames) => (B, freq_bins, frames)
        # But PyTorch conv_transpose1d expects (B, in_channels, input_length)
        real_part = real_part  # (B, freq_bins, frames)
        imag_part = imag_part

        # real iSTFT => convolve with "backward_real", "backward_imag", and sum
        # We'll do 2 conv_transpose calls, each giving (B, 1, time),
        # then add them => (B, 1, time).
        real_rec = F.conv_transpose1d(
            real_part,
            self.weight_backward_real,  # shape (freq_bins, 1, filter_length)
            bias=None,
            stride=self.hop_length,
            padding=0,
        )
        imag_rec = F.conv_transpose1d(
            imag_part,
            self.weight_backward_imag,
            bias=None,
            stride=self.hop_length,
            padding=0,
        )
        # sum => (B, 1, time)
        waveform = real_rec - imag_rec  # typical real iFFT has minus for imaginary part

        # If we used "center=True" in forward, we should remove pad
        if self.center:
            pad_len = self.n_fft // 2
            # Because of transposed convolution, total length might have extra samples
            # We remove `pad_len` from start & end if possible
            waveform = waveform[..., pad_len:-pad_len]

        # If a specific length is desired, clamp
        if length is not None:
            waveform = waveform[..., :length]

        # shape => (B, T)
        return waveform

    def forward(self, x: torch.Tensor):
        """
        Full STFT -> iSTFT pass: returns time-domain reconstruction.
        Same interface as your original code.
        """
        mag, phase = self.transform(x)
        return self.inverse(mag, phase, length=x.shape[-1])


class Stylish(nn.Module):
    def __init__(self, model_config, device):
        super(Stylish, self).__init__()
        self.device = device
        self.model = build_model(model_config)
        for key in self.model:
            self.model[key].to(device).eval()
            for p in self.model[key].parameters():
                p.requires_grad = False

    def decoding_single(
        self,
        text_encoding,
        duration,
        pitch,
        energy,
        style,
        probing=False,
    ):
        mel, f0_curve = self.model.decoder(
            text_encoding @ duration, pitch, energy, style, probing=probing
        )
        prediction = self.model.generator(
            mel=mel, style=style, pitch=f0_curve, energy=energy
        )
        # prediction = self.model.decoder(
        #     text_encoding @ duration, pitch, energy, style, probing=probing
        # )
        return prediction

    def duration_predictor(
        self, duration_encoding, prosody_embedding, text_lengths, text_mask
    ):
        d = self.model.duration_predictor.text_encoder(
            duration_encoding, prosody_embedding, text_lengths, text_mask
        )
        x, _ = self.model.duration_predictor.lstm(d)
        duration = self.model.duration_predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1)

        pred_dur = torch.round(duration).clamp(min=1).long().squeeze()
        indices = torch.repeat_interleave(
            torch.arange(duration_encoding.shape[2], device=self.device), pred_dur
        )
        pred_aln_trg = torch.zeros(
            (duration_encoding.shape[2], indices.shape[0]), device=self.device
        )
        pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
        pred_aln_trg = pred_aln_trg.unsqueeze(0).to(self.device)

        prosody = d.permute(0, 2, 1) @ pred_aln_trg
        return pred_aln_trg, prosody

    def forward(self, texts, text_lengths, text_mask, sentence_embedding):
        text_encoding = self.model.text_encoder(texts, text_lengths, text_mask)
        style_embedding = self.model.textual_style_encoder(sentence_embedding)
        prosody_embedding = self.model.textual_prosody_encoder(sentence_embedding)

        plbert_embedding = self.model.bert(texts, attention_mask=(~text_mask).int())
        duration_encoding = self.model.bert_encoder(plbert_embedding).permute(0, 2, 1)

        duration_prediction, prosody = self.duration_predictor(
            duration_encoding,
            prosody_embedding,
            text_lengths,
            text_mask,
        )
        # duration_prediction, prosody = self.duration_prediction, self.prosody
        pitch_prediction, energy_prediction = self.model.pitch_energy_predictor(
            prosody, prosody_embedding
        )
        mel, f0_curve = self.model.decoder(
            text_encoding @ duration_prediction,
            pitch_prediction,
            energy_prediction,
            style_embedding,
            probing=False,
        )
        return mel, style_embedding, f0_curve, energy_prediction
        """return (
            text_encoding,
            duration_prediction,
            pitch_prediction,
            energy_prediction,
            style_embedding,
        )"""
        """prediction = self.decoding_single(
            text_encoding,
            duration_prediction,
            pitch_prediction,
            energy_prediction,
            style_embedding,
        )
        return prediction.audio.squeeze()"""


class Generator(torch.nn.Module):
    def __init__(self, generator):
        super(Generator, self).__init__()
        self.generator = generator

    def forward(self, mel, style, pitch, energy):
        return self.generator(
            mel=mel, style=style, pitch=pitch, energy=energy
        ).audio.squeeze()


model_config = load_model_config_yaml("/content/stylish-tts/config/model.yml")
text_cleaner = TextCleaner(model_config.symbol)
sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").cpu()
model = Stylish(model_config, "cuda").eval()
model.model.generator.stft = CustomSTFT(
    filter_length=model.model.generator.gen_istft_n_fft,
    hop_length=model.model.generator.gen_istft_hop_size,
    win_length=model.model.generator.gen_istft_n_fft,
)
model.model.generator.stft.cuda().eval()
generator = Generator(model.model.generator)

texts = torch.tensor(text_cleaner("ɑɐɒæɓʙβɔɗɖðʤəɘɚɛɜɝɞɟʄɡɠ")).unsqueeze(0).cuda()
text_lengths = torch.zeros([1], dtype=int).cuda()
text_lengths[0] = texts.shape[1]
text_mask = torch.ones(1, texts.shape[1], dtype=bool).cuda()
sentence_embedding = (
    torch.from_numpy(
        sbert.encode(
            [
                "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."
            ],
            show_progress_bar=False,
        )
    )
    .float()
    .cuda()
)

inputs = (texts, text_lengths, text_mask, sentence_embedding)
with torch.no_grad():
    torch.onnx.export(
        model,
        inputs,
        opset_version=14,
        f="stylish.onnx",
        input_names=["texts", "text_lengths", "text_mask", "sentence_embedding"],
        output_names=["waveform"],
        dynamic_axes={
            "texts": {0: "batch_size", 1: "num_token"},
            "text_mask": {0: "batch_size", 1: "num_token"},
            "waveform": {0: "num_samples"},
        },
    )


input_shapes = (
    torch.Size([1, 512, 1150]),
    torch.Size([1, 128]),
    torch.Size([1, 1150]),
    torch.Size([1, 1150]),
)
input_dtypes = torch.float32, torch.float32, torch.float32, torch.float32
input_names = "mel, style, pitch, energy".split(", ")
with torch.no_grad():
    torch.onnx.export(
        generator,
        tuple(
            [
                torch.ones(input_shape, dtype=dtype).cuda()
                for input_shape, dtype in zip(input_shapes, input_dtypes)
            ]
        ),
        opset_version=19,
        f="ringformer.onnx",
        input_names=input_names,
        output_names=["waveform"],
        dynamic_axes=dict(
            {
                k: {
                    i: f"dim_{i}"
                    for i, d in enumerate(v)
                    if d > 1 and d != 512 and d != 640
                }
                for k, v in zip(input_names, input_shapes)
            }
        ),
        do_constant_folding=False,
    )
