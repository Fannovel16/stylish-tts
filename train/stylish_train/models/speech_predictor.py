import torch
from .text_encoder import TextEncoder, Encoder
from .text_style_encoder import TextStyleEncoder
from .prosody_encoder import ProsodyEncoder
from .duration_predictor import DurationPredictor
from .pitch_energy_predictor import PitchEnergyPredictor
from .decoder import Decoder
from .generator import Generator
from .hubert_encoder import HubertEncoder
from .mel_style_encoder import MelStyleEncoder


class SpeechPredictor(torch.nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.text_encoder = TextEncoder(
            inter_dim=model_config.inter_dim, config=model_config.text_encoder
        )

        self.style_encoder = TextStyleEncoder(
            model_config.inter_dim,
            model_config.style_dim,
            model_config.style_encoder,
        )

        self.mel_style_encoder =  MelStyleEncoder(
            model_config.n_mels,
            model_config.style_dim,
            model_config.mel_style_encoder.max_channels,
            model_config.mel_style_encoder.skip_downsample,
        )

        self.decoder = Decoder(
            dim_in=model_config.inter_dim,
            style_dim=model_config.style_dim,
            dim_out=model_config.generator.input_dim,
            hidden_dim=model_config.decoder.hidden_dim,
            residual_dim=model_config.decoder.residual_dim,
        )

        # self.generator = Generator(
        #     style_dim=model_config.style_dim,
        #     resblock_kernel_sizes=model_config.generator.resblock_kernel_sizes,
        #     upsample_rates=model_config.generator.upsample_rates,
        #     upsample_initial_channel=model_config.generator.input_dim,
        #     upsample_last_channel=model_config.generator.upsample_last_channel,
        #     resblock_dilation_sizes=model_config.generator.resblock_dilation_sizes,
        #     upsample_kernel_sizes=model_config.generator.upsample_kernel_sizes,
        #     gen_istft_n_fft=model_config.generator.gen_istft_n_fft,
        #     gen_istft_hop_size=model_config.generator.gen_istft_hop_size,
        #     sample_rate=model_config.sample_rate,
        # )
        self.generator = Generator(
            style_dim=model_config.style_dim,
            n_fft=model_config.n_fft,
            win_length=model_config.win_length,
            hop_length=model_config.hop_length,
            config=model_config.generator,
        )

    def forward(self, texts, text_lengths, alignment, pitch, energy, mel):
        text_encoding, _, _ = self.text_encoder(texts, text_lengths)
        # style = self.style_encoder(text_encoding, text_lengths)
        style = self.mel_style_encoder(mel.unsqueeze(1))
        mel, f0_curve = self.decoder(
            text_encoding @ alignment,
            pitch,
            energy,
            style,
        )
        prediction = self.generator(
            mel=mel,
            style=style,
            pitch=f0_curve,
            energy=energy,
        )
        return prediction


class HubertSpeechPredictor(torch.nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.phone_encoder = HubertEncoder(model_config)

        self.style_encoder = TextStyleEncoder(
            model_config.inter_dim,
            model_config.style_dim,
            model_config.style_encoder,
        )

        self.decoder = Decoder(
            dim_in=model_config.inter_dim,
            style_dim=model_config.style_dim,
            dim_out=model_config.generator.input_dim,
            hidden_dim=model_config.decoder.hidden_dim,
            residual_dim=model_config.decoder.residual_dim,
        )

        # self.generator = Generator(
        #     style_dim=model_config.style_dim,
        #     resblock_kernel_sizes=model_config.generator.resblock_kernel_sizes,
        #     upsample_rates=model_config.generator.upsample_rates,
        #     upsample_initial_channel=model_config.generator.upsample_initial_channel,
        #     upsample_last_channel=model_config.generator.upsample_last_channel,
        #     resblock_dilation_sizes=model_config.generator.resblock_dilation_sizes,
        #     upsample_kernel_sizes=model_config.generator.upsample_kernel_sizes,
        #     gen_istft_n_fft=model_config.generator.gen_istft_n_fft,
        #     gen_istft_hop_size=model_config.generator.gen_istft_hop_size,
        #     sample_rate=model_config.sample_rate,
        # )
        self.generator = Generator(
            style_dim=model_config.style_dim,
            n_fft=model_config.n_fft,
            win_length=model_config.win_length,
            hop_length=model_config.hop_length,
            config=model_config.generator,
        )

    def forward(self, phones, phone_lengths, spk_emb, pitch, energy):
        phones = self.phone_encoder(phones, phone_lengths)
        style = self.style_encoder(phones, phone_lengths)
        mel, f0_curve = self.decoder(
            phones,
            pitch,
            energy,
            style,
        )
        prediction = self.generator(
            mel=mel,
            style=style,
            pitch=f0_curve,
            energy=energy,
        )
        return prediction
