import torch
from .text_encoder import TextEncoder
from .text_style_encoder import TextStyleEncoder
from .prosody_encoder import ProsodyEncoder
from .duration_predictor import DurationPredictor
from .pitch_energy_predictor import PitchEnergyPredictor
from .decoder import Decoder
from .generator import Generator
from .hubert_encoder import HubertEncoder
from .flow import PriorEncoder, PosteriorEncoder, ResidualCouplingBlock
from stylish_tts.train.utils import DecoderPrediction, FlowStatistics


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

        self.decoder = Decoder(
            dim_in=model_config.inter_dim,
            style_dim=model_config.style_dim,
            dim_out=model_config.generator.input_dim,
            hidden_dim=model_config.decoder.hidden_dim,
            residual_dim=model_config.decoder.residual_dim,
        )

        flow_hidden_dim = model_config.decoder.hidden_dim // 4
        self.prior_encoder = PriorEncoder(
            model_config.decoder.hidden_dim, flow_hidden_dim
        )
        self.posterior_encoder = PosteriorEncoder(
            flow_hidden_dim,
            flow_hidden_dim,
            3,
            1,
            n_layers=12,
            n_fft=model_config.n_fft,
            win_length=model_config.win_length,
            hop_length=model_config.hop_length // 4,
            gin_channels=model_config.style_dim,
        )
        self.flow = ResidualCouplingBlock(
            flow_hidden_dim,
            flow_hidden_dim,
            5,
            1,
            n_layers=4,
            n_flows=8,
            gin_channels=model_config.style_dim,
            use_transformer_flow=False,
        )
        self.post_flow = torch.nn.Linear(
            flow_hidden_dim, model_config.decoder.hidden_dim
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
        self.upsampler = torch.nn.Upsample(scale_factor=4, mode="linear")
        self.generator = Generator(
            style_dim=model_config.style_dim,
            n_fft=model_config.n_fft,
            win_length=model_config.win_length,
            hop_length=model_config.hop_length // 4,
            config=model_config.generator,
        )

    def forward(self, texts, text_lengths, alignment, pitch, energy, audio_gt=None):
        text_encoding, _, _ = self.text_encoder(texts, text_lengths)
        style = self.style_encoder(text_encoding, text_lengths)
        alignment = alignment.repeat_interleave(4, dim=2)
        pitch = self.upsampler(pitch.unsqueeze(1)).squeeze(1)
        energy = self.upsampler(energy.unsqueeze(1)).squeeze(1)

        x, _ = self.decoder(
            text_encoding @ alignment,
            pitch,
            energy,
            style,
        )
        cond = style.unsqueeze(-1)
        z_text, mean_text, logstd_text = self.prior_encoder(x)
        z_text2mel, mean_text2mel, logstd_text2mel = self.flow(
            z_text, mean_text, logstd_text, 1, cond, reverse=True
        )

        if audio_gt is not None:
            z_mel, mean_mel, logstd_mel = self.posterior_encoder(audio_gt, cond)
            z_mel2text, mean_mel2text, logstd_mel2text = self.flow(
                z_mel, mean_mel, logstd_mel, 1, cond, reverse=False
            )
            mel = self.post_flow(z_mel.mT).mT
        else:
            mel = self.post_flow(z_text2mel.mT).mT

        prediction: DecoderPrediction = self.generator(
            mel=mel,
            style=style,
            pitch=pitch,
            energy=energy,
        )

        if audio_gt is not None:
            prediction.text_stats = FlowStatistics(z_text, mean_text, logstd_text)
            prediction.text2mel_stats = FlowStatistics(
                z_text2mel, mean_text2mel, logstd_text2mel
            )
            prediction.mel_stats = FlowStatistics(z_mel, mean_mel, logstd_mel)
            prediction.mel2text_stats = FlowStatistics(
                z_mel2text, mean_mel2text, logstd_mel2text
            )
        return prediction


class HubertSpeechPredictor(torch.nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.phone_encoder = HubertEncoder(model_config)

        self.style_encoder = torch.nn.Sequential(
            torch.nn.Linear(
                model_config.speaker_embedder.hidden_dim, model_config.style_dim * 4
            ),
            torch.nn.Mish(),
            torch.nn.Dropout(0.25),
            torch.nn.Linear(model_config.style_dim * 4, model_config.style_dim * 2),
            torch.nn.Mish(),
            torch.nn.Dropout(0.25),
            torch.nn.Linear(model_config.style_dim * 2, model_config.style_dim),
        )

        self.decoder = Decoder(
            dim_in=model_config.inter_dim,
            style_dim=model_config.style_dim,
            dim_out=model_config.generator.input_dim,
            hidden_dim=model_config.decoder.hidden_dim,
            residual_dim=model_config.decoder.residual_dim,
        )

        flow_hidden_dim = model_config.decoder.hidden_dim // 4
        self.prior_encoder = PriorEncoder(
            model_config.decoder.hidden_dim, flow_hidden_dim
        )
        self.posterior_encoder = PosteriorEncoder(
            flow_hidden_dim,
            flow_hidden_dim,
            3,
            1,
            n_layers=12,
            n_fft=model_config.n_fft,
            win_length=model_config.win_length,
            hop_length=model_config.hop_length // 4,
            gin_channels=model_config.style_dim,
        )
        self.flow = ResidualCouplingBlock(
            flow_hidden_dim,
            flow_hidden_dim,
            5,
            1,
            n_layers=4,
            n_flows=8,
            gin_channels=model_config.style_dim,
            use_transformer_flow=False,
        )
        self.post_flow = torch.nn.Linear(
            flow_hidden_dim, model_config.decoder.hidden_dim
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
            hop_length=model_config.hop_length // 4,
            config=model_config.generator,
        )
        self.upsampler = torch.nn.Upsample(scale_factor=4, mode="linear")

    def forward(self, phones, phone_lengths, spk_emb, pitch, energy, audio_gt=None):
        phones = self.phone_encoder(
            phones.repeat_interleave(4, dim=2), phone_lengths * 4
        )
        style = self.style_encoder(spk_emb)
        pitch = self.upsampler(pitch.unsqueeze(1)).squeeze(1)
        energy = self.upsampler(energy.unsqueeze(1)).squeeze(1)
        x, _ = self.decoder(
            phones,
            pitch,
            energy,
            style,
        )
        cond = style.unsqueeze(-1)
        z_text, mean_text, logstd_text = self.prior_encoder(x)
        z_text2mel, mean_text2mel, logstd_text2mel = self.flow(
            z_text, mean_text, logstd_text, 1, cond, reverse=True
        )

        if audio_gt is not None:
            z_mel, mean_mel, logstd_mel = self.posterior_encoder(audio_gt, cond)
            z_mel2text, mean_mel2text, logstd_mel2text = self.flow(
                z_mel, mean_mel, logstd_mel, 1, cond, reverse=False
            )
            mel = self.post_flow(z_mel.mT).mT
        else:
            mel = self.post_flow(z_text2mel.mT).mT

        prediction: DecoderPrediction = self.generator(
            mel=mel,
            style=style,
            pitch=pitch,
            energy=energy,
        )

        if audio_gt is not None:
            prediction.text_stats = FlowStatistics(z_text, mean_text, logstd_text)
            prediction.text2mel_stats = FlowStatistics(
                z_text2mel, mean_text2mel, logstd_text2mel
            )
            prediction.mel_stats = FlowStatistics(z_mel, mean_mel, logstd_mel)
            prediction.mel2text_stats = FlowStatistics(
                z_mel2text, mean_mel2text, logstd_mel2text
            )
        return prediction
