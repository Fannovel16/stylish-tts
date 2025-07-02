# coding:utf-8

from stylish_lib.config_loader import ModelConfig


from .text_aligner import tdnn_blstm_ctc_model_base

from .discriminators.multi_period import MultiPeriodDiscriminator
from .discriminators.multi_resolution import MultiResolutionDiscriminator
from .discriminators.multi_subband import MultiScaleSubbandCQTDiscriminator

from .duration_predictor import DurationPredictor
from .pitch_energy_predictor import PitchEnergyPredictor

from .feature_extractor import TextFeatureExtractor, HubertFeatureExtractor
from .decoder import Decoder
from .ringformer import RingformerGenerator
import torch.nn as nn

from munch import Munch

import logging

logger = logging.getLogger(__name__)


def build_model(model_config: ModelConfig):
    text_aligner = tdnn_blstm_ctc_model_base(model_config.n_mels, model_config.tokens)
    assert model_config.generator.type in [
        "ringformer",
    ], "Decoder type unknown"

    if model_config.generator.type == "ringformer":
        acoustic_config = model_config.text_acoustic_extractor
        text_acoustic_extractor = TextFeatureExtractor(
            tokens=model_config.tokens,
            inter_dim=acoustic_config.inter_dim,
            style_dim=acoustic_config.style_dim,
            text_encoder_config=acoustic_config.text_encoder,
            style_encoder_config=acoustic_config.style_encoder,
            feature_encoder_config=acoustic_config.feature_encoder,
            encode_feature=False,
        )
        mel_decoder = Decoder(
            dim_in=acoustic_config.inter_dim,
            style_dim=acoustic_config.style_dim,
            dim_out=model_config.generator.upsample_initial_channel,
            hidden_dim=model_config.decoder.hidden_dim,
            residual_dim=model_config.decoder.residual_dim,
        )
        generator = RingformerGenerator(
            style_dim=acoustic_config.style_dim,
            resblock_kernel_sizes=model_config.generator.resblock_kernel_sizes,
            upsample_rates=model_config.generator.upsample_rates,
            upsample_initial_channel=model_config.generator.upsample_initial_channel,
            resblock_dilation_sizes=model_config.generator.resblock_dilation_sizes,
            upsample_kernel_sizes=model_config.generator.upsample_kernel_sizes,
            gen_istft_n_fft=model_config.generator.gen_istft_n_fft,
            gen_istft_hop_size=model_config.generator.gen_istft_hop_size,
            sample_rate=model_config.sample_rate,
            mel_decoder=mel_decoder,
        )
        hubert_acoustic_extractor = HubertFeatureExtractor(
            hubert_dim=model_config.hubert.hidden_dim,
            inter_dim=acoustic_config.inter_dim,
            style_dim=acoustic_config.style_dim,
            text_encoder_config=acoustic_config.text_encoder,
            style_encoder_config=acoustic_config.style_encoder,
            feature_encoder_config=acoustic_config.feature_encoder,
            encode_feature=False,
        )

    duration_config = model_config.text_duration_extractor
    text_duration_extractor = TextFeatureExtractor(
        tokens=model_config.tokens,
        inter_dim=duration_config.inter_dim,
        style_dim=duration_config.style_dim,
        text_encoder_config=duration_config.text_encoder,
        style_encoder_config=duration_config.style_encoder,
        feature_encoder_config=duration_config.feature_encoder,
        encode_feature=True,
    )
    duration_predictor = DurationPredictor(
        inter_dim=duration_config.inter_dim,
        style_dim=duration_config.style_dim,
        max_dur=model_config.duration_predictor.max_dur,
    )

    spectral_config = model_config.text_spectral_extractor
    text_spectral_extractor = TextFeatureExtractor(
        tokens=model_config.tokens,
        inter_dim=spectral_config.inter_dim,
        style_dim=spectral_config.style_dim,
        text_encoder_config=spectral_config.text_encoder,
        style_encoder_config=spectral_config.style_encoder,
        feature_encoder_config=spectral_config.feature_encoder,
        encode_feature=True,
    )
    pitch_energy_predictor = PitchEnergyPredictor(
        style_dim=spectral_config.style_dim,
        d_hid=spectral_config.inter_dim,
        dropout=model_config.pitch_energy_predictor.dropout,
    )

    hubert_spectral_extractor = HubertFeatureExtractor(
        hubert_dim=model_config.hubert.hidden_dim,
        inter_dim=spectral_config.inter_dim,
        style_dim=spectral_config.style_dim,
        text_encoder_config=spectral_config.text_encoder,
        style_encoder_config=spectral_config.style_encoder,
        feature_encoder_config=spectral_config.feature_encoder,
        encode_feature=True,
        f0=False,
    )

    nets = Munch(
        text_acoustic_extractor=text_acoustic_extractor,
        text_duration_extractor=text_duration_extractor,
        text_spectral_extractor=text_spectral_extractor,
        duration_predictor=duration_predictor,
        pitch_energy_predictor=pitch_energy_predictor,
        generator=generator,
        text_aligner=text_aligner,
        mpd=MultiPeriodDiscriminator(),
        msbd=MultiScaleSubbandCQTDiscriminator(sample_rate=model_config.sample_rate),
        mrd=MultiResolutionDiscriminator(),
        hubert_acoustic_extractor=hubert_acoustic_extractor,
        hubert_spectral_extractor=hubert_spectral_extractor,
    )

    return nets
