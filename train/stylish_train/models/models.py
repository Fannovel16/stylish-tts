import torch
from stylish_lib.config_loader import ModelConfig

from .text_aligner import tdnn_blstm_ctc_model_base

from .discriminator import MultiResolutionDiscriminator

from .duration_predictor import DurationPredictor
from .pitch_energy_predictor import PitchEnergyPredictor

from .text_encoder import TextEncoder
from .text_style_encoder import TextStyleEncoder
from .mel_style_encoder import MelStyleEncoder
from .pitch_energy_predictor import PitchEnergyPredictor, HubertPitchEnergyPredictor
from .speech_predictor import SpeechPredictor, HubertSpeechPredictor
from multi_spectrogram import multi_spectrogram_count

from munch import Munch

import logging

logger = logging.getLogger(__name__)


def build_model(model_config: ModelConfig):
    text_aligner = tdnn_blstm_ctc_model_base(
        model_config.n_mels, model_config.text_encoder.tokens
    )

    duration_predictor = DurationPredictor(
        style_dim=model_config.style_dim,
        inter_dim=model_config.inter_dim,
        text_config=model_config.text_encoder,
        style_config=model_config.style_encoder,
        duration_config=model_config.duration_predictor,
    )

    pitch_energy_predictor = PitchEnergyPredictor(
        style_dim=model_config.style_dim,
        inter_dim=model_config.pitch_energy_predictor.inter_dim,
        text_config=model_config.text_encoder,
        style_config=model_config.style_encoder,
        duration_config=model_config.duration_predictor,
        pitch_energy_config=model_config.pitch_energy_predictor,
    )

    pe_text_encoder = TextEncoder(
        inter_dim=model_config.pitch_energy_predictor.inter_dim,
        config=model_config.text_encoder,
    )
    pe_text_style_encoder = TextStyleEncoder(
        model_config.pitch_energy_predictor.inter_dim,
        model_config.style_dim,
        model_config.style_encoder,
    )
    pe_mel_style_encoder = MelStyleEncoder(
        model_config.n_mels,
        model_config.style_dim,
        model_config.mel_style_encoder.max_channels,
        model_config.mel_style_encoder.skip_downsample,
    )

    nets = Munch(
        text_aligner=text_aligner,
        duration_predictor=duration_predictor,
        pitch_energy_predictor=pitch_energy_predictor,
        speech_predictor=SpeechPredictor(model_config),
        mrd=MultiResolutionDiscriminator(discriminator_count=multi_spectrogram_count),
        pe_text_encoder=pe_text_encoder,
        pe_text_style_encoder=pe_text_style_encoder,
        pe_mel_style_encoder=pe_mel_style_encoder,
        hubert_speech_predictor=HubertSpeechPredictor(model_config),
        hubert_pitch_energy_predictor=HubertPitchEnergyPredictor(
            hubert_dim=model_config.hubert.hidden_dim,
            spk_dim=model_config.speaker_embedder.hidden_dim,
            style_dim=model_config.style_dim,
            inter_dim=model_config.inter_dim,
            style_config=model_config.style_encoder,
            pitch_energy_config=model_config.pitch_energy_predictor,
        ),
    )

    return nets
