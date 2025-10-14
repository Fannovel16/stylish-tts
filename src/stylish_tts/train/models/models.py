import torch
from stylish_tts.lib.config_loader import ModelConfig

from .text_aligner import tdnn_blstm_ctc_model_base

from .discriminator import MultiResolutionDiscriminator, MultiPeriodDiscriminator

from .duration_predictor import DurationPredictor
from .pitch_energy_predictor import PitchEnergyPredictor, HubertPitchEnergyPredictor

from .text_encoder import TextEncoder
from .text_style_encoder import TextStyleEncoder
from .mel_style_encoder import MelStyleEncoder
from .speech_predictor import SpeechPredictor
from stylish_tts.train.multi_spectrogram import multi_spectrogram_count
from .cfm.cfm_mel_decoder import CfmMelDecoder
from .cfm.cfm_pitch_predictor import CfmPitchPredictor
from .hubert_encoder import HubertEncoder

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

    cfm_mel_decoder = CfmMelDecoder(
        feat_dim=model_config.n_mels,
        asr_dim=model_config.hubert.hidden_dim,
        spk_dim=model_config.speaker_embedder.hidden_dim,
        hidden_dim=model_config.decoder.hidden_dim,
    )

    cfm_pitch_predictor = CfmPitchPredictor(
        asr_dim=model_config.hubert.hidden_dim,
        n_mels=model_config.n_mels,
    )

    hubert_encoder = HubertEncoder(model_config)

    nets = Munch(
        text_aligner=text_aligner,
        duration_predictor=duration_predictor,
        pitch_energy_predictor=pitch_energy_predictor,
        speech_predictor=SpeechPredictor(model_config),
        mrd=MultiResolutionDiscriminator(discriminator_count=multi_spectrogram_count),
        mpd=MultiPeriodDiscriminator((2, 3, 5, 7, 11)),
        pe_text_encoder=pe_text_encoder,
        pe_text_style_encoder=pe_text_style_encoder,
        pe_mel_style_encoder=pe_mel_style_encoder,
        hubert_encoder=hubert_encoder,
        cfm_mel_decoder=cfm_mel_decoder,
        cfm_pitch_predictor=cfm_pitch_predictor,
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
