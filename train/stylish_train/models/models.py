# coding:utf-8


import math

import safetensors.torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from reformer_pytorch import ReformerLM, Autopadder

from config_loader import ModelConfig


from .text_aligner import tdnn_blstm_ctc_model_base
from .plbert import PLBERT

from .discriminators.multi_period import MultiPeriodDiscriminator
from .discriminators.multi_resolution import MultiResolutionDiscriminator
from .discriminators.multi_subband import MultiScaleSubbandCQTDiscriminator
from .discriminators.multi_stft import MultiScaleSTFTDiscriminator

from .duration_predictor import DurationPredictor
from .pitch_energy_predictor import PitchEnergyPredictor

# from .text_encoder import TextEncoder
from .text_encoder import TextMelGenerator, TextMelClassifier, TextEncoder
from .style_encoder import StyleEncoder
from .decoder.mel_decoder import MelDecoder
from .decoder.freev import FreevGenerator
from .decoder.ringformer import RingformerGenerator

from munch import Munch
import safetensors
from huggingface_hub import hf_hub_download

import logging

logger = logging.getLogger(__name__)


class TextualStyleEncoder(nn.Module):
    """Linear layer with merciful load_state_dict."""

    def __init__(self, sbert_output_dim, style_dim):
        super(TextualStyleEncoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(sbert_output_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, style_dim),
        )

    def forward(self, x):
        return self.model(x)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if strict:
            try:
                return super().load_state_dict(state_dict, strict, assign)
            except:
                logger.warning(
                    "The state dict from the checkpoint of TextualStyleEncoder is not compatible. Ignore this message if the sbert is intentionally changed."
                )


def build_model(model_config: ModelConfig, sbert_output_dim: int):
    bert = PLBERT(
        vocab_size=model_config.text_encoder.n_token,
        **{
            k: v
            for k, v in model_config.plbert.model_dump().items()
            if k not in ["enabled", "path"]
        },
    )

    text_aligner = tdnn_blstm_ctc_model_base(
        model_config.n_mels, model_config.text_encoder.n_token
    )

    assert model_config.decoder.type in [
        # "istftnet",
        # "hifigan",
        "ringformer",
        # "vocos",
        "freev",
    ], "Decoder type unknown"

    if model_config.decoder.type == "ringformer":
        decoder = MelDecoder(
            dim_in=model_config.inter_dim,
            style_dim=model_config.style_dim,
            dim_out=model_config.inter_dim,
        )
        generator = RingformerGenerator(
            style_dim=model_config.style_dim,
            resblock_kernel_sizes=model_config.decoder.resblock_kernel_sizes,
            upsample_rates=model_config.decoder.upsample_rates,
            upsample_initial_channel=model_config.decoder.upsample_initial_channel,
            resblock_dilation_sizes=model_config.decoder.resblock_dilation_sizes,
            upsample_kernel_sizes=model_config.decoder.upsample_kernel_sizes,
            gen_istft_n_fft=model_config.decoder.gen_istft_n_fft,
            gen_istft_hop_size=model_config.decoder.gen_istft_hop_size,
            sample_rate=model_config.sample_rate,
        )
    else:
        decoder = MelDecoder()
        generator = FreevGenerator()

    text_encoder = TextEncoder(
        channels=model_config.inter_dim,
        kernel_size=model_config.text_encoder.kernel_size,
        depth=model_config.text_encoder.n_layer,
        n_symbols=model_config.text_encoder.n_token,
    )

    # text_mel_generator = TextMelGenerator(
    #     dim_in=model_config.n_mels,
    #     hidden_dim=512,
    #     num_heads=16,
    #     num_layers=10,
    # )

    # text_encoder = Autopadder(ReformerLM(
    #     num_tokens = model_config.text_encoder.n_token,
    #     # emb_dim = 128,
    #     dim = model_config.inter_dim,
    #     dim_head = 64,
    #     heads = 16,
    #     depth = 10,
    #     ff_mult = 4,
    #     max_seq_len = 2304,
    #     return_embeddings = True
    # ))
    # text_encoder = TextEncoder(
    #     num_tokens=model_config.text_encoder.n_token,
    #     inter_dim=model_config.inter_dim,
    #     num_heads=8,
    #     num_layers=4,
    # )

    # text_mel_classifier = TextMelClassifier(
    #     inter_dim=model_config.inter_dim, n_mels=model_config.n_mels
    # )

    duration_predictor = DurationPredictor(
        style_dim=model_config.style_dim,
        d_hid=model_config.inter_dim,
        nlayers=model_config.duration_predictor.n_layer,
        max_dur=model_config.duration_predictor.max_dur,
        dropout=model_config.duration_predictor.dropout,
    )

    pitch_energy_predictor = PitchEnergyPredictor(
        style_dim=model_config.style_dim,
        d_hid=model_config.inter_dim,
        dropout=model_config.pitch_energy_predictor.dropout,
    )

    # predictor = ProsodyPredictor(
    #    style_dim=model_config.style_dim,
    #    d_hid=model_config.prosody_predictor.hidden_dim,
    #    nlayers=model_config.prosody_predictor.n_layer,
    #    max_dur=model_config.prosody_predictor.max_dur,
    #    dropout=model_config.prosody_predictor.dropout,
    # )

    style_encoder = StyleEncoder(
        dim_in=model_config.style_encoder.dim_in,
        style_dim=model_config.style_dim,
        max_conv_dim=model_config.style_encoder.hidden_dim,
        skip_downsamples=model_config.style_encoder.skip_downsamples,
    )
    # style_encoder = StyleEncoder(
    #     mel_dim=model_config.n_mels,
    #     style_dim=model_config.style_dim,
    #     # TODO Add config values
    #     hidden_dim=512,
    #     num_heads=8,
    #     num_layers=6,
    # )
    predictor_encoder = StyleEncoder(
        dim_in=model_config.style_encoder.dim_in,
        style_dim=model_config.style_dim,
        max_conv_dim=model_config.style_encoder.hidden_dim,
        skip_downsamples=model_config.style_encoder.skip_downsamples,
    )
    # predictor_encoder = StyleEncoder(
    #     mel_dim=model_config.n_mels,
    #     style_dim=model_config.style_dim,
    #     # TODO Add config values
    #     hidden_dim=512,
    #     num_heads=8,
    #     num_layers=6,
    # )

    nets = Munch(
        bert=bert,
        bert_encoder=nn.Linear(bert.config.hidden_size, model_config.inter_dim),
        # predictor=predictor,
        duration_predictor=duration_predictor,
        pitch_energy_predictor=pitch_energy_predictor,
        decoder=decoder,
        generator=generator,
        text_encoder=text_encoder,
        # text_mel_generator=text_mel_generator,
        # text_mel_classifier=text_mel_classifier,
        textual_prosody_encoder=TextualStyleEncoder(
            sbert_output_dim,  # model_config.embedding_encoder.dim_in,
            model_config.style_dim,
        ),
        textual_style_encoder=TextualStyleEncoder(
            sbert_output_dim,  # model_config.embedding_encoder.dim_in,
            model_config.style_dim,
        ),
        acoustic_prosody_encoder=predictor_encoder,
        acoustic_style_encoder=style_encoder,
        # diffusion=diffusion,
        text_aligner=text_aligner,
        # pitch_extractor=pitch_extractor,
        mpd=MultiPeriodDiscriminator(),
        msbd=MultiScaleSubbandCQTDiscriminator(sample_rate=model_config.sample_rate),
        mrd=MultiResolutionDiscriminator(),
        mstftd=MultiScaleSTFTDiscriminator(),
        # slm discriminator head
        # wd=WavLMDiscriminator(
        #    model_config.slm.hidden,
        #    model_config.slm.nlayers,
        #    model_config.slm.initial_channel,
        # ),
    )

    return nets  # , kdiffusion


def load_defaults(train, model):
    with train.accelerator.main_process_first():
        # Load pretrained text_aligner
        # if train.model_config.n_mels == 80:
        #     params = safetensors.torch.load_file(
        #         hf_hub_download(
        #             repo_id="stylish-tts/text_aligner",
        #             filename="text_aligner.safetensors",
        #         )
        #     )
        #     model.text_aligner.load_state_dict(params)

        # Load pretrained pitch_extractor
        # params = safetensors.torch.load_file(
        # hf_hub_download(
        # repo_id="stylish-tts/pitch_extractor",
        # filename="pitch_extractor.safetensors",
        # )
        # )
        # model.pitch_extractor.load_state_dict(params)

        # Load pretrained PLBERT
        if train.model_config.plbert.enabled:
            params = safetensors.torch.load_file(train.model_config.plbert.path)
            model.bert.load_state_dict(params, strict=False)
