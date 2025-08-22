from stylish_lib.config_loader import Config, ModelConfig
from batch_manager import BatchManager
from typing import Optional, Any
import os.path as osp
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
import logging
from torch.utils.data import DataLoader
from losses import (
    GeneratorLoss,
    DiscriminatorLoss,
    WavLMLoss,
    MultiResolutionSTFTLoss,
    CTCLossWithLabelPriors,
    MagPhaseLoss,
)
from torch.utils.tensorboard.writer import SummaryWriter
from stylish_lib.text_utils import TextCleaner
import torchaudio, torch
import torch.nn as nn
from transformers import HubertModel
from models.vevo.vevo_utils import build_vevo_inference_pipeline
from transformers import AutoFeatureExtractor, WhisperModel
from models.vevo_token_predictor import ByteTokenizer


class Manifest:
    def __init__(self) -> None:
        self.current_epoch: int = 0
        self.current_step: int = 0
        self.steps_per_epoch: int = 0
        self.current_total_step: int = 0
        self.total_trained_audio_seconds: float = 0.0
        self.stage: str = "first"
        self.best_loss: float = float("inf")
        self.training_log: list = []

    def state_dict(self) -> dict:
        return self.__dict__.copy()

    def load_state_dict(self, state: dict) -> None:
        for key, value in state.items():
            if hasattr(self, key):
                setattr(self, key, value)


class HubertModelWithFinalProj(HubertModel):
    def __init__(self, config):
        super().__init__(config)
        self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)


class AdaptiveHubert(nn.Module):
    def __init__(self, hubert_path: str, model_sr: int, hubert_sr: int):
        super().__init__()
        self.model = HubertModelWithFinalProj.from_pretrained(hubert_path)
        self.sr = hubert_sr
        self.resample = torchaudio.transforms.Resample(model_sr, hubert_sr)

    def forward(self, wave, time_dim):
        xs = []
        wave = self.resample(wave)
        for i in range(wave.shape[0]):
            audio = wave[i : i + 1, :]  # (1, time)

            if audio.shape[1] >= self.sr * 5:
                # Split the audio into two halves
                mid = audio.shape[1] // 2
                segments = [audio[:, :mid], audio[:, mid:]]
                segment_outputs = []

                for segment in segments:
                    x = self.model(segment)["last_hidden_state"].transpose(-1, -2)
                    x = torch.nn.functional.interpolate(
                        x,
                        size=time_dim // 2,
                        mode="nearest",
                    ).transpose(-1, -2)
                    segment_outputs.append(x)

                # Concatenate the two halves along the time dimension
                x = torch.cat(segment_outputs, dim=1)

            else:
                x = self.model(audio)["last_hidden_state"].transpose(-1, -2)
                x = torch.nn.functional.interpolate(
                    x,
                    size=time_dim,
                    mode="nearest",
                ).transpose(-1, -2)

            xs.append(x)

        return torch.cat(xs, 0)


"""class AdaptiveQuantizedHubert(nn.Module):
    def __init__(self, device, model_sr: int, hubert_sr: int):
        super().__init__()
        self.vevo = build_vevo_inference_pipeline(device)
        self.sr = hubert_sr
        self.resample = torchaudio.transforms.Resample(model_sr, hubert_sr).to(device)

    def forward(self, wave, time_dim):
        xs, all_codes = [], []
        wave = self.resample(wave)
        for i in range(wave.shape[0]):
            audio = wave[i : i + 1, :]  # (1, time)

            if audio.shape[1] >= self.sr * 5:
                # Split the audio into two halves
                mid = audio.shape[1] // 2
                segments = [audio[:, :mid], audio[:, mid:]]
                segment_outputs = []
                code_outputs = []

                for segment in segments:
                    x, _codes = self.vevo.extract_hubert_quantized(
                        self.vevo.content_tokenizer,
                        segment,
                        token_type=self.vevo.ar_cfg.model.vc_input_token_type,
                    )
                    x = torch.nn.functional.interpolate(
                        x.transpose(-1, -2),
                        size=time_dim // 2,
                        mode="nearest",
                    ).transpose(-1, -2)
                    segment_outputs.append(x)
                    code_outputs.append(_codes)

                # Concatenate the two halves along the time dimension
                x = torch.cat(segment_outputs, dim=1)
                codes = torch.cat(code_outputs, dim=1)

            else:
                x, codes = self.vevo.extract_hubert_quantized(
                    self.vevo.content_tokenizer,
                    audio,
                    token_type=self.vevo.ar_cfg.model.vc_input_token_type,
                )
                x = torch.nn.functional.interpolate(
                    x.transpose(-1, -2),
                    size=time_dim,
                    mode="nearest",
                ).transpose(-1, -2)

            xs.append(x)
            all_codes.append(codes)

        return torch.cat(xs, 0), torch.cat(all_codes, 0)"""


"""class AdaptiveWhisperEncoder(nn.Module):
    def __init__(self, whisper_name, model_sr, device):
        super().__init__()
        self.device = device
        self.whisper_model = (
            WhisperModel.from_pretrained(whisper_name).to(self.device).eval()
        )
        del self.whisper_model.decoder
        self.whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(
            whisper_name
        )
        self.resample = torchaudio.transforms.Resample(model_sr, 16000).to(self.device)

    def forward(self, wave, time_dim):
        wave = self.resample(wave)
        all_input_features = []
        for i in range(wave.shape[0]):
            audio = wave[i, :]
            inputs = self.whisper_feature_extractor(
                [audio.cpu().numpy()],
                return_tensors="pt",
                return_attention_mask=True,
                sampling_rate=16000,
            )
            input_features = self.whisper_model._mask_input_features(
                inputs.input_features, attention_mask=inputs.attention_mask
            ).to(self.device)
            all_input_features.append(input_features)
        all_input_features = torch.cat(all_input_features, 0)
        outputs = self.whisper_model.encoder(
            all_input_features.to(self.whisper_model.encoder.dtype),
            head_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        all_features = outputs.last_hidden_state.to(torch.float32)
        all_features = all_features[:, : audio.size(-1) // 320 + 1]
        all_features = torch.nn.functional.interpolate(
            all_features.transpose(-1, -2),
            size=time_dim,
            mode="nearest",
        ).transpose(-1, -2)
        return all_features"""


class TrainContext:
    def __init__(
        self,
        stage_name: str,
        base_out_dir: str,
        config: Config,
        model_config: ModelConfig,
        logger: logging.Logger,
    ) -> None:
        import stage

        self.base_output_dir: str = base_out_dir
        self.out_dir: str = ""
        self.reset_out_dir(stage_name)
        self.config: Config = config
        self.model_config: ModelConfig = model_config
        self.batch_manager: Optional[BatchManager] = None
        self.stage: Optional[stage.Stage] = None
        self.manifest: Manifest = Manifest()
        self.writer: Optional[SummaryWriter] = None

        ddp_kwargs = DistributedDataParallelKwargs(
            broadcast_buffers=False, find_unused_parameters=True
        )
        self.accelerator = Accelerator(
            project_dir=self.base_output_dir,
            split_batches=True,
            kwargs_handlers=[ddp_kwargs],
            mixed_precision=self.config.training.mixed_precision,
            step_scheduler_with_optimizer=False,
        )
        self.accelerator.even_batches = False

        if self.accelerator.is_main_process:
            self.writer = SummaryWriter(self.out_dir + "/tensorboard")

        # TODO Replace these with json files, pickling is bad
        self.accelerator.register_for_checkpointing(self.config)
        self.accelerator.register_for_checkpointing(self.model_config)
        self.accelerator.register_for_checkpointing(self.manifest)

        self.val_dataloader: Optional[DataLoader] = None

        self.model: Optional[Any] = None

        self.logger: logging.Logger = logger

        # Losses
        self.generator_loss: Optional[GeneratorLoss] = None  # Generator Loss
        self.discriminator_loss: Optional[DiscriminatorLoss] = (
            None  # Discriminator Loss
        )
        self.wavlm_loss: Optional[WavLMLoss] = None  # WavLM Loss
        self.stft_loss: MultiResolutionSTFTLoss = MultiResolutionSTFTLoss(
            sample_rate=self.model_config.sample_rate
        ).to(self.config.training.device)
        self.align_loss: CTCLossWithLabelPriors = CTCLossWithLabelPriors(
            prior_scaling_factor=0.3, blank=model_config.tokens
        )
        self.magphase_loss: MagPhaseLoss = MagPhaseLoss(
            n_fft=self.model_config.generator.gen_istft_n_fft,
            hop_length=self.model_config.generator.gen_istft_hop_size,
        ).to(self.config.training.device)

        self.text_cleaner = TextCleaner(self.model_config.symbol)

        self.to_mel = torchaudio.transforms.MelSpectrogram(
            n_mels=self.model_config.n_mels,
            n_fft=self.model_config.n_fft,
            win_length=self.model_config.win_length,
            hop_length=self.model_config.hop_length,
            sample_rate=self.model_config.sample_rate,
        ).to(self.config.training.device)

        hubert_config = self.model_config.hubert
        self.hubert = (
            AdaptiveHubert(
                hubert_config.model,
                self.model_config.sample_rate,
                hubert_config.sr,
            )
            .to(self.config.training.device)
            .eval()
        )
        # with self.accelerator.main_process_first():
        #     self.hubert = AdaptiveQuantizedHubert(
        #         self.config.training.device,
        #         self.model_config.sample_rate,
        #         hubert_config.sr,
        #     )
        self.byte_tokenizer = ByteTokenizer
        # with self.accelerator.main_process_first():
        #     self.whisper = AdaptiveWhisperEncoder(
        #         "openai/whisper-small",
        #         self.model_config.sample_rate,
        #         self.config.training.device,
        #     )

    def reset_out_dir(self, stage_name):
        self.out_dir = osp.join(self.base_output_dir, stage_name)
