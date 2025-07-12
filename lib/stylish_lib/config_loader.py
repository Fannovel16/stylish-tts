from pydantic import BaseModel, Field
from typing import List, Union, Literal
from pathlib import Path
import yaml
import json
import logging

logger = logging.getLogger(__name__)


######## General Configuration ########


class TrainingConfig(BaseModel):
    """
    Training configuration parameters.
    """

    log_interval: int = Field(..., description="Interval (in steps) for logging.")
    save_interval: int = Field(
        ..., description="Interval (in steps) for saving checkpoints."
    )
    val_interval: int = Field(..., description="Interval (in steps) for validation.")
    device: str = Field(..., description="Computational device (e.g., 'cuda').")
    mixed_precision: str = Field(..., description="accelerator use bf16 or fp16 or no")


class TrainingStageConfig(BaseModel):
    """
    Training stage configuration that defines the number of epochs, probe batch max and learning rate for one stage.
    """

    epochs: int = Field(default=10, description="Number of epochs of this stage.")
    probe_batch_max: int = Field(
        default=32, description="Maximum batch size to attempt during bin probing."
    )
    lr: float = Field(default=1e-4, description="General learning rate.")


class TrainingPlanConfig(BaseModel):
    """
    Training plan configuration for every stage.
    """

    # text_encoder: TrainingStageConfig = Field(
    #     default_factory=TrainingStageConfig,
    #     description="Configuration for pretraining the text encoder model."
    # )
    # vocoder: TrainingStageConfig = Field(
    #     default_factory=TrainingStageConfig,
    #     description="Configuration for the vocoder pretraining stage."
    # )
    text_encoder: TrainingStageConfig = None
    vocoder: TrainingStageConfig = None
    alignment: TrainingStageConfig = Field(
        default_factory=TrainingStageConfig,
        description="Configuration for pretraining the text alignment model.",
    )
    acoustic: TrainingStageConfig = Field(
        default_factory=TrainingStageConfig,
        description="Configuration for joint training of acoustic models stage (second stage).",
    )
    spectral: TrainingStageConfig = Field(
        default_factory=TrainingStageConfig,
        description="Configuration for training of spectral models stage.",
    )
    textual: TrainingStageConfig = Field(
        default_factory=TrainingStageConfig,
        description="Configuration for training of textual models stage.",
    )
    textual_acoustic: TrainingStageConfig = Field(
        default_factory=TrainingStageConfig,
        description="Configuration for training of textual acoustic models stage.",
    )
    pre_hubert_quantizer: TrainingStageConfig = Field(
        default_factory=TrainingStageConfig,
        description="Configuration for pretraining stage of HuBERT Residual VQ.",
    )
    pre_vq_indexer: TrainingStageConfig = Field(
        default_factory=TrainingStageConfig,
        description="Configuration for pretraining stage of HuBERT Codebook Index Predictor.",
    )

    def get_stage(self, name: str) -> TrainingStageConfig:
        try:
            return getattr(self, name)
        except AttributeError:
            raise ValueError(f"Stage '{name}' not found.")


class DatasetConfig(BaseModel):
    """
    Dataset configuration parameters.
    """

    train_data: str = Field(..., description="Path to the training data list.")
    val_data: str = Field(..., description="Path to the validation data list.")
    wav_path: str = Field(..., description="Directory containing WAV files.")
    pitch_path: str = Field(
        ...,
        description="Path to the precomputed pitch safetensor file for your segments.",
    )
    alignment_path: str = Field(
        ...,
        description="Path to the precomputed alignment safetensor file for your segments.",
    )


class LossWeightConfig(BaseModel):
    """
    Loss weight configuration for various loss components.
    """

    mel: float = Field(..., description="Weight for mel spetral convergence loss.")
    generator: float = Field(..., description="Weight for generator loss.")
    slm: float = Field(
        ..., description="Weight for speech-language model feature matching loss."
    )
    pitch: float = Field(..., description="Weight for F0 pitch reconstruction loss.")
    energy: float = Field(..., description="Weight for energy reconstruction loss.")
    duration: float = Field(..., description="Weight for duration loss.")
    duration_ce: float = Field(
        ..., description="Weight for duration cross-entropy loss."
    )
    style: float = Field(..., description="Weight for style reconstruction loss.")
    magphase: float = Field(..., description="Weight for magnitude/phase loss.")
    confidence: float = Field(..., description="Weight for alignment confidence")
    align_loss: float = Field(..., description="Weight for alignment loss")
    discriminator: float = Field(..., description="Weight for discriminator loss")


######## Model Configuration ########


class SymbolConfig(BaseModel):
    """
    Configuration for text processing symbols with validation
    """

    pad: str = Field(..., description="Padding symbol for sequence alignment")
    punctuation: str = Field(..., description="Supported punctuation marks")
    letters: str = Field(..., description="Latin alphabet letters")
    letters_ipa: str = Field(
        ..., description="IPA phonetic characters including diacritics"
    )


class TextAlignerConfig(BaseModel):
    """
    Configuration for the text aligner component.
    """

    hidden_dim: int = Field(
        ..., description="Dimension of the hidden layers in the text aligner."
    )
    token_embedding_dim: int = Field(
        ..., description="Dimension of the token embeddings in the text aligner."
    )


class DecoderConfig(BaseModel):
    """
    Configuration for decoder.
    """

    hidden_dim: int = Field(..., description="Hidden dimension")
    residual_dim: int = Field(..., description="Residual shortcut dimension")


class RingformerGeneratorConfig(BaseModel):
    """
    Configuration for Ringformer generator.
    """

    type: Literal["ringformer"] = "ringformer"
    resblock_kernel_sizes: List[int] = Field(
        ..., description="Kernel sizes for residual blocks."
    )
    upsample_rates: List[int] = Field(
        ..., description="Upsample rates for each upsampling layer."
    )
    upsample_initial_channel: int = Field(
        ..., description="Initial channel count for upsampling."
    )
    resblock_dilation_sizes: List[List[int]] = Field(
        ..., description="Dilation sizes for residual blocks."
    )
    upsample_kernel_sizes: List[int] = Field(
        ..., description="Kernel sizes for the upsampling layers."
    )
    gen_istft_n_fft: int = Field(..., description="FFT size for iSTFT generator.")
    gen_istft_hop_size: int = Field(..., description="Hop size for iSTFT generator.")
    depth: int = Field(..., description="Number of conformer blocks in model")


class ValidationConfig(BaseModel):
    """
    Setup samples to use during validation
    """

    sample_count: int = Field(
        ..., description="Number of samples to generate during validation."
    )
    force_samples: list = Field(
        default=[], description="List of segments to use during validation."
    )


class TextEncoderConfig(BaseModel):
    """
    Text encoder configuration parameters.
    """

    hidden_dim: int = Field(..., description="Hidden dimension")
    filter_channels: int = Field(..., description="Filter Channel for encoder")
    heads: int = Field(..., description="Number of attention heads")
    layers: int = Field(..., description="Number of layers in the text encoder.")
    kernel_size: int = Field(..., description="Kernel size for convolution.")
    dropout: float = Field(..., description="Dropout for internal layers")


class StyleEncoderConfig(BaseModel):
    """
    Style encoder configuration parameters.
    """

    layers: int = Field(..., description="Number of convnext blocks to use")


class AdaptiveFeatureEncoderConfig(BaseModel):
    """
    Adaptive style encoder configuration parameters.
    """

    layers: int = Field(..., description="Number of transformer encoder blocks.")
    dropout: float = Field(..., description="Dropout rate for feature encoder.")
    heads: int = Field(..., description="Number of attention heads")
    kernel_size: int = Field(...)


class TextFeatureExtractorConfig(BaseModel):
    """
    Text feature extractor configuration parameters.
    """

    style_dim: int = Field(..., description="Dimension of the style vector.")
    inter_dim: int = Field(
        ..., description="Dimension of the embedding used between models."
    )
    text_encoder: TextEncoderConfig = Field(
        ..., description="Text encoder configuration parameters."
    )
    style_encoder: StyleEncoderConfig = Field(
        ..., description="Style encoder configuration parameters."
    )
    feature_encoder: AdaptiveFeatureEncoderConfig = Field(
        ..., description="Shared feature encoder configuration parameters for feature."
    )


class DurationPredictorConfig(BaseModel):
    """
    Prosody predictor configuration parameters.
    """

    max_dur: int = Field(..., description="Maximum duration of a single phoneme.")


class PitchEnergyPredictorConfig(BaseModel):
    """
    Prosody predictor configuration parameters.
    """

    dropout: float = Field(..., description="Dropout rate for the prosody predictor.")


class SlmConfig(BaseModel):
    """
    Speech Language Model (SLM) configuration parameters.
    """

    model: str = Field(..., description="Identifier or path for the SLM model.")
    sr: int = Field(..., description="Sampling rate used by the SLM.")


class HubertConfig(BaseModel):
    model: str = Field(..., description="Identifier or path for HuBERT model.")
    hidden_dim: int = Field(
        ..., description="Dimension of the last hidden layer in HuBERT."
    )
    sr: int = Field(..., description="Sampling rate used by HuBERT.")


class HubertQuantizer(BaseModel):
    num_quantizers: int = Field(default=4, description="Number of codebooks.")
    codebook_size: int = Field(default=32, description="Number of rows in a codebook.")
    kmeans_init: bool = Field(
        default=True, description="Initialization with k-means clustering."
    )
    kmeans_iters: int = Field(
        default=10, description="Iteration of k-means clustering."
    )
    decay: float = Field(
        default=0.5,
        description="EMA decay, lower means the dictionary will change faster.",
    )
    commitment_weight: float = Field(default=1.0, description="Commitment weight.")
    quantize_dropout: bool = Field(default=True, description="Enable dropout.")
    quantize_dropout_cutoff_index: int = Field(
        default=1,
        description="Specify which position (0-based) of codebook to start randomly dropout.",
    )
    threshold_ema_dead_code: int = Field(
        default=2,
        description="Should actively replace any codes having EMA cluster size less than.",
    )
    rotation_trick: bool = Field(default=True, description="Rotation trick.")


class Config(BaseModel):
    """
    Top-level configuration model that encompasses all settings.
    """

    training: TrainingConfig = Field(
        ..., description="Training configuration parameters."
    )
    training_plan: TrainingPlanConfig = Field(
        ..., description="Training plan configuration parameters."
    )
    validation: ValidationConfig = Field(
        ..., description="Validation configuration parameters."
    )
    dataset: DatasetConfig = Field(..., description="Dataset configuration parameters.")
    loss_weight: LossWeightConfig = Field(
        ..., description="Loss weight configuration for various loss components."
    )

    def state_dict(self) -> dict:
        return self.model_dump()

    def load_state_dict(self, state: dict) -> None:
        self = self.model_copy(update=state)


class ModelConfig(BaseModel):
    multispeaker: bool = Field(
        ..., description="Indicates if the model supports multispeaker input."
    )
    n_mels: int = Field(..., description="Number of mel frequency bins.")
    sample_rate: int = Field(..., description="Sample rate for audio preprocessing.")
    n_fft: int = Field(..., description="FFT size for spectrogram computation.")
    win_length: int = Field(
        ..., description="Window length for spectrogram computation."
    )
    hop_length: int = Field(..., description="Hop length for spectrogram computation.")
    tokens: int = Field(..., description="Number of phoneme tokens.")

    text_aligner: TextAlignerConfig = Field(
        ..., description="Configuration for the text aligner component."
    )
    decoder: DecoderConfig = Field(..., description="Decoder configuration parameters.")
    generator: Union[RingformerGeneratorConfig,] = Field(
        ..., description="Generator (vocoder) configuration parameters."
    )
    text_acoustic_extractor: TextFeatureExtractorConfig = Field(
        ..., description="Configuration for acoustic branch extractor."
    )
    text_duration_extractor: TextFeatureExtractorConfig = Field(
        ..., description="Configuration for duration branch extractor."
    )
    text_spectral_extractor: TextFeatureExtractorConfig = Field(
        ..., description="Configuration for spectral branch extractor."
    )
    duration_predictor: DurationPredictorConfig = Field(
        ..., description="Duration predictor configuration parameters."
    )
    pitch_energy_predictor: PitchEnergyPredictorConfig = Field(
        ..., description="Pitch/Energy predictor configuration parameters."
    )
    slm: SlmConfig = Field(
        ..., description="Speech Language Model (SLM) configuration parameters."
    )
    hubert: HubertConfig = Field(..., description="HuBERT configuration parameters.")
    hubert_quantizer: HubertQuantizer = Field(
        ...,
        default_factory=HubertQuantizer,
        description="HuBERT Residual VQ parameters.",
    )
    symbol: SymbolConfig = Field(..., description="Text processing symbols.")

    def state_dict(self) -> dict:
        return self.model_dump()

    def load_state_dict(self, state: dict) -> None:
        self = self.model_copy(update=state)


def load_config_yaml(config_path: str) -> Config:
    """
    Load a configuration file from the specified path.

    Args:
        config_path (Path): Path to the configuration file.

    Returns:
        Config: Parsed configuration object.
    """
    path = Path(config_path)
    # Load the YAML file into a dictionary
    with path.open("r", encoding="utf-8") as file:
        config_dict = yaml.safe_load(file)

    # Parse and validate the configuration dictionary
    return Config.model_validate(config_dict)


def load_model_config_yaml(config_path: str) -> ModelConfig:
    """
    Load a configuration file from the specified path.

    Args:
        config_path (Path): Path to the configuration file.

    Returns:
        Config: Parsed configuration object.
    """
    path = Path(config_path)
    # Load the YAML file into a dictionary
    with path.open("r", encoding="utf-8") as file:
        config_dict = yaml.safe_load(file)

    # Parse and validate the configuration dictionary
    return ModelConfig.model_validate(config_dict)


def dump_to_string(config: Config) -> str:
    """
    Dump the configuration object to a string.

    Args:
        config (Config): Configuration object to dump.

    Returns:
        str: JSON string representation of the configuration object.
    """
    return config.model_dump_json()  # Dumps as a compact single-line JSON string


def load_from_string(config_str: str) -> Config:
    """
    Load a configuration object from a JSON string.

    Args:
        config_str (str): JSON string representation of the configuration object.

    Returns:
        Config: Parsed configuration object.
    """
    # Convert JSON string to a dictionary and validate it with the Config model
    return Config.model_validate(json.loads(config_str))


# Example usage:
if __name__ == "__main__":
    # Path to the YAML configuration file
    config_path = Path("Configs/new.config.yml")

    # Load the YAML file into a dictionary
    with config_path.open("r", encoding="utf-8") as file:
        config_dict = yaml.safe_load(file)
    # Parse and validate the configuration dictionary
    config = Config.model_validate(config_dict)

    # For Pydantic v2, use model_dump_json (or use json.dumps(config.model_dump(), indent=2))
    test = dump_to_string(config)
    logging.debug(test)

    # Load a configuration object from a JSON string
    config_loaded = load_from_string(test)
    logging.debug(config_loaded)
