import torch
import torchaudio
from huggingface_hub import PyTorchModelHubMixin
from torch import Tensor, nn
from torch.nn import functional as F
from torchaudio.models.wav2vec2 import components
from torchaudio.pipelines import HUBERT_BASE

FINETUNING_HUBERT_CONFIG = {
    "encoder_projection_dropout": 0,
    "encoder_attention_dropout": 0,
    "encoder_ff_interm_dropout": 0.1,
    "encoder_dropout": 0,
    "encoder_layer_drop": 0.1,  # In torchaudio: 0.05
    "mask_prob": 0.75,  # In torchaudio: 0.65
    "mask_channel_prob": 0.5,
    "mask_channel_length": 10,  # In torchaudio and fairseq: 64. This is the value for pretraining.
    "num_classes": 500,  # Number of classes during HuBERT pretraining.
}


class Tokenizer:
    # fmt:off
    PHONEMES = {
        "SIL": 0, "AA": 1, "AE": 2, "AH": 3, "AO": 4, "AW": 5, "AY": 6, "B": 7,
        "CH": 8, "D": 9, "DH": 10, "EH": 11, "ER": 12, "EY": 13, "F": 14, "G": 15,
        "HH": 16, "IH": 17, "IY": 18, "JH": 19, "K": 20, "L": 21, "M": 22, "N": 23,
        "NG": 24, "OW": 25, "OY": 26, "P": 27, "R": 28, "S": 29, "SH": 30, "T": 31,
        "TH": 32, "UH": 33, "UW": 34, "V": 35, "W": 36, "Y": 37, "Z": 38, "ZH": 39,
    }
    # fmt:on

    def __init__(self, with_blank: bool = False) -> None:
        self.token_to_id = self.PHONEMES | {"<pad>": self.pad_id}
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.with_blank = with_blank

    @property
    def vocab_size(self) -> int:
        if self.with_blank:
            return len(self.PHONEMES) + 1
        return len(self.PHONEMES)

    @property
    def silence_id(self) -> int:
        return self.PHONEMES["SIL"]

    @property
    def pad_id(self) -> int:
        return len(self.PHONEMES)

    def encode(self, phones: list[str] | str) -> torch.LongTensor:
        if isinstance(phones, str):
            phones = phones.split(" ")
        return torch.LongTensor([self.token_to_id[phone] for phone in phones])

    def decode(self, tokens: list[int]) -> str:
        return " ".join(
            self.id_to_token[int(token)] for token in tokens if token < self.pad_id
        )


class HuBERTPhoneme(nn.Module, PyTorchModelHubMixin):
    def __init__(self, freeze_encoder: bool = True, ctc_training: bool = False) -> None:
        """Initialize the model.

        Parameters
        ----------
        freeze_encoder : bool, optional
            Whether to freeze the Transformer encoder of HuBERT, by default True.
            The convolutional layers are always frozen.
        """
        super().__init__()
        self.model = torchaudio.models.hubert_pretrain_base(**FINETUNING_HUBERT_CONFIG)
        self.model.wav2vec2.load_state_dict(HUBERT_BASE.get_model().state_dict())
        self.aux = nn.Linear(
            HUBERT_BASE._params["encoder_embed_dim"],
            Tokenizer(with_blank=ctc_training).vocab_size,
        )
        self.freeze_encoder = freeze_encoder
        self.ctc_training = ctc_training

    def forward(
        self, waveforms: Tensor, lengths: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        """Extract logits during training, with masking."""
        if self.freeze_encoder:
            with torch.no_grad():
                x, out_len = self.model.wav2vec2.feature_extractor(waveforms, lengths)
                padding_mask = components._get_padding_mask(x, out_len)
                x, attention_mask = self.model.wav2vec2.encoder._preprocess(x, out_len)
                x, _ = self.model.mask_generator(x, padding_mask)
                x = self.model.wav2vec2.encoder.transformer(
                    x, attention_mask=attention_mask
                )
        else:
            with torch.no_grad():
                x, out_len = self.model.wav2vec2.feature_extractor(waveforms, lengths)
                padding_mask = components._get_padding_mask(x, out_len)
            x, attention_mask = self.model.wav2vec2.encoder._preprocess(x, out_len)
            x, _ = self.model.mask_generator(x, padding_mask)
            x = self.model.wav2vec2.encoder.transformer(
                x, attention_mask=attention_mask
            )
        logits = self.aux(x)
        return logits, out_len

    def inference(
        self, waveforms: Tensor, lengths: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        """Extract logits during inference. No masking is applied."""
        x, out_len = self.model.wav2vec2(waveforms, lengths)
        logits = self.aux(x)
        return logits, out_len

    @torch.jit.export
    def extract_features(
        self, waveforms: Tensor, lengths: Tensor | None = None
    ) -> tuple[list[Tensor], Tensor | None]:
        """Extract features from intermediate layers. No masking is applied."""
        x, out_len = self.model.wav2vec2.extract_features(waveforms, lengths)
        x.append(self.aux(x[-1]))
        return x, out_len

    def train(self, mode: bool = True) -> "HuBERTPhoneme":
        """Override the train method to set the encoder in eval mode if it is frozen."""
        if self.freeze_encoder:
            self.model.wav2vec2.eval()
        else:
            self.model.wav2vec2.train(mode)
        self.aux.train(mode)
        return self
