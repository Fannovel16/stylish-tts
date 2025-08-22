from transformers import T5Config, ByT5Tokenizer, T5ForConditionalGeneration
from dataclasses import dataclass
from transformers import AutoTokenizer
import torch
from .conformer import Conformer
from .text_aligner import CTCModel
from utils import sequence_mask
from misaki.vi import ViCleaner


@dataclass
class DataCollatorWithPadding:
    tokenizer: AutoTokenizer
    device: torch.device
    padding = True

    def __call__(self, features):
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        words = [feature["input_ids"] for feature in features]
        prons = [feature["labels"] for feature in features]

        batch = self.tokenizer(
            words,
            padding=self.padding,
            add_special_tokens=False,
            return_attention_mask=True,
            return_tensors="pt",
        )
        pron_batch = self.tokenizer(
            prons,
            padding=self.padding,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        # replace padding with -100 to ignore loss correctly
        batch["labels"] = pron_batch["input_ids"].masked_fill(
            pron_batch.attention_mask.ne(1), -100
        )

        return {k: v.to(self.device) for k, v in batch.items()}


def phoneme_idx_to_token(i):
    return f"<{i}>"


import torch
from torch.nn.utils.rnn import pad_sequence


class ByteTokenizer:
    """
    ByT5 tokenizer without special tokens
    """

    vocab_size = 256
    vi_cleaner = ViCleaner()

    @classmethod
    def encode(cls, text):
        return tuple(cls.vi_cleaner.clean_text(text.lower()).encode("utf-8"))

    @classmethod
    def decode(cls, tokens):
        return bytes(tokens).decode("utf-8", errors="replace").lower()

    @classmethod
    def batch_encode(cls, texts):
        input_ids, input_lengths = [], []
        for text in texts:
            token_ids = cls.encode(text)
            input_ids.append(torch.tensor(token_ids, dtype=torch.long))
            input_lengths.append(len(token_ids))
        return pad_sequence(input_ids, batch_first=True), torch.tensor(
            input_lengths, dtype=torch.long
        )

    @classmethod
    def batch_decode(cls, input_ids, input_lengths):
        texts = []
        for input_ids, input_length in zip(input_ids, input_lengths):
            texts.append(cls.decode(input_ids[:input_length]))
        return texts


class ConformerTextEncoder(torch.nn.Module):
    def __init__(self, num_tokens, hidden_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_tokens, hidden_dim)
        self.model = Conformer(
            dim=hidden_dim,
            depth=4,
            heads=4,
            dim_head=hidden_dim // 4,
            conv_kernel_size=21,
        )

    def forward(self, input, lengths):
        x = self.embedding(input).transpose(-1, -2)
        x_mask = sequence_mask(lengths, x.size(2)).to(x.dtype)
        x = self.model(x.transpose(-1, -2), x_mask)
        return x, None


def build_model(num_tokens=ByteTokenizer.vocab_size, hidden_dim=128, num_classes=32):
    model = CTCModel(
        encoder=ConformerTextEncoder(num_tokens, hidden_dim),
        encoder_output_layer=torch.nn.Linear(hidden_dim, num_classes),
        n_token=num_classes,
        n_mels=num_tokens,
    )
    return model


"""def build_tokenizer_model(
    num_vevo_tokens=32,
    num_decoder_layers=4,
    num_layers=4 * 2,
    d_model=256,
    d_kv=64,
    d_ff=256 * 4,
):
    byt5_tokenzier = ByT5Tokenizer.from_pretrained("google/byt5-small")
    byt5_tokenzier.add_tokens([phoneme_idx_to_token(i) for i in range(num_vevo_tokens)])
    byt5_config = T5Config.from_pretrained("google/byt5-small")
    byt5_config.vocab_size = len(byt5_tokenzier)
    byt5_config.num_decoder_layers = num_decoder_layers
    byt5_config.num_layers = num_layers
    byt5_config.d_model = d_model
    byt5_config.d_kv = d_kv
    byt5_config.d_ff = d_ff
    byt5_model = T5ForConditionalGeneration(byt5_config)
    return byt5_tokenzier, byt5_model"""
