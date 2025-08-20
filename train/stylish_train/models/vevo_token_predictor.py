from transformers import T5Config, ByT5Tokenizer, T5ForConditionalGeneration
from dataclasses import dataclass
from transformers import AutoTokenizer
import torch


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
    return f"<PH{i}>"


def build_model_tokenizer(
    num_vevo_tokens=32,
    num_decoder_layers=2,
    num_layers=2,
    d_model=512,
    d_kv=64,
    d_ff=512,
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
    return byt5_model, byt5_tokenzier
