import torch
import torch.nn as nn
from utils import length_to_mask
from stylish_lib.config_loader import ModelConfig
from stylish_lib.text_utils import TextCleaner
from models.export_model import ExportModel
from models.stft import STFT
from utils import length_to_mask
import onnx
from collections import defaultdict
from prettytable import PrettyTable


def add_meta_data_onnx(filename, key, value):
    model = onnx.load(filename)
    meta = model.metadata_props.add()
    meta.key = key
    meta.value = value
    onnx.save(model, filename)


def count_parameters(model):
    table = PrettyTable(["Module", "Parameters (M)"])
    summary = defaultdict(float)
    total_params = 0

    for name, parameter in model.named_parameters():
        module = ".".join(name.split(".")[:2])
        summary[module] += parameter.numel() / 1_000_000
        total_params += parameter.numel() / 1_000_000

    for module, params in summary.items():
        table.add_row([module, f"{params:.3}M"])

    print(table)
    print(f"Total Trainable Params: {total_params:,.2f}M")
    return total_params


def convert_to_onnx(model_config: ModelConfig, out_dir, model_in, device):
    text_cleaner = TextCleaner(model_config.symbol)
    model = ExportModel(**model_in, device=device).eval()
    stft = STFT(
        filter_length=model.generator.gen_istft_n_fft,
        hop_length=model.generator.gen_istft_hop_size,
        win_length=model.generator.gen_istft_n_fft,
    )
    model.generator.stft = stft.to(device).eval()
    print("The model's inference-only modules")
    count_parameters(model)

    tokens = (
        torch.tensor(
            text_cleaner(
                "ðˈiːz wˈɜː tˈuː hˈæv ˈæn ɪnˈɔːɹməs ˈɪmpækt , nˈɑːt ˈoʊnliː bɪkˈɔz ðˈeɪ wˈɜː əsˈoʊsiːˌeɪtᵻd wˈɪð kˈɑːnstəntˌiːn ,"
            )
        )
        .unsqueeze(0)
        .to(device)
    )
    texts = torch.zeros([1, tokens.shape[1] + 2], dtype=int).to(device)
    texts[0, 1 : tokens.shape[1] + 1] = tokens
    text_lengths = torch.zeros([1], dtype=int).to(device)
    text_lengths[0] = tokens.shape[1] + 2
    text_mask = length_to_mask(text_lengths)

    filename = f"{out_dir}/stylish.onnx"
    inputs = (texts, text_lengths)
    with torch.no_grad():
        torch.onnx.export(
            model,
            inputs,
            opset_version=14,
            f=filename,
            input_names=["texts", "text_lengths"],
            output_names=["waveform"],
            dynamic_axes={
                "texts": {1: "num_token"},
                "waveform": {0: "num_samples"},
            },
        )
    add_meta_data_onnx(filename, "model_config", model_config.model_dump_json())
    return filename
