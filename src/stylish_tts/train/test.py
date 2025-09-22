import os, sys
import os.path as osp
import click
import importlib.resources

sys.path.insert(0, osp.realpath(osp.join(__file__, "../../..")))
from stylish_tts.lib.config_loader import load_config_yaml, load_model_config_yaml
import stylish_tts.train.config as config
import logging
from stylish_tts.train.models.models import build_model
from prettytable import PrettyTable
from collections import defaultdict
import torch
from time import perf_counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_config(config_path):
    if osp.exists(config_path):
        config = load_config_yaml(config_path)
    else:
        # TODO: we may be able to pull it out of the model if a model is passed in instead
        logger.error(f"Config file not found at {config_path}")
        exit(1)
    return config


def get_model_config(model_config_path):
    if len(model_config_path) == 0:
        path = importlib.resources.files(config) / "model.yml"
        f_model = path.open("r", encoding="utf-8")
    else:
        if osp.exists(model_config_path):
            f_model = open(model_config_path, "r", encoding="utf-8")
        else:
            logger.error(f"Config file not found at {model_config_path}")
            exit(1)
    result = load_model_config_yaml(f_model)
    f_model.close()
    return result


def count_parameters(model):
    table = PrettyTable(["Module", "Parameters (M)"])
    summary = defaultdict(float)
    total_params = 0

    for name, parameter in model.named_parameters():
        module = ".".join(name.split(".")[:1])
        summary[module] += parameter.numel() / 1_000_000
        total_params += parameter.numel() / 1_000_000

    for module, params in summary.items():
        table.add_row([module, f"{params:.3}M"])

    print(table)
    print(f"Total Trainable Params: {total_params:,.2f}M")
    return total_params


model_config = get_model_config("")
model = build_model(model_config)


class Model(torch.nn.Module):
    def __init__(self, **args):
        super().__init__()
        self.models = torch.nn.ModuleDict(args)

    def forward(self, text, text_length, alignment):
        pe_text_encoding, _, _ = self.models.pe_text_encoder(text, text_length)
        pe_text_style = self.models.pe_text_style_encoder(pe_text_encoding, text_length)
        pred_pitch, pred_energy = self.models.pitch_energy_predictor(
            pe_text_encoding, text_length, alignment, pe_text_style
        )
        return self.models.speech_predictor(
            text, text_length, alignment, pred_pitch, pred_energy
        )


model = Model(
    speech_predictor=model.speech_predictor,
    duration_predictor=model.duration_predictor,
    pitch_energy_predictor=model.pitch_energy_predictor,
    pe_text_encoder=model.pe_text_encoder,
    pe_text_style_encoder=model.pe_text_style_encoder,
)
from stylish_tts.train.models.cfm_mel_decoder import CfmMelDecoder

model = CfmMelDecoder(hidden_dim=512)
count_parameters(model)
b, t, c, ph = 1, 10 * 80, 768, 100
import numpy as np
import psutil

pid = os.getpid()
python_process = psutil.Process(pid)
memory_use = python_process.memory_info()[0] / 2.0**30
args = (
    torch.rand(b, c, t),
    torch.rand(b, t),
    torch.rand(b, t) * 500,
    torch.rand(b, 10000),
    1,
    0.3,
)
print("memory use:", memory_use)
ts = []
for _ in range(10):
    start = perf_counter()
    model(*args)
    memory_use = python_process.memory_info()[0] / 2.0**30
    print("memory use:", memory_use)
    ts.append(perf_counter() - start)
print(np.mean(ts) / b)
