from stylish_tts.train.cli import get_config, get_model_config
from stylish_tts.train.dataprep.align_text import tqdm_wrapper, audio_list
from stylish_tts.train.models.pretrained import (
    AdaptiveMioCodec,
    AdaptiveKanadeCodec,
    AdaptiveVevoCodec,
)
from pathlib import Path
from safetensors.torch import save_file
import torch


def generate_codes(config_path, model_config_path, code_type):
    config = get_config(config_path)
    model_config = get_model_config(model_config_path)

    root = Path(config.dataset.path)
    wavdir = root / config.dataset.wav_path
    if code_type == "kanade":
        model = AdaptiveKanadeCodec(24_000, extract_all=True).cuda().eval()
        val_codes, val_globals = calculate_kanade_codes(
            model, "Val set", root / config.dataset.val_data, wavdir, model_config
        )
        train_codes, train_globals = calculate_kanade_codes(
            model, "Train set", root / config.dataset.train_data, wavdir, model_config
        )
        save_file(val_codes | train_codes, root / "codes.safetensors")
        save_file(val_globals | train_globals, root / "globals.safetensors")
    elif code_type == "vevo":
        model = AdaptiveVevoCodec(24_000).bfloat16().cuda().eval()
        val_codes = calculate_vevo_codes(
            model, "Val set", root / config.dataset.val_data, wavdir, model_config
        )
        train_codes = calculate_vevo_codes(
            model, "Train set", root / config.dataset.train_data, wavdir, model_config
        )
        save_file(val_codes | train_codes, root / "vevo_codes.safetensors")
    else:
        raise NotImplementedError(code_type)


@torch.no_grad()
def calculate_kanade_codes(
    model: AdaptiveKanadeCodec, label, path, wavdir, model_config
):
    codes, globals = {}, {}
    with path.open("r", encoding="utf-8") as f:
        total_segments = sum(1 for _ in f)
        iterator = tqdm_wrapper(
            audio_list(path, wavdir, model_config),
            total=total_segments,
            desc="Processing " + label,
            color="MAGENTA",
        )
        for name, text_raw, wave in iterator:
            wave = torch.from_numpy(wave).float().cuda().unsqueeze(0)
            result = model.encode(wave)
            codes[name], globals[name] = (
                result.content_token_indices.cpu(),
                result.global_embedding.cpu(),
            )
    return codes, globals


@torch.no_grad()
def calculate_vevo_codes(model: AdaptiveVevoCodec, label, path, wavdir, model_config):
    codes = {}
    with path.open("r", encoding="utf-8") as f:
        total_segments = sum(1 for _ in f)
        iterator = tqdm_wrapper(
            audio_list(path, wavdir, model_config),
            total=total_segments,
            desc="Processing " + label,
            color="MAGENTA",
        )
        for name, text_raw, wave in iterator:
            wave = torch.from_numpy(wave).bfloat16().cuda().unsqueeze(0)
            codes[name] = model(wave, 1)[0]
    return codes
