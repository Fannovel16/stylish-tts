from stylish_tts.train.cli import get_config, get_model_config
from stylish_tts.train.dataprep.align_text import tqdm_wrapper, audio_list
from stylish_tts.train.models.pretrained import AdaptiveKanadeCodec
from pathlib import Path
from safetensors.torch import save_file
import torch


def generate_codes(config_path, model_config_path):
    model = AdaptiveKanadeCodec(24_000).cuda().eval()
    config = get_config(config_path)
    model_config = get_model_config(model_config_path)

    root = Path(config.dataset.path)
    wavdir = root / config.dataset.wav_path
    val_codes, val_globals = calculate_codes(
        model, "Val set", root / config.dataset.val_data, wavdir, model_config
    )
    train_codes, train_globals = calculate_codes(
        model, "Train set", root / config.dataset.train_data, wavdir, model_config
    )
    save_file(val_codes | train_codes, root / "codes.safetensors")
    save_file(val_globals | train_globals, root / "globals.safetensors")


@torch.no_grad()
def calculate_codes(model: AdaptiveKanadeCodec, label, path, wavdir, model_config):
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
