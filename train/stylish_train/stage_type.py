import random
from typing import Callable, List, Optional, Tuple
import torch
from torch.nn import functional as F
import torchaudio
from einops import rearrange
from loss_log import LossLog, build_loss_log
from utils import print_gpu_vram, log_norm
from typing import List
from losses import multi_phase_loss


stages = {}


def is_valid_stage(name):
    return name in stages


def valid_stage_list():
    return list(stages.keys())


class StageType:
    def __init__(
        self,
        next_stage: Optional[str],
        train_fn: Callable,
        validate_fn: Callable,
        train_models: List[str],
        eval_models: List[str],
        discriminators: List[str],
        inputs: List[str],
    ):
        self.next_stage: Optional[str] = next_stage
        self.train_fn: Callable = train_fn
        self.validate_fn: Callable = validate_fn
        self.train_models: List[str] = train_models
        self.eval_models: List[str] = eval_models
        self.discriminators = discriminators
        self.inputs: List[str] = inputs


def make_list(tensor: torch.Tensor) -> List[torch.Tensor]:
    result = []
    for i in range(tensor.shape[0]):
        result.append(tensor[i])
    return result


##### Alignment #####


def train_alignment(
    batch, model, train, probing
) -> Tuple[LossLog, Optional[torch.Tensor]]:
    log = build_loss_log(train)
    mel, mel_length = calculate_mel(batch.audio_gt, train.to_align_mel)
    mel = rearrange(mel, "b f t -> b t f")
    ctc, _ = model.text_aligner(mel, mel_length)
    train.stage.optimizer.zero_grad()
    loss_ctc = train.align_loss(
        ctc, batch.text, mel_length, batch.text_length, step_type="train"
    )

    log.add_loss(
        "align_loss",
        loss_ctc,
    )
    train.accelerator.backward(log.backwards_loss())
    return log.detach(), None, None


@torch.no_grad()
def validate_alignment(batch, train):
    log = build_loss_log(train)
    mel, mel_length = calculate_mel(batch.audio_gt, train.to_align_mel)
    mel = rearrange(mel, "b f t -> b t f")
    ctc, _ = train.model.text_aligner(mel, mel_length)
    train.stage.optimizer.zero_grad()

    loss_ctc = train.align_loss(
        ctc, batch.text, mel_length, batch.text_length, step_type="eval"
    )

    blank = train.model_config.text_encoder.tokens
    logprobs = rearrange(ctc, "t b k -> b t k")
    confidence_total = 0.0
    confidence_count = 0
    for i in range(mel.shape[0]):
        _, scores = torchaudio.functional.forced_align(
            log_probs=logprobs[i].unsqueeze(0).contiguous(),
            targets=batch.text[i, : batch.text_length[i].item()].unsqueeze(0),
            input_lengths=mel_length[i].unsqueeze(0),
            target_lengths=batch.text_length[i].unsqueeze(0),
            blank=blank,
        )
        confidence_total += scores.exp().sum()
        confidence_count += scores.shape[-1]
    log.add_loss("confidence", confidence_total / confidence_count)
    log.add_loss("align_loss", loss_ctc)
    return log, None, None, None


stages["alignment"] = StageType(
    next_stage=None,
    train_fn=train_alignment,
    validate_fn=validate_alignment,
    train_models=["text_aligner"],
    eval_models=[],
    discriminators=[],
    inputs=[
        "text",
        "text_length",
        "audio_gt",
    ],
)

##### Acoustic #####


def train_acoustic(
    batch, model, train, probing
) -> Tuple[LossLog, Optional[torch.Tensor]]:
    with train.accelerator.autocast():
        print_gpu_vram("init")
        mel, _ = calculate_mel(batch.audio_gt, train.to_mel)
        with torch.no_grad():
            energy = log_norm(mel.unsqueeze(1)).squeeze(1)
        spk_emb = train.speaker_embedder(batch.audio_gt)
        pred = model.speech_predictor(
            batch.text, batch.text_length, batch.alignment, batch.pitch, energy, spk_emb
        )
        # pred_pitch, pred_energy = model.pitch_energy_predictor(
        #     batch.text, batch.text_length, batch.alignment
        # )
        print_gpu_vram("predicted")
        train.stage.optimizer.zero_grad()

        log = build_loss_log(train)
        target_spec, pred_spec, target_phase, pred_phase = train.multi_spectrogram(
            target=batch.audio_gt, pred=pred.audio.squeeze(1)
        )
        train.stft_loss(target_list=target_spec, pred_list=pred_spec, log=log)
        log.add_loss(
            "multi_phase",
            multi_phase_loss(pred_phase, target_phase, train.model_config.n_fft),
        )
        print_gpu_vram("stft_loss")
        gan_inputs = dict(
            mrd={"target_list": target_spec, "pred_list": pred_spec},
        )
        log.add_loss(
            "generator",
            train.generator_loss(**gan_inputs).mean(),
        )
        print_gpu_vram("generator_loss")
        log.add_loss(
            "slm",
            train.wavlm_loss(batch.audio_gt.detach(), pred.audio),
        )
        print_gpu_vram("slm_loss")
        train.magphase_loss(pred, batch.audio_gt, log)
        print_gpu_vram("magphase_loss")

        # log.add_loss(
        #     "pitch",
        #     torch.nn.functional.smooth_l1_loss(batch.pitch, pred_pitch),
        # )
        # log.add_loss(
        #     "energy",
        #     torch.nn.functional.smooth_l1_loss(energy, pred_energy),
        # )
        train.accelerator.backward(log.backwards_loss())
        print_gpu_vram("backward")

    return log.detach(), detach_all(gan_inputs)


@torch.no_grad()
def validate_acoustic(batch, train):
    mel, _ = calculate_mel(batch.audio_gt, train.to_mel)
    energy = log_norm(mel.unsqueeze(1)).squeeze(1)
    # pred_pitch, pred_energy = train.model.pitch_energy_predictor(
    #     batch.text, batch.text_length, batch.alignment
    # )
    spk_emb = train.speaker_embedder(batch.audio_gt)
    pred = train.model.speech_predictor(
        batch.text, batch.text_length, batch.alignment, batch.pitch, energy, spk_emb
    )
    log = build_loss_log(train)
    target_spec, pred_spec, target_phase, pred_phase = train.multi_spectrogram(
        target=batch.audio_gt, pred=pred.audio.squeeze(1)
    )
    train.stft_loss(target_list=target_spec, pred_list=pred_spec, log=log)
    log.add_loss(
        "multi_phase",
        multi_phase_loss(pred_phase, target_phase, train.model_config.n_fft),
    )
    # log.add_loss(
    #     "pitch",
    #     torch.nn.functional.smooth_l1_loss(batch.pitch, pred_pitch),
    # )
    # log.add_loss(
    #     "energy",
    #     torch.nn.functional.smooth_l1_loss(energy, pred_energy),
    # )
    return log, batch.alignment[0], make_list(pred.audio), batch.audio_gt


stages["acoustic"] = StageType(
    next_stage="textual",
    train_fn=train_acoustic,
    validate_fn=validate_acoustic,
    train_models=["speech_predictor"],
    eval_models=[],
    # discriminators=[],
    discriminators=["mrd"],
    inputs=[
        "text",
        "text_length",
        "audio_gt",
        "pitch",
        "alignment",
    ],
)

##### Textual #####


def train_textual(
    batch, model, train, probing
) -> Tuple[LossLog, Optional[torch.Tensor]]:
    with train.accelerator.autocast():
        mel, _ = calculate_mel(batch.audio_gt, train.to_mel)
        pe_text_encoding, _, _ = model.pe_text_encoder(batch.text, batch.text_length)
        # pe_text_style = model.pe_text_style_encoder(pe_text_encoding, batch.text_length)
        pe_mel_style = model.pe_mel_style_encoder(mel.unsqueeze(1))
        pred_pitch, pred_energy = model.pitch_energy_predictor(
            pe_text_encoding, batch.text_length, batch.alignment, pe_mel_style
        )
        # pred = model.speech_predictor(
        #     batch.text, batch.text_length, batch.alignment, pred_pitch, pred_energy
        # )
        with torch.no_grad():
            energy = log_norm(mel.unsqueeze(1)).squeeze(1)
        train.stage.optimizer.zero_grad()
        log = build_loss_log(train)
        # train.stft_loss(pred.audio.squeeze(1), batch.audio_gt, log)
        # log.add_loss(
        #     "generator",
        #     train.generator_loss(
        #         batch.audio_gt.detach().unsqueeze(1).float(), pred.audio, ["mrd"]
        #     ).mean(),
        # )
        log.add_loss(
            "pitch",
            torch.nn.functional.smooth_l1_loss(batch.pitch, pred_pitch),
        )
        log.add_loss(
            "energy",
            torch.nn.functional.smooth_l1_loss(energy, pred_energy),
        )
        # log.add_loss(
        #     "style",
        #     torch.nn.functional.smooth_l1_loss(pe_text_style, pe_mel_style)
        # )
        train.accelerator.backward(log.backwards_loss())

    return log.detach(), {}


@torch.no_grad()
def validate_textual(batch, train):
    mel, _ = calculate_mel(batch.audio_gt, train.to_mel)
    pe_text_encoding, _, _ = train.model.pe_text_encoder(batch.text, batch.text_length)
    # pe_text_style = train.model.pe_text_style_encoder(pe_text_encoding, batch.text_length)
    pe_mel_style = train.model.pe_mel_style_encoder(mel.unsqueeze(1))
    pred_pitch, pred_energy = train.model.pitch_energy_predictor(
        pe_text_encoding, batch.text_length, batch.alignment, pe_mel_style
    )
    pred = train.model.speech_predictor(
        batch.text, batch.text_length, batch.alignment, pred_pitch, pred_energy
    )
    energy = log_norm(mel.unsqueeze(1)).squeeze(1)
    log = build_loss_log(train)
    target_spec, pred_spec, target_phase, pred_phase = train.multi_spectrogram(
        target=batch.audio_gt, pred=pred.audio.squeeze(1)
    )
    train.stft_loss(target_list=target_spec, pred_list=pred_spec, log=log)
    log.add_loss(
        "pitch",
        torch.nn.functional.smooth_l1_loss(batch.pitch, pred_pitch),
    )
    log.add_loss("energy", torch.nn.functional.smooth_l1_loss(energy, pred_energy))
    # log.add_loss(
    #     "style",
    #     torch.nn.functional.smooth_l1_loss(pe_text_style, pe_mel_style)
    # )
    return log, batch.alignment[0], make_list(pred.audio), batch.audio_gt


stages["textual"] = StageType(
    next_stage="style",
    train_fn=train_textual,
    validate_fn=validate_textual,
    train_models=[
        "pitch_energy_predictor",
        "pe_text_encoder",
        "pe_mel_style_encoder",
    ],
    eval_models=["speech_predictor"],
    discriminators=[],
    # discriminators=["mrd"],
    inputs=[
        "text",
        "text_length",
        "audio_gt",
        "pitch",
        "alignment",
    ],
)

##### Style #####


def train_style(batch, model, train, probing) -> Tuple[LossLog, Optional[torch.Tensor]]:
    with train.accelerator.autocast():
        mel, _ = calculate_mel(batch.audio_gt, train.to_mel)
        pe_text_encoding, _, _ = model.pe_text_encoder(batch.text, batch.text_length)
        pe_text_style = model.pe_text_style_encoder(pe_text_encoding, batch.text_length)
        pe_mel_style = model.pe_mel_style_encoder(mel.unsqueeze(1))
        pred_pitch, pred_energy = train.model.pitch_energy_predictor(
            pe_text_encoding, batch.text_length, batch.alignment, pe_text_style
        )
        energy = log_norm(mel.unsqueeze(1)).squeeze(1)

        train.stage.optimizer.zero_grad()
        log = build_loss_log(train)
        log.add_loss(
            "style",
            torch.nn.functional.smooth_l1_loss(pe_text_style, pe_mel_style) * 10,
        )
        log.add_loss(
            "pitch",
            torch.nn.functional.smooth_l1_loss(batch.pitch, pred_pitch),
        )
        log.add_loss("energy", torch.nn.functional.smooth_l1_loss(energy, pred_energy))
        train.accelerator.backward(log.backwards_loss())

    return log.detach(), {}


@torch.no_grad()
def validate_style(batch, train):
    mel, _ = calculate_mel(batch.audio_gt, train.to_mel)
    pe_text_encoding, _, _ = train.model.pe_text_encoder(batch.text, batch.text_length)
    pe_text_style = train.model.pe_text_style_encoder(
        pe_text_encoding, batch.text_length
    )
    pe_mel_style = train.model.pe_mel_style_encoder(mel.unsqueeze(1))
    pred_pitch, pred_energy = train.model.pitch_energy_predictor(
        pe_text_encoding, batch.text_length, batch.alignment, pe_text_style
    )
    pred = train.model.speech_predictor(
        batch.text, batch.text_length, batch.alignment, pred_pitch, pred_energy
    )
    energy = log_norm(mel.unsqueeze(1)).squeeze(1)
    log = build_loss_log(train)
    target_spec, pred_spec, target_phase, pred_phase = train.multi_spectrogram(
        target=batch.audio_gt, pred=pred.audio.squeeze(1)
    )
    train.stft_loss(target_list=target_spec, pred_list=pred_spec, log=log)
    log.add_loss(
        "pitch",
        torch.nn.functional.smooth_l1_loss(batch.pitch, pred_pitch),
    )
    log.add_loss("energy", torch.nn.functional.smooth_l1_loss(energy, pred_energy))
    log.add_loss(
        "style", torch.nn.functional.smooth_l1_loss(pe_text_style, pe_mel_style) * 10
    )
    return log, batch.alignment[0], make_list(pred.audio), batch.audio_gt


stages["style"] = StageType(
    next_stage="duration",
    train_fn=train_style,
    validate_fn=validate_style,
    train_models=[
        "pe_text_style_encoder",
    ],
    eval_models=[
        "pe_mel_style_encoder",
        "pitch_energy_predictor",
        "pe_text_encoder",
        "speech_predictor",
    ],
    discriminators=[],
    inputs=[
        "text",
        "text_length",
        "audio_gt",
        "pitch",
        "alignment",
    ],
)


##### Duration #####


def train_duration(
    batch, model, train, probing
) -> Tuple[LossLog, Optional[torch.Tensor]]:
    targets = train.duration_processor.align_to_class(batch.alignment)
    duration = model.duration_predictor(batch.text, batch.text_length)
    train.stage.optimizer.zero_grad()
    loss_ce, loss_cdw = train.duration_loss(duration, targets, batch.text_length)

    log = build_loss_log(train)
    log.add_loss("duration_ce", loss_ce)
    log.add_loss("duration", loss_cdw)
    train.accelerator.backward(log.backwards_loss())

    return log.detach(), {}


@torch.no_grad()
def validate_duration(batch, train):
    pe_text_encoding, _, _ = train.model.pe_text_encoder(batch.text, batch.text_length)
    pe_text_style = train.model.pe_text_style_encoder(
        pe_text_encoding, batch.text_length
    )
    duration = train.model.duration_predictor(batch.text, batch.text_length)
    results = []
    for i in range(duration.shape[0]):
        dur = train.duration_processor.prediction_to_duration(
            duration[i], batch.text_length[i]
        )
        dur = dur[: batch.text_length[i]]
        alignment = train.duration_processor.duration_to_alignment(dur)
        alignment = rearrange(alignment, "t a -> 1 t a")
        pred_pitch, pred_energy = train.model.pitch_energy_predictor(
            pe_text_encoding[i : i + 1, :, : batch.text_length[i]],
            batch.text_length[i : i + 1],
            alignment,
            pe_text_style[i : i + 1],
        )
        pred = train.model.speech_predictor(
            batch.text[i : i + 1, : batch.text_length[i]],
            batch.text_length[i : i + 1],
            alignment,
            pred_pitch,
            pred_energy,
        )
        audio = rearrange(pred.audio, "1 1 l -> l")
        results.append(audio)
    log = build_loss_log(train)
    loss_ce, loss_cdw = train.duration_loss(
        duration,
        train.duration_processor.align_to_class(batch.alignment),
        batch.text_length,
    )
    log.add_loss("duration_ce", loss_ce)
    log.add_loss("duration", loss_cdw)
    # log.add_loss("duration", loss_dur)

    return log.detach(), alignment[0], results, batch.audio_gt


stages["duration"] = StageType(
    next_stage=None,
    train_fn=train_duration,
    validate_fn=validate_duration,
    train_models=[
        "duration_predictor",
    ],
    eval_models=[
        "pitch_energy_predictor",
        "speech_predictor",
        "pe_text_encoder",
        "pe_text_style_encoder",
    ],
    discriminators=[],
    inputs=[
        "text",
        "text_length",
        "alignment",
        "audio_gt",
    ],
)


##### HuBERT Acoustic #####


def pred_ssl_features(train, batch, time_dim):
    phones = train.hubert(batch.audio_gt, time_dim)
    spk_emb = train.speaker_embedder(batch.audio_gt)
    return phones, spk_emb


def train_hubert_acoustic(
    batch, model, train, probing
) -> Tuple[LossLog, Optional[torch.Tensor]]:
    with train.accelerator.autocast():
        print_gpu_vram("init")
        mel, mel_lengths = calculate_mel(batch.audio_gt, train.to_mel)
        with torch.no_grad():
            energy = log_norm(mel.unsqueeze(1)).squeeze(1)
        phones, spk_emb = pred_ssl_features(train, batch, mel.shape[-1])
        pred = model.hubert_speech_predictor(
            phones, mel_lengths, spk_emb, batch.pitch, energy
        )
        pred_pitch, pred_energy = model.hubert_pitch_energy_predictor(
            phones, mel_lengths, spk_emb
        )
        print_gpu_vram("predicted")
        train.stage.optimizer.zero_grad()

        log = build_loss_log(train)
        target_spec, pred_spec, target_phase, pred_phase = train.multi_spectrogram(
            target=batch.audio_gt, pred=pred.audio.squeeze(1)
        )
        train.stft_loss(target_list=target_spec, pred_list=pred_spec, log=log)
        log.add_loss(
            "multi_phase",
            multi_phase_loss(pred_phase, target_phase, train.model_config.n_fft),
        )
        print_gpu_vram("stft_loss")
        gan_inputs = dict(
            mrd={"target_list": target_spec, "pred_list": pred_spec},
        )
        log.add_loss(
            "generator",
            train.generator_loss(**gan_inputs).mean(),
        )
        print_gpu_vram("generator_loss")
        log.add_loss(
            "slm",
            train.wavlm_loss(batch.audio_gt.detach(), pred.audio),
        )
        print_gpu_vram("slm_loss")
        train.magphase_loss(pred, batch.audio_gt, log)
        print_gpu_vram("magphase_loss")

        log.add_loss(
            "pitch",
            torch.nn.functional.smooth_l1_loss(batch.pitch, pred_pitch),
        )
        with torch.no_grad():
            energy = log_norm(mel.unsqueeze(1)).squeeze(1)
        log.add_loss(
            "energy",
            torch.nn.functional.smooth_l1_loss(energy, pred_energy),
        )
        train.accelerator.backward(log.backwards_loss())
        print_gpu_vram("backward")

    return log.detach(), detach_all(gan_inputs)


@torch.no_grad()
def validate_hubert_acoustic(batch, train):
    model = train.model
    mel, mel_lengths = calculate_mel(batch.audio_gt, train.to_mel)
    with torch.no_grad():
        energy = log_norm(mel.unsqueeze(1)).squeeze(1)
    phones, spk_emb = pred_ssl_features(train, batch, mel.shape[-1])
    pred = model.hubert_speech_predictor(
        phones, mel_lengths, spk_emb, batch.pitch, energy
    )

    log = build_loss_log(train)
    target_spec, pred_spec, target_phase, pred_phase = train.multi_spectrogram(
        target=batch.audio_gt, pred=pred.audio.squeeze(1)
    )
    train.stft_loss(target_list=target_spec, pred_list=pred_spec, log=log)
    log.add_loss(
        "multi_phase",
        multi_phase_loss(pred_phase, target_phase, train.model_config.n_fft),
    )
    pred_pitch, pred_energy = model.hubert_pitch_energy_predictor(
        phones, mel_lengths, spk_emb
    )
    log.add_loss(
        "pitch",
        torch.nn.functional.smooth_l1_loss(batch.pitch, pred_pitch),
    )
    with torch.no_grad():
        energy = log_norm(mel.unsqueeze(1)).squeeze(1)
    log.add_loss(
        "energy",
        torch.nn.functional.smooth_l1_loss(energy, pred_energy),
    )
    return log, batch.alignment[0], make_list(pred.audio), batch.audio_gt


stages["hubert_acoustic"] = StageType(
    next_stage="hubert_textual",
    train_fn=train_hubert_acoustic,
    validate_fn=validate_hubert_acoustic,
    train_models=["hubert_speech_predictor"],
    eval_models=[],
    # discriminators=[],
    discriminators=["mrd"],
    inputs=[
        "text",
        "text_length",
        "audio_gt",
        "pitch",
        "alignment",
    ],
)

##### HuBERT textual #####


def train_hubert_textual(
    batch, model, train, probing
) -> Tuple[LossLog, Optional[torch.Tensor]]:
    with train.accelerator.autocast():
        print_gpu_vram("init")
        mel, mel_lengths = calculate_mel(batch.audio_gt, train.to_mel)
        phones, spk_emb = pred_ssl_features(train, batch, mel.shape[-1])
        pred_pitch, pred_energy = model.hubert_pitch_energy_predictor(
            phones, mel_lengths, spk_emb
        )
        pred = model.hubert_speech_predictor(
            phones, mel_lengths, spk_emb, pred_pitch, pred_energy
        )
        print_gpu_vram("predicted")
        train.stage.optimizer.zero_grad()

        log = build_loss_log(train)
        target_spec, pred_spec, target_phase, pred_phase = train.multi_spectrogram(
            target=batch.audio_gt, pred=pred.audio.squeeze(1)
        )
        train.stft_loss(target_list=target_spec, pred_list=pred_spec, log=log)
        log.add_loss(
            "multi_phase",
            multi_phase_loss(pred_phase, target_phase, train.model_config.n_fft),
        )
        print_gpu_vram("stft_loss")
        gan_inputs = dict(
            mrd={"target_list": target_spec, "pred_list": pred_spec},
        )
        log.add_loss(
            "generator",
            train.generator_loss(**gan_inputs).mean(),
        )
        print_gpu_vram("generator_loss")
        log.add_loss(
            "slm",
            train.wavlm_loss(batch.audio_gt.detach(), pred.audio),
        )
        print_gpu_vram("slm_loss")
        train.magphase_loss(pred, batch.audio_gt, log)
        print_gpu_vram("magphase_loss")

        log.add_loss(
            "pitch",
            torch.nn.functional.smooth_l1_loss(batch.pitch, pred_pitch),
        )
        with torch.no_grad():
            energy = log_norm(mel.unsqueeze(1)).squeeze(1)
        log.add_loss(
            "energy",
            torch.nn.functional.smooth_l1_loss(energy, pred_energy),
        )
        train.accelerator.backward(log.backwards_loss())

    return (
        log.detach(),  # None, None
        detach_all(gan_inputs),
    )


@torch.no_grad()
def validate_hubert_textual(batch, train):
    model = train.model
    mel, mel_lengths = calculate_mel(batch.audio_gt, train.to_mel)
    phones, spk_emb = pred_ssl_features(train, batch, mel.shape[-1])
    pred_pitch, pred_energy = model.hubert_pitch_energy_predictor(
        phones, mel_lengths, spk_emb
    )
    pred = model.hubert_speech_predictor(
        phones, mel_lengths, spk_emb, pred_pitch, pred_energy
    )

    log = build_loss_log(train)
    target_spec, pred_spec, target_phase, pred_phase = train.multi_spectrogram(
        target=batch.audio_gt, pred=pred.audio.squeeze(1)
    )
    train.stft_loss(target_list=target_spec, pred_list=pred_spec, log=log)
    log.add_loss(
        "multi_phase",
        multi_phase_loss(pred_phase, target_phase, train.model_config.n_fft),
    )
    return log, batch.alignment[0], make_list(pred.audio), batch.audio_gt


stages["hubert_textual"] = StageType(
    next_stage=None,
    train_fn=train_hubert_textual,
    validate_fn=validate_hubert_textual,
    train_models=["hubert_pitch_energy_predictor"],
    eval_models=["hubert_speech_predictor"],
    # discriminators=[],
    discriminators=["mrd"],
    inputs=[
        "text",
        "text_length",
        "audio_gt",
        "pitch",
        "alignment",
    ],
)


##### Joint #####


def train_joint(batch, model, train, probing) -> Tuple[LossLog, Optional[torch.Tensor]]:
    with train.accelerator.autocast():
        print_gpu_vram("init")
        mel, _ = calculate_mel(batch.audio_gt, train.to_mel)
        with torch.no_grad():
            energy = log_norm(mel.unsqueeze(1)).squeeze(1)
        pe_mel_style = train.model.pe_mel_style_encoder(mel.unsqueeze(1))
        pe_text_encoding, _, _ = train.model.pe_text_encoder(
            batch.text, batch.text_length
        )
        pe_text_style = train.model.pe_text_style_encoder(
            pe_text_encoding, batch.text_length
        )
        pred_pitch, pred_energy = train.model.pitch_energy_predictor(
            pe_text_encoding, batch.text_length, batch.alignment, pe_text_style
        )
        pred = model.speech_predictor(
            batch.text, batch.text_length, batch.alignment, pred_pitch, pred_energy
        )
        # pred_pitch, pred_energy = model.pitch_energy_predictor(
        #     batch.text, batch.text_length, batch.alignment
        # )
        print_gpu_vram("predicted")
        train.stage.optimizer.zero_grad()

        log = build_loss_log(train)
        target_spec, pred_spec, target_phase, pred_phase = train.multi_spectrogram(
            target=batch.audio_gt, pred=pred.audio.squeeze(1)
        )
        train.stft_loss(target_list=target_spec, pred_list=pred_spec, log=log)
        print_gpu_vram("stft_loss")
        log.add_loss(
            "generator",
            train.generator_loss(
                mrd={"target_list": target_spec, "pred_list": pred_spec},
            ).mean(),
        )
        print_gpu_vram("generator_loss")
        log.add_loss(
            "slm",
            train.wavlm_loss(batch.audio_gt.detach(), pred.audio),
        )
        print_gpu_vram("slm_loss")
        train.magphase_loss(pred, batch.audio_gt, log)
        print_gpu_vram("magphase_loss")

        log.add_loss(
            "style",
            torch.nn.functional.smooth_l1_loss(pe_text_style, pe_mel_style) * 10,
        )
        log.add_loss(
            "pitch",
            torch.nn.functional.smooth_l1_loss(batch.pitch, pred_pitch),
        )
        log.add_loss(
            "energy",
            torch.nn.functional.smooth_l1_loss(energy, pred_energy),
        )
        train.accelerator.backward(log.backwards_loss())
        print_gpu_vram("backward")

    return (
        log.detach(),  # None, None
        detach_all(target_spec),
        detach_all(pred_spec),
    )  # pred.audio.detach()


@torch.no_grad()
def validate_joint(batch, train):
    mel, _ = calculate_mel(batch.audio_gt, train.to_mel)
    pe_text_encoding, _, _ = train.model.pe_text_encoder(batch.text, batch.text_length)
    pe_text_style = train.model.pe_text_style_encoder(
        pe_text_encoding, batch.text_length
    )
    pred_pitch, pred_energy = train.model.pitch_energy_predictor(
        pe_text_encoding, batch.text_length, batch.alignment, pe_text_style
    )
    pred = train.model.speech_predictor(
        batch.text, batch.text_length, batch.alignment, pred_pitch, pred_energy
    )
    energy = log_norm(mel.unsqueeze(1)).squeeze(1)
    log = build_loss_log(train)
    target_spec, pred_spec, target_phase, pred_phase = train.multi_spectrogram(
        target=batch.audio_gt, pred=pred.audio.squeeze(1)
    )
    train.stft_loss(target_list=target_spec, pred_list=pred_spec, log=log)
    log.add_loss(
        "pitch",
        torch.nn.functional.smooth_l1_loss(batch.pitch, pred_pitch),
    )
    log.add_loss("energy", torch.nn.functional.smooth_l1_loss(energy, pred_energy))
    return log, batch.alignment[0], make_list(pred.audio), batch.audio_gt


stages["joint"] = StageType(
    next_stage=None,
    train_fn=train_joint,
    validate_fn=validate_joint,
    train_models=[
        "pe_text_style_encoder",
        "pitch_energy_predictor",
        "pe_text_encoder",
        "speech_predictor",
    ],
    eval_models=["pe_mel_style_encoder"],
    discriminators=["mrd"],
    # discriminators=[],
    inputs=[
        "text",
        "text_length",
        "audio_gt",
        "pitch",
        "alignment",
    ],
)


#########################


def detach_all(gan_inputs):
    result = {}
    for k, input in gan_inputs.items():
        if type(input["target_list"]) == torch.Tensor:
            result[k] = {
                "target_list": input["target_list"].detach(),
                "pred_list": input["pred_list"].detach(),
            }
        else:
            result[k] = {
                "target_list": [target.detach() for target in input["target_list"]],
                "pred_list": [pred.detach() for pred in input["pred_list"]],
            }
    return result


@torch.no_grad()
def calculate_mel(audio, to_mel):
    mean, std = -4, 4
    mel = to_mel(audio)
    mel = (torch.log(1e-5 + mel) - mean) / std
    # STFT returns audio_len // hop_len + 1, so we strip off the extra here
    mel = mel[:, :, : (mel.shape[-1] - mel.shape[-1] % 2)]
    mel_length = torch.full(
        [audio.shape[0]], mel.shape[2], dtype=torch.long, device=audio.device
    )
    return mel, mel_length
