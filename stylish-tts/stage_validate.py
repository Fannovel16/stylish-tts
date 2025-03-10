import torch

from batch_context import BatchContext
from loss_log import build_loss_log


@torch.no_grad()
def validate_acoustic(batch, train):
    state = BatchContext(train, train.model, batch.text_length)
    pred = state.acoustic_prediction_single(batch)
    log = build_loss_log(train)
    log.add_loss("mel", train.stft_loss(pred.audio, batch.audio_gt.unsqueeze(1)))
    return log, state.get_attention(), pred.audio[0], batch.audio_gt[0]


@torch.no_grad()
def validate_textual(batch, train):
    state = BatchContext(train, train.model, batch.text_length)
    pred = state.textual_prediction_single(batch)
    energy = state.acoustic_energy(batch.mel)
    log = build_loss_log(train)
    log.add_loss("mel", train.stft_loss(pred.audio, batch.audio_gt.unsqueeze(1)))
    log.add_loss(
        "F0",
        torch.nn.functional.smooth_l1_loss(batch.pitch, state.pitch_prediction) / 10,
    )
    log.add_loss(
        "norm", torch.nn.functional.smooth_l1_loss(energy, state.energy_prediction)
    )
    return log, state.get_attention(), pred.audio[0], batch.audio_gt[0]
