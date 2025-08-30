import random
import torch
import torchaudio
from torch.nn import functional as F
from einops import rearrange

from batch_context import BatchContext
from loss_log import build_loss_log
from losses import compute_duration_ce_loss
from utils import length_to_mask


@torch.no_grad()
def validate_alignment(batch, train):
    log = build_loss_log(train)
    mel = rearrange(batch.align_mel, "b f t -> b t f")
    ctc, _ = train.model.text_aligner(mel, batch.mel_length)
    train.stage.optimizer.zero_grad()

    loss_ctc = train.align_loss(
        ctc, batch.text, batch.mel_length // 2, batch.text_length, step_type="eval"
    )

    blank = train.model_config.tokens
    logprobs = rearrange(ctc, "t b k -> b t k")
    confidence_total = 0.0
    confidence_count = 0
    for i in range(mel.shape[0]):
        _, scores = torchaudio.functional.forced_align(
            log_probs=logprobs[i].unsqueeze(0).contiguous(),
            targets=batch.text[i, : batch.text_length[i].item()].unsqueeze(0),
            input_lengths=batch.mel_length[i].unsqueeze(0) // 2,
            target_lengths=batch.text_length[i].unsqueeze(0),
            blank=blank,
        )
        confidence_total += scores.exp().sum()
        confidence_count += scores.shape[-1]
    log.add_loss("confidence", confidence_total / confidence_count)
    log.add_loss("align_loss", loss_ctc)
    return log, None, None, None


@torch.no_grad()
def validate_acoustic(batch, train):
    state = BatchContext(train=train, model=train.model)
    pred = state.acoustic_prediction_single(batch)
    log = build_loss_log(train)
    train.stft_loss(pred.audio.squeeze(1), batch.audio_gt, log)
    return log, batch.alignment[0], pred.audio, batch.audio_gt


@torch.no_grad()
def validate_textual(batch, train):
    state = BatchContext(train=train, model=train.model)
    pred = state.textual_prediction_single(batch)
    energy = state.acoustic_energy(batch.mel)
    log = build_loss_log(train)
    train.stft_loss(pred.audio.squeeze(1), batch.audio_gt, log)
    log.add_loss(
        "pitch",
        F.smooth_l1_loss(batch.pitch, state.pitch_prediction),
    )
    log.add_loss("energy", F.smooth_l1_loss(energy, state.energy_prediction))
    loss_ce, loss_dur = compute_duration_ce_loss(
        state.duration_prediction,
        batch.alignment.sum(dim=-1),
        batch.text_length,
    )
    log.add_loss("duration_ce", loss_ce)
    log.add_loss("duration", loss_dur)
    return log, batch.alignment[0], pred.audio, batch.audio_gt


@torch.no_grad()
def validate_spectral(batch, train):
    state = BatchContext(train=train, model=train.model)
    pred = state.spectral_prediction_single(batch)
    energy = state.acoustic_energy(batch.mel)
    log = build_loss_log(train)
    train.stft_loss(pred.audio.squeeze(1), batch.audio_gt, log)
    log.add_loss(
        "pitch",
        F.smooth_l1_loss(batch.pitch, state.pitch_prediction),
    )
    log.add_loss("energy", F.smooth_l1_loss(energy, state.energy_prediction))
    return log, batch.alignment[0], pred.audio, batch.audio_gt


@torch.no_grad()
def validate_pre_hubert_quantizer(batch, train):
    state = BatchContext(train=train, model=train.model)
    state.pre_hubert_quantizer(batch)
    train.stage.optimizer.zero_grad()
    log = build_loss_log(train)
    log.add_loss(
        "hubert_distil_l1",
        F.smooth_l1_loss(state.phones_prediction, state.phones, beta=0.5),
    )
    return log, None, None, None


@torch.no_grad()
def validate_pre_feature_synthesizer(batch, train):
    state = BatchContext(train=train, model=train.model)
    state.pre_feature_synthesizer(batch)
    log = build_loss_log(train)
    # log.add_loss("hubert_code_ce", state.compute_feature_synthesizer_loss(batch))
    log.add_loss(
        "hubert_distil_l2",
        F.mse_loss(
            state.phones_prediction,
            state.phones,
        ),
    )
    return log, None, None, None


@torch.no_grad()
def validate_pre_vevo_token_predictor(batch, train):
    state = BatchContext(train=train, model=train.model)
    ctc, grapheme_ids, grapheme_lengths, pphones, pphone_lengths = (
        state.pre_vevo_token_predictor(batch, training=False)
    )
    log = build_loss_log(train)
    if ctc is None:
        return log, None, None, None
    loss_ctc = train.align_loss(
        ctc, pphones, grapheme_lengths, pphone_lengths, step_type="eval"
    )
    log.add_loss(
        "pphone_ctc",
        loss_ctc,
    )
    ctc = rearrange(ctc, "t b k -> b t k")
    log.add_loss(
        "pphone_per",
        state.wer_metric.compute(
            predictions=[
                " ".join(
                    str(x)
                    for x in state.duration_reduction_func(_ctc.argmax(-1), 1).tolist()
                )
                for _ctc in ctc
            ],
            references=[
                " ".join(str(x) for x in _pphones.tolist()) for _pphones in pphones
            ],
        ),
    )
    return log, None, None, None


@torch.no_grad()
def validate_mspin(batch, train):
    state = BatchContext(train=train, model=train.model)
    metrics = state.pre_mspin(batch)
    log = build_loss_log(train)
    num_codewords = train.model.mspin.loss.num_vars
    log.add_loss("code_ce", metrics["loss_ce"])
    log.add_loss("norm_code_perplexity", metrics["code_perplexity"] / num_codewords)
    log.add_loss("norm_prob_perplexity", metrics["prob_perplexity"] / num_codewords)
    log.add_loss("acc_both_view", metrics["acc"])
    log.add_loss("acc_view_0", metrics["acc_1"])
    log.add_loss("acc_view_1", metrics["acc_2"])
    return log, None, None, None
