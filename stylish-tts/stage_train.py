import torch
from batch_context import BatchContext
from loss_log import LossLog, build_loss_log
from losses import magphase_loss, compute_duration_ce_loss


def train_pre_acoustic(batch, model, train) -> LossLog:
    state = BatchContext(train, model, batch.text_length)
    with train.accelerator.autocast():
        pred = state.acoustic_prediction_single(batch, use_random_mono=True)
        train.stage.optimizer.zero_grad()
        log = build_loss_log(train)
        log.add_loss(
            "mel",
            train.stft_loss(pred.audio, batch.audio_gt.unsqueeze(1)),
        )
        if pred.magnitude is not None and pred.phase is not None:
            log.add_loss(
                "magphase",
                magphase_loss(pred.magnitude, pred.phase, batch.audio_gt),
            )
        train.accelerator.backward(log.total())
    return log.detach()


def train_acoustic(batch, model, train) -> LossLog:
    state = BatchContext(train, model, batch.text_length)
    with train.accelerator.autocast():
        pred = state.acoustic_prediction_single(batch)
        train.stage.optimizer.zero_grad()
        d_loss = train.discriminator_loss(
            batch.audio_gt.detach().unsqueeze(1).float(), pred.audio.detach()
        ).mean()
        train.accelerator.backward(d_loss)
        train.stage.optimizer.step("msd")
        train.stage.optimizer.step("mpd")
        train.stage.optimizer.zero_grad()

        log = build_loss_log(train)
        log.add_loss(
            "mel",
            train.stft_loss(pred.audio, batch.audio_gt.unsqueeze(1)),
        )
        log.add_loss(
            "gen",
            train.generator_loss(
                batch.audio_gt.detach().unsqueeze(1).float(), pred.audio
            ).mean(),
        )
        log.add_loss(
            "slm",
            train.wavlm_loss(batch.audio_gt.detach(), pred.audio),
        )
        if pred.magnitude is not None and pred.phase is not None:
            log.add_loss(
                "magphase",
                magphase_loss(pred.magnitude, pred.phase, batch.audio_gt),
            )

        loss_s2s = 0
        for pred, text, length in zip(state.s2s_pred, batch.text, batch.text_length):
            loss_s2s += torch.nn.functional.cross_entropy(pred[:length], text[:length])
        loss_s2s /= batch.text.size(0)
        log.add_loss("s2s", loss_s2s)

        log.add_loss(
            "mono", torch.nn.functional.l1_loss(*(state.duration_results)) * 10
        )

        # freev_loss(log, batch, pred, begin, end, batch.audio_gt, train)
        train.accelerator.backward(log.total())
        log.add_loss("discriminator", d_loss)

    return log.detach()


def train_textual(batch, model, train) -> LossLog:
    state = BatchContext(train, model, batch.text_length)
    with train.accelerator.autocast():
        pred = state.textual_prediction_single(batch)
        energy = state.acoustic_energy(batch.mel)
        train.stage.optimizer.zero_grad()
        log = build_loss_log(train)
        log.add_loss(
            "mel",
            train.stft_loss(pred.audio, batch.audio_gt.unsqueeze(1)),
        )
        log.add_loss(
            "slm",
            train.wavlm_loss(batch.audio_gt.detach(), pred.audio),
        )
        if pred.magnitude is not None and pred.phase is not None:
            log.add_loss(
                "magphase",
                magphase_loss(pred.magnitude, pred.phase, batch.audio_gt),
            )
        log.add_loss(
            "F0",
            torch.nn.functional.smooth_l1_loss(batch.pitch, state.pitch_prediction)
            / 10,
        )
        log.add_loss(
            "norm", torch.nn.functional.smooth_l1_loss(energy, state.energy_prediction)
        )
        loss_ce, loss_dur = compute_duration_ce_loss(
            state.duration_prediction,
            state.duration_results[1].sum(dim=-1),
            batch.text_length,
        )
        log.add_loss("duration_ce", loss_ce)
        log.add_loss("duration", loss_dur)
        train.accelerator.backward(log.total())

    return log.detach()
