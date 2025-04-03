import math
import random
from typing import List, Tuple
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
from transformers import AutoModel
import numpy as np
from models import penn
import huggingface_hub


class SpectralConvergenceLoss(torch.nn.Module):
    """Spectral convergence loss module."""

    def __init__(self):
        """Initilize spectral convergence loss module."""
        super(SpectralConvergenceLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Spectral convergence loss value.
        """
        num = torch.linalg.matrix_norm(y_mag - x_mag)
        denom = torch.linalg.matrix_norm(y_mag)
        return (num / denom).mean()


class STFTLoss(torch.nn.Module):
    """STFT loss module."""

    def __init__(
        self,
        *,
        fft_size,
        shift_size,
        win_length,
        window,
        n_mels,
        sample_rate,
    ):
        """Initialize STFT loss module."""
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.to_mel = torchaudio.transforms.MelSpectrogram(
            n_mels=n_mels,
            sample_rate=sample_rate,
            n_fft=fft_size,
            win_length=win_length,
            hop_length=shift_size,
            window_fn=window,
        )

        self.spectral_convergence_loss = SpectralConvergenceLoss()

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        x_mag = self.to_mel(x)
        x_norm = torch.log(torch.norm(x_mag, dim=2))
        x_log = torch.log(1 + x_mag)

        y_mag = self.to_mel(y)
        y_norm = torch.log(torch.norm(y_mag, dim=2))
        y_log = torch.log(1 + y_mag)

        sc_loss = F.mse_loss(x_log, y_log) * 2
        # sc_loss = self.spectral_convergence_loss(x_log, y_log)
        energy_loss = F.smooth_l1_loss(x_norm, y_norm)
        # log_loss = F.l1_loss(x_log, y_log)
        return sc_loss, energy_loss  # , log_loss


class Resolution:
    def __init__(self, *, fft, hop, window, mels):
        self.fft = fft
        self.hop = hop
        self.window = window
        self.mels = mels


resolutions = [
    Resolution(fft=256, hop=31, window=67, mels=40),
    Resolution(fft=256, hop=67, window=127, mels=40),
    Resolution(fft=512, hop=127, window=257, mels=80),
    Resolution(fft=1024, hop=257, window=509, mels=120),
    Resolution(fft=2048, hop=509, window=1021, mels=120),
    Resolution(fft=4096, hop=1021, window=2053, mels=120),
]


class MultiResolutionSTFTLoss(torch.nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(
        self,
        *,
        # fft_sizes=[1024, 2048, 512],
        # hop_sizes=[120, 240, 50],
        # win_lengths=[600, 1200, 240],
        resolution_list=resolutions,
        window=torch.hann_window,
        sample_rate,
        # n_mels,
    ):
        """Initialize Multi resolution STFT loss module.
        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.
        """
        super(MultiResolutionSTFTLoss, self).__init__()
        self.stft_losses = torch.nn.ModuleList()
        for item in resolution_list:
            self.stft_losses += [
                STFTLoss(
                    fft_size=item.fft,
                    shift_size=item.hop,
                    win_length=item.window,
                    window=window,
                    sample_rate=sample_rate,
                    n_mels=item.mels,
                )
            ]

    def forward(self, x, y, log):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.
        """
        sc_loss = 0.0
        energy_loss = 0.0
        # log_loss = 0.0
        for f in self.stft_losses:
            # sc_l, energy_l, log_l = f(x, y)
            sc_l, energy_l = f(x, y)
            sc_loss += sc_l
            energy_loss += energy_l
            # log_loss += log_l
        sc_loss /= len(self.stft_losses)
        energy_loss /= len(self.stft_losses)
        # log_loss /= len(self.stft_losses)

        log.add_loss("mel", sc_loss)
        # log.add_loss("mel_energy", energy_loss)
        # log.add_loss("mel_log", log_loss)

        return sc_loss, energy_loss  # , log_loss


mp_window = torch.hann_window(20).to("cuda")


def magphase_loss(mag, phase, gt):
    result = 0.0
    if mag is not None and phase is not None:
        y_stft = torch.stft(
            gt,
            n_fft=20,
            hop_length=5,
            win_length=20,
            return_complex=True,
            window=mp_window,
        )
        target_mag = torch.abs(y_stft)
        target_phase = torch.angle(y_stft)
        result = torch.nn.functional.l1_loss(
            mag, target_mag
        ) + torch.nn.functional.l1_loss(phase, target_phase)
    return result


def amplitude_loss(log_amplitude_r, log_amplitude_g):
    MSELoss = torch.nn.MSELoss()

    amplitude_loss = MSELoss(log_amplitude_r, log_amplitude_g)

    return amplitude_loss


def anti_wrapping_function(x):
    return torch.abs(x - torch.round(x / (2 * np.pi)) * 2 * np.pi)


def phase_loss(phase_r, phase_g, n_fft, frames):
    GD_matrix = (
        torch.triu(torch.ones(n_fft // 2 + 1, n_fft // 2 + 1), diagonal=1)
        - torch.triu(torch.ones(n_fft // 2 + 1, n_fft // 2 + 1), diagonal=2)
        - torch.eye(n_fft // 2 + 1)
    )
    GD_matrix = GD_matrix.to(phase_g.device)

    GD_r = torch.matmul(phase_r.permute(0, 2, 1), GD_matrix)
    GD_g = torch.matmul(phase_g.permute(0, 2, 1), GD_matrix)

    PTD_matrix = (
        torch.triu(torch.ones(frames, frames), diagonal=1)
        - torch.triu(torch.ones(frames, frames), diagonal=2)
        - torch.eye(frames)
    )
    PTD_matrix = PTD_matrix.to(phase_g.device)

    PTD_r = torch.matmul(phase_r, PTD_matrix)
    PTD_g = torch.matmul(phase_g, PTD_matrix)

    IP_loss = torch.mean(anti_wrapping_function(phase_r - phase_g))
    GD_loss = torch.mean(anti_wrapping_function(GD_r - GD_g))
    PTD_loss = torch.mean(anti_wrapping_function(PTD_r - PTD_g))

    return IP_loss, GD_loss, PTD_loss


def stft_consistency_loss(rea_r, rea_g, imag_r, imag_g):
    C_loss = torch.mean(
        torch.mean((rea_r - rea_g) ** 2 + (imag_r - imag_g) ** 2, (1, 2))
    )

    return C_loss


def amp_phase_spectrum(y, n_fft, hop_size, win_size):
    hann_window = torch.hann_window(win_size).to(y.device)

    stft_spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=True,
        return_complex=True,
    )  # [batch_size, n_fft//2+1, frames, 2]

    log_amplitude = torch.log(
        stft_spec.abs() + 1e-5
    )  # [batch_size, n_fft//2+1, frames]
    phase = stft_spec.angle()  # [batch_size, n_fft//2+1, frames]

    return log_amplitude, phase, stft_spec.real, stft_spec.imag


def freev_loss(log, pred, gt_audio, train):
    if pred.log_amplitude is not None:
        gt_log_amplitude, gt_phase, gt_real, gt_imag = amp_phase_spectrum(
            gt_audio,
            train.model_config.n_fft,
            train.model_config.hop_length,
            train.model_config.win_length,
        )
        loss_amplitude = amplitude_loss(gt_log_amplitude, pred.log_amplitude)

        L_IP, L_GD, L_PTD = phase_loss(
            gt_phase,
            pred.phase,
            train.model_config.n_fft,
            pred.phase.size()[-1],
        )
        # Losses defined on phase spectra
        loss_phase = L_IP + L_GD + L_PTD
        _, _, rea_g_final, imag_g_final = amp_phase_spectrum(
            pred.audio.squeeze(1),
            train.model_config.n_fft,
            train.model_config.hop_length,
            train.model_config.win_length,
        )
        loss_consistency = stft_consistency_loss(
            pred.real, rea_g_final, pred.imaginary, imag_g_final
        )
        loss_real_part = F.l1_loss(gt_real, pred.real)
        loss_imaginary_part = F.l1_loss(gt_imag, pred.imaginary)
        loss_stft_reconstruction = loss_consistency + 2.25 * (
            loss_real_part + loss_imaginary_part
        )
        log.add_loss("amplitude", 4.5 * loss_amplitude)
        log.add_loss("phase", 9 * loss_phase)
        log.add_loss("stft_reconstruction", 2 * loss_stft_reconstruction)


# def feature_loss(fmap_r, fmap_g):
# loss = 0
# for dr, dg in zip(fmap_r, fmap_g):
# for rl, gl in zip(dr, dg):
# loss += torch.mean(torch.abs(rl - gl))
#
# return loss * 2
#
#
# def discriminator_loss(disc_real_outputs, disc_generated_outputs):
# loss = 0
# r_losses = []
# g_losses = []
# for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
# r_loss = torch.mean((1 - dr) ** 2)
# g_loss = torch.mean(dg**2)
# loss += r_loss + g_loss
# r_losses.append(r_loss.item())
# g_losses.append(g_loss.item())
#
# return loss, r_losses, g_losses
#
#
# def generator_loss(disc_outputs):
# loss = 0
# gen_losses = []
# for dg in disc_outputs:
# l = torch.mean((1 - dg) ** 2)
# gen_losses.append(l)
# loss += l
#
# return loss, gen_losses


# """ https://dl.acm.org/doi/abs/10.1145/3573834.3574506 """
#
#
# def discriminator_TPRLS_loss(disc_real_outputs, disc_generated_outputs):
# loss = 0
# for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
# tau = 0.04
# m_DG = torch.median((dr - dg))
# L_rel = torch.mean((((dr - dg) - m_DG) ** 2)[dr < dg + m_DG])
# loss += tau - F.relu(tau - L_rel)
# return loss
#
#
# def generator_TPRLS_loss(disc_real_outputs, disc_generated_outputs):
# loss = 0
# for dg, dr in zip(disc_real_outputs, disc_generated_outputs):
# tau = 0.04
# m_DG = torch.median((dr - dg))
# L_rel = torch.mean((((dr - dg) - m_DG) ** 2)[dr < dg + m_DG])
# loss += tau - F.relu(tau - L_rel)
# return loss
#
#
# class GeneratorLoss(torch.nn.Module):
# def __init__(self, mpd, msd):
# super(GeneratorLoss, self).__init__()
# self.mpd = mpd
# self.msd = msd
#
# def forward(self, y, y_hat):
# y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.mpd(y, y_hat)
# y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.msd(y, y_hat)
# loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
# loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
# loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
# loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
#
# loss_rel = generator_TPRLS_loss(y_df_hat_r, y_df_hat_g) + generator_TPRLS_loss(
# y_ds_hat_r, y_ds_hat_g
# )
#
# loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_rel
#
# return loss_gen_all.mean()
#
#
# class DiscriminatorLoss(torch.nn.Module):
# def __init__(self, mpd, msd):
# super(DiscriminatorLoss, self).__init__()
# self.mpd = mpd
# self.msd = msd
# self.last_loss = 4
#
# def get_disc_lr_multiplier(self):
# ideal_loss = 4.0
# f_max = 2.0
# h_min = 0.1
# x_max = 4.5
# x_min = 3.5
# x = abs(self.last_loss - ideal_loss)
# result = 1.0
# if self.last_loss > ideal_loss:
# x = min(x, x_max)
# result = min(math.pow(f_max, x / x_max), f_max)
# # f_x = tf.clip_by_value(tf.math.pow(f_max, x/x_max), 1.0, f_max)
# else:
# x = max(x, x_min)
# result = max(math.pow(h_min, x / x_min), h_min)
# # h_x = tf.clip_by_value(tf.math.pow(h_min, x/x_min), h_min, 1.0)
# # return tf.cond(loss > ideal_loss, lambda: f_x, lambda: h_x)
# return result
#
# def get_disc_lambda(self):
# return lambda epoch: self.get_disc_lr_multiplier()
#
# def forward(self, y, y_hat):
# # MPD
# y_df_hat_r, y_df_hat_g, _, _ = self.mpd(y, y_hat)
# loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(
# y_df_hat_r, y_df_hat_g
# )
# # MSD
# y_ds_hat_r, y_ds_hat_g, _, _ = self.msd(y, y_hat)
# loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(
# y_ds_hat_r, y_ds_hat_g
# )
#
# loss_rel = discriminator_TPRLS_loss(
# y_df_hat_r, y_df_hat_g
# ) + discriminator_TPRLS_loss(y_ds_hat_r, y_ds_hat_g)
#
# d_loss = loss_disc_s + loss_disc_f + loss_rel
#
# mean = d_loss.mean()
# self.last_loss = mean.item()
# return mean


class GeneratorLoss(torch.nn.Module):
    def __init__(self, *, mpd, mrd, msbd, mstftd):
        super(GeneratorLoss, self).__init__()
        self.generators = torch.nn.ModuleList(
            [
                GeneratorLossHelper(mpd, 1.0),
                GeneratorLossHelper(mrd, 0.2),
                GeneratorLossHelper(msbd, 1.0),
                GeneratorLossHelper(mstftd, 1.0),
            ]
        )

    def forward(self, audio_gt, audio):
        # return self.generators[index](audio_gt, audio)
        loss = 0
        for index in range(2, len(self.generators)):
            loss += self.generators[index](audio_gt, audio)
        return loss


class DiscriminatorLoss(torch.nn.Module):
    def __init__(self, *, mpd, mrd, msbd, mstftd):
        super(DiscriminatorLoss, self).__init__()
        self.discriminators = torch.nn.ModuleDict(
            {
                "mpd": DiscriminatorLossHelper(mpd, 1.0),
                "mrd": DiscriminatorLossHelper(mrd, 0.2),
                "msbd": DiscriminatorLossHelper(msbd, 1.0),
                "mstftd": DiscriminatorLossHelper(mstftd, 1.0),
            }
        )

    def get_disc_lr_multiplier(self, key):
        return self.discriminators[key].get_disc_lr_multiplier()

    def forward(self, audio_gt, audio):
        # key = list(self.discriminators.keys())[index]
        # return self.discriminators[key](audio_gt, audio)
        loss = 0
        for key in self.discriminators.keys():
            if key != "mpd" and key != "mrd":
                loss += self.discriminators[key](audio_gt, audio)
        return loss


class DiscriminatorLossHelper(torch.nn.Module):
    """
    Discriminator Loss module. Calculates the loss for the discriminator based on real and generated outputs.
    """

    def __init__(self, model, weight):
        super(DiscriminatorLossHelper, self).__init__()
        self.model = model
        self.weight = weight
        self.last_loss = 0.5

    def get_disc_lr_multiplier(self):
        ideal_loss = 2.0
        f_max = 4.0
        h_min = 0.1
        x_max = 0.05
        x_min = 0.05
        x = abs(self.last_loss - ideal_loss)
        result = 1.0
        if self.last_loss > ideal_loss:
            result = min(math.pow(f_max, x / x_max), f_max)
        else:
            result = max(math.pow(h_min, x / x_min), h_min)
        return result

    def discriminator_loss(
        self,
        disc_real_outputs: List[torch.Tensor],
        disc_generated_outputs: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            disc_real_outputs (List[Tensor]): List of discriminator outputs for real samples.
            disc_generated_outputs (List[Tensor]): List of discriminator outputs for generated samples.

        Returns:
            Tuple[Tensor, List[Tensor], List[Tensor]]: A tuple containing the total loss, a list of loss values from
                                                       the sub-discriminators for real outputs, and a list of
                                                       loss values for generated outputs.
        """
        loss = torch.zeros(
            1, device=disc_real_outputs[0].device, dtype=disc_real_outputs[0].dtype
        )
        r_losses = []
        g_losses = []
        for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
            r_loss = torch.mean(torch.clamp(1 - dr, min=0))
            g_loss = torch.mean(torch.clamp(1 + dg, min=0))
            loss += r_loss + g_loss
            r_losses.append(r_loss)
            g_losses.append(g_loss)

        return loss, r_losses, g_losses

    def forward(self, audio_gt, audio):
        real_score, gen_score, _, _ = self.model(y=audio_gt, y_hat=audio)
        loss, loss_real, _ = self.discriminator_loss(
            disc_real_outputs=real_score, disc_generated_outputs=gen_score
        )
        loss /= len(loss_real)
        self.last_loss = self.last_loss * 0.95 + loss.item() * 0.05
        return loss * self.weight


class GeneratorLossHelper(torch.nn.Module):
    """
    Generator Loss module. Calculates the loss for the generator based on discriminator outputs.
    """

    def __init__(self, model, weight):
        super(GeneratorLossHelper, self).__init__()
        self.model = model
        self.weight = weight

    def generator_loss(
        self, disc_outputs: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            disc_outputs (List[Tensor]): List of discriminator outputs.

        Returns:
            Tuple[Tensor, List[Tensor]]: Tuple containing the total loss and a list of loss values from
                                         the sub-discriminators
        """
        loss = torch.zeros(
            1, device=disc_outputs[0].device, dtype=disc_outputs[0].dtype
        )
        gen_losses = []
        for dg in disc_outputs:
            item = torch.mean(torch.clamp(1 - dg, min=0))
            gen_losses.append(item)
            loss += item

        return loss, gen_losses

    def feature_matching_loss(
        self, fmap_r: List[List[torch.Tensor]], fmap_g: List[List[torch.Tensor]]
    ) -> torch.Tensor:
        """
        Args:
            fmap_r (List[List[Tensor]]): List of feature maps from real samples.
            fmap_g (List[List[Tensor]]): List of feature maps from generated samples.

        Returns:
            Tensor: The calculated feature matching loss.
        """
        loss = torch.zeros(1, device=fmap_r[0][0].device, dtype=fmap_r[0][0].dtype)
        for dr, dg in zip(fmap_r, fmap_g):
            for rl, gl in zip(dr, dg):
                loss += torch.mean(torch.abs(rl - gl))

        return loss

    def forward(self, audio_gt, audio):
        _, gen_score, fmap_rs, fmap_gs = self.model(y=audio_gt, y_hat=audio)
        loss_gen, list_loss_gen = self.generator_loss(disc_outputs=gen_score)
        loss_gen = loss_gen / len(list_loss_gen)
        loss_fm = self.feature_matching_loss(fmap_r=fmap_rs, fmap_g=fmap_gs)
        loss_fm = loss_fm / len(fmap_rs)
        return (loss_gen + loss_fm) * self.weight


class WavLMLoss(torch.nn.Module):
    def __init__(self, model, model_sr, slm_sr=16000):
        super(WavLMLoss, self).__init__()
        self.wavlm = AutoModel.from_pretrained(model)
        self.resample = torchaudio.transforms.Resample(model_sr, slm_sr)

    def forward(self, wav, y_rec):
        with torch.no_grad():
            wav_16 = self.resample(wav)
            wav_embeddings = self.wavlm(
                input_values=wav_16, output_hidden_states=True
            ).hidden_states
            wav_tensor = torch.stack(wav_embeddings)
        y_rec_16 = self.resample(y_rec)
        y_rec_embeddings = self.wavlm(
            input_values=y_rec_16.squeeze(1), output_hidden_states=True
        ).hidden_states
        y_rec_tensor = torch.stack(y_rec_embeddings)
        return torch.nn.functional.l1_loss(wav_tensor, y_rec_tensor)


class PitchReconstructionLoss(torch.nn.Module):
    def __init__(self, *, hop_length, win_length):
        super(PitchReconstructionLoss, self).__init__()
        checkpoint = huggingface_hub.hf_hub_download(
            "maxrmorrison/fcnf0-plus-plus", "fcnf0++.pt"
        )
        states = torch.load(checkpoint, map_location="cpu")
        self.model = penn.Fcnf0()
        self.model.load_state_dict(states["model"])
        self.unfold = torch.nn.Unfold(kernel_size=(1, 1024), stride=(1, 80))
        self.resample = torchaudio.transforms.Resample(24000, 8000)

    def predict(self, x):
        # x = x[index:index+1]
        # x = self.resample(x)
        # x = x.unsqueeze(1)
        # frames = self.unfold(x)
        # frames = frames.transpose(0, 1)
        # return self.model(frames.unsqueeze(1))

        # frames = frames.unsqueeze(2)
        # result = []
        # for i in range(frames.shape[0]):
        # result.append(self.model(frames[i]))
        # return torch.stack(result)

        x = self.resample(x)
        x = x.unsqueeze(1).unsqueeze(1)
        frames = self.unfold(x)
        frames = frames.transpose(1, 2)
        frames = frames.flatten(0, 1)
        return self.model(frames.unsqueeze(1))

    def forward(self, audio_gt, audio, log):
        # index = random.randrange(0, audio.shape[0])
        with torch.no_grad():
            x_pred = self.predict(audio_gt)
        y_pred = self.predict(audio)
        result = F.l1_loss(x_pred, y_pred)
        log.add_loss("pitch_reconstruction", result)
        return result


def compute_duration_ce_loss(
    duration_prediction: List[torch.Tensor],
    duration: List[torch.Tensor],
    text_length: List[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the duration and binary cross-entropy losses over a batch.
    Returns (loss_ce, loss_dur).
    """
    loss_ce = 0
    loss_dur = 0
    for pred, dur, length in zip(duration_prediction, duration, text_length):
        pred = pred[:length, :]
        dur = dur[:length].long()
        target = torch.zeros_like(pred)
        for i in range(target.shape[0]):
            target[i, : dur[i]] = 1
        dur_pred = torch.sigmoid(pred).sum(dim=1)
        loss_dur += F.l1_loss(dur_pred[1 : length - 1], dur[1 : length - 1])
        loss_ce += F.binary_cross_entropy_with_logits(pred.flatten(), target.flatten())
    n = len(text_length)
    return loss_ce / n, loss_dur / n
