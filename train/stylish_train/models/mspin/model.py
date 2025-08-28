from transformers import Wav2Vec2BertModel, AutoFeatureExtractor
import torch
import torch.nn as nn
import torchaudio
import numpy as np
from scipy.signal import sosfilt
from .swav_vq_dis import SwavVQDisentangle
from .audio import params2sos, change_gender, change_gender_f0


def freeze_module(m: nn.Module) -> None:
    for p in m.parameters():
        p.requires_grad = False


def unfreeze_module(m: nn.Module) -> None:
    for p in m.parameters():
        p.requires_grad = True


class MSpin(nn.Module):
    def __init__(
        self,
        w2v_bert_path="facebook/w2v-bert-2.0",
        learnable_encoder_layers=[22, 23, 24],  # Three last layers
        codebook_dim=256,
        codebook_size=2048,
        global_sr=24000,
        nansy_both=False,
    ):
        super().__init__()
        self.model = Wav2Vec2BertModel.from_pretrained(w2v_bert_path)
        self.model.config.apply_spec_augment = False
        freeze_module(self.model)
        for i in learnable_encoder_layers:
            if i == 0:
                unfreeze_module(self.model.feature_projection)
                unfreeze_module(self.model.masked_spec_embed)
                unfreeze_module(self.model.encoder.embed_positions)
                unfreeze_module(self.model.encoder.dropout)
            else:
                unfreeze_module(self.model.encoder.layers[i - 1])

        self.pred_head = nn.Linear(
            self.model.config.hidden_size,
            codebook_dim,
        )
        self.loss = SwavVQDisentangle(
            dim=codebook_dim,
            num_vars=codebook_size,
            epsilon=0.02,
            sinkhorn_iters=3,
            temp=0.1,
            l2_norm=True,
            prob_ratio=1.0,
        )

        self.global_sr = global_sr
        self.processor = AutoFeatureExtractor.from_pretrained(w2v_bert_path)
        self.encoder_sr = self.processor.sampling_rate
        self.resample = torchaudio.transforms.Resample(self.global_sr, self.encoder_sr)

    def extract_feature(self, wav):
        device = wav.device
        wav = self.resample(wav).cpu().numpy()
        wav = self.processor(
            wav,
            sampling_rate=self.encoder_sr,
            return_tensors="pt",
        )["input_features"]
        wav = wav.to(device)
        return self.model(wav).last_hidden_state

    def forward(self, view0, view1=None):
        x0 = self.extract_feature(view0)
        if view1 is None:
            return x0, None
        x1 = self.extract_feature(view1)
        metrics = self.loss.cal_loss(
            self.pred_head(x0).squeeze(1), self.pred_head(x1).squeeze(1)
        )
        return x0, metrics


# https://github.com/vectominist/spin/blob/main/src/data/dataset.py
Qmin, Qmax = 2, 5


class Nansy:
    def __init__(self, global_sr):
        self.rng = np.random.default_rng()
        self.Fc = np.exp(np.linspace(np.log(60), np.log(7600), 10))
        self.global_sr = global_sr

    def random_eq(self, wav, sr):
        z = self.rng.uniform(0, 1, size=(10,))
        Q = Qmin * (Qmax / Qmin) ** z
        G = self.rng.uniform(-12, 12, size=(10,))
        sos = params2sos(G, self.Fc, Q, sr)
        wav = sosfilt(sos, wav)
        return wav

    def random_formant_f0(self, wav, sr, lo, hi):
        ratio_fs = self.rng.uniform(1, 1.4)
        coin = self.rng.random() > 0.5
        ratio_fs = coin * ratio_fs + (1 - coin) * (1 / ratio_fs)

        ratio_ps = self.rng.uniform(1, 2)
        coin = self.rng.random() > 0.5
        ratio_ps = coin * ratio_ps + (1 - coin) * (1 / ratio_ps)

        ratio_pr = self.rng.uniform(1, 1.5)
        coin = self.rng.random() > 0.5
        ratio_pr = coin * ratio_pr + (1 - coin) * (1 / ratio_pr)

        ss = change_gender(wav, sr, lo, hi, ratio_fs, ratio_ps, ratio_pr)

        return ss

    def fixed_formant_f0(self, wav, sr, lo, hi):
        ratio_fs, f0_med, ratio_pr = 0.8, 100, 0.8
        ss = change_gender_f0(wav, sr, lo, hi, ratio_fs, f0_med, ratio_pr)
        return ss

    def __call__(self, wav, f0):
        device = wav.device
        wav, f0 = wav.cpu().numpy(), f0.cpu().numpy()
        sr = self.global_sr
        wav_p = []
        for _wav, _f0 in zip(wav, f0):
            _f0 = np.ma.MaskedArray(_f0, mask=_f0 <= 0)
            _wav_p = _wav
            if _f0.count() > 0:
                lo, hi = int(_f0.mean()), int(_f0.max())
                _wav_p = self.random_formant_f0(_wav_p, sr, lo, hi)
                print("Positive pitch not found, skip random_formant_f0")
            _wav_p = self.random_eq(_wav_p, sr)
            _wav_p = np.clip(_wav_p, -1.0, 1.0)
            wav_p.append(_wav_p)
        wav_p = torch.from_numpy(np.stack(wav_p, axis=0)).to(device)
        return wav_p
