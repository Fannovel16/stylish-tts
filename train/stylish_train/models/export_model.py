import torch


class ExportModel(torch.nn.Module):
    def __init__(
        self,
        *,
        hubert_code_predictor,
        hubert_acoustic_extractor,
        hubert_spectral_extractor,
        pitch_energy_predictor,
        text_duration_extractor,
        duration_predictor,
        generator,
        device="cuda",
        **kwargs
    ):
        super(ExportModel, self).__init__()

        for model in [
            hubert_code_predictor,
            hubert_acoustic_extractor,
            hubert_spectral_extractor,
            pitch_energy_predictor,
            text_duration_extractor,
            duration_predictor,
            generator,
        ]:
            model.to(device).eval()
            for p in model.parameters():
                p.requires_grad = False

        self.device = device
        self.hubert_code_predictor = hubert_code_predictor
        self.hubert_acoustic_extractor = hubert_acoustic_extractor
        self.hubert_spectral_extractor = hubert_spectral_extractor
        self.pitch_energy_predictor = pitch_energy_predictor
        self.text_duration_extractor = text_duration_extractor
        self.duration_predictor = duration_predictor
        self.generator = generator

    def duration_to_alignment(self, duration):
        duration = torch.sigmoid(duration).sum(dim=-1)

        pred_dur = torch.round(duration).clamp(min=1).long().squeeze()
        indices = torch.repeat_interleave(
            torch.arange(duration.shape[1], device=self.device), pred_dur
        )
        pred_aln_trg = torch.zeros(
            (duration.shape[1], indices.shape[0]), device=self.device
        )
        pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
        pred_aln_trg = pred_aln_trg.unsqueeze(0).to(self.device)

        return pred_aln_trg

    def forward(self, texts, text_lengths):
        duration_features, _ = self.text_duration_extractor(texts, text_lengths)
        duration = self.duration_predictor(
            duration_features,
        )
        alignment = self.duration_to_alignment(duration)
        phones = self.hubert_code_predictor(texts, text_lengths, alignment)

        half_mel_lengths = (
            torch.ones([texts.shape[0]], device=texts.device) * phones.shape[1]
        )
        acoustic_features, acoustic_styles = self.hubert_acoustic_extractor(
            phones, half_mel_lengths
        )
        spectral_features, spectral_styles = self.hubert_spectral_extractor(
            phones, half_mel_lengths
        )

        pitch, energy = self.pitch_energy_predictor(
            spectral_features.transpose(-1, -2),
            spectral_styles,
        )
        prediction = self.generator(
            acoustic_features,
            acoustic_styles,
            pitch,
            energy,
        )
        return prediction.audio.squeeze()
