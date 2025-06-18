import torch


class ExportModel(torch.nn.Module):
    def __init__(
        self,
        *,
        text_acoustic_extractor,
        text_duration_extractor,
        text_spectral_extractor,
        duration_predictor,
        pitch_energy_predictor,
        generator,
        device="cuda",
        **kwargs
    ):
        super(ExportModel, self).__init__()

        for model in [
            text_acoustic_extractor,
            text_duration_extractor,
            text_spectral_extractor,
            duration_predictor,
            pitch_energy_predictor,
            generator,
        ]:
            model.to(device).eval()
            for p in model.parameters():
                p.requires_grad = False

        self.device = device
        self.text_acoustic_extractor = text_acoustic_extractor
        self.text_duration_extractor = text_duration_extractor
        self.text_spectral_extractor = text_spectral_extractor
        self.duration_predictor = duration_predictor
        self.pitch_energy_predictor = pitch_energy_predictor
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
        acoustic_features, acoustic_styles = self.text_acoustic_extractor(
            texts, text_lengths
        )
        duration_features, _ = self.text_duration_extractor(texts, text_lengths)
        spectral_features, spectral_styles = self.text_spectral_extractor(
            texts, text_lengths
        )
        duration = self.duration_predictor(duration_features)
        alignment = self.duration_to_alignment(duration)
        pitch, energy = self.pitch_energy_predictor(
            spectral_features.transpose(-1, -2) @ alignment,
            spectral_styles @ alignment,
        )
        prediction = self.generator(
            acoustic_features @ alignment,
            acoustic_styles @ alignment,
            pitch,
            energy,
        )
        return prediction.audio.squeeze()
