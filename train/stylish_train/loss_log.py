from __future__ import annotations
import logging
import torch
from typing import Optional
import train_context

logger = logging.getLogger(__name__)


def build_loss_log(train: train_context.TrainContext):
    return LossLog(train.logger, train.writer, train.config.loss_weight)


class LossLog:
    def __init__(self, logger, writer, loss_weight):
        self.logger = logger
        self.writer = writer
        self.weights = loss_weight
        self.weight_dict = loss_weight.model_dump()
        self.metrics = {}
        self.total_loss = None
        self.codebook_indices = None
        self.codebook_size = None

    def total(self):
        if self.total_loss is None:
            self.calculate_metrics()
        return self.total_loss

    def broadcast(self, manifest, stage, validation=False):
        if self.total_loss is None:
            self.calculate_metrics()
        loss_list = [f"{k}: {v:.3f}" for k, v in self.metrics.items()]
        loss_string = f"loss: {self.total_loss:.3f}, " + ", ".join(loss_list)
        if validation:
            writer_type = "eval"
            best_string = ""
            if manifest.best_loss != float("inf"):
                best_string = f", (best was {manifest.best_loss})"
            self.logger.info(
                f"Validation step {manifest.current_total_step}: "
                + loss_string
                + best_string
            )
        else:
            writer_type = "train"
            if stage == "acoustic":
                lr = stage.optimizer.optimizers["text_acoustic_extractor"].param_groups[
                    0
                ]["lr"]
            else:
                lr = stage.optimizer.optimizers["text_duration_extractor"].param_groups[
                    0
                ]["lr"]
            lr_string = f", lr: {lr:.7f}"
            self.logger.info(
                f"Epoch [{manifest.current_epoch}/{stage.max_epoch}], "
                + f"Step [{manifest.current_step}/{manifest.steps_per_epoch}], "
                + loss_string
                + lr_string
            )
        self.writer.add_scalar(
            f"{writer_type}/loss", self.total_loss, manifest.current_total_step
        )
        for key, value in self.metrics.items():
            self.writer.add_scalar(
                f"{writer_type}/{key}", value, manifest.current_total_step
            )

        if (
            stage == "pre_cvpl_bert"
            and self.codebook_indices is not None
            and self.codebook_size is not None
        ):
            step = manifest.current_total_step
            groups, B, T, Q = self.codebook_indices.shape

            for g in range(groups):
                for q in range(Q):
                    ids = self.codebook_indices[g, :, :, q].reshape(-1)
                    ids = ids[ids != -1]  # exclude dropped positions

                    if ids.numel() == 0:
                        continue

                    # Histogram
                    hist = torch.bincount(ids, minlength=self.codebook_size).float()
                    self.writer.add_histogram(
                        f"{writer_type}/codebook_hist/group_{g}_q_{q}",
                        hist,
                        step,
                    )

                    # Number of codes used
                    num_used = (hist > 0).sum().item()
                    self.writer.add_scalar(
                        f"{writer_type}/codebook_used/group_{g}_q_{q}",
                        num_used,
                        step,
                    )

                    # Entropy
                    probs = hist / hist.sum()
                    ent = -(probs * torch.log(probs + 1e-9)).sum().item()
                    self.writer.add_scalar(
                        f"{writer_type}/codebook_entropy/group_{g}_q_{q}",
                        ent,
                        step,
                    )

                    self.logger.info(
                        f"[Codebook] group_{g}_q_{q} used={num_used}, entropy={ent:.2f}"
                    )

            # Clear codebook state after logging
            self.codebook_indices = None
            self.codebook_size = None

    def weight(self, key: str):
        if key in self.weight_dict:
            return self.weight_dict[key]
        else:
            logging.error(f"WARNING: Unknown weight for key {key}, defaulting to 1")
            logging.debug(f"self.weights: {self.weights}")
            return 1

    def calculate_metrics(self):
        total = 0
        total_weight = 0
        for key, value in self.metrics.items():
            weight = self.weight(key)
            loss = value * weight
            total += loss
            total_weight += weight
        self.total_loss = total

    def backwards_loss(self):
        total = 0
        for key, value in self.metrics.items():
            if key == "generator" or key == "align_loss":
                loss = value
            else:
                loss = value / value.detach()
            weight = self.weight(key)
            total += loss * weight
        return total

    def detach(self):
        for key, value in self.metrics.items():
            if torch.is_tensor(value):
                self.metrics[key] = value.item()
        if torch.is_tensor(self.total_loss):
            self.total_loss = self.total_loss.item()
        return self

    def add_loss(self, key, value):
        self.metrics[key] = value
        self.total_loss = None

    def add_codebook_indices(self, indices, codebook_size):
        self.codebook_indices = indices
        self.codebook_size = codebook_size


def combine_logs(loglist) -> Optional[LossLog]:
    result = None
    if len(loglist) > 0:
        result = LossLog(loglist[0].logger, loglist[0].writer, loglist[0].weights)
        totals = {}
        counts = {}
        for log in loglist:
            for key in log.metrics.keys():
                if key not in totals:
                    totals[key] = 0
                    counts[key] = 0
                totals[key] += log.metrics[key]
                counts[key] += 1
        for key in totals.keys():
            result.metrics[key] = totals[key] / counts[key]
    return result
