"""Benchmark interface for the validated CardioFlow architecture."""

import torch
from torch import nn
from torch.nn import functional as F
from .core.model import StrictConductionSpikeV8


class CardioFlow(nn.Module):
    def __init__(self, task, num_classes, population_width=96, dropout=0.15, **backbone_options):
        super().__init__()
        self.task = task
        if task not in {"classification", "delineation"}:
            raise ValueError(task)
        if task == "delineation" and num_classes != 4:
            raise ValueError("Delineation uses background, P, QRS, and T")
        self.backbone = StrictConductionSpikeV8(
            num_classes=num_classes if task == "classification" else 3,
            task="classification" if task == "classification" else "segmentation",
            sample_rate=100, population_width=population_width, dropout=dropout,
            **backbone_options,
        )

    def set_training_progress(self, progress):
        setter = getattr(self.backbone, "set_training_progress", None)
        if callable(setter):
            setter(progress)

    def forward(self, x, lead_mask=None, valid_mask=None, target_length=None):
        if lead_mask is not None and lead_mask.ndim == 2:
            lead_mask = lead_mask.unsqueeze(-1)
        result = self.backbone(x, valid_time_mask=valid_mask, observed_lead_mask=lead_mask)
        if self.task == "delineation":
            foreground = result["logits"]
            logits = torch.cat((torch.zeros_like(foreground[:, :1]), foreground), dim=1)
            if target_length is not None and logits.shape[-1] != target_length:
                logits = F.interpolate(logits, size=target_length, mode="linear", align_corners=False)
            result["logits"] = logits
        return result


def build_model(task, num_classes, config=None):
    return CardioFlow(task=task, num_classes=num_classes, **(config or {}))
