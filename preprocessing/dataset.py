"""Uniform model inputs for classification and native-grid delineation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .constants import CLASSES, PROTOCOLS, SPLITS
from .download import sha256


def dataset_info(root: str | Path, dataset: str) -> dict:
    if dataset not in CLASSES:
        raise ValueError(f"Unknown dataset: {dataset}")
    return json.loads((Path(root) / dataset / "meta.json").read_text(encoding="utf-8"))


class CardioDataset(Dataset):
    """Read a prepared split without modifying the underlying cache.

    ``root`` is the parent of the ptbxl/cpsc2018/ludb directories.
    LUDB targets use class IDs 0=background, 1=P, 2=QRS, 3=T.
    """

    def __init__(self, root: str | Path, dataset: str, split: str, augment: bool = False, *, verify: bool = True):
        if dataset not in CLASSES or split not in SPLITS:
            raise ValueError(f"Unknown dataset/split: {dataset}/{split}")
        self.root = Path(root) / dataset
        self.dataset = dataset
        self.split = split
        self.augment = bool(augment and split == "train")
        self.meta = dataset_info(root, dataset)
        if self.meta.get("format_version") != 1 or self.meta.get("protocol") != PROTOCOLS[dataset]:
            raise ValueError("Unsupported prepared dataset format/protocol")
        self.class_names = tuple(self.meta["class_names"])
        if self.class_names != CLASSES[dataset]:
            raise ValueError("Prepared dataset class order differs from the task definition")
        self.task = self.meta["task"]
        self.num_classes = len(self.class_names)
        self.sample_rate = int(self.meta["sample_rate"])
        self.target_sample_rate = int(self.meta["target_sample_rate"])
        folder = self.root / split
        self.manifest_sha256 = sha256(self.root / "meta.json")
        files = self.meta["splits"][split]["files"]
        if verify:
            for name in ("signals.npy", "targets.npy", "records.json"):
                if sha256(folder / name) != files[name]:
                    raise ValueError(f"Prepared cache checksum mismatch: {dataset}/{split}/{name}")
        self.signals = np.load(folder / "signals.npy", mmap_mode="r", allow_pickle=False)
        self.targets = np.load(folder / "targets.npy", mmap_mode="r", allow_pickle=False)
        self.records = json.loads((folder / "records.json").read_text(encoding="utf-8"))
        count = int(self.meta["splits"][split]["records"])
        if len(self.records) != count or self.signals.shape != (count, 12, 1000):
            raise ValueError("Prepared signal/metadata dimensions do not match the manifest")
        expected_targets = (count, 5000) if dataset == "ludb" else (count, self.num_classes)
        if self.targets.shape != expected_targets:
            raise ValueError("Prepared target dimensions do not match the task")
        normalization = self.meta["normalization"]
        self.mean = float(normalization.get("mean", 0.0))
        self.std = float(normalization.get("std", 1.0))
        if not np.isfinite([self.mean, self.std]).all() or self.std <= 0:
            raise ValueError("Invalid normalization statistics")

    def __len__(self) -> int:
        return len(self.records)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state.pop("signals", None)
        state.pop("targets", None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        folder = self.root / self.split
        self.signals = np.load(folder / "signals.npy", mmap_mode="r", allow_pickle=False)
        self.targets = np.load(folder / "targets.npy", mmap_mode="r", allow_pickle=False)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records[index]
        x = torch.from_numpy(np.asarray(self.signals[index], dtype=np.float32).copy())
        x = (x - self.mean) / self.std
        valid_mask = torch.zeros((1, 1000), dtype=torch.float32)
        valid_mask[:, :int(self.meta["valid_length"])] = 1.0
        lead_mask = torch.ones(12, dtype=torch.float32)
        if self.dataset == "ludb":
            lead_mask.zero_()
            lead_mask[int(row["lead_index"])] = 1.0
        observed = lead_mask[:, None] * valid_mask
        if self.augment:
            x = x * torch.empty(()).uniform_(0.9, 1.1)
            x = x + 0.005 * torch.randn_like(x) * observed
        x = x * observed
        dtype = np.int64 if self.dataset == "ludb" else np.float32
        target = torch.from_numpy(np.asarray(self.targets[index], dtype=dtype).copy())
        item = {
            "x": x, "y": target, "valid_mask": valid_mask, "lead_mask": lead_mask,
            "record_id": str(row["record_id"]), "group_id": str(row["group_id"]),
            "index": int(row["source_index"]),
        }
        if self.dataset == "ludb":
            item["label_mask"] = torch.ones(5000, dtype=torch.bool)
        return item

    def provenance(self) -> dict:
        return {
            "dataset": self.dataset, "split": self.split, "protocol": self.meta["protocol"],
            "manifest_sha256": self.manifest_sha256,
            "files": self.meta["splits"][self.split]["files"],
            "normalization": self.meta["normalization"],
            "class_names": list(self.class_names),
        }
