"""Minimal CardioFlow training example."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.nn import functional as F
from torch.utils.data import DataLoader

from model.model import build_model
from preprocessing.dataset import CardioDataset


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = value
    return result


def load_cardioflow_config(path: str | Path, dataset: str) -> tuple[dict, dict, float]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    model_config = dict(config.get("model", {}))
    training = dict(config.get("training", {}))
    training = merge(training, config.get("dataset_training", {}).get(dataset, {}))
    reconstruction_weight = float(config.get("reconstruction_weight", 0.0))
    return model_config, training, reconstruction_weight


def compute_class_weights(dataset: CardioDataset, exponent: float, cap: float, device: torch.device) -> torch.Tensor:
    if dataset.task == "classification":
        positive = np.zeros(dataset.num_classes, dtype=np.float64)
        for start in range(0, len(dataset), 1024):
            positive += np.asarray(dataset.targets[start:start + 1024], dtype=float).sum(0)
        weights = ((len(dataset) - positive) / np.maximum(positive, 1)) ** exponent
    else:
        counts = np.zeros(4, dtype=np.float64)
        for start in range(0, len(dataset), 32):
            states = np.asarray(dataset.targets[start:start + 32])
            counts += np.bincount(states.ravel(), minlength=4)
        weights = (counts.sum() / np.maximum(counts, 1)) ** exponent
        weights /= weights.mean()
    weights = np.clip(weights, 1.0, cap)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def move_batch(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def training_loss(
    output: dict,
    batch: dict,
    task: str,
    weights: torch.Tensor,
    dice_weight: float,
    reconstruction_weight: float,
) -> torch.Tensor:
    logits = output["logits"]
    target = batch["y"]
    if task == "classification":
        loss = F.binary_cross_entropy_with_logits(logits, target.float(), pos_weight=weights)
    else:
        mask = batch.get("label_mask", torch.ones_like(target, dtype=torch.bool)).bool()
        errors = F.cross_entropy(logits, target.long(), weight=weights, reduction="none")
        denominator = weights[target.long()][mask].sum()
        loss = errors[mask].sum() / denominator.clamp_min(1e-12)
        probability = logits.softmax(1)[:, 1:] * mask.unsqueeze(1)
        truth = F.one_hot(target.long(), 4).permute(0, 2, 1)[:, 1:].to(probability) * mask.unsqueeze(1)
        intersection = (probability * truth).sum((0, 2))
        union = probability.sum((0, 2)) + truth.sum((0, 2))
        loss = loss + dice_weight * (1.0 - ((2.0 * intersection + 1.0) / (union + 1.0)).mean())
    if reconstruction_weight:
        reconstruction = output["reconstruction"]
        target_x = output.get("reconstruction_target", batch["x"])
        mask = batch["valid_mask"] * batch["lead_mask"].unsqueeze(-1)
        reconstruction_loss = F.smooth_l1_loss(reconstruction, target_x, reduction="none")
        loss = loss + reconstruction_weight * (reconstruction_loss * mask).sum() / mask.sum().clamp_min(1.0)
    return loss


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model_config, training, reconstruction_weight = load_cardioflow_config(args.config, args.dataset)
    dataset = CardioDataset(args.data_root, args.dataset, "train", augment=bool(training.get("augment", False)))
    batch_size = int(args.batch_size or training.get("batch_size", 32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=int(args.num_workers))
    model = build_model(dataset.task, dataset.num_classes, model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 1e-3)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    exponent_key = "class_balance_exponent" if dataset.task == "classification" else "segmentation_class_balance_exponent"
    weights = compute_class_weights(
        dataset,
        float(training.get(exponent_key, 0.75 if dataset.task == "classification" else 0.25)),
        float(training.get("class_balance_cap", 12.0)),
        device,
    )
    max_batches = None if args.max_batches <= 0 else int(args.max_batches)
    history = []
    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        batches = 0
        for batch in loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            target_length = batch["y"].shape[-1] if dataset.task == "delineation" else None
            output = model(batch["x"], lead_mask=batch["lead_mask"], valid_mask=batch["valid_mask"], target_length=target_length)
            loss = training_loss(
                output,
                batch,
                dataset.task,
                weights,
                float(training.get("dice_weight", 0.5)),
                reconstruction_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training.get("grad_clip", 1.0)))
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batches += 1
            if max_batches is not None and batches >= max_batches:
                break
        average_loss = total_loss / max(batches, 1)
        record = {"epoch": epoch, "loss": average_loss, "batches": batches}
        history.append(record)
        print(json.dumps(record), flush=True)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model_config,
            "dataset": args.dataset,
            "task": dataset.task,
            "class_names": list(dataset.class_names),
            "history": history,
        },
        output_path,
    )
    print(f"Saved checkpoint to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("ptbxl", "cpsc2018", "ludb"), default="ptbxl")
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--config", default="config/cardioflow.yaml")
    parser.add_argument("--output", default="outputs/cardioflow_demo.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
