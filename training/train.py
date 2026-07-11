"""Train the Subway Surfers action classifier and save it as a .pth file.

Usage:
    python training/train.py [--epochs N] [--batch-size N] [--lr F]

The trained weights (plus preprocessing config and class order) are written to
``models/subway_surfers_cnn.pth`` so the live-play code can load them.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import (
    SubwayDataset,
    class_weights,
    find_samples,
    stratified_split,
)
from model import (
    CLASSES,
    INPUT_CHANNELS,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    NORM_MEAN,
    NORM_STD,
    build_model,
)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
SCREENS_ROOT = os.path.join(REPO_ROOT, "screen_collector", "screens")
MODELS_DIR = os.path.join(REPO_ROOT, "models")
CKPT_PATH = os.path.join(MODELS_DIR, "subway_surfers_cnn.pth")
METRICS_PATH = os.path.join(MODELS_DIR, "training_metrics.json")


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    n_classes = len(CLASSES)
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    correct = total = 0
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        pred = logits.argmax(1).cpu().numpy()
        y = yb.numpy()
        for t, p in zip(y, pred):
            confusion[t, p] += 1
        correct += int((pred == y).sum())
        total += len(y)
    acc = correct / max(total, 1)
    # Balanced accuracy = mean per-class recall (robust to NONE dominance).
    per_class_recall = []
    for c in range(n_classes):
        denom = confusion[c].sum()
        per_class_recall.append(confusion[c, c] / denom if denom else 0.0)
    balanced = float(np.mean(per_class_recall))
    return acc, balanced, per_class_recall, confusion


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    torch.set_num_threads(os.cpu_count() or 4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(MODELS_DIR, exist_ok=True)

    print(f"Scanning {SCREENS_ROOT} ...")
    samples = find_samples(SCREENS_ROOT)
    if not samples:
        raise SystemExit(f"No screenshots found under {SCREENS_ROOT}")
    train_samples, val_samples = stratified_split(
        samples, val_frac=args.val_frac, seed=args.seed
    )
    print(f"Total {len(samples)} | train {len(train_samples)} | val {len(val_samples)}")

    print("Loading + caching images into RAM ...")
    t0 = time.time()
    train_ds = SubwayDataset(train_samples, augment=True)
    val_ds = SubwayDataset(val_samples, augment=False)
    print(f"Cached in {time.time() - t0:.1f}s")

    # Oversample rare classes so each batch is roughly balanced.
    train_labels = train_ds.labels
    counts = np.bincount(train_labels, minlength=len(CLASSES))
    inv = 1.0 / np.maximum(counts, 1)
    sample_w = inv[train_labels]
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_w, dtype=torch.double),
        num_samples=len(train_labels),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=2,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2,
    )

    model = build_model().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    # Class-weighted loss as a second defense against imbalance.
    weights = class_weights(train_samples).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_balanced = -1.0
    history = []
    print("\nTraining ...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        t_ep = time.time()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(yb)
            seen += len(yb)
        scheduler.step()

        train_loss = running / max(seen, 1)
        acc, balanced, recalls, _ = evaluate(model, val_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_acc": round(acc, 4),
                "val_balanced_acc": round(balanced, 4),
            }
        )
        marker = ""
        if balanced > best_balanced:
            best_balanced = balanced
            save_checkpoint(model, args, n_params, acc, balanced)
            marker = "  <- saved (best)"
        print(
            f"epoch {epoch:2d}/{args.epochs}  "
            f"loss {train_loss:.4f}  val_acc {acc:.3f}  "
            f"bal_acc {balanced:.3f}  ({time.time() - t_ep:.1f}s){marker}"
        )

    # Final report using the best checkpoint.
    print("\nReloading best checkpoint for final report ...")
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    acc, balanced, recalls, confusion = evaluate(model, val_loader, device)
    print(f"\nBest model  val_acc={acc:.3f}  balanced_acc={balanced:.3f}")
    print("Per-class recall:")
    for c, r in zip(CLASSES, recalls):
        print(f"  {c:6s} {r:.3f}")
    print("\nConfusion matrix (rows=true, cols=pred):")
    print("        " + "".join(f"{c:>7s}" for c in CLASSES))
    for i, c in enumerate(CLASSES):
        print(f"{c:6s} " + "".join(f"{confusion[i, j]:7d}" for j in range(len(CLASSES))))

    with open(METRICS_PATH, "w") as f:
        json.dump(
            {
                "history": history,
                "final": {
                    "val_acc": acc,
                    "val_balanced_acc": balanced,
                    "per_class_recall": dict(zip(CLASSES, recalls)),
                    "confusion_matrix": confusion.tolist(),
                },
                "num_params": n_params,
                "num_samples": len(samples),
            },
            f,
            indent=2,
        )
    print(f"\nSaved checkpoint -> {CKPT_PATH}")
    print(f"Saved metrics    -> {METRICS_PATH}")


def save_checkpoint(model, args, n_params, acc, balanced) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "classes": CLASSES,
            "input": {
                "channels": INPUT_CHANNELS,
                "height": INPUT_HEIGHT,
                "width": INPUT_WIDTH,
            },
            "norm": {"mean": list(NORM_MEAN), "std": list(NORM_STD)},
            "arch": "SubwayCNN",
            "num_params": n_params,
            "val_acc": acc,
            "val_balanced_acc": balanced,
        },
        CKPT_PATH,
    )


if __name__ == "__main__":
    main()
