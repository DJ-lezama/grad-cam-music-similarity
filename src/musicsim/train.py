"""Training loop (same hyper-parameters as the original run)."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import config as C
from .model import SiameseNet, get_device


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-4             # the original run used 1e-4 (the earlier 1e-3 optimizer was replaced)
    plateau_patience: int = 3    # ReduceLROnPlateau(factor=0.1)
    early_stop_patience: int = 8
    num_workers: int = 2
    seed: int = 0


def _run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    total, n = 0.0, 0
    use_amp = device.type == "cuda"
    with torch.set_grad_enabled(training):
        for a, b, y in tqdm(loader, leave=False, desc="train" if training else "val"):
            a, b, y = a.to(device), b.to(device), y.to(device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = criterion(model(a, b), y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            total += loss.item() * y.size(0)
            n += y.size(0)
    return total / max(n, 1)


def train(train_ds, val_ds, cfg: TrainConfig = TrainConfig(), ckpt_path: Path = C.CHECKPOINT,
          device: torch.device | None = None) -> list[dict]:
    """Train with early stopping on validation loss; saves the best state_dict."""
    torch.manual_seed(cfg.seed)
    device = device or get_device()
    train_loader = DataLoader(train_ds, cfg.batch_size, shuffle=True, drop_last=True,
                              num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    model = SiameseNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1,
                                                           patience=cfg.plateau_patience)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    ckpt_path = Path(ckpt_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    best, bad, history = float("inf"), 0, []
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        tr = _run_epoch(model, train_loader, criterion, device, optimizer, scaler)
        va = _run_epoch(model, val_loader, criterion, device)
        scheduler.step(va)
        improved = va < best
        history.append(dict(epoch=epoch, train_loss=tr, val_loss=va,
                            lr=optimizer.param_groups[0]["lr"], saved=improved))
        print(f"epoch {epoch:02d}  train {tr:.4f}  val {va:.4f}  "
              f"lr {optimizer.param_groups[0]['lr']:.0e}  {time.time() - t0:.0f}s"
              + ("  *saved*" if improved else ""))
        if improved:
            best, bad = va, 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            bad += 1
            if bad >= cfg.early_stop_patience:
                print("early stopping")
                break
    return history
