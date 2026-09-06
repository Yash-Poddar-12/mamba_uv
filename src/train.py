"""
Usage (from the project root, with your venv activated):

    python -m src.train --config configs/config.yaml

Trains Mamba-UV on AQUA20. Reads everything (data mode, model size, loss
weights, imbalance handling) from the YAML config -- edit that file rather
than passing a long list of flags.
"""
import argparse
import json
import math
import os

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data.dataset import build_folds
from src.losses import MambaUVLoss
from src.models.mamba_uv import MambaUV
from src.utils.metrics import classification_report_dict
from src.utils.seed import set_seed


def build_scheduler(optimizer, warmup_epochs: int, total_epochs: int, steps_per_epoch: int):
    warmup_steps = max(1, warmup_epochs * steps_per_epoch)
    total_steps = max(warmup_steps + 1, total_epochs * steps_per_epoch)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate_loader(model, loader, device, num_classes: int) -> dict:
    model.eval()
    all_probs, all_preds, all_targets = [], [], []
    for imgs, targets in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
        all_preds.append(probs.argmax(axis=1))
        all_targets.append(targets.numpy())
    all_probs = np.concatenate(all_probs)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    return classification_report_dict(all_targets, all_preds, all_probs, num_classes)


def train_one_fold(cfg: dict, fold, fold_idx: int, device: torch.device) -> dict:
    model = MambaUV(cfg).to(device)
    print(f"[Fold {fold_idx}] Trainable parameters: {model.count_parameters():,} "
          f"(paper target: < 3.5M -- adjust model.d_model / n_blocks in the config if over)")
    print(f"[Fold {fold_idx}] SSM backend: "
          f"{'mamba_ssm (CUDA kernel)' if model.blocks[0].using_official_kernel else 'pure-PyTorch fallback'}")

    class_weights = fold.class_weights.to(device)
    criterion = MambaUVLoss(cfg, class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"]
    )
    scheduler = build_scheduler(
        optimizer, cfg["train"]["warmup_epochs"], cfg["train"]["epochs"], len(fold.train_loader)
    )
    use_amp = cfg["train"]["mixed_precision"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    scan_inv_frac = cfg["losses"]["scan_invariance_shuffle_frac"] \
        if cfg["losses"]["use_scan_invariance_loss"] else 0.0

    ckpt_dir = os.path.join(cfg["train"]["checkpoint_dir"], f"fold_{fold_idx}")
    os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(cfg["train"]["log_dir"], f"fold_{fold_idx}"))

    resume_path = os.path.join(ckpt_dir, "last_checkpoint.pt")
    start_epoch, best_f1, patience_counter, global_step = 0, -1.0, 0, 0

    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_f1 = ckpt["best_f1"]
        patience_counter = ckpt["patience_counter"]
        global_step = ckpt["global_step"]
        print(f"[Fold {fold_idx}] Resuming from checkpoint: epoch {start_epoch}, "
              f"best_f1 so far = {best_f1:.4f}")

    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        model.train()
        pbar = tqdm(fold.train_loader, desc=f"Fold {fold_idx} | Epoch {epoch + 1}/{cfg['train']['epochs']}")
        for imgs, targets in pbar:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits, aux = model(imgs, return_aux=True, scan_invariance_shuffle_frac=scan_inv_frac)
                loss, logs = criterion(logits, targets, aux)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip_norm"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            pbar.set_postfix(loss=f"{logs['loss_total']:.4f}")
            for k, v in logs.items():
                writer.add_scalar(f"train/{k}", v, global_step)
            global_step += 1

        val_metrics = evaluate_loader(model, fold.val_loader, device, fold.num_classes)
        for k, v in val_metrics.items():
            writer.add_scalar(f"val/{k}", v, epoch)
        print(f"[Fold {fold_idx}] Epoch {epoch + 1}: "
              f"val_f1={val_metrics['f1_macro']:.4f} val_top1={val_metrics['top1_acc']:.4f} "
              f"val_top3={val_metrics['top3_acc']:.4f}")

        stop_early = False
        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best.pt"))
        else:
            patience_counter += 1
            if patience_counter >= cfg["train"]["early_stopping_patience"]:
                print(f"[Fold {fold_idx}] Early stopping at epoch {epoch + 1}")
                stop_early = True

        # Full training-state checkpoint, saved every epoch regardless of
        # whether it was the best -- lets an interrupted run (reboot, sleep,
        # crash) resume from here instead of restarting from epoch 0.
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_f1": best_f1,
            "patience_counter": patience_counter,
            "global_step": global_step,
        }, resume_path)

        if stop_early:
            break

    writer.close()
    model.load_state_dict(torch.load(os.path.join(ckpt_dir, "best.pt"), map_location=device))

    if fold.test_loader is not None:
        test_metrics = evaluate_loader(model, fold.test_loader, device, fold.num_classes)
        print(f"[Fold {fold_idx}] TEST metrics: {test_metrics}")
        return test_metrics

    return {"best_val_f1": best_f1, **evaluate_loader(model, fold.val_loader, device, fold.num_classes)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["data"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cpu" and cfg["mamba_backend"]["prefer_mamba_ssm"]:
        print("WARNING: no CUDA device found. mamba_ssm needs a GPU -- the "
              "pure-PyTorch fallback SSM will be used automatically, but it "
              "is slow on CPU. Expect this run to be for correctness-checking only.")

    folds = build_folds(cfg)
    print(f"Built {len(folds)} fold(s) in '{cfg['data']['mode']}' mode. "
          f"Class names: {folds[0].class_names}")

    results = []
    for i, fold in enumerate(folds):
        results.append(train_one_fold(cfg, fold, i, device))

    if len(results) > 1:
        keys = results[0].keys()
        summary = {k: (float(np.mean([r[k] for r in results])), float(np.std([r[k] for r in results])))
                   for k in keys}
        print("\n==== 5-fold CV summary (mean, std) ====")
        for k, (mean, std) in summary.items():
            print(f"  {k}: {mean:.4f} +/- {std:.4f}")
        results = {"per_fold": results, "summary": summary}
    else:
        results = results[0]

    os.makedirs(cfg["train"]["checkpoint_dir"], exist_ok=True)
    with open(os.path.join(cfg["train"]["checkpoint_dir"], "results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()