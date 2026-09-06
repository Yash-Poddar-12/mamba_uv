"""
Usage:
    python -m src.evaluate --config configs/config.yaml \
        --checkpoint checkpoints/fold_0/best.pt \
        --split test

Prints precision/recall/F1 (macro), Top-1/Top-3 accuracy, and saves a
confusion matrix image next to the checkpoint.
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from src.data.dataset import build_folds
from src.models.mamba_uv import MambaUV
from src.utils.metrics import classification_report_dict, get_confusion_matrix


@torch.no_grad()
def run_eval(model, loader, device, num_classes):
    model.eval()
    all_probs, all_preds, all_targets = [], [], []
    for imgs, targets in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
        all_preds.append(probs.argmax(axis=1))
        all_targets.append(targets.numpy())
    return (
        np.concatenate(all_targets),
        np.concatenate(all_preds),
        np.concatenate(all_probs),
    )


def plot_confusion_matrix(cm: np.ndarray, class_names, out_path: str):
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=6)
    ax.set_yticklabels(class_names, fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Mamba-UV confusion matrix")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds = build_folds(cfg)
    fold = folds[args.fold]

    loader = fold.test_loader if args.split == "test" else fold.val_loader
    if loader is None:
        raise ValueError(
            f"No '{args.split}' loader available for this fold (mode='{cfg['data']['mode']}'). "
            "Use --split val in kfold mode."
        )

    model = MambaUV(cfg).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    y_true, y_pred, y_probs = run_eval(model, loader, device, fold.num_classes)
    report = classification_report_dict(y_true, y_pred, y_probs, fold.num_classes)
    print(f"\n==== Results on '{args.split}' split ====")
    for k, v in report.items():
        print(f"  {k}: {v:.4f}")

    cm = get_confusion_matrix(y_true, y_pred, fold.num_classes)
    out_path = os.path.join(os.path.dirname(args.checkpoint), f"confusion_matrix_{args.split}.png")
    plot_confusion_matrix(cm, fold.class_names, out_path)
    print(f"Confusion matrix saved to {out_path}")


if __name__ == "__main__":
    main()
