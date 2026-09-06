from typing import Dict

import numpy as np
import torch
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    top_k_accuracy_score,
)


@torch.no_grad()
def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    """Top-k accuracy for a single batch of logits (B, num_classes)."""
    topk = logits.topk(k, dim=1).indices              # (B, k)
    correct = topk.eq(targets.unsqueeze(1)).any(dim=1)
    return correct.float().mean().item()


def classification_report_dict(
    y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray, num_classes: int
) -> Dict[str, float]:
    """Precision / Recall / F1 (macro) + Top-1 / Top-3 accuracy, per Sec 3.7."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    top1 = top_k_accuracy_score(
        y_true, y_probs, k=1, labels=list(range(num_classes))
    )
    top3 = top_k_accuracy_score(
        y_true, y_probs, k=min(3, num_classes), labels=list(range(num_classes))
    )
    return {
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "top1_acc": top1,
        "top3_acc": top3,
    }


def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
