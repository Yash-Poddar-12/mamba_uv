from typing import Sequence

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


def compute_class_counts(labels: Sequence[int], num_classes: int) -> np.ndarray:
    counts = np.zeros(num_classes, dtype=np.int64)
    for l in labels:
        counts[l] += 1
    return counts


def compute_class_weights(labels: Sequence[int], num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights, normalised so the mean weight is 1.0.
    Feed directly into nn.CrossEntropyLoss(weight=...)."""
    counts = compute_class_counts(labels, num_classes)
    counts = np.clip(counts, a_min=1, a_max=None)   # avoid div-by-zero for empty classes
    inv_freq = 1.0 / counts
    weights = inv_freq * (num_classes / inv_freq.sum())
    return torch.tensor(weights, dtype=torch.float32)


def make_weighted_sampler(labels: Sequence[int], num_classes: int) -> WeightedRandomSampler:
    """Per-sample weights for WeightedRandomSampler so minority-class images
    are drawn roughly as often as majority-class ones during training."""
    counts = compute_class_counts(labels, num_classes)
    counts = np.clip(counts, a_min=1, a_max=None)
    class_weight = 1.0 / counts
    sample_weights = np.array([class_weight[l] for l in labels], dtype=np.float64)
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )


def minority_mask_from_counts(
    labels: Sequence[int], num_classes: int, threshold_fraction: float
) -> np.ndarray:
    """True for samples whose class has fewer than `threshold_fraction` of the
    largest class's count -> used to route samples through heavier augmentation."""
    counts = compute_class_counts(labels, num_classes)
    max_count = counts.max() if counts.max() > 0 else 1
    minority_classes = set(np.where(counts < threshold_fraction * max_count)[0].tolist())
    return np.array([l in minority_classes for l in labels], dtype=bool)
