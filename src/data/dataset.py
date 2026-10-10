"""
Loads AQUA20 either from the HuggingFace Hub (taufiktrf/AQUA20) or from a
local folder you've already downloaded/exported (train/<class>/*.jpg,
test/<class>/*.jpg -- the classic ImageFolder layout), and produces
PyTorch DataLoaders for both the "holdout" (official train/test split) and
"kfold" (5-fold stratified CV, matching the paper's eval protocol) modes.

Set data.source in configs/config.yaml to "huggingface" (default) or
"local" (point data.local_root at the folder containing train/ and test/).

IMPORTANT: images are kept in [0, 1] (Resize -> ToTensor, NO mean/std
normalization). The Degradation Map Estimator (src/models/dme.py) relies on
physically meaningful pixel intensities (channel ratios, dark-channel
minima) -- shifting/scaling those with ImageNet-style normalization would
corrupt the physics signal it computes.
"""
import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image, ImageFile
from datasets import load_dataset, concatenate_datasets, DatasetDict
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader

# scraped / iNaturalist downloads occasionally contain truncated files --
# load what's there instead of crashing a DataLoader worker mid-epoch
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

from src.data.class_balance import (
    compute_class_weights,
    make_weighted_sampler,
    minority_mask_from_counts,
)


# ----------------------------------------------------------------------
# Transforms
# ----------------------------------------------------------------------
def build_transforms(cfg) -> "tuple[Callable, Callable, Callable]":
    """Returns (train_transform, minority_train_transform, eval_transform)."""
    size = cfg["data"]["image_size"]
    aug = cfg["augmentation"]

    base_train = [
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(p=aug["horizontal_flip_p"]),
        transforms.RandomRotation(degrees=aug["rotation_degrees"]),
        transforms.ColorJitter(**aug["color_jitter"]),
        transforms.ToTensor(),  # -> [0, 1], no Normalize (see module docstring)
        transforms.RandomErasing(p=aug["random_erasing_p"]),
    ]
    train_transform = transforms.Compose(base_train)

    minority_train = [
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(p=aug["horizontal_flip_p"]),
        transforms.RandomRotation(degrees=aug["minority_extra_rotation_degrees"]),
        transforms.ColorJitter(**aug["color_jitter"]),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.ToTensor(),
        transforms.RandomErasing(p=aug["minority_random_erasing_p"]),
    ]
    minority_transform = transforms.Compose(minority_train)

    eval_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    return train_transform, minority_transform, eval_transform


# ----------------------------------------------------------------------
# Torch Dataset wrapper around a HuggingFace split
# ----------------------------------------------------------------------
class HFImageDataset(Dataset):
    def __init__(
        self,
        hf_split,
        indices: Optional[Sequence[int]],
        transform: Callable,
        minority_mask: Optional[np.ndarray] = None,
        minority_transform: Optional[Callable] = None,
    ):
        self.hf_split = hf_split
        self.indices = list(indices) if indices is not None else list(range(len(hf_split)))
        self.transform = transform
        self.minority_mask = minority_mask
        self.minority_transform = minority_transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        item = self.hf_split[idx]
        img: Image.Image = item["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        label = int(item["label"])

        if self.minority_mask is not None and self.minority_mask[i] and self.minority_transform is not None:
            img_t = self.minority_transform(img)
        else:
            img_t = self.transform(img)
        return img_t, label

    def labels(self) -> List[int]:
        return [int(self.hf_split[idx]["label"]) for idx in self.indices]


class LocalFolderImageDataset(Dataset):
    """Same behaviour as HFImageDataset, but reads from a list of
    (filepath, label) tuples produced by torchvision.datasets.ImageFolder
    -- used when data.source == 'local'."""

    def __init__(
        self,
        samples: Sequence[tuple],
        indices: Optional[Sequence[int]],
        transform: Callable,
        minority_mask: Optional[np.ndarray] = None,
        minority_transform: Optional[Callable] = None,
    ):
        self.samples = samples
        self.indices = list(indices) if indices is not None else list(range(len(samples)))
        self.transform = transform
        self.minority_mask = minority_mask
        self.minority_transform = minority_transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        path, label = self.samples[idx]
        try:
            img = default_loader(path)  # PIL, already RGB (handles RGBA/L/P PNGs)
        except Exception as e:  # corrupt file -> use the next sample instead of crashing
            print(f"[WARN] unreadable image skipped: {path} ({e})")
            return self.__getitem__((i + 1) % len(self))
        if self.minority_mask is not None and self.minority_mask[i] and self.minority_transform is not None:
            img_t = self.minority_transform(img)
        else:
            img_t = self.transform(img)
        return img_t, int(label)


@dataclass
class Fold:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: Optional[DataLoader]
    class_names: List[str]
    class_weights: torch.Tensor
    num_classes: int


def _make_synthetic_dataset(cfg, class_names: List[str], transform: Callable) -> Optional[Dataset]:
    """Optional: fold in images produced by synthetic_augment_openai.py.
    Expects <synthetic_data_dir>/<class_name>/*.jpg with class_name matching
    the AQUA20 label names exactly."""
    root = cfg["data"].get("synthetic_data_dir")
    if not root:
        return None

    folder = ImageFolder(root, transform=transform)
    # ImageFolder assigns its own 0..N label ids based on alphabetical
    # subfolder order -- remap them onto AQUA20's label ids.
    name_to_aqua20_id = {name: i for i, name in enumerate(class_names)}
    remap = {folder.class_to_idx[name]: name_to_aqua20_id[name]
             for name in folder.classes if name in name_to_aqua20_id}

    class _Remapped(Dataset):
        def __init__(self, base, remap):
            self.base, self.remap = base, remap

        def __len__(self):
            return len(self.base)

        def __getitem__(self, i):
            x, y = self.base[i]
            return x, self.remap[y]

    return _Remapped(folder, remap)


def _loader(dataset: Dataset, cfg, shuffle: bool, sampler=None) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        drop_last=shuffle,
    )



# ----------------------------------------------------------------------
# Flat <root>/<class_name>/*.jpg|png layout (no train/test split on disk)
# ----------------------------------------------------------------------
def _safe_stratified_split(labels, frac: float, seed: int):
    """Per-class split that never crashes on tiny classes.
    Returns (keep_idx, held_idx), both sorted index arrays into `labels`.
    - class with >= 2 samples: at least 1 sample in each side
    - class with 1 sample: stays entirely in keep_idx"""
    rng = np.random.RandomState(seed)
    labels = np.asarray(labels)
    keep, held = [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        n = len(idx)
        if n < 2:
            keep.extend(idx.tolist())
            continue
        n_held = min(max(int(round(n * frac)), 1), n - 1)
        held.extend(idx[:n_held].tolist())
        keep.extend(idx[n_held:].tolist())
    return np.array(sorted(keep), dtype=np.int64), np.array(sorted(held), dtype=np.int64)


def _holdout_split(labels, frac: float, seed: int):
    """sklearn's stratified split when it is possible (keeps the original
    AQUA20 behaviour identical); per-class fallback when some class is too
    small for sklearn (needs >= 2 samples per class)."""
    labels = np.asarray(labels)
    idx = np.arange(len(labels))
    try:
        return train_test_split(idx, test_size=frac, stratify=labels, random_state=seed)
    except ValueError:
        print("[data] a class has < 2 train samples -> using per-class safe split for val")
        return _safe_stratified_split(labels, frac, seed)


def _scan_flat_folder(root: str):
    """Scan <root>/<class_name>/<image files>. Returns (class_names, samples)
    with samples = [(path, label_id), ...]. Class order is alphabetical, so
    label ids are stable between train.py and evaluate.py."""
    if not os.path.isdir(root):
        raise FileNotFoundError(f"data.flat_root does not exist: {root}")
    class_dirs = sorted(e.name for e in os.scandir(root) if e.is_dir())
    class_names: List[str] = []
    samples: List[tuple] = []
    empty = []
    for name in class_dirs:
        cdir = os.path.join(root, name)
        files = sorted(f for f in os.listdir(cdir)
                       if os.path.splitext(f)[1].lower() in IMG_EXTS)
        if not files:
            empty.append(name)
            continue
        label = len(class_names)
        class_names.append(name)
        samples.extend((os.path.join(cdir, f), label) for f in files)
    if empty:
        print(f"[data] skipped {len(empty)} class folder(s) with no images: {empty}")

    counts = np.bincount([l for _, l in samples], minlength=len(class_names))
    order = np.argsort(counts)
    print(f"[data] flat dataset: {len(class_names)} classes, {len(samples)} images | "
          f"per-class min/median/max = {counts.min()}/{int(np.median(counts))}/{counts.max()} | "
          f"classes with <10 images: {int((counts < 10).sum())}")
    print("[data] 8 smallest classes: " +
          ", ".join(f"{class_names[i]}({counts[i]})" for i in order[:8]))
    return class_names, samples


def _load_pool(cfg):
    """Returns (class_names, train_pool, test_pool, kind) where kind is
    'hf' or 'local' and each pool is the object build_folds' dataset
    factories know how to index into."""
    source = cfg["data"].get("source", "huggingface")

    if source == "local":
        import os
        root = cfg["data"]["local_root"]
        train_folder = ImageFolder(os.path.join(root, "train"))
        test_folder = ImageFolder(os.path.join(root, "test"))
        assert train_folder.classes == test_folder.classes, (
            f"train/test class folders don't match:\n"
            f"  train: {train_folder.classes}\n  test: {test_folder.classes}"
        )
        return train_folder.classes, train_folder.samples, test_folder.samples, "local"

    elif source == "flat":
        class_names, samples = _scan_flat_folder(cfg["data"]["flat_root"])
        labels = np.array([l for _, l in samples])
        trainval_idx, test_idx = _safe_stratified_split(
            labels, cfg["data"].get("test_fraction", 0.15), cfg["data"]["seed"]
        )
        train_pool = [samples[i] for i in trainval_idx]
        test_pool = [samples[i] for i in test_idx]
        print(f"[data] split -> train+val: {len(train_pool)} | test: {len(test_pool)}")
        return class_names, train_pool, test_pool, "local"

    elif source == "huggingface":
        ds: DatasetDict = load_dataset(cfg["data"]["hf_dataset_id"])
        class_names = ds["train"].features["label"].names
        return class_names, ds["train"], ds["test"], "hf"

    else:
        raise ValueError(f"Unknown data.source: {source!r} (use 'huggingface', 'local' or 'flat')")


def _make_dataset(kind, pool, indices, transform, minority_mask=None, minority_transform=None):
    if kind == "hf":
        return HFImageDataset(pool, indices, transform, minority_mask, minority_transform)
    return LocalFolderImageDataset(pool, indices, transform, minority_mask, minority_transform)


def _pool_labels(kind, pool) -> np.ndarray:
    if kind == "hf":
        return np.array(pool["label"])
    return np.array([label for _, label in pool])


def _combine_pools(kind, pool_a, pool_b):
    if kind == "hf":
        return concatenate_datasets([pool_a, pool_b])
    return list(pool_a) + list(pool_b)


def build_folds(cfg) -> List[Fold]:
    """Returns a list with a single Fold for mode == 'holdout', or
    cfg.data.k_folds Folds for mode == 'kfold'."""
    seed = cfg["data"]["seed"]
    class_names, train_pool, test_pool, kind = _load_pool(cfg)
    # num_classes always follows the data on disk; the model head and the
    # loss read cfg["data"]["num_classes"], so overwrite it here.
    num_classes = len(class_names)
    if cfg["data"].get("num_classes") != num_classes:
        print(f"[data] num_classes: config said {cfg['data'].get('num_classes')}, "
              f"data has {num_classes} -> using {num_classes}")
    cfg["data"]["num_classes"] = num_classes

    train_tf, minority_tf, eval_tf = build_transforms(cfg)
    threshold = cfg["augmentation"]["minority_boost_threshold"]

    folds: List[Fold] = []

    if cfg["data"]["mode"] == "holdout":
        all_labels = _pool_labels(kind, train_pool)
        all_idx = np.arange(len(all_labels))
        train_idx, val_idx = _holdout_split(all_labels, cfg["data"]["val_fraction"], seed)

        train_labels_subset = all_labels[train_idx]
        minority_mask = minority_mask_from_counts(train_labels_subset, num_classes, threshold)

        train_ds = _make_dataset(kind, train_pool, train_idx, train_tf, minority_mask, minority_tf)
        val_ds = _make_dataset(kind, train_pool, val_idx, eval_tf)
        test_ds = _make_dataset(kind, test_pool, None, eval_tf)

        synthetic_ds = _make_synthetic_dataset(cfg, class_names, train_tf)
        full_train = ConcatDataset([train_ds, synthetic_ds]) if synthetic_ds else train_ds

        class_weights = compute_class_weights(train_labels_subset, num_classes)
        sampler = make_weighted_sampler(train_labels_subset, num_classes) \
            if cfg["imbalance"]["use_weighted_sampler"] else None

        folds.append(Fold(
            train_loader=_loader(full_train, cfg, shuffle=(sampler is None), sampler=sampler),
            val_loader=_loader(val_ds, cfg, shuffle=False),
            test_loader=_loader(test_ds, cfg, shuffle=False),
            class_names=class_names,
            class_weights=class_weights,
            num_classes=num_classes,
        ))
        return folds

    elif cfg["data"]["mode"] == "kfold":
        combined = _combine_pools(kind, train_pool, test_pool)
        labels = _pool_labels(kind, combined)
        skf = StratifiedKFold(n_splits=cfg["data"]["k_folds"], shuffle=True, random_state=seed)

        for train_idx, val_idx in skf.split(np.zeros(len(labels)), labels):
            train_labels_subset = labels[train_idx]
            minority_mask = minority_mask_from_counts(train_labels_subset, num_classes, threshold)

            train_ds = _make_dataset(kind, combined, train_idx, train_tf, minority_mask, minority_tf)
            val_ds = _make_dataset(kind, combined, val_idx, eval_tf)

            synthetic_ds = _make_synthetic_dataset(cfg, class_names, train_tf)
            full_train = ConcatDataset([train_ds, synthetic_ds]) if synthetic_ds else train_ds

            class_weights = compute_class_weights(train_labels_subset, num_classes)
            sampler = make_weighted_sampler(train_labels_subset, num_classes) \
                if cfg["imbalance"]["use_weighted_sampler"] else None

            folds.append(Fold(
                train_loader=_loader(full_train, cfg, shuffle=(sampler is None), sampler=sampler),
                val_loader=_loader(val_ds, cfg, shuffle=False),
                test_loader=None,   # in kfold mode, val IS the held-out estimate for that fold
                class_names=class_names,
                class_weights=class_weights,
                num_classes=num_classes,
            ))
        return folds

    else:
        raise ValueError(f"Unknown data.mode: {cfg['data']['mode']!r} (use 'holdout' or 'kfold')")