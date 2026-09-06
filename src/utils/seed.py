import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Seed python/numpy/torch (CPU + all CUDA devices) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
