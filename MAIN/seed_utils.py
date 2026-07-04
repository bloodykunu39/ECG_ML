"""
seed_utils.py
-------------
Shared seeding utilities for multi-seed experiments.
Seeds are centred on the original fixed seed (42).
"""
import random
import numpy as np
import torch

# 5 seeds centred on 42  (reviewer-reproducible, easy to report)
SEEDS = [40, 41, 42, 43, 44]


def set_seed(seed: int) -> None:
    """Set all RNG sources deterministically for a given seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
