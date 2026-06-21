"""
Reproducibility utilities.

Provides a single ``set_seed`` function that seeds Python's ``random``, NumPy,
PyTorch CPU, and (if available) CUDA generators, and configures cuDNN for
deterministic behaviour.
"""

import random
import numpy as np
import torch


def set_seed(seed: int):
    """
    Seed all random-number generators for reproducible runs.

    Sets seeds for:
    - Python ``random``
    - NumPy
    - PyTorch (CPU)
    - PyTorch CUDA (all GPUs)
    - cuDNN flags (deterministic mode on, benchmark off)

    Parameters
    ----------
    seed : int
        The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
