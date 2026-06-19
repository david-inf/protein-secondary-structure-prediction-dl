"""Utilities."""

import logging
import torch
import numpy as np


# Define logger
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

def set_seeds(seed: int, deterministic: bool = False) -> None:
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy` and `torch`
    (taken from huggingface transformers.set_seed())

    Args:
        seed (`int`):
            The seed to set.
        deterministic (`bool`, *optional*, defaults to `False`):
            Whether to use deterministic algorithms where available. Can slow down training.
    """
    import random
    import torch
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def N(x: torch.Tensor) -> np.ndarray:
    """
    -> Detach from computational graph
    -> Send back to cpu as numpy.ndarray
    """
    return x.detach().cpu().numpy()
