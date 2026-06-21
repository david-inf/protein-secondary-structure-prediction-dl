"""Utilities."""

import logging
from typing import Tuple
import torch
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

"""TODO:
- [x] Recall, precision, F1
"""


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


def build_model(opts):
    """Factory function to build a model based on the provided options."""
    if opts.model_name == "simple1dcnn":
        from simplecnn import Simple1DCNN
        return Simple1DCNN()
    elif opts.model_name == "customcnn":
        from customcnn import CustomCNN
        return CustomCNN()
    else:
        raise ValueError(f"Unknown model type: {opts.model_name}")


def accuracy_q8(logits: np.ndarray, targets: np.ndarray) -> float:
    """Calculate the Q8 accuracy."""
    # having logits with shape (B, T, K) we take the argmax on the K dim
    pred = np.argmax(logits, axis=2)
    assert np.all((pred >= 0) & (pred < 8)), "Predictions should be in range 0-7."
    # Check against groundtruth and average the result
    acc = np.mean(pred == targets)
    return acc


def metrics_q8(logits: np.ndarray, targets: np.ndarray, average: str = 'micro') -> Tuple:
    """Calculate the Recall, Precision, and F1 on Q8."""
    pred: np.ndarray = np.argmax(logits, axis=2)
    # assert np.all((pred >= 0) & (pred < 8)), "Predictions should be in range 0-7."
    # Reshape to 1D arrays for metric calculations
    pred = pred.flatten()
    targets = targets.flatten()

    if average is None:
        # Return per-class metrics
        num_classes = 8

        # This will return arrays of shape (8,) for each metric
        # NOTE: might happen that some classes are not present in the batch
        recall = recall_score(
            targets, pred, average=None, labels=list(range(num_classes)), zero_division=0)
        precision = precision_score(
            targets, pred, average=None, labels=list(range(num_classes)), zero_division=0)
        f1 = f1_score(
            targets, pred, average=None, labels=list(range(num_classes)), zero_division=0)

        return recall, precision, f1

    recall = recall_score(targets, pred, average=average)
    precision = precision_score(targets, pred, average=average)
    f1 = f1_score(targets, pred, average=average)

    return recall, precision, f1
