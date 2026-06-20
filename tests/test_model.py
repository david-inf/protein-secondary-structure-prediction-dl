"""Test model forward pass."""

import sys
sys.path.insert(0, ".")
sys.path.insert(0, "scripts")


def test_simple1dcnn():
    """Test Simple1DCNN model (CPU)."""
    import torch
    from torch import Tensor
    from torch.nn import functional as F
    import numpy as np
    from scripts.models import Simple1DCNN
    from scripts.utils import N

    batch_size = 32
    seq_length = 700
    num_classes = 8
    inputs = torch.randn(batch_size, seq_length, 46)
    targets = torch.empty(batch_size, seq_length, dtype=torch.long).random_(num_classes)

    # Test forward pass
    model = Simple1DCNN()
    logits, loss = model(inputs, targets)
    assert isinstance(logits, Tensor), "Expected logits to be a Tensor."
    assert isinstance(loss, Tensor), "Expected loss value to be Tensor."

    assert logits.shape == (batch_size, seq_length, num_classes), f"Expected (batch_size, 700, 8) got {logits.shape}"
    # Check if loss computation is correct
    assert loss.ndim == 0, f"Loss function value expected to be a scalar, got {loss.shape}."
    assert loss.item() >= 0, "Loss should be non-negative."

    manual_loss = F.cross_entropy(
        input=logits.reshape(-1, logits.size(-1)),  # (B*T, K)
        target=targets.view(-1),  # (B*T,)
    )
    assert torch.isclose(loss, manual_loss), "Loss computation mismatch."

    # NOTE: check function accuracy_q8 in scripts/utils.py
    # Test Q8 accuracy calculation
    # having outputs (logits) with shape (B, T, K) we take the argmax on the K dim
    pred = np.argmax(N(logits), axis=2)
    assert isinstance(pred, np.ndarray), f"Expected numpy array, got {type(pred)}."
    assert pred.shape == (batch_size, seq_length), f"Expected (B, T) got {pred.shape}."
    assert np.all((pred >= 0) & (pred < 8)), "Predictions should be in range 0-7."
    # Now we check against groundtruth, calculating the number of correct predictions
    correct = pred == N(targets)
    acc = np.mean(correct)
    assert isinstance(acc, float)
