"""Model definitions."""

from typing import Tuple, Optional
from torch import Tensor, nn
import torch.nn.functional as F

"""TODO:
- [ ] Loss function calculation inside forward pass?
"""


def build_model(opts):
    """Factory function to build a model based on the provided options."""
    if opts.model_name == "simple1dcnn":
        return Simple1DCNN()
    else:
        raise ValueError(f"Unknown model type: {opts.model_name}")


class Simple1DCNN(nn.Module):
    """A simple 1D CNN for sequence classification."""

    def __init__(
        self,
        in_channels: int = 46,
        num_classes: int = 8,
        bias: bool = True
    ):
        super(Simple1DCNN, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels, 64, 3, padding='same', bias=bias)
        self.conv2 = nn.Conv1d(
            64, 32, 3, padding='same', bias=bias)

        self.classifier = nn.Conv1d(
            32, num_classes, 3, padding='same', bias=bias)

    def forward(
        self,
        x: Tensor,  # (B, T, C)
        targets: Optional[Tensor] = None  # (B, T)
    ) -> Tuple[Tensor, Optional[Tensor]]:
        # x shape (batch_size, sequence_length=700, in_channels=46)
        # C size works as the embedding dimension in language models
        x = x.permute(0, 2, 1)  # (B, C, T)

        # 1D convolution will be applied to the last dimension, we want
        # that to be the sequence lenght (aminoacids) and the the features
        # in this case channels, will be modified
        x = F.relu(self.conv1(x))  # (B, C_new, T)
        x = F.relu(self.conv2(x))  # (B, C_new, T)

        logits: Tensor = self.classifier(x)  # (B, K, T)
        logits = logits.permute(0, 2, 1)  # (B, T, K)

        if targets is not None:
            # Compute loss function value with provided targets
            loss = F.cross_entropy(
                input=logits.reshape(-1, logits.size(-1)),  # (B*T, K)
                target=targets.view(-1),  # (B*T,)
                # see data pipeline, this will exclude padding tokens from loss
                # computation, resulting between B and B*T effective samples
                # as for language models, we have more than B sample per each batch
                ignore_index=-100,
            )  # will result in a scalar tensor
            return logits, loss

        return logits, None
