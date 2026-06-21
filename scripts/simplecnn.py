"""Baseline CNN model definition."""

from typing import Tuple, Optional
from torch import Tensor, nn
import torch.nn.functional as F

"""TODO:
- [ ] Loss function calculation inside forward pass?
"""


class SimpleBlock(nn.Module):
    """Conv1d -> BatchNorm -> ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size, bias):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels, out_channels, kernel_size, padding='same', bias=bias),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class Simple1DCNN(nn.Module):
    """
    Simple 1D CNN for 8-class protein secondary structure prediction.

    Input: (batch_size, seq_lenght, in_features) e.g. (B, 700, 46)
    Output (logits): (batch_size, seq_length, 8) + optional loss
    """

    def __init__(
        self,
        in_features: int = 46,
        num_classes: int = 8,
        hidden_channels: int = 128,
        kernel_size: int = 7,  # tokens window
        bias: bool = True,
        num_layers: int = 3,
    ):
        super().__init__()

        # Encoder backbone
        layers = []
        for i in range(num_layers):
            in_ch = in_features if i == 0 else hidden_channels
            layers.append(SimpleBlock(in_ch, hidden_channels, kernel_size, bias))
        self.encoder = nn.Sequential(*layers)

        # CLassifier layer
        self.classifier = nn.Conv1d(
            hidden_channels, num_classes, 1, padding='same', bias=bias)

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
        # x = F.relu(self.conv1(x))  # (B, C_new, T)
        # x = F.relu(self.conv2(x))  # (B, C_new, T)
        x = self.encoder(x)

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
