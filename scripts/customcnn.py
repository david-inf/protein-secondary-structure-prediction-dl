"""Improving over baseline."""

from typing import Optional, Tuple
from dataclasses import dataclass
import torch
from torch import Tensor, nn
import torch.nn.functional as F

# TODO: since we're computing the loss in the forward pass,
# this allows to maybe use a custom loss function just by
# using this model instead of a yaml parameter


@dataclass
class ModelArgs:
    # Vocab will be the length of the amino acids one-hot encoding
    # 21 amino acids + 1 special token (padding)
    vocab_size: int = 22
    embd_dim: int = 64

    pssm_dim: int = 22  # PSSM + padding
    terminals_dim: int = 2  # N- and C- terminals

    conv_num_layers: int = 5
    conv_hidden_channels: int = 256
    conv_kernel_size: int = 7
    bias: bool = True
    num_classes: int = 8


class Block(nn.Module):
    """Conv1d -> BatchNorm -> ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size, bias, layer_idx=0):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels, out_channels, kernel_size,
                padding='same', bias=bias,
                dilation=2**layer_idx
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class CustomCNN(nn.Module):
    """
    Custom model fusing aminoacid embeddings with PSSM and terminals.

    1) Amino acids embeddings branch
       Creates a representation of the amino acids using an embedding layer.
    2) PSSM projection branch
       Projects the PSSM features to match the embedding dimension.
    3) Fusion of embeddings and PSSM
       
    4) Convolutional layers
       Takes the fused (concatenated) features.
    5) Classifier

    This way I'm treating the dataset as the actual amino acid sequence +
    I'm adding extra info using these PSSM features and terminals. Though the
    dataset might be too small for such a training.
    """

    def __init__(self, args: Optional[ModelArgs] = None):
        super().__init__()

        if args is None:
            args = ModelArgs()
        self.args = args

        # Embeddings branch
        # TODO: might be interesting to inspect
        self.token_embd = nn.Embedding(
            num_embeddings=args.vocab_size,
            embedding_dim=args.embd_dim,
            # map NoSeq to 0 vector, might be redundant with ignore_index in loss function
            padding_idx=21,
        )

        # PSSM projection branch (expands 24 features to embd_dim)
        # TODO: might be interesting to inspect
        self.pssm_proj = nn.Sequential(
            nn.Linear(args.pssm_dim, args.embd_dim, bias=args.bias),
            nn.ReLU()
        )

        # Concatenated embeddings + PSSM + terminals
        cnn_input_dim = 2 * args.embd_dim + args.terminals_dim

        # Input projection and convolutional layers
        self.input_proj = nn.Linear(
            cnn_input_dim, args.conv_hidden_channels, bias=args.bias)

        self.conv_layers = nn.ModuleList(
            Block(
                in_channels=args.conv_hidden_channels, out_channels=args.conv_hidden_channels,
                kernel_size=args.conv_kernel_size, bias=args.bias, layer_idx=i
            )
            for i in range(args.conv_num_layers)
        )

        # Final classification head
        # TODO: maybe an MLP?
        self.classifier = nn.Conv1d(
            args.conv_hidden_channels, args.num_classes, kernel_size=1, padding='same', bias=args.bias)

    def forward(
        self,
        x: Tensor,  # (B, T, C) considering 46 channels as input features
        targets: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass:
        - Embeddings branch
        - PSSM projection branch
        - Fusion
        - Convolutional layers
        - Classifier
        """
        args = self.args
        # 0) Decompose input

        # take the first 22 channels as the one-hot encoded amino acids
        # we'll see this as a tokenization step already performed
        aa = x[:, :, :22]  # (B, T, 22)
        assert aa.shape[2] == 22, f"Expected 22 features for amino acids one-hot encoding, got {aa.shape[2]}"
        # from one-hot to token ids
        aa_idx = aa.argmax(dim=2)  # (B, T) in 0-21 range

        # take terminals
        terminals = x[:, :, 22:24]  # (B, T, 2)
        assert terminals.shape[2] == args.terminals_dim, f"Expected {args.terminals_dim} features for terminals, got {terminals.shape[2]}"

        # take the remaining 22 channels PSSM features
        pssm = x[:, :, 24:]  # (B, T, 22)
        assert pssm.shape[2] == 22, f"Expected 22 features for PSSM, got {pssm.shape[2]} from input with {x.shape[2]} features."

        # 1) Amino acids embeddings branch
        aa_embd = self.token_embd(aa_idx)  # (B, T, d)

        # 2) PSSM projection branch
        pssm_proj = self.pssm_proj(pssm)  # (B, T, d)

        # 3) Embeddings, PSSM, and terminals fusion -> (B, T, 2d + 2)
        x = torch.cat((aa_embd, pssm_proj, terminals), dim=2)
        assert x.shape[2] == 2 * args.embd_dim + args.terminals_dim, f"Expected 2 * {args.embd_dim} + {args.terminals_dim} features after fusion, got {x.shape[2]}"
        x = self.input_proj(x)  # (B, T, hidden)

        # 4) Convolutional layers and classification head
        x = x.permute(0, 2, 1)  # (B, hidden, T)
        layer: Block
        for layer in self.conv_layers:
            x = x + layer(x)  # skip connections

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
