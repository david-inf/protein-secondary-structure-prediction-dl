"""Training utilities."""

from utils import LOG, set_seeds, N, accuracy_q8, metrics_q8

from dataclasses import dataclass
from typing import Optional, Tuple
# import time
from tqdm import tqdm
import torch
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


"""TODO:
- [ ] Checkpointing utilities
- [x] Validation loader
"""


@dataclass
class TrainArgs:
    """Arguments for training."""

    seed: int = 42
    epochs: int = 10
    batch_size: int = 32
    # Optimizer stuff
    learning_rate: float = 1e-3
    momentum: float = 0.0
    # Simple training on a single device (CPU or GPU).
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class Trainer:
    def __init__(
        self,
        args: TrainArgs,
        model: nn.Module,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader] = None
    ):

        if args is None:
            LOG.warning("No training arguments provided. Using default values.")
            args = TrainArgs()
        if model is None:
            raise RuntimeError("`Trainer` requires a model argument.")
        if train_loader is None:
            raise RuntimeError("`Trainer` requires a train_loader argument.")

        self.args = args

        # Model
        self.model = model.to(self.args.device)
        LOG.info(f"Training on device: {self.args.device}")

        # Check CUDA version version and PyTorch version for compatibility
        if self.args.device == "cuda":
            cuda_version = torch.version.cuda
            torch_version = torch.__version__
            LOG.info(f"CUDA version: {cuda_version}")
            LOG.info(f"PyTorch version: {torch_version}\n")

        # Data loaders
        self.train_loader = train_loader
        self.eval_loader = eval_loader

        # Set a fixed optimizer for now.
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
        )

    def train(self) -> None:
        """Training loop."""
        # train_start_time = time.time()
        args = self.args

        losses, accs = [], []

        set_seeds(args.seed)
        LOG.info(f"Random seeds set to {args.seed}.")

        for epoch in range(1, args.epochs + 1):
            # epoch_start_time = time.time()

            with tqdm(self.train_loader, unit='batch') as tepoch:
                for batch_idx, (inputs, targets) in enumerate(tepoch):
                    self.model.train()
                    tepoch.set_description(f"{epoch:03d}")

                    # Fetch data and move to device
                    inputs, targets = inputs.to(args.device), targets.to(args.device)

                    # Forward pass
                    loss: Tensor
                    logits, loss = self.model(inputs, targets)

                    # Calculate metrics
                    losses.append(N(loss))
                    acc = accuracy_q8(N(logits), N(targets))
                    accs.append(acc)

                    # Backward pass and optimization step
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    if batch_idx % 20 == 0 or batch_idx == len(self.train_loader) - 1:
                        # Training metrics
                        train_loss = np.mean(losses[-20:])
                        train_acc = np.mean(accs[-20:])
                        _, _, train_f1 = metrics_q8(N(logits), N(targets))
                        # Validation metrics
                        val_loss, val_acc = None, None
                        if self.eval_loader is not None:
                            val_loss, val_acc = self.evaluate()

                        tepoch.set_postfix(
                            train_loss=train_loss, train_acc=train_acc,
                            train_f1=train_f1)
                        tepoch.update()        

        # Final validation step with report

        # Training completed
        # runtime = time.time() - train_start_time

    @torch.no_grad()
    def evaluate(self, print_report: bool = False) -> Tuple[float, float]:
        """
        Model validation pipeline.

        Return validation loss and accuracy.
        """
        losses, accs = [], []
        args = self.args
        self.model.eval()

        for inputs, targets in self.eval_loader:
            inputs, targets = inputs.to(args.device), targets.to(args.device)

            logits, loss = self.model(inputs, targets)

            losses.append(N(loss))
            acc = accuracy_q8(N(logits), N(targets))
            accs.append(acc)

        return np.mean(losses), np.mean(accs)
