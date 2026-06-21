"""Training utilities."""

from utils import LOG, set_seeds, N, accuracy_q8, metrics_q8

from dataclasses import dataclass
from typing import Optional, Tuple, List
from pathlib import Path
import time
from tqdm import tqdm
import torch
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


"""TODO:
- [x] Checkpointing utilities
- [ ] Early stopping strategy
- [x] Validation loader
- [ ] Plot learning curves
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
    # Checkpointing
    checkpoint_path: str = "results/ckpts/checkpoint.pt"
    # Logs
    logs_path: str = "results/logs/logs.txt"
    # Learning curves
    curves_path: str = "results/curves/learning_curves.png"


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
        train_start_time = time.time()
        args = self.args

        losses, accs = [], []
        val_losses, val_accs = [], []

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
                    # NOTE: each batch might have a different number of samples due to padding
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
                        train_loss = np.mean(losses[-5:])
                        train_acc = np.mean(accs[-5:])
                        _, _, train_f1 = metrics_q8(N(logits), N(targets))
                        # Validation metrics
                        val_loss, val_acc = None, None
                        if self.eval_loader is not None:
                            val_loss, val_acc = self.evaluate()
                            val_losses.append(val_loss)

                        tepoch.set_postfix(
                            train_loss=train_loss, train_acc=train_acc,
                            train_f1=train_f1)
                        tepoch.update()

        # Final validation step with report in a log file
        runtime = time.time() - train_start_time
        if self.eval_loader is not None:
            _, _ = self.evaluate(print_report=True)

        # Learning curves
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        axs[0].plot(losses, label='Training Loss')
        axs[0].plot(val_losses, label='Validation Loss', color='orange')
        axs[0].set_title('Training/Validation Loss Curve')
        axs[0].set_xlabel('Step')
        axs[0].set_ylabel('Loss')
        axs[0].grid()
        axs[0].set_yscale('log')
        axs[0].legend()
        axs[1].plot(accs, label='Training Accuracy', color='green')
        axs[1].set_title('Training Accuracy Curve')
        axs[1].set_xlabel('Step')
        axs[1].set_ylabel('Accuracy')
        axs[1].grid()
        plt.tight_layout()
        curves_path = Path(args.curves_path)
        curves_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(curves_path)
        LOG.info(f"Learning curves saved to {curves_path}")

        # Training completed
        self.save_checkpoint(args.checkpoint_path)

    @torch.no_grad()
    def evaluate(self, print_report: bool = False) -> Tuple[float, float]:
        """
        Model validation pipeline.

        Return validation loss and accuracy.
        """
        losses, accs = [], []
        # will be a list of f1 scores for each class
        recall_scores, precision_scores, f1_scores = [], [], []

        all_targets, all_preds = [], []

        args = self.args
        self.model.eval()

        for inputs, targets in self.eval_loader:
            inputs, targets = inputs.to(args.device), targets.to(args.device)

            logits, loss = self.model(inputs, targets)

            losses.append(N(loss))
            acc = accuracy_q8(N(logits), N(targets))
            accs.append(acc)

            if print_report:
                all_targets.append(N(targets).flatten())
                all_preds.append(np.argmax(N(logits), axis=-1).flatten())
            # report = classification_report(
            #     N(targets).flatten(), np.argmax(N(logits), axis=-1).flatten(),
            #     labels=list(range(8)), output_dict=True, zero_division=0)
            # LOG.info(f"Classification report:\n{report}")
            # cm = confusion_matrix(
            #     N(targets).flatten(), np.argmax(N(logits), axis=-1).flatten(),
            #     labels=list(range(8)))
            # LOG.info(f"Confusion matrix:\n{cm}")

            # Calculate f1 score for each class
            recall, precision, f1 = metrics_q8(N(logits), N(targets), average=None)
            recall_scores.append(recall)
            precision_scores.append(precision)
            f1_scores.append(f1)

        if print_report:
            report_path = Path(self.args.logs_path)
            report_path.parent.mkdir(parents=True, exist_ok=True)

            with report_path.open("w", encoding="utf-8") as report_file:
                report_file.write(
                    f"Recall scores for each class:\n {np.round(np.mean(recall_scores, axis=0), 4)}\n")
                report_file.write(
                    f"Precision scores for each class:\n {np.round(np.mean(precision_scores, axis=0), 4)}\n")
                report_file.write(
                    f"F1 scores for each class:\n {np.round(np.mean(f1_scores, axis=0), 4)}\n")

                report_file.write(
                    f"Overall accuracy: {np.mean(accs):.4f}\n")
                report_file.write(
                    f"Overall loss: {np.mean(losses):.4f}\n")

                report_file.write(
                    f"Classification report:\n{classification_report(np.concatenate(all_targets), np.concatenate(all_preds), labels=list(range(8)), zero_division=0)}\n")
                report_file.write(
                    f"Confusion matrix:\n{confusion_matrix(np.concatenate(all_targets), np.concatenate(all_preds), labels=list(range(8)))}\n")

            LOG.info(f"Validation report saved to {report_path}")

        return np.mean(losses), np.mean(accs)

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        LOG.info(f"Checkpoint saved to {path}.")

    def learning_curves(self, losses: List[float]) -> None:
        """Plot learning curves for training loss."""
        return
