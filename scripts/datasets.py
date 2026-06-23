"""Utilities for data loading and preprocessing.
- Data should be in the data/ folder
"""

from utils import LOG
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple
from collections import defaultdict, OrderedDict
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt


"""TODO:
- [ ] Do an offline preprocessing to create dataset splits in advance?
- [x] Maybe leave the NoSeq feature in the data, and let the model learn to ignore it?
- [x] Collate function for addressing padding and one-hot encoding of labels
- [x] Plot class distribution in the dataset
- [ ] Normalize data?
"""

IGNORE_INDEX = -100  # for CrossEntropyLoss, ignore the NoSeq class in the labels

SPLITS = {
    "cullpdb": {
        "train": np.arange(0, 5430),
        "test": np.arange(5435, 5690),
        "eval": np.arange(5690, 5926),
    },
    # "cb513": {
    #     "test": np.arange(0, 514),
    # },
    "cullpdb_filtered": {
        "train": np.arange(0, int(0.8 * 5365)),
        "eval": np.arange(int(0.8 * 5365), 5365),
    }
}


def my_collate(batch: List[Tuple[Tensor, Tensor]]):
    """Custom collate function to separate features and labels."""
    # Get Xs
    data = [item[0] for item in batch]  # shape (batch_size, 700, 45)

    # Ys from one-hot to integer
    targets = []  # will result in shape (batch_size, 700)
    for item in batch:
        # y has shape (700, 9)
        y = item[1].argmax(dim=1)  # shape (700,)
        # check if padding token is present
        mask = (y == 8)  # NoSeq token is class 8
        y[mask] = IGNORE_INDEX  # set padding token to IGNORE_INDEX for CrossEntropyLoss
        targets.append(y)

    return torch.stack(data), torch.stack(targets)


@dataclass
class DataArgs:
    """Arguments for data loading and preprocessing."""

    # Data paths
    dataset_name: str = "cullpdb"  # cullpdb, cullpdb_filtered, cb513
    split: str = "train"  # train, eval, test

    # Data loading parameters
    seed: int = 42
    batch_size: int = 32
    num_workers: int = 2
    shuffle: bool = True


class DataPipeline:
    """Data loading and preprocessing pipeline."""

    def __init__(
        self,
        args: Optional[DataArgs] = None
    ):

        if args is None:
            LOG.warning("No data arguments provided. Using default values.")
            args = DataArgs()

        self.args = args

        # Take the corresponding dataset path based on the provided dataset name
        self.dataset_path: Path
        if self.args.dataset_name == "cullpdb":
            self.dataset_path = Path("data/cullpdb+profile_5926.npy")
        elif self.args.dataset_name == "cullpdb_filtered":
            self.dataset_path = Path("data/cullpdb+profile_5926_filtered.npy")
        elif self.args.dataset_name == "cb513":
            self.dataset_path = Path("data/cb513+profile_split1.npy")
        if self.dataset_path is None:
            raise ValueError(f"Cannot find dataset path for '{self.args.dataset_name}'.")

    def run(self):
        """Run the data pipeline."""
        LOG.info("Running data pipeline...")

        # 1. Load data
        dataset = self._load_data(self.dataset_path)

        # 2. Build data loader
        dataloader = self._build_loader(dataset)

        return dataloader

    def _load_data(self, path: Path) -> TensorDataset:
        """Load data from disk."""
        LOG.info(" Loading data...")

        # Load .npy file
        data = np.load(path, allow_pickle=True)
        LOG.info(f"  Data loaded from '{path}'. Shape: {data.shape}")

        # Check shape and content
        assert isinstance(data, np.ndarray), "Data should be a numpy array."
        assert data.ndim == 2, "Data should be a 2D array."
        assert data.shape[1] == 700 * 57, "Data should have shape (n_samples, 700 * 57)."
        # Reshape to (n_samples, context_length, n_embd)
        data = data.reshape(-1, 700, 57)
        LOG.info(f"  Data reshaped to (n_samples, 700, 57). New shape: {data.shape}")

        # Take dataset split
        if self.args.dataset_name in SPLITS:
            # get predefined split indices for the dataset
            split_indices = SPLITS[self.args.dataset_name].get(self.args.split)
            if split_indices is None:
                raise ValueError(f"Unknown split '{self.args.split}' for dataset '{self.args.dataset_name}'.")
            data = data[split_indices]
            LOG.info(f"  Data split '{self.args.split}' selected. New shape: {data.shape}")
        else:
            LOG.warning(f"  No predefined splits for dataset '{self.args.dataset_name}'. Using full dataset.")

        # Define Xs and Ys (both will incude the NoSeq feature)
        Xs = np.concatenate([
            data[:, :, :22],    # amino-acid residues (one-hot encoding) | 21 features + 1 (NoSeq)
            data[:, :, 31:33],  # N- and C- terminals | 2 features
            data[:, :, 35:57],  # PSSM features | 22 features
        ], axis=-1)  # 46 features in total

        # Normalization (not much improvement)
        # Xs_flat = Xs.reshape(-1, Xs.shape[-1])  # shape (n_samples * 700, 46)
        # mean = Xs_flat.mean(axis=0)
        # std = np.where(Xs_flat.std(axis=0) < 1e-6, 1, Xs_flat.std(axis=0))
        # Xs = (Xs - mean) / std  # normalize each feature to have mean 0 and std 1
        # LOG.info(f"  Data normalized. Mean: {mean}, Std: {std}")

        # One-hot encoding of secondary structure labels | 8 classes + 1 (NoSeq)
        # a collate function will use the NoSeq to change label to -100 when detected
        Ys = data[:, :, 22:31]

        dataset = TensorDataset(torch.from_numpy(Xs).float(), torch.from_numpy(Ys).float())
        return dataset

    def _build_loader(self, dataset: TensorDataset) -> DataLoader:
        """Build a PyTorch DataLoader from the dataset."""
        LOG.info(" Building DataLoader...")

        # Create DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=self.args.shuffle,
            num_workers=self.args.num_workers,
            pin_memory=True,
            collate_fn=my_collate,
        )
        LOG.info(f"  DataLoader created with batch size {self.args.batch_size}.")

        return dataloader


if __name__ == "__main__":
    # Preprocessing .npy.gz dataset files, save them to .npy
    paths = [
        Path("data/cullpdb+profile_5926.npy.gz"),
        Path("data/cullpdb+profile_5926_filtered.npy.gz"),
        Path("data/cb513+profile_split1.npy.gz")
    ]
    for path in paths:
        if not path.exists():
            LOG.error(f"File '{path}' not found. Please download it and place it in the 'data/' folder.")
            exit(1)

        data = np.load(path, allow_pickle=True)
        np.save(path.with_suffix(""), data)

        # Check data shape and content
        data_npy = np.load(path.with_suffix(""), allow_pickle=True)
        assert isinstance(data_npy, np.ndarray), "Data should be a numpy array."
        assert data_npy.ndim == 2, "Data should be a 2D array."
        assert data_npy.shape[1] == 700 * 57, "Data should have shape (n_samples, 700 * 57)."

        LOG.info(f"Preprocessed '{path}'. Shape: {data_npy.shape}.")
    LOG.info("Data preprocessing completed.")


    # Inspection
    import argparse
    parser = argparse.ArgumentParser(description="Inspect dataset class distribution.")
    parser.add_argument("--dataset_name", type=str, default="cullpdb", choices=["cullpdb", "cullpdb_filtered"])
    args = parser.parse_args()

    pipe_args = DataArgs(dataset_name=args.dataset_name, split="train")
    pipe = DataPipeline(args=pipe_args)
    loader = pipe.run()

    distrib = defaultdict(int)
    targets: Tensor
    for _, targets in loader:
        # targets with shape (B, T)
        vals, counts = targets.unique(return_counts=True)
        for i, val in enumerate(vals):
            if val.item() == IGNORE_INDEX:
                continue  # skip padding tokens
            distrib[f"{int(val.item())}"] += counts[i].item()
    assert len(distrib) == 8, "There should be 8 classes (0-7) in the distribution."
    # sort keys
    distrib_sort = OrderedDict(sorted(distrib.items(), key=lambda item: int(item[0])))

    LOG.info(f"Class distribution in {pipe_args.dataset_name} | split: {pipe_args.split}")
    for cls, count in distrib_sort.items():
        LOG.info(f" Class {cls}: {count} samples")

    # Dump class weights for cross-entropy loss
    num_samples = sum(distrib.values())
    LOG.info(f"Total samples: {num_samples}")
    class_weights = []
    for i in range(8):
        weight = num_samples / (8 * distrib.get(str(i), 1e-6))
        if weight < 10.:
            class_weights.append(weight)
        else:
            class_weights.append(10.)  # cap weights to avoid instability during training
    torch.save(torch.tensor(class_weights), f"results/distrib/{pipe_args.dataset_name}_weights.pt")
    LOG.info(f"Class weights for CrossEntropyLoss: {class_weights}")

    # Plot distribution of classes (tokens)
    plt.bar(distrib_sort.keys(), distrib_sort.values())
    plt.xlabel("Class (token) index")
    plt.ylabel("Count")
    plt.title(f"Class distribution in {pipe_args.dataset_name} | split: {pipe_args.split}")
    plt.grid(True, ls="--", lw=0.5)
    plt.yscale("log")  # log scale for better visibility of underrepresented classes
    path = Path("results/distrib")
    path.mkdir(parents=True, exist_ok=True)
    plt.savefig(path / f"{pipe_args.dataset_name}_{pipe_args.split}.png")

    # print("Checking sample 0, amino-acid (token) 0:")
    # x, y = dataset[0]  # take x and y from first sample
    # print(f" x shape: {x.shape} | y shape: {y.shape}")
    # print(f" y (one-hot) -> {y[0, :9]}")
    # print(f" [0, 21] -> {x[0, :21]}")
    # print(f" [31, 33] -> {x[0, 21:23]}")
    # print(f" [35, 57] -> {x[0, 23:45]}")

    # print("\nChecking sample 700, amino-acid (token) 700:")
    # print(f" y (one-hot) -> {y[-1, :9]}")
    # print(f" [0, 21] -> {x[-1, :21]}")
    # print(f" [31, 33] -> {x[-1, 21:23]}")
    # print(f" [35, 57] -> {x[-1, 23:45]}")
