"""Utilities for data loading and preprocessing.
- Data should be in the data/ folder
"""

from utils import LOG
from dataclasses import dataclass
from pathlib import Path
import numpy as np


"""TODO:
- [ ] Do an offline preprocessing to create dataset splits in advance?
"""


@dataclass
class DataArgs:
    """Arguments for data loading and preprocessing."""

    # Data paths
    data_path: Path = Path("data")
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
        args: DataArgs
    ):

        if args is None:
            LOG.warning("No data arguments provided. Using default values.")
            args = DataArgs()

        self.args = args

    def run(self):
        """Run the data pipeline."""
        LOG.info("Running data pipeline...")

        # 1. Load data
        dataset = self._load_data()
        # 2. Build data loader

    def _load_data(self) -> np.ndarray:
        """Load data from disk."""
        LOG.info(" Loading data...")

        # Load npy file
        data = np.load(self.args.data_path, allow_pickle=True)
        LOG.info(f"  Data loaded from '{self.args.data_path}'. Shape: {data.shape}")

        # Check shape and content
        assert isinstance(data, np.ndarray), "Data should be a numpy array."
        assert data.ndim == 2, "Data should be a 2D array."
        assert data.shape[1] == 700 * 57, "Data should have shape (n_samples, 700 * 57)."
        # Reshape to (n_samples, 700, 57)
        n_samples = data.shape[0]
        data = data.reshape(n_samples, 700, 57)
        LOG.info(f"  Data reshaped to (n_samples, 700, 57). New shape: {data.shape}")

        # Take dataset split

        return data

    def _build_loader(self):
        pass


if __name__ == "__main__":
    # Preprocessing .npy.gz dataset files, save them to .npy
    # paths = [
    #     Path("data/cullpdb+profile_5926.npy.gz"),
    #     Path("data/cullpdb+profile_5926_filtered.npy.gz"),
    #     Path("data/cb513+profile_split1.npy.gz")
    # ]
    # for path in paths:
    #     if not path.exists():
    #         LOG.error(f"File '{path}' not found. Please download it and place it in the 'data/' folder.")
    #         exit(1)

    #     data = np.load(path, allow_pickle=True)
    #     np.save(path.with_suffix(""), data)

    #     # Check data shape and content
    #     data_npy = np.load(path.with_suffix(""), allow_pickle=True)
    #     assert isinstance(data_npy, np.ndarray), "Data should be a numpy array."
    #     assert data_npy.ndim == 2, "Data should be a 2D array."
    #     assert data_npy.shape[1] == 700 * 57, "Data should have shape (n_samples, 700 * 57)."

    #     LOG.info(f"Preprocessed '{path}'. Shape: {data_npy.shape}.")
    # LOG.info("Data preprocessing completed.")

    data_args = DataArgs(
        # data_path=Path("data/cb513+profile_split1.npy")
        data_path=Path("data/cullpdb+profile_5926.npy")
        # data_path=Path("data/cullpdb+profile_5926_filtered.npy")
    )
    data_pipeline = DataPipeline(data_args)
    data_pipeline.run()
