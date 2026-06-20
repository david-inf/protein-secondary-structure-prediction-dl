"""Test dataset loading and preprocessing pipeline."""

import sys
sys.path.insert(0, ".")
sys.path.insert(0, "scripts")


def test_loading():
    """Test loading of datasets."""
    from scripts.datasets import DataPipeline, DataArgs
    import torch
    from torch.utils.data import TensorDataset

    # Test loading cullpdb dataset
    args = DataArgs(dataset_name="cullpdb", split="train")
    pipeline = DataPipeline(args)
    assert pipeline.dataset_path is not None, "Dataset path should not be None."
    assert pipeline.dataset_path.exists(), f"Dataset path {pipeline.dataset_path} does not exist."
    assert str(pipeline.dataset_path) == "data/cullpdb+profile_5926.npy", f"Unexpected dataset file name: {pipeline.dataset_path.name}"

    dataset = pipeline._load_data(pipeline.dataset_path)
    assert dataset is not None, "Loaded dataset should not be None."
    assert isinstance(dataset, TensorDataset), f"Expected dataset to be a TensorDataset, got {type(dataset)}."

    # Check dataset size
    assert len(dataset) == 5430, f"Expected 5430 samples, got {len(dataset)}."
    # Check that each sample has the correct shape
    for i in range(5):  # check first 5 samples
        x, y = dataset[i]
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        assert x.shape == (700, 46), f"Expected features shape (700, 46), got {x.shape} for sample {i}."
        assert y.shape == (700, 9), f"Expected labels shape (700, 9), got {y.shape} for sample {i}."


def test_collate_fn():
    """Test custom collate function."""
    from scripts.datasets import my_collate
    import torch
    from torch import Tensor

    # Create a dummy batch of 2 samples
    batch = [
        (torch.randn(700, 46), torch.randint(0, 2, (700, 9))),
        (torch.randn(700, 46), torch.randint(0, 2, (700, 9))),
    ]

    data, targets = my_collate(batch)
    assert isinstance(data, Tensor), f"Expected data to be a Tensor, got {type(data)}."
    assert isinstance(targets, Tensor), f"Expected targets to be a Tensor, got {type(targets)}."

    assert data.shape == (2, 700, 46), f"Expected data shape (2, 700, 46), got {data.shape}."
    assert targets.shape == (2, 700), f"Expected targets shape (2, 700), got {targets.shape}."
    # Check that targets have values in the expected range (-100 for padding, 0-7 for classes)
    assert torch.all((targets == -100) | ((targets >= 0) & (targets < 8))), "Targets should be -100 for padding or in the range 0-7 for classes."


def test_dataloader():
    """Test DataLoader creation."""
    from scripts.datasets import DataPipeline, DataArgs
    from torch.utils.data import DataLoader

    args = DataArgs(dataset_name="cullpdb", split="train")
    pipeline = DataPipeline(args)
    loader = pipeline.run()
    assert loader is not None, "Returned loader should not be None."
    assert isinstance(loader, DataLoader), f"Expected DataLoader object, got {type(loader)}."

    trainiter = iter(loader)  # make dataloader an iterator (it's an iterable)
    data, targets = next(trainiter)
    assert data.shape == (args.batch_size, 700, 46), f"Expected (32, 700, 46), got {data.shape}."
    assert targets.shape == (args.batch_size, 700), f"Expected (32, 700), got {targets.shape}."
