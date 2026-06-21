# Predicting protein secondary structure using deep learning

This repository implements deep-learning models to predict protein secondary structure from sequence and profile features. The work follows the experimental setup of established literature and provides data preprocessing, model definitions, training loops, and evaluation utilities.

**Key idea:** assign one of eight secondary-structure labels to each amino-acid residue in a protein sequence. This task can be seen as a sequence labeling problem, where the input is a sequence of amino acids (and associated features) and the output is a sequence of secondary structure labels.

**Paper reference:** https://arxiv.org/abs/1403.1347

**Overview**
This project provides:
- Data pipeline to convert compressed numpy dataset files into processed arrays suitable for training.
- Model architectures in `scripts/models.py` and training logic in `scripts/train.py` and `scripts/trainer.py`.
- Unit tests for dataset processing and model components under `tests/`.

**Project structure**
- `scripts/` : data pipeline, model definitions, training and testing scripts (`datasets.py`, `models.py`, `train.py`, `trainer.py`, `test.py`).
- `data/` : place source `*.npy.gz` files here; processed `*.npy` will be written here by the pipeline.
- `results/` : output directory for checkpoints and logs.
- `tests/` : unit tests (`test_datasets.py`, `test_model.py`).
- `config/` : example config files (e.g., `cullpdb_simple1dcnn.yaml`).
- `main.py` : optional entry point.

**Install**
Used https://docs.astral.sh/uv/ for environment management, but you can use any Python environment manager using dependencies from `pyproject.toml`.

```bash
# Install UV if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh
```


## Dataset

Get the dataset from https://mega.nz/folder/xct0XSpA#SKz72JtnSAaX61QLMC_JNg. This project expects the following compressed numpy dataset files (provided externally) to be placed in the `data/` folder:
- `cb513+profile_split1.npy.gz` — CB513 test split
- `cullpdb+profile_5926_filtered.npy.gz` — filtered Cull PDB (train+eval for CB513 evaluation)
- `cullpdb+profile_5926.npy.gz` — Cull PDB (train, eval, test)

The repository contains a data pipeline that will convert `*.npy.gz` files into `*.npy` and reshape them for training. To run the conversion:

```bash
uv run scripts/datasets.py
```

After running the pipeline you should find processed `*.npy` files in `data/`. You can also simply place the processed `*.npy` files in `data/` if you already have them.

Dataset sizes (samples, features):

- `cullpdb+profile_5926`: 5926 samples, 39900 features per sample (reshaped to length 700 × feature_dim 57)
- `cullpdb+profile_5926_filtered`: 5365 samples
- `cb513+profile_split1`: 514 samples

Run dataset unit tests:

```bash
# Test data loading pipeline
uv run pytest tests/test_datasets.py
```


## Models

Run model unit tests:

```bash
# Test model input/output shapes and forward pass
uv run pytest tests/test_model.py
```

**Simple1DCNN**

Takes the amino-acid sequence (one-hot encoded), terminals, and PSSM features as input and predicts secondary structure labels. The architecture is a simple 1D CNN with configurable number of convolutional layers, kernel sizes, and hidden channels. See `scripts/simplecnn.py` for details.

**CustomCNN**

Takes the amino-acid sequence (one-hot encoded), terminals, and PSSM features as input and predicts secondary structure labels in separate branches, then fuses, and applies a configurable number of convolutional layers, kernel sizes, and hidden channels.

- Amino-acid sequence is processed through an embedding layer.
- PSSM features are projected to the same embedding dimension through a linear layer.

Once we have the embeddings from both branches, we concatenate them with the terminals features and feed them into a configurable number of convolutional layers, followed by a final classification head. See `scripts/customcnn.py` for details.

## Training

**Simple usage**
Train a model using a configuration file (example):

```bash
uv run scripts/train.py --config cullpdb_simple1dcnn.yaml

uv run scripts/train.py --config cullpdb_customcnn.yaml
```

**Configuration**

Training and model settings are stored in YAML config files under `config/`. Edit or add new configs to try different architectures and hyperparameters.

**Results**

```bash
uv run scripts/test.py --config cullpdb_simple1dcnn.yaml

uv run scripts/test.py --config cullpdb_customcnn.yaml
```

| Model       | cullpdb | cb513 |
| ----------- | ------- | ----- |
| Simple1DCNN | .       | .     |
| CustomCNN   | .       | .     |
