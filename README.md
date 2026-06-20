# Predicting protein secondary structure using deep learning

Based on [Arxiv](https://arxiv.org/abs/1403.1347)


## :one: Problem Statement

Predicting protein secondary structure is a fundamental problem in protein structure prediction.

- Labeling the secondary structure state of each amino-acid residue


### Setup

This project uses [uv]


### Dataset

We evaluate the model on two datasets, one derived from the PISCES Cull PDB server (`cullpdb`) and the other is the CB513 dataset (`cb513`). The `cullpdb` comes also in a filtered version which removes redundant samples with the `cb513` benchmark. Training and evaluation for these two datasets is conducted as follows:

1. Training, validation, and testing on train, eval, and test splits from `cullpdb+profile_5926`
2. Training and validation on train and eval splits from `cullpdb+profile_5926_filtered` and testing on `cb513`

Each protein samples comes in a sequence of amino-acid residues, the task is to label each residue with one of the eight secondary structure states.

You need to create a `data/` containing the following files:
- `cb513+profile_split1.npy.gz` the CB513 test split
- `cullpdb+profile_5926_filtered.npy.gz` the filtered Cull PDB dataset (train and eval splits for CB513 evaluation)
- `cullpdb+profile_5926.npy.gz` the Cull PDB dataset (train, eval, test splits)

The dataset files saved as `npy.gz` will be converted to `npy` files by the data pipeline, run the following command:

```bash
uv run scripts/datasets.py
```

You should have the `npy` files in the `data/` folder after running the command. Datasets have the following sizes:

| Name                            | Size        |
| ------------------------------- | ----------- |
| `cullpdb+profile_5926`          | 5926, 39900 |
| `cullpdb+profile_5926_filtered` | 5365, 39900 |
| `cb513+profile_split1`          | 514, 39900  |


Test data pipeline as follows:

```bash
uv run tests/test_datasets.py
```

Once reshaped, we will have for each sample 700 amino-acids and for each 57 features, in this form


### Model


## :two: Training results
