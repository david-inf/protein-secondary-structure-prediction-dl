# Dataset preparation
uv run scripts/datasets.py --dataset_name cullpdb
uv run scripts/datasets.py --dataset_name cullpdb_filtered

# Train and test Simple1DCNN
uv run scripts/train.py --config cullpdb_simple1dcnn.yaml
uv run scripts/train.py --config cullpdb_filtered_simple1dcnn.yaml

uv run scripts/test.py --config cullpdb_simple1dcnn.yaml
uv run scripts/test.py --config cullpdb_filtered_simple1dcnn.yaml

# Train and test CustomCNN
uv run scripts/train.py --config cullpdb_customcnn.yaml
uv run scripts/train.py --config cullpdb_filtered_customcnn.yaml

uv run scripts/test.py --config cullpdb_customcnn.yaml
uv run scripts/test.py --config cullpdb_filtered_customcnn.yaml