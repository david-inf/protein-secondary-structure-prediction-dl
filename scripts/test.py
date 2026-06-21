"""Model testing script."""

from utils import LOG, N, set_seeds, accuracy_q8, metrics_q8
from datasets import DataArgs, DataPipeline
from utils import build_model
from pathlib import Path
from argparse import ArgumentParser, Namespace
import yaml
import torch
import numpy as np

"""TODO:
- [ ] Testing report
"""


@torch.no_grad()
def test(device, test_loader, model) -> None:
    """Model testing function."""
    accs = []

    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        logits, loss = model(inputs, targets)

        acc = accuracy_q8(N(logits), N(targets))
        accs.append(acc)

    LOG.info(f"Accuracy: {np.mean(accs)}")

    # TODO: dump to a log file


def main(opts: Namespace):
    """Testing script entry point."""
    set_seeds(getattr(opts, 'seed', 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load test dataloader
    if opts.dataset_name == "cullpdb_filtered":
        opts.dataset_name = "cb513"
    testdata_args = DataArgs(
        dataset_name=opts.dataset_name,
        split="test"
    )
    testdata_pipeline = DataPipeline(args=testdata_args)
    test_loader = testdata_pipeline.run()

    # Init model
    LOG.info("\nLoading model...")
    model = build_model(opts)
    # Load checkpoint
    if not Path(opts.checkpoint_path).exists():
        raise RuntimeError(f"File {opts.checkpoint_path} does not exist.")
    checkpoint = torch.load(opts.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    LOG.info("Model ready")

    # Launch testing
    test(device, test_loader, model)

    # Sampling a few predictions for qualitative analysis
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits, _ = model(inputs, targets)
            break

        preds = np.argmax(N(logits), axis=2)
        # Alignment for few samples
        trues = N(targets).astype(int)
        out_path = Path(opts.test_alignment_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            n = min(10, len(trues))
            for i in range(n):
                # Print line with true and predicted labels for each sample
                true_line = " ".join(str(int(x)) for x in trues[i])
                pred_line = " ".join(str(int(x)) for x in preds[i])
                f.write(f"# Sample {i}\n")
                f.write(f"TRUE: {true_line}\n")
                f.write(f"PRED: {pred_line}\n\n")
        LOG.info(f"Wrote compact sample alignment to {out_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config", help="YAML configuration file.",
        default="cullpdb_simple1dcnn.yaml")
    args = parser.parse_args()

    config_path = f"scripts/config/{args.config}"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        assert isinstance(config, dict), "Expected config to be a dictionary."
    args = Namespace(**config)

    try:
        set_seeds(getattr(args, 'seed', 42))
        main(args)

    except Exception as e:
        LOG.error(f"An error occurred during testing: {e}")
        raise e
