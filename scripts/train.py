"""Training script."""

from utils import LOG
from datasets import DataArgs, DataPipeline
from models import build_model
from trainer import TrainArgs, Trainer

from argparse import ArgumentParser, Namespace
import yaml

"""TODO:
- [ ] 
"""


def main(opts: Namespace):
    """Training entry point."""
    traindata_args = DataArgs(
        dataset_name=opts.dataset_name,
        split="train"
    )
    traindata_pipeline = DataPipeline(args=traindata_args)
    train_loader = traindata_pipeline.run()

    # evaldata_args = DataArgs(
    #     dataset_name=opts.dataset_name,
    #     split="eval"
    # )
    # evaldata_pipeline = DataPipeline(args=evaldata_args)

    LOG.info("\nLoading model...")
    model = build_model(opts)
    LOG.info(f" Model ready:\n{model}")

    train_args = TrainArgs()
    LOG.info(f"\nTraining config: {train_args}\n")
    trainer = Trainer(
        args=train_args,
        model=model,
        train_loader=train_loader
    )
    trainer.train()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config", help="YAML configuration file.",
        default="scripts/config/cullpdb_simple1dcnn.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        assert isinstance(config, dict), "'config' should be a dict."
    args = Namespace(**config)

    try:
        main(args)

    except Exception as e:
        LOG.error(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise
