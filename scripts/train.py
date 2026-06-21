"""Training script."""

from utils import LOG
from datasets import DataArgs, DataPipeline
from utils import build_model
from trainer import TrainArgs, Trainer

from argparse import ArgumentParser, Namespace
import yaml

"""TODO:
- [ ] 
"""


def main(opts: Namespace):
    """Training entry point."""
    # Load train dataloader
    traindata_args = DataArgs(
        dataset_name=opts.dataset_name,
        split="train"
    )
    traindata_pipeline = DataPipeline(args=traindata_args)
    train_loader = traindata_pipeline.run()

    # Load validation dataloader
    evaldata_args = DataArgs(
        dataset_name=opts.dataset_name,
        split="eval"
    )
    evaldata_pipeline = DataPipeline(args=evaldata_args)
    eval_loader = evaldata_pipeline.run()

    # Init model
    LOG.info("\nLoading model...")
    model = build_model(opts)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    LOG.info(f" Model ready ({n_params:,} params)")
    # print(model)

    # Launch training loop
    train_args = TrainArgs(
        epochs=getattr(opts, 'epochs', 50),
        learning_rate=getattr(opts, 'learning_rate', 0.01),
        checkpoint_path=getattr(opts, 'checkpoint_path', "results/ckpts/checkpoint.pt")
    )
    LOG.info(f"\nTraining config: {train_args}\n")
    trainer = Trainer(
        args=train_args,
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
    )
    trainer.train()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config", help="YAML configuration file.",
        default="cullpdb_simple1dcnn.yaml")
    args = parser.parse_args()

    config_path = f"scripts/config/{args.config}"
    with open(config_path, "r", encoding="utf-8") as f:
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
