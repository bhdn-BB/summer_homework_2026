"""Evaluate a trained timm matcher on held-out test pairs."""

from __future__ import annotations

import click
import torch
from torch.utils.data import DataLoader

from dataset.pair_dataset import SentinelPairDataset
from models.siamese_matcher import TimmSiameseMatcher
from train_matcher import evaluate


@click.command()
@click.option("--checkpoint", required=True, type=click.Path(exists=True))
@click.option("--manifest", required=True, type=click.Path(exists=True))
@click.option("--batch-size", default=None, type=int)
@click.option("--workers", default=None, type=int)
@click.option("--device", default=None)
@click.option("--wandb-project", default=None)
@click.option("--wandb-run-name", default=None)
@click.option("--disable-wandb", is_flag=True)
def cli(checkpoint, manifest, batch_size, workers, device, wandb_project, wandb_run_name, disable_wandb):
    saved = torch.load(checkpoint, map_location="cpu")
    config = saved["config"]
    if batch_size is not None:
        config["batch_size"] = batch_size
    if workers is not None:
        config["workers"] = workers
    if device is not None:
        config["device"] = device

    selected_device = torch.device(
        "cuda" if config["device"] == "auto" and torch.cuda.is_available() else config["device"]
    )
    bands = tuple(item.strip() for item in config["bands"].split(","))
    dataset = SentinelPairDataset(manifest, "test", config["image_size"], bands, train=False)
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["workers"],
        pin_memory=selected_device.type == "cuda",
    )
    model = TimmSiameseMatcher(config["backbone"], config["embedding_dim"], pretrained=False).to(selected_device)
    model.load_state_dict(saved["model"])
    objective = config.get("objective", "bce")
    margin = config.get("margin", 1.0)
    threshold = saved.get("threshold", 0.5)
    metrics, _, _ = evaluate(model, loader, selected_device, objective, margin, threshold)
    click.echo({"test": metrics})

    if not disable_wandb:
        import wandb
        run = wandb.init(
            project=wandb_project or config["wandb_project"],
            name=wandb_run_name or f"{config.get('wandb_run_name') or 'matcher'}-test",
            config=config,
        )
        run.log({f"test/{key}": value for key, value in metrics.items()})
        run.finish()


if __name__ == "__main__":
    cli()
