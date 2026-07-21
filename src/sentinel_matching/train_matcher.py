"""Train the timm Siamese matcher from a generated pair manifest."""

from __future__ import annotations

import json
from pathlib import Path

import click
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.pair_dataset import SentinelPairDataset
from models.siamese_matcher import TimmSiameseMatcher
from configs.default import DEFAULT_MATCHING_CONFIG


def contrastive_loss(left, right, labels, margin):
    distances = torch.nn.functional.pairwise_distance(left, right)
    positive = labels * distances.pow(2)
    negative = (1.0 - labels) * torch.clamp(margin - distances, min=0.0).pow(2)
    return (positive + negative).mean()


def scores_to_metrics(scores, targets, threshold):
    predictions = [score >= threshold for score in scores]
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(targets, predictions),
        "precision": precision_score(targets, predictions, zero_division=0),
        "recall": recall_score(targets, predictions, zero_division=0),
        "f1": f1_score(targets, predictions, zero_division=0),
    }


def find_best_threshold(scores, targets):
    candidates = sorted({0.05 + index * 0.05 for index in range(19)} | {0.5})
    best = max((scores_to_metrics(scores, targets, threshold) for threshold in candidates), key=lambda item: item["f1"])
    return best["threshold"]


def evaluate(model, loader, device, objective="bce", margin=1.0, threshold=0.5, description="Validation"):
    model.eval()
    scores, targets = [], []
    total_loss = 0.0
    bce = nn.BCEWithLogitsLoss()
    with torch.inference_mode():
        for batch in tqdm(loader, desc=description, leave=False):
            left = model.encode(batch["image_left"].to(device))
            right = model.encode(batch["image_right"].to(device))
            labels = batch["label"].to(device)
            if objective == "bce":
                outputs = model.forward_embeddings(left, right)
                batch_scores = outputs.sigmoid()
                loss = bce(outputs, labels)
            else:
                distances = torch.nn.functional.pairwise_distance(left, right)
                batch_scores = torch.exp(-distances)
                loss = contrastive_loss(left, right, labels, margin)
            total_loss += loss.item() * len(labels)
            scores.extend(batch_scores.cpu().tolist())
            targets.extend(labels.cpu().numpy())
    metrics = scores_to_metrics(scores, targets, threshold)
    metrics["loss"] = total_loss / max(len(loader.dataset), 1)
    return metrics, scores, targets


def train(config):
    device = torch.device("cuda" if config["device"] == "auto" and torch.cuda.is_available() else config["device"])
    bands = tuple(item.strip() for item in config["bands"].split(","))
    train_set = SentinelPairDataset(config["manifest"], "train", config["image_size"], bands, train=True)
    validation_set = SentinelPairDataset(config["manifest"], "validation", config["image_size"], bands, train=False)
    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True, num_workers=config["workers"], pin_memory=device.type == "cuda")
    validation_loader = DataLoader(validation_set, batch_size=config["batch_size"], shuffle=False, num_workers=config["workers"], pin_memory=device.type == "cuda")
    model = TimmSiameseMatcher(config["backbone"], config["embedding_dim"], pretrained=True).to(device)
    print(
        json.dumps(
            {
                "training_device": str(device),
                "backbone": config["backbone"],
                "train_pairs": len(train_set),
                "validation_pairs": len(validation_set),
                "batch_size": config["batch_size"],
                "epochs": config["epochs"],
            }
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    loss_fn = nn.BCEWithLogitsLoss()
    best_f1 = -1.0
    output = Path(config["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    wandb_run = None
    if not config["disable_wandb"]:
        import wandb
        wandb_run = wandb.init(project=config["wandb_project"], name=config["wandb_run_name"], config=config)

    for epoch in range(config["epochs"]):
        model.train()
        train_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['epochs']} train")
        for batch in progress:
            optimizer.zero_grad(set_to_none=True)
            left = model.encode(batch["image_left"].to(device))
            right = model.encode(batch["image_right"].to(device))
            labels = batch["label"].to(device)
            if config["objective"] == "bce":
                loss = loss_fn(model.forward_embeddings(left, right), labels)
            else:
                loss = contrastive_loss(left, right, labels, config["margin"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(batch["label"])
            progress.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()
        _, validation_scores, validation_targets = evaluate(
            model, validation_loader, device, config["objective"], config["margin"], description=f"Epoch {epoch + 1}/{config['epochs']} validation"
        )
        threshold = find_best_threshold(validation_scores, validation_targets)
        metrics, _, _ = evaluate(
            model, validation_loader, device, config["objective"], config["margin"], threshold, description=f"Epoch {epoch + 1}/{config['epochs']} validation metrics"
        )
        train_loss /= max(len(train_loader.dataset), 1)
        epoch_metrics = {"epoch": epoch + 1, "train_loss": train_loss, "learning_rate": scheduler.get_last_lr()[0], **metrics}
        print(json.dumps(epoch_metrics))
        if wandb_run is not None:
            wandb_run.log(epoch_metrics, step=epoch + 1)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save({"model": model.state_dict(), "config": config, "metrics": metrics, "threshold": threshold}, output)
    if wandb_run is not None:
        wandb_run.finish()


@click.command()
@click.option("--manifest", required=True, type=click.Path(exists=True))
@click.option("--output", default=DEFAULT_MATCHING_CONFIG["output"], show_default=True)
@click.option("--backbone", default=DEFAULT_MATCHING_CONFIG["backbone"], show_default=True)
@click.option("--embedding-dim", default=DEFAULT_MATCHING_CONFIG["embedding_dim"], type=int, show_default=True)
@click.option("--image-size", default=DEFAULT_MATCHING_CONFIG["image_size"], type=int, show_default=True)
@click.option("--bands", default=DEFAULT_MATCHING_CONFIG["bands"], show_default=True)
@click.option("--batch-size", default=DEFAULT_MATCHING_CONFIG["batch_size"], type=int, show_default=True)
@click.option("--epochs", default=DEFAULT_MATCHING_CONFIG["epochs"], type=int, show_default=True)
@click.option("--learning-rate", default=DEFAULT_MATCHING_CONFIG["learning_rate"], type=float, show_default=True)
@click.option("--weight-decay", default=DEFAULT_MATCHING_CONFIG["weight_decay"], type=float, show_default=True)
@click.option("--objective", type=click.Choice(["bce", "contrastive"]), default="bce", show_default=True)
@click.option("--margin", default=1.0, type=float, show_default=True, help="Contrastive-loss margin.")
@click.option("--workers", default=DEFAULT_MATCHING_CONFIG["workers"], type=int, show_default=True)
@click.option("--device", default=DEFAULT_MATCHING_CONFIG["device"], show_default=True)
@click.option("--wandb-project", default=DEFAULT_MATCHING_CONFIG["wandb_project"], show_default=True)
@click.option("--wandb-run-name", default=DEFAULT_MATCHING_CONFIG["wandb_run_name"])
@click.option("--disable-wandb", is_flag=True)
def cli(manifest, output, backbone, embedding_dim, image_size, bands, batch_size, epochs, learning_rate, weight_decay, objective, margin, workers, device, wandb_project, wandb_run_name, disable_wandb):
    train(locals())


if __name__ == "__main__":
    cli()
