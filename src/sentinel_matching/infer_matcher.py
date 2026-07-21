"""Run a trained timm Siamese matcher on one image pair."""

from __future__ import annotations

import click
import time
import torch

from dataset.pair_dataset import SentinelPairDataset
from models.siamese_matcher import TimmSiameseMatcher
from configs.default import DEFAULT_MATCHING_CONFIG


@click.command()
@click.option("--checkpoint", "checkpoint_path", required=True, type=click.Path(exists=True))
@click.option("--left", "left_path", required=True, type=click.Path(exists=True))
@click.option("--right", "right_path", required=True, type=click.Path(exists=True))
@click.option("--backbone", default=DEFAULT_MATCHING_CONFIG["backbone"], show_default=True)
@click.option("--embedding-dim", default=DEFAULT_MATCHING_CONFIG["embedding_dim"], type=int, show_default=True)
@click.option("--image-size", default=DEFAULT_MATCHING_CONFIG["image_size"], type=int, show_default=True)
@click.option("--bands", "bands_value", default=DEFAULT_MATCHING_CONFIG["bands"], show_default=True)
@click.option("--device", "device_name", default=DEFAULT_MATCHING_CONFIG["device"], show_default=True)
@click.option("--threshold", type=float, default=None, help="Override the validation-tuned match threshold.")
def cli(checkpoint_path, left_path, right_path, backbone, embedding_dim, image_size, bands_value, device_name, threshold):
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else device_name)
    model = TimmSiameseMatcher(backbone, embedding_dim, pretrained=False).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    bands = tuple(item.strip() for item in bands_value.split(","))

    # Reuse the dataset preprocessing without requiring a manifest file.
    transform = SentinelPairDataset._build_transform(image_size, train=False)
    from match_images import read_sentinel_scene
    left = transform(read_sentinel_scene(left_path, bands)).unsqueeze(0).to(device)
    right = transform(read_sentinel_scene(right_path, bands)).unsqueeze(0).to(device)
    started = time.perf_counter()
    with torch.inference_mode():
        left_embedding = model.encode(left)
        right_embedding = model.encode(right)
        objective = checkpoint.get("config", {}).get("objective", "bce")
        if objective == "contrastive":
            score = torch.exp(-torch.nn.functional.pairwise_distance(left_embedding, right_embedding)).item()
        else:
            score = model.forward_embeddings(left_embedding, right_embedding).sigmoid().item()
    selected_threshold = threshold if threshold is not None else checkpoint.get("threshold", 0.5)
    print(f"objective={objective}")
    print(f"match_score={score:.6f}")
    print(f"threshold={selected_threshold:.6f}")
    print(f"is_match={score >= selected_threshold}")
    print(f"inference_time_seconds={time.perf_counter() - started:.4f}")


if __name__ == "__main__":
    cli()
