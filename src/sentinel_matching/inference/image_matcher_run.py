"""Run either a classical image matcher or a trained Siamese checkpoint."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import click

MATCHER_ROOT = Path(__file__).resolve().parents[1]
if str(MATCHER_ROOT) not in sys.path:
    sys.path.insert(0, str(MATCHER_ROOT))


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


@click.command()
@click.option("--left", required=True, type=click.Path(exists=True))
@click.option("--right", required=True, type=click.Path(exists=True))
@click.option("--method", type=click.Choice(["sift", "orb"]))
@click.option("--checkpoint", type=click.Path(exists=True, file_okay=True))
@click.option("--output-dir", required=True, type=click.Path())
@click.option("--bands", default="B04,B03,B02", show_default=True)
@click.option("--max-features", default=5000, type=int, show_default=True)
@click.option("--max-side", default=1600, type=int, show_default=True)
@click.option("--ransac-threshold", default=5.0, type=float, show_default=True)
@click.option("--backbone", default="convnext_tiny", show_default=True)
@click.option("--embedding-dim", default=256, type=int, show_default=True)
@click.option("--image-size", default=224, type=int, show_default=True)
@click.option("--threshold", type=float, default=None)
@click.option("--device", "device_name", type=click.Choice(["auto", "cpu", "cuda"]), default="auto", show_default=True)
def cli(left, right, method, checkpoint, output_dir, bands, max_features, max_side, ransac_threshold, backbone, embedding_dim, image_size, threshold, device_name):
    """Match LEFT and RIGHT with exactly one classical method or checkpoint."""
    if (method is None) == (checkpoint is None):
        raise click.UsageError("Choose exactly one of --method or --checkpoint.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    if checkpoint:
        import torch
        from dataset.pair_dataset import SentinelPairDataset
        from match_images import read_sentinel_scene
        from models.siamese_matcher import TimmSiameseMatcher

        selected_device = "cuda" if device_name == "auto" and torch.cuda.is_available() else "cpu" if device_name == "auto" else device_name
        if selected_device == "cuda" and not torch.cuda.is_available():
            raise click.ClickException("CUDA is not available.")
        saved = torch.load(checkpoint, map_location=selected_device)
        config = saved.get("config", {})
        model = TimmSiameseMatcher(backbone, embedding_dim, pretrained=False).to(selected_device).eval()
        model.load_state_dict(saved["model"])
        transform = SentinelPairDataset._build_transform(image_size, train=False)
        selected_bands = tuple(item.strip() for item in bands.split(",") if item.strip())
        left_image = transform(read_sentinel_scene(left, selected_bands)).unsqueeze(0).to(selected_device)
        right_image = transform(read_sentinel_scene(right, selected_bands)).unsqueeze(0).to(selected_device)
        started = time.perf_counter()
        with torch.inference_mode():
            left_embedding = model.encode(left_image)
            right_embedding = model.encode(right_image)
            if config.get("objective", "bce") == "contrastive":
                score = torch.exp(-torch.nn.functional.pairwise_distance(left_embedding, right_embedding)).item()
            else:
                score = model.forward_embeddings(left_embedding, right_embedding).sigmoid().item()
        selected_threshold = threshold if threshold is not None else saved.get("threshold", 0.5)
        metrics = {"mode": "checkpoint", "checkpoint": str(checkpoint), "objective": config.get("objective", "bce"), "score": float(score), "threshold": float(selected_threshold), "is_match": bool(score >= selected_threshold), "inference_time_seconds": time.perf_counter() - started}
        path = output / "checkpoint_metrics.json"
    else:
        from match_images import run

        visualization = output / f"{method}_matches.png"
        started = time.perf_counter()
        result = run(left, right, method, str(visualization), max_features, max_side, ransac_threshold, tuple(item.strip() for item in bands.split(",") if item.strip()))
        metrics = {"mode": "classical", "method": result.method, "keypoints_left": result.keypoints_left, "keypoints_right": result.keypoints_right, "candidate_matches": result.candidate_matches, "inlier_matches": result.inlier_matches, "inlier_ratio": result.inlier_ratio, "inference_time_seconds": time.perf_counter() - started, "visualization": str(visualization)}
        path = output / f"{method}_metrics.json"

    save_json(path, metrics)
    click.echo(json.dumps({"metrics": str(path), **metrics}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    cli()
