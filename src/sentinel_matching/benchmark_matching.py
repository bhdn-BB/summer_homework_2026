"""Benchmark classical and learned matchers on large seasonal Sentinel scenes.

The script processes corresponding tiles instead of shrinking a whole scene to
one small image. This preserves local detail and makes the runtime comparison
between methods reproducible on Kaggle.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import cv2
import click
import numpy as np
from tqdm import tqdm

from configs.default import DEFAULT_MATCHING_CONFIG

from match_images import MatchResult, draw_matches, match_pair, read_sentinel_scene


def tile_coordinates(height: int, width: int, tile_size: int, overlap: int):
    """Yield a complete overlapping grid, including the final border tiles."""
    if overlap >= tile_size:
        raise ValueError("overlap must be smaller than tile_size")
    step = tile_size - overlap
    y_starts = list(range(0, max(height - tile_size, 0) + 1, step))
    x_starts = list(range(0, max(width - tile_size, 0) + 1, step))
    if not y_starts or y_starts[-1] + tile_size < height:
        y_starts.append(max(height - tile_size, 0))
    if not x_starts or x_starts[-1] + tile_size < width:
        x_starts.append(max(width - tile_size, 0))

    for y in sorted(set(y_starts)):
        for x in sorted(set(x_starts)):
            yield x, y, min(x + tile_size, width), min(y + tile_size, height)


def aggregate_classical(
    left: np.ndarray,
    right: np.ndarray,
    method: str,
    tile_size: int,
    overlap: int,
    max_features: int,
    ransac_threshold: float,
    output_path: str,
) -> dict:
    """Run SIFT/ORB over corresponding tiles and save the strongest tile view."""
    started = time.perf_counter()
    total_keypoints_left = 0
    total_keypoints_right = 0
    total_candidates = 0
    total_inliers = 0
    best_tile = None
    best_matches = []

    tiles = list(tile_coordinates(*left.shape[:2], tile_size, overlap))
    for x1, y1, x2, y2 in tqdm(tiles, desc=f"{method.upper()} tiles"):
        left_tile = left[y1:y2, x1:x2]
        right_tile = right[y1:y2, x1:x2]
        result, inliers, keypoints_left, keypoints_right = match_pair(
            left_tile,
            right_tile,
            method=method,
            max_features=max_features,
            max_side=tile_size,
            ransac_threshold=ransac_threshold,
        )
        total_keypoints_left += result.keypoints_left
        total_keypoints_right += result.keypoints_right
        total_candidates += result.candidate_matches
        total_inliers += result.inlier_matches
        if best_tile is None or result.inlier_matches > best_tile.inlier_matches:
            best_tile = result
            best_matches = (left_tile, right_tile, keypoints_left, keypoints_right, inliers)

    if best_tile is not None:
        draw_matches(*best_matches, output_path)

    return {
        "method": method,
        "keypoints_left": total_keypoints_left,
        "keypoints_right": total_keypoints_right,
        "candidate_matches": total_candidates,
        "inlier_matches": total_inliers,
        "inlier_ratio": total_inliers / max(total_candidates, 1),
        "inference_time_seconds": time.perf_counter() - started,
        "visualization": output_path,
    }


def load_loftr(device: str):
    """Load Kornia LoFTR only when the learned method is requested."""
    try:
        import torch
        from kornia.feature import LoFTR
    except ImportError as exc:
        raise ImportError("LoFTR requires torch and kornia: python -m pip install torch kornia") from exc

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LoFTR(pretrained="outdoor").to(device).eval()
    return model, torch, device


def aggregate_loftr(
    left: np.ndarray,
    right: np.ndarray,
    tile_size: int,
    overlap: int,
    ransac_threshold: float,
    device: str,
    output_path: str,
) -> dict:
    """Run pretrained LoFTR on corresponding tiles and verify matches with RANSAC."""
    model, torch, device = load_loftr(device)
    started = time.perf_counter()
    total_matches = 0
    total_inliers = 0
    best = None

    tiles = list(tile_coordinates(*left.shape[:2], tile_size, overlap))
    for x1, y1, x2, y2 in tqdm(tiles, desc="LOFTR tiles"):
        left_tile = left[y1:y2, x1:x2]
        right_tile = right[y1:y2, x1:x2]
        left_gray = cv2.cvtColor(left_tile, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_tile, cv2.COLOR_BGR2GRAY)
        inputs = {
            "image0": torch.from_numpy(left_gray / 255.0).float()[None, None].to(device),
            "image1": torch.from_numpy(right_gray / 255.0).float()[None, None].to(device),
        }
        with torch.inference_mode():
            prediction = model(inputs)

        points_left = prediction["keypoints0"].detach().cpu().numpy()
        points_right = prediction["keypoints1"].detach().cpu().numpy()
        inlier_mask = np.zeros(len(points_left), dtype=bool)
        if len(points_left) >= 4:
            _, mask = cv2.findHomography(points_left, points_right, cv2.RANSAC, ransac_threshold)
            if mask is not None:
                inlier_mask = mask.ravel().astype(bool)
        inliers = int(inlier_mask.sum())
        total_matches += len(points_left)
        total_inliers += inliers
        if best is None or inliers > best[0]:
            best = (inliers, left_tile, right_tile, points_left, points_right, inlier_mask)

    if best is not None:
        _, left_tile, right_tile, points_left, points_right, inlier_mask = best
        keypoints_left = [cv2.KeyPoint(float(x), float(y), 3) for x, y in points_left]
        keypoints_right = [cv2.KeyPoint(float(x), float(y), 3) for x, y in points_right]
        matches = [cv2.DMatch(i, i, 0) for i, keep in enumerate(inlier_mask) if keep]
        draw_matches(left_tile, right_tile, keypoints_left, keypoints_right, matches, output_path)

    return {
        "method": "loftr",
        "keypoints_left": total_matches,
        "keypoints_right": total_matches,
        "candidate_matches": total_matches,
        "inlier_matches": total_inliers,
        "inlier_ratio": total_inliers / max(total_matches, 1),
        "inference_time_seconds": time.perf_counter() - started,
        "visualization": output_path,
    }


def load_lightglue(device: str):
    """Load SuperPoint and LightGlue only when this optional method is requested."""
    try:
        import torch
        from lightglue import LightGlue, SuperPoint
        from lightglue.utils import numpy_image_to_torch
    except ImportError as exc:
        raise ImportError(
            "SuperPoint + LightGlue requires the optional LightGlue package. "
            "Install the pinned Git dependency from requirements.txt."
        ) from exc

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)
    return extractor, matcher, numpy_image_to_torch, torch, device


def aggregate_lightglue(
    left: np.ndarray,
    right: np.ndarray,
    tile_size: int,
    overlap: int,
    ransac_threshold: float,
    device: str,
    output_path: str,
) -> dict:
    """Benchmark SuperPoint + LightGlue on the identical tile grid as other methods."""
    extractor, matcher, numpy_image_to_torch, torch, device = load_lightglue(device)
    started = time.perf_counter()
    total_keypoints_left = 0
    total_keypoints_right = 0
    total_candidates = 0
    total_inliers = 0
    best = None

    tiles = list(tile_coordinates(*left.shape[:2], tile_size, overlap))
    for x1, y1, x2, y2 in tqdm(tiles, desc="SUPERPOINT_LIGHTGLUE tiles"):
        left_tile = left[y1:y2, x1:x2]
        right_tile = right[y1:y2, x1:x2]
        left_rgb = cv2.cvtColor(left_tile, cv2.COLOR_BGR2RGB)
        right_rgb = cv2.cvtColor(right_tile, cv2.COLOR_BGR2RGB)
        with torch.inference_mode():
            features_left = extractor.extract(numpy_image_to_torch(left_rgb).to(device))
            features_right = extractor.extract(numpy_image_to_torch(right_rgb).to(device))
            prediction = matcher({"image0": features_left, "image1": features_right})

        keypoints_left = features_left["keypoints"][0].detach().cpu().numpy()
        keypoints_right = features_right["keypoints"][0].detach().cpu().numpy()
        pairs = prediction["matches"][0].detach().cpu().numpy()
        points_left = keypoints_left[pairs[:, 0]] if len(pairs) else np.empty((0, 2), dtype=np.float32)
        points_right = keypoints_right[pairs[:, 1]] if len(pairs) else np.empty((0, 2), dtype=np.float32)
        inlier_mask = np.zeros(len(points_left), dtype=bool)
        if len(points_left) >= 4:
            _, mask = cv2.findHomography(points_left, points_right, cv2.RANSAC, ransac_threshold)
            if mask is not None:
                inlier_mask = mask.ravel().astype(bool)
        inliers = int(inlier_mask.sum())
        total_keypoints_left += len(keypoints_left)
        total_keypoints_right += len(keypoints_right)
        total_candidates += len(points_left)
        total_inliers += inliers
        if best is None or inliers > best[0]:
            best = (inliers, left_tile, right_tile, points_left, points_right, inlier_mask)

    if best is not None:
        _, left_tile, right_tile, points_left, points_right, inlier_mask = best
        visual_keypoints_left = [cv2.KeyPoint(float(x), float(y), 3) for x, y in points_left]
        visual_keypoints_right = [cv2.KeyPoint(float(x), float(y), 3) for x, y in points_right]
        matches = [cv2.DMatch(index, index, 0) for index, keep in enumerate(inlier_mask) if keep]
        draw_matches(left_tile, right_tile, visual_keypoints_left, visual_keypoints_right, matches, output_path)

    return {
        "method": "superpoint_lightglue",
        "keypoints_left": total_keypoints_left,
        "keypoints_right": total_keypoints_right,
        "candidate_matches": total_candidates,
        "inlier_matches": total_inliers,
        "inlier_ratio": total_inliers / max(total_candidates, 1),
        "inference_time_seconds": time.perf_counter() - started,
        "visualization": output_path,
    }


def save_results(results: list[dict], output_path: str) -> None:
    """Save a compact CSV result table for the experiment report."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)


@click.command()
@click.option("--left", required=True, type=click.Path(exists=True))
@click.option("--right", required=True, type=click.Path(exists=True))
@click.option("--methods", default=DEFAULT_MATCHING_CONFIG["methods"], show_default=True)
@click.option("--bands", "bands_value", default=DEFAULT_MATCHING_CONFIG["bands"], show_default=True)
@click.option("--tile-size", default=DEFAULT_MATCHING_CONFIG["tile_size"], type=int, show_default=True)
@click.option("--overlap", default=DEFAULT_MATCHING_CONFIG["overlap"], type=int, show_default=True)
@click.option("--max-features", default=DEFAULT_MATCHING_CONFIG["max_features"], type=int, show_default=True)
@click.option("--ransac-threshold", default=DEFAULT_MATCHING_CONFIG["ransac_threshold"], type=float, show_default=True)
@click.option("--device", default=DEFAULT_MATCHING_CONFIG["device"], type=click.Choice(["auto", "cpu", "cuda"]), show_default=True)
@click.option("--output-dir", default=DEFAULT_MATCHING_CONFIG["benchmark_output_dir"], show_default=True)
def cli(left, right, methods, bands_value, tile_size, overlap, max_features, ransac_threshold, device, output_dir):
    bands = tuple(item.strip() for item in bands_value.split(",") if item.strip())
    if len(bands) != 3:
        raise ValueError("--bands must contain exactly three band names")
    left = read_sentinel_scene(left, bands)
    right = read_sentinel_scene(right, bands)
    output_dir = Path(output_dir)
    results = []
    for method in [item.strip().lower() for item in methods.split(",") if item.strip()]:
        output_path = str(output_dir / f"{method}_best_tile.png")
        if method in {"sift", "orb"}:
            result = aggregate_classical(left, right, method, tile_size, overlap, max_features, ransac_threshold, output_path)
        elif method == "loftr":
            result = aggregate_loftr(left, right, tile_size, overlap, ransac_threshold, device, output_path)
        elif method == "superpoint_lightglue":
            result = aggregate_lightglue(left, right, tile_size, overlap, ransac_threshold, device, output_path)
        else:
            raise ValueError(f"Unsupported method: {method}")
        results.append(result)
        print(result)
    save_results(results, str(output_dir / "benchmark.csv"))
    print(f"results={output_dir / 'benchmark.csv'}")


if __name__ == "__main__":
    cli()
