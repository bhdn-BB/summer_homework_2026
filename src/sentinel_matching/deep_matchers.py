from __future__ import annotations

import time
from pathlib import Path

import click
import cv2
import numpy as np

from match_images import read_sentinel_scene, resize_for_features


def geometric_filter(points_left, points_right, threshold):
    if len(points_left) < 4:
        return np.zeros(len(points_left), dtype=bool), None
    homography, mask = cv2.findHomography(
        np.asarray(points_left, dtype=np.float32),
        np.asarray(points_right, dtype=np.float32),
        cv2.USAC_MAGSAC,
        threshold,
        confidence=0.999,
        maxIters=10000,
    )
    return np.zeros(len(points_left), dtype=bool) if mask is None else mask.ravel().astype(bool), homography


def save_visual(left, right, points_left, points_right, inlier_mask, output):
    keypoints_left = [cv2.KeyPoint(float(x), float(y), 3) for x, y in points_left]
    keypoints_right = [cv2.KeyPoint(float(x), float(y), 3) for x, y in points_right]
    matches = [cv2.DMatch(i, i, 0) for i, keep in enumerate(inlier_mask) if keep]
    canvas = cv2.drawMatches(
        left,
        keypoints_left,
        right,
        keypoints_right,
        matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), canvas):
        raise OSError(str(target))


def run_loftr(left, right, device, threshold, output):
    import torch
    from kornia.feature import LoFTR

    model = LoFTR(pretrained="outdoor").to(device).eval()
    left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    inputs = {
        "image0": torch.from_numpy(left_gray / 255.0).float()[None, None].to(device),
        "image1": torch.from_numpy(right_gray / 255.0).float()[None, None].to(device),
    }
    started = time.perf_counter()
    with torch.inference_mode():
        prediction = model(inputs)
    points_left = prediction["keypoints0"].detach().cpu().numpy()
    points_right = prediction["keypoints1"].detach().cpu().numpy()
    inlier_mask, homography = geometric_filter(points_left, points_right, threshold)
    save_visual(left, right, points_left, points_right, inlier_mask, output)
    return {
        "method": "loftr",
        "candidate_matches": int(len(points_left)),
        "inlier_matches": int(inlier_mask.sum()),
        "inlier_ratio": float(inlier_mask.mean()) if len(inlier_mask) else 0.0,
        "elapsed_seconds": time.perf_counter() - started,
        "homography": homography,
    }


def run_lightglue(left, right, device, threshold, output):
    import torch
    from lightglue import LightGlue, SuperPoint
    from lightglue.utils import numpy_image_to_torch

    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)
    left_rgb = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
    right_rgb = cv2.cvtColor(right, cv2.COLOR_BGR2RGB)
    started = time.perf_counter()
    with torch.inference_mode():
        features_left = extractor.extract(numpy_image_to_torch(left_rgb).to(device))
        features_right = extractor.extract(numpy_image_to_torch(right_rgb).to(device))
        matches = matcher({"image0": features_left, "image1": features_right})
    keypoints_left = features_left["keypoints"][0].detach().cpu().numpy()
    keypoints_right = features_right["keypoints"][0].detach().cpu().numpy()
    pairs = matches["matches"][0].detach().cpu().numpy()
    points_left = keypoints_left[pairs[:, 0]] if len(pairs) else np.empty((0, 2), dtype=np.float32)
    points_right = keypoints_right[pairs[:, 1]] if len(pairs) else np.empty((0, 2), dtype=np.float32)
    inlier_mask, homography = geometric_filter(points_left, points_right, threshold)
    save_visual(left, right, points_left, points_right, inlier_mask, output)
    return {
        "method": "superpoint_lightglue",
        "candidate_matches": int(len(points_left)),
        "inlier_matches": int(inlier_mask.sum()),
        "inlier_ratio": float(inlier_mask.mean()) if len(inlier_mask) else 0.0,
        "elapsed_seconds": time.perf_counter() - started,
        "homography": homography,
    }


@click.command()
@click.option("--left", required=True, type=click.Path(exists=True))
@click.option("--right", required=True, type=click.Path(exists=True))
@click.option("--method", type=click.Choice(["loftr", "superpoint_lightglue"]), required=True)
@click.option("--bands", default="B04,B03,B02", show_default=True)
@click.option("--max-side", default=1600, type=int, show_default=True)
@click.option("--ransac-threshold", default=3.0, type=float, show_default=True)
@click.option("--device", default="auto", type=click.Choice(["auto", "cpu", "cuda"]), show_default=True)
@click.option("--output", required=True, type=click.Path())
def cli(left, right, method, bands, max_side, ransac_threshold, device, output):
    import torch

    selected_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
    left_image = read_sentinel_scene(left, tuple(x.strip() for x in bands.split(",")))
    right_image = read_sentinel_scene(right, tuple(x.strip() for x in bands.split(",")))
    left_image, _ = resize_for_features(left_image, max_side)
    right_image, _ = resize_for_features(right_image, max_side)
    if method == "loftr":
        result = run_loftr(left_image, right_image, selected_device, ransac_threshold, output)
    else:
        result = run_lightglue(left_image, right_image, selected_device, ransac_threshold, output)
    result.pop("homography", None)
    result["visualization"] = output
    click.echo(result)


if __name__ == "__main__":
    cli()
