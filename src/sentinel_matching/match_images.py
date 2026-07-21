"""Match two seasonal satellite images and save a visual diagnostic.

The default SIFT pipeline is a strong classical baseline for seasonal image
matching. ORB is included as a faster CPU baseline. Large images are resized
for feature extraction while the output visualization keeps the original
image proportions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import click
import numpy as np
from tqdm import tqdm

from configs.default import DEFAULT_MATCHING_CONFIG


@dataclass
class MatchResult:
    method: str
    keypoints_left: int
    keypoints_right: int
    candidate_matches: int
    inlier_matches: int
    inlier_ratio: float
    homography: np.ndarray | None
    elapsed_seconds: float


def read_image(path: str) -> np.ndarray:
    """Read an image as an 8-bit 3-channel BGR array.

    Sentinel-2 exports can be grayscale, RGB, or 16-bit. Converting all input
    variants here keeps the matching code independent from the file format.
    """
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] > 3:
        image = image[:, :, :3]

    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return image


def read_sentinel_scene(path: str, bands: tuple[str, ...] = ("B04", "B03", "B02")) -> np.ndarray:
    """Read a Sentinel SAFE directory or a regular image into an 8-bit composite."""
    source = Path(path)
    if not source.is_dir():
        return read_image(path)

    try:
        import rasterio
    except ImportError as exc:
        raise ImportError("Reading SAFE directories requires rasterio: python -m pip install rasterio") from exc

    band_arrays = []
    reference_shape = None
    for band in bands:
        matches = sorted(source.rglob(f"*_{band}.jp2"))
        if not matches:
            raise FileNotFoundError(f"Band {band} was not found under SAFE directory: {path}")
        with rasterio.open(matches[0]) as dataset:
            array = dataset.read(1).astype(np.float32)
        if reference_shape is None:
            reference_shape = array.shape
        elif array.shape != reference_shape:
            array = cv2.resize(array, (reference_shape[1], reference_shape[0]), interpolation=cv2.INTER_LINEAR)
        band_arrays.append(array)

    composite = np.stack(band_arrays, axis=-1)
    output = np.zeros_like(composite, dtype=np.uint8)
    for channel in range(composite.shape[-1]):
        values = composite[:, :, channel]
        low, high = np.percentile(values, [2, 98])
        if high <= low:
            output[:, :, channel] = np.clip(values, 0, 255).astype(np.uint8)
        else:
            output[:, :, channel] = np.clip((values - low) / (high - low) * 255, 0, 255).astype(np.uint8)
    return output


def resize_for_features(image: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    """Resize an image for stable runtime and return its coordinate scale."""
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale == 1.0:
        return image, scale

    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return resized, scale


def create_detector(method: str, max_features: int):
    """Build the requested local feature detector and its matching strategy."""
    if method == "sift":
        detector = cv2.SIFT_create(nfeatures=max_features)
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        ratio = 0.75
    elif method == "orb":
        detector = cv2.ORB_create(nfeatures=max_features, fastThreshold=10)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        ratio = 0.80
    else:
        raise ValueError("method must be either 'sift' or 'orb'")
    return detector, matcher, ratio


def match_pair(
    left: np.ndarray,
    right: np.ndarray,
    method: str = "sift",
    max_features: int = 5000,
    max_side: int = 2400,
    ransac_threshold: float = 5.0,
) -> tuple[MatchResult, list[cv2.DMatch], list[cv2.KeyPoint], list[cv2.KeyPoint]]:
    """Detect, match, and geometrically verify local features."""
    started = time.perf_counter()
    left_small, left_scale = resize_for_features(left, max_side)
    right_small, right_scale = resize_for_features(right, max_side)
    detector, matcher, ratio = create_detector(method, max_features)

    left_gray = cv2.cvtColor(left_small, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_small, cv2.COLOR_BGR2GRAY)
    keypoints_left, descriptors_left = detector.detectAndCompute(left_gray, None)
    keypoints_right, descriptors_right = detector.detectAndCompute(right_gray, None)
    if descriptors_left is None or descriptors_right is None:
        return MatchResult(method, len(keypoints_left), len(keypoints_right), 0, 0, 0.0, None, time.perf_counter() - started), [], keypoints_left, keypoints_right

    raw_matches = matcher.knnMatch(descriptors_left, descriptors_right, k=2)
    good_matches = [
        pair[0]
        for pair in raw_matches
        if len(pair) >= 2 and pair[0].distance < ratio * pair[1].distance
    ]

    homography = None
    inlier_matches = []
    if len(good_matches) >= 4:
        points_left = np.float32([keypoints_left[m.queryIdx].pt for m in good_matches])
        points_right = np.float32([keypoints_right[m.trainIdx].pt for m in good_matches])
        homography, mask = cv2.findHomography(points_left, points_right, cv2.RANSAC, ransac_threshold)
        if mask is not None:
            inlier_matches = [match for match, keep in zip(good_matches, mask.ravel()) if keep]

    result = MatchResult(
        method=method,
        keypoints_left=len(keypoints_left),
        keypoints_right=len(keypoints_right),
        candidate_matches=len(good_matches),
        inlier_matches=len(inlier_matches),
        inlier_ratio=len(inlier_matches) / max(len(good_matches), 1),
        homography=homography,
        elapsed_seconds=time.perf_counter() - started,
    )
    return result, inlier_matches, keypoints_left, keypoints_right


def draw_matches(
    left: np.ndarray,
    right: np.ndarray,
    keypoints_left: list[cv2.KeyPoint],
    keypoints_right: list[cv2.KeyPoint],
    matches: list[cv2.DMatch],
    output_path: str,
) -> None:
    """Write a side-by-side match visualization with only geometric inliers."""
    canvas = cv2.drawMatches(
        left,
        keypoints_left,
        right,
        keypoints_right,
        matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), canvas):
        raise OSError(f"Could not write visualization: {output}")


def run(left_path: str, right_path: str, method: str, output_path: str, max_features: int, max_side: int, ransac_threshold: float, bands: tuple[str, ...]) -> MatchResult:
    """Run matching, print metrics, and save the diagnostic image."""
    progress = tqdm(total=4, desc=f"{method.upper()} matching")
    left = read_sentinel_scene(left_path, bands)
    progress.update(1)
    right = read_sentinel_scene(right_path, bands)
    progress.update(1)
    result, inliers, keypoints_left, keypoints_right = match_pair(
        left,
        right,
        method=method,
        max_features=max_features,
        max_side=max_side,
        ransac_threshold=ransac_threshold,
    )
    progress.update(1)
    left_for_display, _ = resize_for_features(left, max_side)
    right_for_display, _ = resize_for_features(right, max_side)
    draw_matches(left_for_display, right_for_display, keypoints_left, keypoints_right, inliers, output_path)
    progress.update(1)
    progress.close()
    print(f"method={result.method}")
    print(f"keypoints_left={result.keypoints_left}")
    print(f"keypoints_right={result.keypoints_right}")
    print(f"candidate_matches={result.candidate_matches}")
    print(f"inlier_matches={result.inlier_matches}")
    print(f"inlier_ratio={result.inlier_ratio:.4f}")
    print(f"inference_time_seconds={result.elapsed_seconds:.4f}")
    print(f"visualization={output_path}")
    return result


@click.command()
@click.option("--left", required=True, type=click.Path(exists=True), help="Reference-season image or SAFE directory.")
@click.option("--right", required=True, type=click.Path(exists=True), help="Target-season image or SAFE directory.")
@click.option("--method", type=click.Choice(["sift", "orb"]), default=DEFAULT_MATCHING_CONFIG["method"], show_default=True)
@click.option("--output", default=DEFAULT_MATCHING_CONFIG["single_output"], show_default=True)
@click.option("--max-features", default=DEFAULT_MATCHING_CONFIG["max_features"], type=int, show_default=True)
@click.option("--max-side", default=DEFAULT_MATCHING_CONFIG["max_side"], type=int, show_default=True)
@click.option("--ransac-threshold", default=DEFAULT_MATCHING_CONFIG["ransac_threshold"], type=float, show_default=True)
@click.option("--bands", "bands_value", default=DEFAULT_MATCHING_CONFIG["bands"], show_default=True)
def cli(left, right, method, output, max_features, max_side, ransac_threshold, bands_value):
    bands = tuple(item.strip() for item in bands_value.split(",") if item.strip())
    if len(bands) != 3:
        raise ValueError("--bands must contain exactly three band names")
    run(left, right, method, output, max_features, max_side, ransac_threshold, bands)


if __name__ == "__main__":
    cli()
