"""Build a labeled image-pair manifest from Sentinel-2 SAFE scenes."""

from __future__ import annotations

import csv
import random
import re
from itertools import combinations
from pathlib import Path

import click
from tqdm import tqdm

from configs.default import DEFAULT_MATCHING_CONFIG


SCENE_PATTERN = re.compile(r"_(\d{8})T\d+.*?_T(\d{2}[A-Z]{3})_")


def scene_metadata(path: Path) -> tuple[str, str] | None:
    """Extract acquisition date and Sentinel tile id from a SAFE directory."""
    match = SCENE_PATTERN.search(path.name)
    if match is None:
        return None
    return match.group(1), match.group(2)


def build_pairs(root: str, max_positive_pairs_per_tile: int, negative_ratio: int, seed: int) -> list[dict]:
    """Create positive same-tile pairs and hard negatives from different tiles."""
    rng = random.Random(seed)
    scenes_by_tile: dict[str, list[tuple[str, str]]] = {}
    for scene in tqdm(sorted(Path(root).rglob("*.SAFE")), desc="Scanning SAFE scenes"):
        metadata = scene_metadata(scene)
        if metadata is None:
            continue
        date, tile = metadata
        scenes_by_tile.setdefault(tile, []).append((date, str(scene)))

    positive_pairs = []
    for tile, scenes in scenes_by_tile.items():
        tile_pairs = [
            {"left": left[1], "right": right[1], "label": 1, "tile": tile}
            for left, right in combinations(sorted(scenes), 2)
        ]
        if len(tile_pairs) > max_positive_pairs_per_tile:
            tile_pairs = rng.sample(tile_pairs, max_positive_pairs_per_tile)
        positive_pairs.extend(tile_pairs)

    if not positive_pairs:
        raise ValueError("No same-tile pairs found. Check the Kaggle SAFE root and scene names.")

    negatives = []
    tiles = list(scenes_by_tile)
    for pair in positive_pairs:
        other_tile = rng.choice([tile for tile in tiles if tile != pair["tile"]]) if len(tiles) > 1 else None
        if other_tile is None:
            continue
        left_scene = rng.choice(scenes_by_tile[pair["tile"]])[1]
        right_scene = rng.choice(scenes_by_tile[other_tile])[1]
        negatives.append({"left": left_scene, "right": right_scene, "label": 0, "tile": f"{pair['tile']}-{other_tile}"})
        if len(negatives) >= len(positive_pairs) * negative_ratio:
            break
    return positive_pairs + negatives


def save_manifest(rows: list[dict], output_path: str, seed: int, allow_pair_level_fallback: bool) -> None:
    """Split by base tile groups, preventing geography leakage across splits."""
    rng = random.Random(seed)
    groups = sorted({tile for row in rows for tile in row["tile"].split("-")})
    split_rows = []

    if len(groups) >= 3:
        rng.shuffle(groups)
        train_end = max(1, int(len(groups) * 0.7))
        validation_end = max(train_end + 1, int(len(groups) * 0.85))
        split_by_group = {
            group: "train" if index < train_end else "validation" if index < validation_end else "test"
            for index, group in enumerate(groups)
        }
        for row in rows:
            row_tiles = row["tile"].split("-")
            row_splits = {split_by_group[tile] for tile in row_tiles}
            if len(row_splits) == 1:
                row["split"] = row_splits.pop()
                split_rows.append(row)
    else:
        if not allow_pair_level_fallback:
            raise ValueError(
                "At least three geographic tiles are required for leakage-safe train/validation/test splits. "
                "Use --allow-pair-level-fallback only for a smoke test; do not report its metrics as geographic generalization."
            )
        if len(rows) < 3:
            raise ValueError("At least three pairs are required for a small-dataset split.")
        print("WARNING: fewer than three geographic tiles were found. Using pair-level fallback; this split is not geography-leakage-safe.")
        shuffled_rows = list(rows)
        rng.shuffle(shuffled_rows)
        train_end = max(1, int(len(shuffled_rows) * 0.7))
        validation_end = max(train_end + 1, int(len(shuffled_rows) * 0.85))
        for index, row in enumerate(shuffled_rows):
            row["split"] = "train" if index < train_end else "validation" if index < validation_end else "test"
            split_rows.append(row)

    split_counts = {split: sum(row["split"] == split for row in split_rows) for split in ("train", "validation", "test")}
    if any(count == 0 for count in split_counts.values()):
        raise ValueError(f"The manifest split is empty: {split_counts}. Add more geographic tiles or reduce pair constraints.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["left", "right", "label", "tile", "split"])
        writer.writeheader()
        writer.writerows(split_rows)


@click.command()
@click.option("--root", required=True, type=click.Path(exists=True, file_okay=False), help="Kaggle directory containing SAFE scenes.")
@click.option("--output", default=DEFAULT_MATCHING_CONFIG["manifest_output"], show_default=True)
@click.option("--max-positive-pairs-per-tile", default=DEFAULT_MATCHING_CONFIG["max_positive_pairs_per_tile"], type=int, show_default=True)
@click.option("--negative-ratio", default=DEFAULT_MATCHING_CONFIG["negative_ratio"], type=int, show_default=True)
@click.option("--seed", default=DEFAULT_MATCHING_CONFIG["seed"], type=int, show_default=True)
@click.option("--allow-pair-level-fallback", is_flag=True, help="Allow a leakage-prone split only for smoke tests with fewer than three tiles.")
def cli(root, output, max_positive_pairs_per_tile, negative_ratio, seed, allow_pair_level_fallback):
    pairs = build_pairs(root, max_positive_pairs_per_tile, negative_ratio, seed)
    save_manifest(pairs, output, seed, allow_pair_level_fallback)
    click.echo(f"saved={output} pairs={len(pairs)} positives={sum(row['label'] == 1 for row in pairs)} negatives={sum(row['label'] == 0 for row in pairs)}")


if __name__ == "__main__":
    cli()
