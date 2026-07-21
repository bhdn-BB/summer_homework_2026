"""Regenerate Sentinel matching benchmark charts from the aggregate CSV."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


MATCHER_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = MATCHER_ROOT / "data" / "kaggle_benchmark" / "all_methods_metrics.csv"
DEFAULT_OUTPUT_DIR = MATCHER_ROOT / "assets"
COLORS = ["#4c9ad4", "#ff9f43", "#8e6bbd", "#2ca25f"]


def display_name(row: dict[str, float | str]) -> str:
    method = str(row["method"])
    return "SUPERPOINT + LIGHTGLUE" if method == "superpoint_lightglue" else method.upper()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, float | str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No benchmark rows found in {path}")
    for row in rows:
        for column in ("candidate_matches", "inlier_matches", "inlier_ratio", "inference_time_seconds"):
            row[column] = float(row[column])
    return rows


def save_inlier_chart(rows: list[dict[str, float | str]], output: Path) -> None:
    names = [display_name(row) for row in rows]
    values = [float(row["inlier_matches"]) for row in rows]
    figure, axis = plt.subplots(figsize=(7, 4.5))
    bars = axis.bar(names, values, color=COLORS[:len(names)])
    axis.bar_label(bars, labels=[f"{value:,.0f}" for value in values], padding=3)
    axis.set_ylabel("Geometrically verified matches")
    axis.set_title("Sentinel-2 matching: verified inliers")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)


def save_tradeoff_chart(rows: list[dict[str, float | str]], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 4.5))
    for row in rows:
        name = display_name(row)
        seconds = float(row["inference_time_seconds"])
        ratio = float(row["inlier_ratio"])
        inliers = float(row["inlier_matches"])
        axis.scatter(seconds, ratio, s=max(70, inliers / 18), alpha=0.75)
        axis.annotate(name, (seconds, ratio), xytext=(6, 5), textcoords="offset points")
    axis.set_xlabel("Runtime (seconds)")
    axis.set_ylabel("Inlier ratio")
    axis.set_title("Quality-speed trade-off (bubble = inliers)")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)


def save_efficiency_chart(rows: list[dict[str, float | str]], output: Path) -> None:
    names = [display_name(row) for row in rows]
    values = [float(row["inlier_matches"]) / float(row["inference_time_seconds"]) for row in rows]
    figure, axis = plt.subplots(figsize=(7, 4.5))
    bars = axis.bar(names, values, color=COLORS[:len(names)])
    axis.bar_label(bars, labels=[f"{value:.1f}" for value in values], padding=3)
    axis.set_ylabel("Verified inliers per second")
    axis.set_title("Sentinel-2 matching efficiency")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_inlier_chart(rows, args.output_dir / "benchmark_inliers.png")
    save_tradeoff_chart(rows, args.output_dir / "benchmark_tradeoff.png")
    save_efficiency_chart(rows, args.output_dir / "benchmark_efficiency.png")
    print(f"saved_charts={args.output_dir}")


if __name__ == "__main__":
    main()
