"""Create the recorded NER quality-versus-latency chart."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt


NER_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = NER_ROOT / "data" / "results" / "ner_model_tradeoff.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [
        ("SpanMarker", 0.7930, 0.1216, "#6a3d9a", "o"),
        ("BERT CE", 0.7847, 0.0056, "#1f77b4", "o"),
        ("BERT weighted CE", 0.7722, 0.0055, "#4c9ad4", "o"),
        ("BERT CE (val)", 0.8193, 2.0374 / 402, "#1f77b4", "D"),
        ("BERT wCE (val)", 0.7822, 2.0381 / 402, "#4c9ad4", "D"),
        ("Synthetic BERT (val)", 0.8184, 1.4613 / 402, "#e76f51", "D"),
        ("GLiNER large", 0.7050, 0.0467, "#2ca02c", "o"),
        ("GLiNER medium", 0.6910, 0.0257, "#66a95c", "o"),
        ("GLiNER small", 0.6987, 0.0160, "#98c97b", "o"),
    ]
    label_options = {
        "BERT CE (val)": {"xytext": (8, -12), "ha": "left"},
        "BERT wCE (val)": {"xytext": (8, -12), "ha": "left"},
        "Synthetic BERT (val)": {"xytext": (-8, 8), "ha": "right"},
    }

    figure, axis = plt.subplots(figsize=(10, 6))
    for name, f1, seconds, color, marker in rows:
        axis.scatter(seconds, f1, s=110, color=color, marker=marker, edgecolor="white", linewidth=0.8, zorder=3)
        options = label_options.get(name, {"xytext": (7, 5), "ha": "left"})
        axis.annotate(name, (seconds, f1), xytext=options["xytext"], textcoords="offset points", ha=options["ha"])

    axis.set_xscale("log")
    axis.set_xlabel("Average inference time per sample (seconds, log scale)")
    axis.set_ylabel("Entity-level F1")
    axis.set_title("Mountain NER: quality versus inference latency", pad=14)
    axis.set_ylim(0.685, 0.826)
    axis.grid(alpha=0.25, zorder=0)
    axis.text(
        0.02,
        0.02,
        "Diamonds: BERT validation results; circles: held-out test results.",
        transform=axis.transAxes,
        fontsize=8,
        va="bottom",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(args.output, dpi=180)
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
