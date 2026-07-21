"""Evaluate the rule-based mountain matcher on a saved Few-NERD split."""

from __future__ import annotations

import json
import time
from pathlib import Path

import click
from datasets import load_from_disk
from tqdm import tqdm

from match_names import MountainNameMatcher

try:
    from ner.utils.ner_bio import seqeval_metrics, spans_to_bio_labels
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from ner.utils.ner_bio import seqeval_metrics, spans_to_bio_labels


@click.command()
@click.option("--data-path", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--catalog", "catalog_paths", required=True, multiple=True, type=click.Path(exists=True))
@click.option("--output", required=True, type=click.Path())
@click.option("--fuzzy-threshold", default=0.0, type=click.FloatRange(0.0, 1.0), show_default=True)
def cli(data_path: str, catalog_paths: tuple[str, ...], output: str, fuzzy_threshold: float) -> None:
    """Evaluate the matcher on a Dataset saved with Hugging Face save_to_disk."""
    dataset = load_from_disk(data_path)
    if not {"sentence", "tokens", "labels"}.issubset(dataset.column_names):
        raise click.ClickException("Dataset must contain sentence, tokens, and labels columns.")

    matcher = MountainNameMatcher(list(catalog_paths), fuzzy_threshold)
    true_labels = []
    predicted_labels = []
    predicted_entities = 0
    started = time.perf_counter()

    for index, example in enumerate(tqdm(dataset, desc="Evaluating mountain matcher")):
        matches = matcher.match_text(example["sentence"], f"test[{index}]")
        spans = [{"start": item.start, "end": item.end} for item in matches]
        true_labels.append(example["labels"])
        predicted_labels.append(spans_to_bio_labels(example["tokens"], spans))
        predicted_entities += len(matches)

    elapsed = time.perf_counter() - started
    metrics = seqeval_metrics(true_labels, predicted_labels)
    result = {
        "method": "catalog_union_exact_normalized",
        "test_examples": len(dataset),
        "catalogs": list(catalog_paths),
        "unique_catalog_names": len(matcher.canonical_names),
        "fuzzy_threshold": fuzzy_threshold,
        "predicted_entities": predicted_entities,
        "accuracy": metrics["overall_accuracy"],
        "precision": metrics["overall_precision"],
        "recall": metrics["overall_recall"],
        "f1": metrics["overall_f1"],
        "inference_time_seconds": elapsed,
        "avg_inference_time_seconds": elapsed / len(dataset),
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    click.echo(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    cli()
