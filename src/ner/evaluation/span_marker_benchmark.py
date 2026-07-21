"""Evaluate several pretrained BERT inference checkpoints on one dataset split."""
import csv
import json
import sys
import time
from pathlib import Path

import click
import torch
from datasets import load_from_disk
try:
    import optuna
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Missing dependency: install Optuna with `python -m pip install optuna`."
    ) from exc
from tqdm import tqdm

try:
    from span_marker import SpanMarkerModel
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Missing dependency: install SpanMarker with `python -m pip install span-marker`."
    ) from exc

try:
    from ner.utils.ner_bio import seqeval_metrics, word_spans_to_bio_labels
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from ner.utils.ner_bio import seqeval_metrics, word_spans_to_bio_labels


def run_experiment(
    data_path: str,
    tuning_split: str,
    evaluation_split: str,
    model_names: tuple[str, ...],
    label_name: str,
    batch_sizes: tuple[int, ...],
    n_trials: int,
    threshold_low: float,
    threshold_high: float,
    max_tuning_examples: int | None,
    max_evaluation_examples: int | None,
    output_path: str | None,
    show_progress: bool,
    device_name: str,
) -> None:
    """Run inference, evaluate BIO labels, print results, and optionally save them."""
    if tuning_split == evaluation_split:
        raise click.BadParameter("--tuning-split and --evaluation-split must be different.")
    if threshold_low >= threshold_high:
        raise click.BadParameter("--threshold-low must be smaller than --threshold-high.")

    dataset = load_from_disk(data_path)
    required_splits = {tuning_split, evaluation_split}
    missing_splits = required_splits.difference(dataset.keys())

    if missing_splits:
        raise click.ClickException(f"Missing dataset split(s): {sorted(missing_splits)}")

    tuning_dataset = dataset[tuning_split]
    evaluation_dataset = dataset[evaluation_split]

    if max_tuning_examples is not None:
        tuning_dataset = tuning_dataset.select(
            range(min(max_tuning_examples, len(tuning_dataset)))
        )

    if max_evaluation_examples is not None:
        evaluation_dataset = evaluation_dataset.select(
            range(min(max_evaluation_examples, len(evaluation_dataset)))
        )

    required_columns = {"tokens", "labels"}

    for split_name, split_dataset in (
            (tuning_split, tuning_dataset), (evaluation_split, evaluation_dataset)
    ):
        missing_columns = required_columns.difference(split_dataset.column_names)

        if missing_columns:
            raise click.ClickException(
                f"Split '{split_name}' is missing columns: {sorted(missing_columns)}"
            )
        for example in split_dataset.select(range(min(100, len(split_dataset)))):

            if len(example["tokens"]) != len(example["labels"]):
                raise click.ClickException(
                    f"Split '{split_name}' has a tokens/labels length mismatch."
                )

            if not set(example["labels"]).issubset({0, 1, 2}):
                raise click.ClickException(
                    f"Split '{split_name}' contains labels outside the expected BIO ids {{0, 1, 2}}."
                )

    if not tuning_dataset or not evaluation_dataset:
        raise click.ClickException("Tuning and evaluation splits must not be empty.")

    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available() else device_name
    )

    if device.type == "cuda" and not torch.cuda.is_available():
        raise click.ClickException("CUDA was requested but is not available.")

    print("experiment_parameters:")
    print(f"data_path={data_path}")
    print(f"tuning_split={tuning_split}")
    print(f"evaluation_split={evaluation_split}")
    print(f"models={list(model_names)}")
    print(f"label_name={label_name}")
    print(f"batch_sizes={list(batch_sizes)}")
    print(f"n_trials={n_trials}")
    print(f"threshold_range=({threshold_low}, {threshold_high})")
    print(f"tuning_examples={len(tuning_dataset)}")
    print(f"evaluation_examples={len(evaluation_dataset)}")
    print(f"device={device}")

    results = []

    for model_name in tqdm(model_names, desc="Pretrained BERT inference models"):
        model = SpanMarkerModel.from_pretrained(model_name)
        model.to(device)
        for batch_size in batch_sizes:
            tuning_predictions = model.predict(
                [example["tokens"] for example in tuning_dataset],
                batch_size=batch_size,
                show_progress_bar=show_progress,
            )

            study = optuna.create_study(direction="maximize")
            study.optimize(
                lambda trial: seqeval_metrics(
                    [example["labels"] for example in tuning_dataset],
                    [
                        word_spans_to_bio_labels(
                            example["tokens"],
                            [
                                entity
                                for entity in prediction
                                if entity["label"] == label_name
                                and entity["score"] >= trial.suggest_float(
                                    "score_threshold", threshold_low, threshold_high
                                )
                            ],
                        )
                        for example, prediction in zip(tuning_dataset, tuning_predictions)
                    ],
                )["overall_f1"],
                n_trials=n_trials,
                show_progress_bar=False,
            )
            selected_threshold = study.best_params["score_threshold"]
            tuning_f1 = study.best_value
            print(
                f"optuna_best_params: model={model_name} batch_size={batch_size} "
                f"params={study.best_params} tuning_f1={tuning_f1:.4f}"
            )

            started_at = time.perf_counter()
            evaluation_predictions = model.predict(
                [example["tokens"] for example in evaluation_dataset],
                batch_size=batch_size,
                show_progress_bar=show_progress,
            )
            inference_time = time.perf_counter() - started_at

            true_labels = [example["labels"] for example in evaluation_dataset]
            predicted_labels = [
                word_spans_to_bio_labels(
                    example["tokens"],
                    [
                        entity
                        for entity in prediction
                        if entity["label"] == label_name and entity["score"] >= selected_threshold
                    ],
                )
                for example, prediction in zip(evaluation_dataset, evaluation_predictions)
            ]
            metrics = seqeval_metrics(true_labels, predicted_labels)
            result = {
                "model": model_name,
                "tuning_split": tuning_split,
                "evaluation_split": evaluation_split,
                "tuning_examples": len(tuning_dataset),
                "evaluation_examples": len(evaluation_dataset),
                "batch_size": batch_size,
                "n_trials": n_trials,
                "score_threshold": selected_threshold,
                "tuning_f1": tuning_f1,
                "accuracy": metrics["overall_accuracy"],
                "precision": metrics["overall_precision"],
                "recall": metrics["overall_recall"],
                "f1": metrics["overall_f1"],
                "benchmark_inference_time": inference_time,
                "avg_benchmark_time_per_request": inference_time / max(len(evaluation_dataset), 1),
            }
            results.append(result)
            print(
                f"model={model_name} tuning_split={tuning_split} evaluation_split={evaluation_split} "
                f"batch_size={batch_size} trials={n_trials} threshold={selected_threshold:.4f} "
                f"tuning_f1={tuning_f1:.4f} accuracy={result['accuracy']:.4f} "
                f"precision={result['precision']:.4f} recall={result['recall']:.4f} "
                f"f1={result['f1']:.4f} inference_time={inference_time:.3f}s "
                f"avg_request={result['avg_benchmark_time_per_request']:.4f}s"
            )

    if output_path is None:
        return

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".json":
        output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    else:
        with output.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
    print(f"saved_results={output}")


@click.command()
@click.option(
    "--data-path",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Few-NERD dataset saved with datasets.save_to_disk.",
)
@click.option("--tuning-split", required=True, type=click.Choice(["train", "validation", "test"]))
@click.option("--evaluation-split", required=True, type=click.Choice(["train", "validation", "test"]))
@click.option("--model", "model_names", required=True, multiple=True, help="Repeat for each pretrained BERT inference model.")
@click.option("--label-name", required=True)
@click.option(
    "--batch-size",
    "batch_sizes",
    required=True,
    multiple=True,
    type=click.IntRange(min=1),
    help="Repeat to compare inference batch sizes.",
)
@click.option("--n-trials", required=True, type=click.IntRange(min=1))
@click.option("--threshold-low", required=True, type=click.FloatRange(min=0.0, max=1.0))
@click.option("--threshold-high", required=True, type=click.FloatRange(min=0.0, max=1.0))
@click.option("--max-tuning-examples", type=click.IntRange(min=1))
@click.option("--max-evaluation-examples", type=click.IntRange(min=1))
@click.option("--output-path", type=click.Path(), help="Optional .csv or .json output path.")
@click.option("--show-progress/--no-show-progress", default=True, show_default=True)
@click.option("--device", "device_name", default="auto", type=click.Choice(["auto", "cpu", "cuda"]), show_default=True)
def cli(
        data_path,
        tuning_split,
        evaluation_split,
        model_names,
        label_name,
        batch_sizes,
        n_trials,
        threshold_low,
        threshold_high,
        max_tuning_examples,
        max_evaluation_examples,
        output_path,
        show_progress,
        device_name
):
    """Tune and evaluate the requested pretrained BERT inference models from Click options."""
    run_experiment(
        data_path=data_path,
        tuning_split=tuning_split,
        evaluation_split=evaluation_split,
        model_names=model_names,
        label_name=label_name,
        batch_sizes=batch_sizes,
        n_trials=n_trials,
        threshold_low=threshold_low,
        threshold_high=threshold_high,
        max_tuning_examples=max_tuning_examples,
        max_evaluation_examples=max_evaluation_examples,
        output_path=output_path,
        show_progress=show_progress,
        device_name=device_name,
    )


if __name__ == "__main__":
    cli()
