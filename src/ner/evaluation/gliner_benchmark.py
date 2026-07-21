"""Tune and evaluate several GLiNER models in one reproducible experiment."""
import csv
import json
import sys
import time
from pathlib import Path

import click
import optuna
import torch
from datasets import load_from_disk
from gliner import GLiNER
from tqdm import tqdm

try:
    from ner.utils.ner_bio import seqeval_metrics, spans_to_bio_labels
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from ner.utils.ner_bio import seqeval_metrics, spans_to_bio_labels


def run_experiments(
    data_path: str,
    model_names: tuple[str, ...],
    label_text: str,
    tuning_split: str,
    evaluation_split: str,
    n_trials: int,
    threshold_low: float,
    threshold_high: float,
    max_tuning_examples: int | None,
    max_evaluation_examples: int | None,
    output_path: str,
    show_progress: bool,
    device_name: str,
) -> None:
    """Tune every model, evaluate it, print metrics, and save a result file."""
    if tuning_split == evaluation_split:
        raise click.BadParameter("--tuning-split and --evaluation-split must be different.")
    if threshold_low >= threshold_high:
        raise click.BadParameter("--threshold-low must be smaller than --threshold-high.")

    dataset = load_from_disk(data_path)
    missing_splits = {tuning_split, evaluation_split}.difference(dataset.keys())

    if missing_splits:
        raise click.ClickException(f"Missing dataset split(s): {sorted(missing_splits)}")

    tuning_dataset = dataset[tuning_split]
    evaluation_dataset = dataset[evaluation_split]

    if max_tuning_examples is not None:
        tuning_dataset = tuning_dataset.select(range(min(max_tuning_examples, len(tuning_dataset))))

    if max_evaluation_examples is not None:
        evaluation_dataset = evaluation_dataset.select(range(min(max_evaluation_examples, len(evaluation_dataset))))

    required_columns = {"sentence", "tokens", "labels"}

    if (not required_columns.issubset(tuning_dataset.column_names) or
            not required_columns.issubset(evaluation_dataset.column_names)):
        raise click.ClickException("Both splits must contain sentence, tokens, and labels columns.")

    if not tuning_dataset or not evaluation_dataset:
        raise click.ClickException("Tuning and evaluation splits must not be empty.")

    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available() else device_name
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise click.ClickException("CUDA was requested but is not available.")

    print("experiment_parameters:")
    print(f"data_path={data_path}")
    print(f"models={list(model_names)}")
    print(f"label_text={label_text}")
    print(f"tuning_split={tuning_split} examples={len(tuning_dataset)}")
    print(f"evaluation_split={evaluation_split} examples={len(evaluation_dataset)}")
    print(f"n_trials={n_trials}")
    print(f"threshold_range=({threshold_low}, {threshold_high})")
    print(f"device={device}")

    results = []
    for model_name in model_names:
        print(f"\nloading_model={model_name}")
        model = GLiNER.from_pretrained(model_name)
        model.to(device)

        # initial inference to select optimal threshold
        tuning_predictions = [
            model.predict_entities(example["sentence"], [label_text], threshold=0.0)
            for example in tqdm(tuning_dataset, desc=f"{model_name} tuning", disable=not show_progress)
        ]
        true_tuning_labels = [example["labels"] for example in tuning_dataset]

        study = optuna.create_study(
            direction="maximize",
            study_name=f"gliner_{model_name.replace('/', '_')}"
        )

        trial_records = []
        for trial_number in range(n_trials):
            trial = study.ask()
            threshold = trial.suggest_float("threshold", threshold_low, threshold_high)
            tuning_predicted_labels = [
                spans_to_bio_labels(
                    example["tokens"],
                    [entity for entity in prediction if entity["score"] >= threshold],
                )
                for example, prediction in zip(tuning_dataset, tuning_predictions)
            ]
            tuning_metrics = seqeval_metrics(true_tuning_labels, tuning_predicted_labels)
            study.tell(trial, tuning_metrics["overall_f1"])
            trial_record = {
                "trial_number": trial_number,
                "threshold": threshold,
                "tuning_f1": tuning_metrics["overall_f1"],
            }
            trial_records.append(trial_record)
            print(
                f"trial model={model_name} number={trial_number} "
                f"threshold={threshold:.4f} tuning_f1={tuning_metrics['overall_f1']:.4f}"
            )

        selected_threshold = study.best_params["threshold"]
        started_at = time.perf_counter()
        evaluation_predictions = [
            model.predict_entities(example["sentence"], [label_text], threshold=selected_threshold)
            for example in tqdm(evaluation_dataset, desc=f"{model_name} evaluation", disable=not show_progress)
        ]
        inference_time = time.perf_counter() - started_at
        true_evaluation_labels = [example["labels"] for example in evaluation_dataset]
        predicted_evaluation_labels = [
            spans_to_bio_labels(
                example["tokens"],
                prediction,
            )
            for example, prediction in zip(evaluation_dataset, evaluation_predictions)
        ]
        evaluation_metrics = seqeval_metrics(true_evaluation_labels, predicted_evaluation_labels)
        result = {
            "model": model_name,
            "label_text": label_text,
            "tuning_split": tuning_split,
            "evaluation_split": evaluation_split,
            "n_trials": n_trials,
            "threshold": selected_threshold,
            "tuning_f1": study.best_value,
            "accuracy": evaluation_metrics["overall_accuracy"],
            "precision": evaluation_metrics["overall_precision"],
            "recall": evaluation_metrics["overall_recall"],
            "f1": evaluation_metrics["overall_f1"],
            "benchmark_inference_time": inference_time,
            "avg_benchmark_time_per_request": inference_time / len(evaluation_dataset),
            "trials": trial_records,
        }
        results.append(result)
        print(f"optuna_best_params model={model_name} threshold={selected_threshold:.4f}")
        print(
            f"final_result model={model_name} accuracy={result['accuracy']:.4f} "
            f"precision={result['precision']:.4f} recall={result['recall']:.4f} "
            f"f1={result['f1']:.4f} inference_time={inference_time:.3f}s "
            f"avg_request={result['avg_benchmark_time_per_request']:.4f}s"
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".csv":
        csv_rows = [{key: value for key, value in result.items() if key != "trials"} for result in results]
        with output.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
    else:
        output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"saved_results={output.resolve()}")


@click.command()
@click.option("--data-path", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--model", "model_names", required=True, multiple=True, help="Repeat for every GLiNER model.")
@click.option("--label-text", required=True)
@click.option("--tuning-split", required=True, type=click.Choice(["train", "validation", "test"]))
@click.option("--evaluation-split", required=True, type=click.Choice(["train", "validation", "test"]))
@click.option("--n-trials", required=True, type=click.IntRange(min=1))
@click.option("--threshold-low", required=True, type=click.FloatRange(min=0.0, max=1.0))
@click.option("--threshold-high", required=True, type=click.FloatRange(min=0.0, max=1.0))
@click.option("--max-tuning-examples", type=click.IntRange(min=1))
@click.option("--max-evaluation-examples", type=click.IntRange(min=1))
@click.option("--output-path", required=True, type=click.Path())
@click.option("--show-progress/--no-show-progress", default=True, show_default=True)
@click.option("--device", "device_name", default="auto", type=click.Choice(["auto", "cpu", "cuda"]), show_default=True)
def cli(
        data_path,
        model_names,
        label_text,
        tuning_split,
        evaluation_split,
        n_trials,
        threshold_low,
        threshold_high,
        max_tuning_examples,
        max_evaluation_examples,
        output_path,
        show_progress,
        device_name
):
    """Tune and evaluate each requested GLiNER model from Click options."""
    run_experiments(
        data_path,
        model_names,
        label_text,
        tuning_split,
        evaluation_split,
        n_trials,
        threshold_low,
        threshold_high,
        max_tuning_examples,
        max_evaluation_examples,
        output_path,
        show_progress,
        device_name
    )


if __name__ == "__main__":
    cli()
