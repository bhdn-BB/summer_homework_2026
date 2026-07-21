"""Evaluate a fine-tuned BERT checkpoint and optionally infer one sentence."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import click
import torch
from transformers import AutoModelForTokenClassification
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import Trainer
from transformers import TrainingArguments

SRC_ROOT = str(Path(__file__).resolve().parents[2])
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from ner.fine_tuning.bert_config import get_fine_tuning_config
from ner.utils.bert_finetuning import compute_token_metrics, load_and_tokenize_dataset


def run_experiments(
    model_path: str,
    data_path: str,
    evaluation_split: str,
    max_examples: int | None,
    sentence: str | None,
    device_name: str,
    disable_wandb: bool,
) -> None:
    """Run held-out evaluation and optional single-sentence inference."""
    config = get_fine_tuning_config()
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available() else device_name
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise click.ClickException("CUDA was requested but is not available.")
    if not Path(model_path).exists():
        raise click.ClickException(f"Model path does not exist: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_path).to(device)
    _, tokenized_dataset = load_and_tokenize_dataset(
        data_path=data_path,
        tokenizer=tokenizer,
        train_split=evaluation_split,
        eval_split=evaluation_split,
        max_eval_examples=max_examples,
        max_length=config["max_length"],
    )
    report_to = ["none"] if disable_wandb else config["report_to"]
    if not disable_wandb:
        os.environ.setdefault("WANDB_PROJECT", config["wandb_project"])
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(Path(model_path) / "evaluation"),
            per_device_eval_batch_size=config["eval_batch_size"],
            report_to=report_to,
            run_name=config["wandb_run_name"],
        ),
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
        compute_metrics=compute_token_metrics,
    )
    metrics = trainer.evaluate(
        tokenized_dataset[evaluation_split],
        metric_key_prefix=evaluation_split,
    )

    print("bert_parameters:")
    print(f"  model_path={model_path}")
    print(f"  evaluation_split={evaluation_split}")
    print(f"  examples={len(tokenized_dataset[evaluation_split])}")
    print(f"  device={device}")
    for name, value in metrics.items():
        print(f"{name}={value}")

    if sentence is None:
        return

    inputs = tokenizer(
        sentence.split(),
        return_tensors="pt",
        is_split_into_words=True,
        truncation=True,
        max_length=config["max_length"],
    )
    model_inputs = {key: value.to(device) for key, value in inputs.items()}
    model.eval()
    with torch.inference_mode():
        predictions = model(**model_inputs).logits.argmax(dim=-1).squeeze().tolist()
    if isinstance(predictions, int):
        predictions = [predictions]
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())
    labels = [model.config.id2label[int(label_id)] for label_id in predictions]

    print("sentence_predictions:")
    merged = []
    for token, label in zip(tokens, labels):
        if token in {"[CLS]", "[SEP]", "[PAD]"}:
            continue
        if token.startswith("##") and merged:
            merged[-1]["token"] += token[2:]
        else:
            merged.append({"token": token, "label": label})
    for item in merged:
        print(f"{item['token']}: {item['label']}")


@click.command()
@click.option("--model-path", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--data-path", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--evaluation-split", required=True, type=click.Choice(["train", "validation", "test"]))
@click.option("--max-examples", type=click.IntRange(min=1))
@click.option("--sentence", help="Optional sentence for token-level inference.")
@click.option("--device", "device_name", default="auto", type=click.Choice(["auto", "cpu", "cuda"]), show_default=True)
@click.option("--disable-wandb", is_flag=True)
def cli(model_path, data_path, evaluation_split, max_examples, sentence, device_name, disable_wandb):
    """Evaluate a saved checkpoint and optionally infer one input sentence."""
    run_experiments(model_path, data_path, evaluation_split, max_examples, sentence, device_name, disable_wandb)


if __name__ == "__main__":
    cli()
