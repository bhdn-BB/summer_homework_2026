import sys
import os
from pathlib import Path
from inspect import signature

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
from ner.fine_tuning.trainer import MountainTrainer
from ner.utils.bert_finetuning import (
    compute_class_weights,
    compute_token_metrics,
    load_and_tokenize_dataset,
    setup_file_logging,
)


def needs_class_weights(config: dict) -> bool:
    """Return whether the selected loss requires precomputed class weights."""
    return (
        config["loss_name"] == "weighted_cross_entropy"
        or (config["loss_name"] == "focal" and config["focal_use_class_weights"])
    )


def resolve_runtime_paths(config: dict) -> dict:
    """Redirect writable runtime artifacts away from read-only dataset mounts."""
    resolved = config.copy()
    output_dir = Path(resolved["output_dir"])
    save_model_path = Path(resolved["save_model_path"])
    log_file = Path(resolved["log_file"])

    if str(log_file).startswith("/kaggle/input/"):
        resolved["log_file"] = str(output_dir / "logs" / log_file.name)

    if str(output_dir).startswith("/kaggle/input/"):
        resolved["output_dir"] = str(Path("/kaggle/working") / output_dir.name)
    if str(save_model_path).startswith("/kaggle/input/"):
        resolved["save_model_path"] = str(Path("/kaggle/working") / save_model_path.name)

    return resolved


def training_arguments(config: dict) -> TrainingArguments:
    """Build Hugging Face training arguments from the Python config dictionary."""
    kwargs = {
        "output_dir": config["output_dir"],
        "logging_strategy": config["logging_strategy"],
        "logging_steps": config["logging_steps"],
        "save_strategy": config["save_strategy"],
        "load_best_model_at_end": config["load_best_model_at_end"],
        "metric_for_best_model": config["best_model_metric"],
        "greater_is_better": config["greater_is_better"],
        "learning_rate": config["learning_rate"],
        "per_device_train_batch_size": config["train_batch_size"],
        "per_device_eval_batch_size": config["eval_batch_size"],
        "num_train_epochs": config["num_epochs"],
        "weight_decay": config["weight_decay"],
        "save_total_limit": config["save_total_limit"],
        "seed": config["seed"],
        "report_to": config["report_to"],
        "run_name": config["wandb_run_name"],
        "gradient_accumulation_steps": config["gradient_accumulation_steps"],
        "fp16": config["fp16"],
    }
    eval_arg_name = (
        "eval_strategy"
        if "eval_strategy" in signature(TrainingArguments.__init__).parameters
        else "evaluation_strategy"
    )
    kwargs[eval_arg_name] = config["eval_strategy"]
    if config["device"] == "cpu":
        if "use_cpu" in signature(TrainingArguments.__init__).parameters:
            kwargs["use_cpu"] = True
        elif "no_cuda" in signature(TrainingArguments.__init__).parameters:
            kwargs["no_cuda"] = True

    return TrainingArguments(**kwargs)


def train(config: dict) -> None:
    """Train, validate, test, and save one configured token-classification model."""
    if config["device"] == "cuda" and not torch.cuda.is_available():
        raise click.ClickException("CUDA was requested but is not available.")
    if config["device"] == "cpu" and config["fp16"]:
        raise click.ClickException("fp16 is only supported when training on CUDA.")
    if config["report_to"] != ["none"]:
        os.environ.setdefault("WANDB_PROJECT", config["wandb_project"])
    setup_file_logging(config["log_file"])
    print(f"training_device={config['device']}")

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(
        config["model_name"],
        num_labels=config["num_labels"],
        id2label=config["id2label"],
        label2id=config["label2id"],
        ignore_mismatched_sizes=True,
    )

    dataset, tokenized_dataset = load_and_tokenize_dataset(
        data_path=config["data_path"],
        tokenizer=tokenizer,
        train_split=config["train_split"],
        eval_split=config["eval_split"],
        max_length=config["max_length"],
        max_train_examples=config["max_train_examples"],
        max_eval_examples=config["max_eval_examples"],
    )

    class_weights = None
    if needs_class_weights(config):
        class_weights = compute_class_weights(
            dataset_split=dataset[config["train_split"]],
            num_labels=config["num_labels"],
            normalize=config["normalize_class_weights"],
        )

    trainer_kwargs = {
        "model": model,
        "args": training_arguments(config),
        "train_dataset": tokenized_dataset[config["train_split"]],
        "eval_dataset": tokenized_dataset[config["eval_split"]],
        "data_collator": DataCollatorForTokenClassification(tokenizer=tokenizer),
        "compute_metrics": compute_token_metrics,
        "loss_config": config,
        "class_weights": class_weights,
    }
    processor_argument = (
        "processing_class"
        if "processing_class" in signature(Trainer.__init__).parameters
        else "tokenizer"
    )
    trainer_kwargs[processor_argument] = tokenizer

    trainer = MountainTrainer(
        **trainer_kwargs,
    )

    trainer.train(resume_from_checkpoint=config["resume_from_checkpoint"])
    print({"validation": trainer.evaluate()})

    _, tokenized_test = load_and_tokenize_dataset(
        data_path=config["data_path"],
        tokenizer=tokenizer,
        train_split=config["test_split"],
        eval_split=config["test_split"],
        max_eval_examples=config["max_test_examples"],
        max_length=config["max_length"],
    )
    test_metrics = trainer.evaluate(
        eval_dataset=tokenized_test[config["test_split"]],
        metric_key_prefix="test",
    )
    print({"test": test_metrics})

    trainer.save_model(config["save_model_path"])
    tokenizer.save_pretrained(config["save_model_path"])

    if config["push_to_hub"]:
        model.push_to_hub(config["hub_model_id"])
        tokenizer.push_to_hub(config["hub_model_id"])


@click.command()
@click.option(
    "--loss-name",
    type=click.Choice(["cross_entropy", "weighted_cross_entropy", "focal"]),
    default=None,
    help="Override configured loss.",
)
@click.option("--data-path", default=None, help="Override the saved DatasetDict path.")
@click.option("--model-name", default=None, help="Override base model.")
@click.option("--output-dir", default=None, help="Override Trainer output directory.")
@click.option("--save-model-path", default=None, help="Override final saved model path.")
@click.option("--resume-from-checkpoint", default=None, help="Checkpoint directory to resume from.")
@click.option("--gradient-accumulation-steps", default=None, type=int)
@click.option("--fp16/--no-fp16", default=None, help="Use mixed precision on a compatible GPU.")
@click.option("--learning-rate", default=None, type=float)
@click.option("--num-epochs", default=None, type=float)
@click.option("--train-batch-size", default=None, type=int)
@click.option("--max-length", default=None, type=int)
@click.option("--max-train-examples", default=None, type=int, help="Quick train subset size.")
@click.option("--max-eval-examples", default=None, type=int, help="Quick eval subset size.")
@click.option("--max-test-examples", default=None, type=int, help="Quick test evaluation subset size.")
@click.option("--device", default=None, type=click.Choice(["auto", "cpu", "cuda"]), help="Training device preference.")
@click.option("--wandb-project", default=None, help="Weights & Biases project name.")
@click.option("--wandb-run-name", default=None, help="Weights & Biases run name.")
@click.option("--disable-wandb", is_flag=True, help="Disable Weights & Biases logging.")
def cli(
    loss_name,
    data_path,
    model_name,
    output_dir,
    save_model_path,
    resume_from_checkpoint,
    gradient_accumulation_steps,
    fp16,
    learning_rate,
    num_epochs,
    train_batch_size,
    max_length,
    max_train_examples,
    max_eval_examples,
    max_test_examples,
    device,
    wandb_project,
    wandb_run_name,
    disable_wandb,
):
    """Apply optional Click overrides and start the configured training run."""
    config = get_fine_tuning_config()

    overrides = {
        "loss_name": loss_name,
        "data_path": data_path,
        "model_name": model_name,
        "output_dir": output_dir,
        "save_model_path": save_model_path,
        "resume_from_checkpoint": resume_from_checkpoint,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "fp16": fp16,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "train_batch_size": train_batch_size,
        "max_length": max_length,
        "max_train_examples": max_train_examples,
        "max_eval_examples": max_eval_examples,
        "max_test_examples": max_test_examples,
        "device": device,
        "wandb_project": wandb_project,
        "wandb_run_name": wandb_run_name,
    }
    config.update({key: value for key, value in overrides.items() if value is not None})
    if config["device"] == "auto":
        config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    config = resolve_runtime_paths(config)
    if wandb_project:
        import os
        os.environ["WANDB_PROJECT"] = wandb_project
    if disable_wandb:
        config["report_to"] = ["none"]

    train(config)


if __name__ == "__main__":
    cli()
