import logging
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk

from ner.utils.ner_bio import seqeval_metrics


def setup_file_logging(log_file: str) -> None:
    """Configure append-only file logging for a fine-tuning run."""
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_file,
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )


def load_and_tokenize_dataset(
    data_path: str,
    tokenizer,
    train_split: str,
    eval_split: str,
    max_length: int = 256,
    max_train_examples: int | None = None,
    max_eval_examples: int | None = None,
):
    """Load a saved dataset, optionally limit splits, and align BIO labels."""
    dataset = load_from_disk(data_path)

    if max_train_examples is not None:
        dataset[train_split] = dataset[train_split].select(
            range(min(max_train_examples, len(dataset[train_split])))
        )

    if max_eval_examples is not None:
        dataset[eval_split] = dataset[eval_split].select(
            range(min(max_eval_examples, len(dataset[eval_split])))
        )

    tokenized_dataset = dataset.map(
        tokenize_and_align_labels,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "max_length": max_length},
        keep_in_memory=True,
        load_from_cache_file=False,
    )

    return dataset, tokenized_dataset


def tokenize_and_align_labels(batch, tokenizer, max_length: int) -> dict:
    """Tokenize split words and keep a BIO label only on each first subtoken."""
    tokenized = tokenizer(
        batch["tokens"],
        truncation=True,
        max_length=max_length,
        is_split_into_words=True,
    )
    aligned_labels = []

    for row_index, labels in enumerate(batch["labels"]):
        word_ids = tokenized.word_ids(batch_index=row_index)
        previous_word_id = None
        row_labels = []

        for word_id in word_ids:
            if word_id is None:
                row_labels.append(-100)
            elif word_id != previous_word_id:
                row_labels.append(labels[word_id])
            else:
                row_labels.append(-100)
            previous_word_id = word_id

        aligned_labels.append(row_labels)

    tokenized["labels"] = aligned_labels
    return tokenized


def compute_token_metrics(eval_pred) -> dict[str, float]:
    """Calculate token accuracy and exact-span BIO precision, recall, and F1."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    true_labels = []
    pred_labels = []

    for prediction_row, label_row in zip(predictions, labels):
        current_true = []
        current_pred = []

        for prediction, label in zip(prediction_row, label_row):
            if label == -100:
                continue

            current_true.append(int(label))
            current_pred.append(int(prediction))

        true_labels.append(current_true)
        pred_labels.append(current_pred)

    results = seqeval_metrics(true_labels, pred_labels)

    return {
        "accuracy": results["overall_accuracy"],
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
    }


def compute_class_weights(dataset_split, num_labels: int, normalize: bool = True) -> torch.Tensor:
    """Calculate inverse-frequency class weights from a labeled dataset split."""
    counts = torch.zeros(num_labels, dtype=torch.float32)

    for example in dataset_split:
        for label in example["labels"]:
            if label >= 0:
                counts[label] += 1

    if torch.any(counts == 0):
        missing = torch.where(counts == 0)[0].tolist()
        raise ValueError(f"Cannot compute class weights, missing labels: {missing}")

    weights = counts.sum() / (num_labels * counts)

    if normalize:
        weights = weights / weights.mean()

    logging.info("Label counts: %s", counts.tolist())
    logging.info("Class weights: %s", weights.tolist())

    return weights
