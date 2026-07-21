"""Run a fine-tuned Hugging Face token-classification checkpoint on one sentence."""

from __future__ import annotations

import json
from pathlib import Path

import click


@click.command()
@click.argument("sentence")
@click.option("--checkpoint", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--label", default=None)
@click.option("--threshold", default=0.5, type=click.FloatRange(0.0, 1.0), show_default=True)
@click.option("--device", type=click.Choice(["auto", "cpu", "cuda"]), default="auto", show_default=True)
def cli(sentence: str, checkpoint: str, label: str | None, threshold: float, device: str) -> None:
    """Extract mountain entities from SENTENCE using CHECKPOINT."""
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    if device == "cuda" and not torch.cuda.is_available():
        raise click.ClickException("CUDA is not available.")
    selected_device = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(
        checkpoint,
        low_cpu_mem_usage=True,
    ).to(selected_device).eval()
    inputs = tokenizer(sentence, return_offsets_mapping=True, return_tensors="pt", truncation=True, max_length=256)
    offsets = inputs.pop("offset_mapping")[0].tolist()
    inputs = {key: value.to(selected_device) for key, value in inputs.items()}
    with torch.inference_mode():
        probabilities = torch.softmax(model(**inputs).logits[0], dim=-1)

    entities = []
    current = None
    for index, (start, end) in enumerate(offsets):
        if start == end:
            continue
        label_id = int(probabilities[index].argmax())
        token_label = model.config.id2label[label_id]
        score = float(probabilities[index, label_id])
        entity_label = token_label.split("-", 1)[-1]
        keep = token_label.upper() != "O" and score >= threshold and (label is None or entity_label.casefold() == label.casefold() or token_label.casefold() == label.casefold())
        if not keep:
            if current:
                entities.append(current)
                current = None
            continue
        starts_new = token_label.upper().startswith("B-") or current is None or current["label"].casefold() != entity_label.casefold()
        if starts_new:
            if current:
                entities.append(current)
            current = {"text": sentence[start:end], "label": entity_label, "score": score, "start": start, "end": end, "_scores": [score]}
        else:
            current["text"] = sentence[current["start"]:end]
            current["end"] = end
            current["_scores"].append(score)
    if current:
        entities.append(current)
    for entity in entities:
        scores = entity.pop("_scores")
        entity["score"] = sum(scores) / len(scores)
    click.echo(json.dumps({"checkpoint": str(Path(checkpoint)), "threshold": threshold, "entities": entities}, ensure_ascii=False))


if __name__ == "__main__":
    cli()
