"""Run GLiNER on one sentence."""

from __future__ import annotations

import json

import click


@click.command()
@click.argument("sentence")
@click.option("--model", "model_name", default="urchade/gliner_small-v2.1", show_default=True)
@click.option("--label", default="mountain", show_default=True)
@click.option("--threshold", default=0.5, type=click.FloatRange(0.0, 1.0), show_default=True)
@click.option("--device", type=click.Choice(["auto", "cpu", "cuda"]), default="auto", show_default=True)
def cli(sentence: str, model_name: str, label: str, threshold: float, device: str) -> None:
    """Extract entities from SENTENCE using GLiNER."""
    import torch
    from gliner import GLiNER

    selected_device = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
    if selected_device == "cuda" and not torch.cuda.is_available():
        raise click.ClickException("CUDA is not available.")
    model = GLiNER.from_pretrained(model_name).to(selected_device)
    entities = model.predict_entities(sentence, [label], threshold=threshold)
    click.echo(json.dumps({"model": model_name, "threshold": threshold, "entities": entities}, ensure_ascii=False))


if __name__ == "__main__":
    cli()
