"""Map mountain mentions in one sentence to catalog names."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ner.name_matching.match_names import MountainNameMatcher


@click.command()
@click.argument("sentence")
@click.option("--catalog", "catalog_paths", multiple=True, required=True, type=click.Path(exists=True))
@click.option("--fuzzy-threshold", default=0.0, type=click.FloatRange(0.0, 1.0), show_default=True)
def cli(sentence: str, catalog_paths: tuple[str, ...], fuzzy_threshold: float) -> None:
    """Map mountain mentions in SENTENCE to canonical catalog names."""
    matcher = MountainNameMatcher(list(catalog_paths), fuzzy_threshold)
    matches = matcher.match_text(sentence)
    click.echo(json.dumps({"catalogs": list(catalog_paths), "matches": [item.__dict__ for item in matches]}, ensure_ascii=False))


if __name__ == "__main__":
    cli()
