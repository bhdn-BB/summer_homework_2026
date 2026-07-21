"""Canonical mountain-name matching pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from tqdm import tqdm


NAME_COLUMNS = {"mountain", "mountain_name", "name", "peak", "peak_name", "canonical_name", "entity"}
GENERIC_PREFIXES = {"mount", "mt", "mountain", "peak", "pico", "pic", "pointe", "point", "monte", "mont", "cerro", "гора", "горы"}
TEXT_SUFFIXES = {".txt", ".text", ".md", ".csv", ".tsv", ".json"}


@dataclass(frozen=True)
class MountainMatch:
    file: str
    mention: str
    canonical_name: str
    start: int
    end: int
    score: float
    match_type: str


class MountainNameMatcher:
    """Match text against the union of all supplied mountain catalogs."""

    def __init__(self, catalog_paths: list[str | Path], fuzzy_threshold: float = 0.0) -> None:
        if not catalog_paths:
            raise ValueError("At least one catalog is required.")
        if not 0.0 <= fuzzy_threshold <= 1.0:
            raise ValueError("fuzzy_threshold must be between 0 and 1.")
        self.fuzzy_threshold = fuzzy_threshold

        # Union combines all catalog names before normalized duplicate removal.
        all_names: set[str] = set()
        for catalog_path in catalog_paths:
            all_names.update(self._read_catalog(Path(catalog_path)))
        if not all_names:
            raise ValueError("The combined catalogs contain no mountain names.")

        normalized_unique = {self.normalize(value): value for value in all_names if self.normalize(value)}
        self.canonical_names = sorted(normalized_unique.values(), key=lambda value: (self.normalize(value), value.casefold()))
        self.alias_to_canonical = self._build_alias_index()
        self.aliases = sorted(self.alias_to_canonical, key=lambda value: (-len(value), value))

    def normalize(self, value: str) -> str:
        """Normalize case, accents, punctuation, apostrophes, hyphens, and whitespace."""
        decomposed = unicodedata.normalize("NFKD", value).casefold()
        clean = []
        for character in decomposed:
            if unicodedata.combining(character):
                continue
            clean.append(character if character.isalnum() or character == "_" else " ")
        return " ".join("".join(clean).split())

    def run(self, input_dir: str | Path, output_path: str | Path) -> list[MountainMatch]:
        """Run discovery, exact matching, optional fuzzy matching, and CSV export."""
        input_path = Path(input_dir)
        if not input_path.is_dir():
            raise ValueError(f"Input directory does not exist: {input_path}")

        # Hugging Face datasets saved with save_to_disk contain Arrow files.
        if list(input_path.glob("*.arrow")):
            return self._run_saved_dataset(input_path, Path(output_path))

        results: list[MountainMatch] = []
        for file_path in sorted(input_path.rglob("*")):
            if not file_path.is_file() or file_path.suffix.lower() not in TEXT_SUFFIXES:
                continue

            text = self._read_text(file_path)
            normalized = self.normalize(text)
            text_mapping = self._build_mapping(text, normalized)

            # Exact normalized matching is the primary and deterministic method.
            file_matches = self._find_exact(text, normalized, text_mapping, file_path.relative_to(input_path))

            # Fuzzy matching is optional and only adds candidates below the exact pass.
            if self.fuzzy_threshold > 0:
                file_matches.extend(self._find_fuzzy(text, normalized, text_mapping, file_path.relative_to(input_path)))

            results.extend(self._select_non_overlapping(file_matches))

        self._write_results(results, Path(output_path))
        return results

    def _run_saved_dataset(self, dataset_path: Path, output_path: Path) -> list[MountainMatch]:
        """Run matching over a Hugging Face save_to_disk dataset."""
        try:
            from datasets import load_from_disk
        except ImportError as exc:
            raise RuntimeError("Install datasets to process Arrow datasets: pip install datasets") from exc

        dataset = load_from_disk(str(dataset_path))
        if "sentence" not in dataset.column_names:
            raise ValueError("The saved dataset must contain a 'sentence' column.")

        results = []
        for index, sentence in tqdm(enumerate(dataset["sentence"]), total=len(dataset), desc="Matching test dataset"):
            results.extend(self.match_text(str(sentence), f"{dataset_path.name}[{index}]"))
        self._write_results(results, output_path)
        print(f"dataset_examples={len(dataset)} matches={len(results)}")
        return results

    def match_text(self, text: str, file_name: str = "sentence") -> list[MountainMatch]:
        """Match one sentence without creating an input file."""
        normalized = self.normalize(text)
        text_mapping = self._build_mapping(text, normalized)
        matches = self._find_exact(text, normalized, text_mapping, Path(file_name))
        if self.fuzzy_threshold > 0:
            matches.extend(self._find_fuzzy(text, normalized, text_mapping, Path(file_name)))
        return self._select_non_overlapping(matches)

    def _read_catalog(self, path: Path) -> set[str]:
        """Read names from a CSV, JSON, or GeoJSON catalog."""
        if not path.is_file():
            raise ValueError(f"Catalog does not exist: {path}")

        if path.suffix.lower() in {".json", ".geojson"}:
            payload = json.loads(path.read_text(encoding="utf-8"))
            records = payload.get("features", []) if payload.get("type") == "FeatureCollection" else payload
            return {
                str(record.get("properties", record).get("name", "")).strip()
                for record in records
                if str(record.get("properties", record).get("name", "")).strip()
            }

        with path.open("r", encoding="utf-8-sig", newline="") as file:
            rows = list(csv.reader(file))
        if not rows:
            return set()

        header = [self.normalize(item).replace(" ", "_") for item in rows[0]]
        has_header = bool(NAME_COLUMNS.intersection(header))
        name_column = next((header.index(item) for item in NAME_COLUMNS if item in header), 0)
        data_rows = rows[1:] if has_header else rows
        return {
            row[name_column].strip()
            for row in data_rows
            if len(row) > name_column and row[name_column].strip()
        }

    def _build_alias_index(self) -> dict[str, str]:
        """Create canonical aliases while excluding ambiguous one-word removals."""
        alias_to_canonical: dict[str, str] = {}
        for canonical_name in self.canonical_names:
            normalized = self.normalize(canonical_name)
            aliases = {normalized} if normalized else set()
            words = normalized.split()
            if len(words) > 2 and words[0] in GENERIC_PREFIXES:
                aliases.add(" ".join(words[1:]))
            for alias in aliases:
                alias_to_canonical.setdefault(alias, canonical_name)
        return alias_to_canonical

    def _read_text(self, path: Path) -> str:
        """Read plain text and serialize JSON into searchable text."""
        raw = path.read_text(encoding="utf-8-sig", errors="replace")
        if path.suffix.lower() == ".json":
            try:
                return json.dumps(json.loads(raw), ensure_ascii=False)
            except json.JSONDecodeError:
                return raw
        return raw

    def _build_mapping(self, original: str, normalized: str) -> list[int]:
        """Map normalized-text positions back to original-text positions."""
        source = []
        source_map = []
        for index, character in enumerate(original):
            decomposed = unicodedata.normalize("NFKD", character).casefold()
            for item in decomposed:
                if not unicodedata.combining(item):
                    source.append(item if item.isalnum() or item == "_" else " ")
                    source_map.append(index)

        raw = "".join(source)
        result = []
        cursor = 0
        for character in normalized:
            while cursor < len(raw) and raw[cursor] != character:
                cursor += 1
            result.append(source_map[cursor] if cursor < len(source_map) else max(len(original) - 1, 0))
            cursor += 1
        return result

    def _find_exact(self, text: str, normalized: str, mapping: list[int], relative_path: Path) -> list[MountainMatch]:
        """Find exact normalized aliases using bounded regex chunks."""
        matches = []
        for start in range(0, len(self.aliases), 400):
            chunk = self.aliases[start : start + 400]
            pattern = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(alias) for alias in chunk) + r")(?!\w)")
            for found in pattern.finditer(normalized):
                original_start = mapping[found.start()]
                original_end = mapping[found.end() - 1] + 1
                if any(character in text[original_start:original_end] for character in "()[]{}"):
                    continue
                alias = found.group(0)
                matches.append(MountainMatch(str(relative_path), text[original_start:original_end], self.alias_to_canonical[alias], original_start, original_end, 1.0, "exact_normalized"))
        return matches

    def _find_fuzzy(self, text: str, normalized: str, mapping: list[int], relative_path: Path) -> list[MountainMatch]:
        """Find optional typo-tolerant candidates with character similarity."""
        words = list(re.finditer(r"\S+", normalized))
        fuzzy_aliases = [(alias, canonical) for alias, canonical in self.alias_to_canonical.items() if len(alias) >= 4]
        matches = []
        for word_start in range(len(words)):
            for alias, canonical in fuzzy_aliases:
                word_end = word_start + len(alias.split())
                if word_end > len(words):
                    continue
                candidate = " ".join(item.group(0) for item in words[word_start:word_end])
                score = SequenceMatcher(None, candidate, alias).ratio()
                if score < self.fuzzy_threshold:
                    continue
                start_norm = words[word_start].start()
                end_norm = words[word_end - 1].end()
                original_start = mapping[start_norm]
                original_end = mapping[end_norm - 1] + 1
                matches.append(MountainMatch(str(relative_path), text[original_start:original_end], canonical, original_start, original_end, score, "fuzzy_normalized"))
        return matches

    def _select_non_overlapping(self, matches: list[MountainMatch]) -> list[MountainMatch]:
        """Keep the highest-confidence, longest match in an overlapping span."""
        selected = []
        for item in sorted(matches, key=lambda value: (-value.score, -(value.end - value.start), value.start)):
            if not any(item.start < other.end and other.start < item.end for other in selected):
                selected.append(item)
        return sorted(selected, key=lambda value: (value.start, value.end))

    def _write_results(self, results: list[MountainMatch], output_path: Path) -> None:
        """Write the canonical mapping table."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["file", "mention", "canonical_name", "start", "end", "score", "match_type"])
            writer.writeheader()
            writer.writerows(item.__dict__ for item in results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Match mountain names using the union of one or more catalogs.")
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--catalog", action="append", required=True, type=Path, help="Repeat for every CSV or GeoJSON catalog.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fuzzy-threshold", type=float, default=0.0)
    args = parser.parse_args()
    matcher = MountainNameMatcher(args.catalog, args.fuzzy_threshold)
    results = matcher.run(args.input_dir, args.output)
    print(json.dumps({"catalogs": [str(path) for path in args.catalog], "unique_names": len(matcher.canonical_names), "matches": len(results), "output": str(args.output.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
