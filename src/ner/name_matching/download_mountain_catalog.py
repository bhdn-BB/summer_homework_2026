"""Download and convert the Open Peaks global mountain catalog to CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from urllib.request import urlopen


URL = "https://raw.githubusercontent.com/open-peaks/data/master/_index.geojson"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with urlopen(URL, timeout=60) as response:
        payload = json.load(response)
    rows = []
    seen = set()
    for feature in payload.get("features", []):
        properties = feature.get("properties", {})
        name = str(properties.get("name", "")).strip()
        key = name.casefold()
        if not name or key in seen:
            continue
        seen.add(key)
        rows.append({
            "name": name,
            "meters": properties.get("meters", ""),
            "feet": properties.get("feet", ""),
            "continent": properties.get("continent", ""),
            "countries": "; ".join(properties.get("countries", []) or []),
            "regions": "; ".join(properties.get("regions", []) or []),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved={args.output} names={len(rows)} source={URL}")


if __name__ == "__main__":
    main()
