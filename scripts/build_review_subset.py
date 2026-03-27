"""
build_review_subset.py — Create a stable manual-review subset for VetQwen.

This helper selects:
  - every urgent case
  - a seeded sample of labeled dog/cat cases

It accepts either processed dataset JSONL files or evaluation prediction JSONL files
and writes a compact JSONL file for clinical review.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from vetqwen_core.jsonl import load_jsonl
from vetqwen_core.text import canonicalize_triage, extract_triage_from_response


def to_review_record(row: dict[str, Any]) -> dict[str, Any]:
    if "messages" in row:
        messages = row.get("messages", [])
        user = next((m["content"] for m in messages if m["role"] == "user"), "")
        reference = next((m["content"] for m in messages if m["role"] == "assistant"), "")
        prediction = None
        meta = dict(row.get("_meta", {}))
    else:
        user = row.get("user", "")
        reference = row.get("reference", "")
        prediction = row.get("prediction")
        meta = dict(row.get("meta", {}))

    triage = canonicalize_triage(meta.get("triage")) or extract_triage_from_response(reference)
    return {
        "row_index": row.get("_row_index"),
        "user": user,
        "reference": reference,
        "prediction": prediction,
        "meta": meta,
        "species": meta.get("species", "unknown"),
        "condition": meta.get("condition"),
        "source": meta.get("source", "unknown"),
        "reference_triage": triage,
        "urgent": bool(triage == "Urgent"),
    }


def select_review_subset(
    records: list[dict[str, Any]],
    target_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    urgent_records = [record for record in records if record["urgent"]]
    labeled_companion = [
        record
        for record in records
        if record["condition"] and record["species"] in {"dog", "cat"}
    ]

    selected = {record["row_index"]: record for record in urgent_records}
    remaining_budget = max(target_size - len(selected), 0)

    if remaining_budget:
        rng = random.Random(seed)
        pool = [
            record for record in labeled_companion if record["row_index"] not in selected
        ]
        sample_size = min(remaining_budget, len(pool))
        for record in rng.sample(pool, sample_size):
            selected[record["row_index"]] = record

    return sorted(selected.values(), key=lambda record: record["row_index"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a stable VetQwen manual-review subset")
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/test.jsonl",
        help="Input JSONL from data/processed or results/*_predictions.jsonl",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/review/manual_review.jsonl",
        help="Output JSONL path for the manual-review subset",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=40,
        help="Target total review size after all urgent cases are included",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for stable labeled-case sampling",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    rows = load_jsonl(input_path)
    for index, row in enumerate(rows):
        row["_row_index"] = index
    review_records = [to_review_record(row) for row in rows]
    subset = select_review_subset(review_records, args.target_size, args.seed)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        for record in subset:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "input": str(input_path),
                "output": str(output_path),
                "selected": len(subset),
                "urgent_included": sum(record["urgent"] for record in subset),
                "seed": args.seed,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
