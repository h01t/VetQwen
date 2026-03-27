"""
run_judge.py — Standalone LLM-as-judge evaluation via Ollama

Reads predictions from a previous evaluation run and sends them to
Ollama for clinical quality scoring. Run this AFTER evaluate.py finishes
and the GPU is free for Ollama.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

from judge_utils import (
    DEFAULT_JUDGE_MODEL,
    DEFAULT_OLLAMA_BASE_URL,
    check_ollama,
    judge_response,
    select_sample_indices,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


def load_predictions(path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def run_judge(args: argparse.Namespace) -> None:
    predictions_path = Path(args.predictions)
    if not predictions_path.exists():
        log.error("Predictions file not found: %s", predictions_path)
        return

    records = load_predictions(str(predictions_path))
    log.info("Loaded %s predictions from %s", len(records), predictions_path)

    ok, detail = check_ollama(args.ollama_url)
    if not ok:
        log.error("Cannot reach Ollama at %s: %s", args.ollama_url, detail)
        log.error("Make sure Ollama is running: ollama serve")
        return

    log.info("Ollama is running. Available models: %s", detail or "unknown")

    indices = select_sample_indices(len(records), args.sample, seed=args.seed)
    if args.sample and args.sample < len(records):
        log.info("Sampling %s predictions for judging", args.sample)

    results: list[dict[str, Any]] = []
    for index, record_index in enumerate(indices, start=1):
        record = records[record_index]
        scores = judge_response(
            record["user"],
            record["prediction"],
            base_url=args.ollama_url,
            model=args.model,
            timeout=120,
        )
        if scores:
            results.append(scores)
        else:
            log.warning("  Sample %s: judge returned no scores", index)

        if index % 10 == 0:
            log.info("  Progress: %s/%s", index, len(indices))
        time.sleep(0.1)

    if not results:
        log.error("No judge scores collected. Check Ollama and model availability.")
        return

    keys = ["clinical_accuracy", "completeness", "tone", "hallucination"]
    averages = {
        key: sum(result.get(key, 0) for result in results) / len(results)
        for key in keys
        if any(key in result for result in results)
    }

    log.info("\n%s", "=" * 60)
    log.info("LLM-AS-JUDGE RESULTS")
    log.info("%s", "=" * 60)
    log.info("Predictions file: %s", predictions_path)
    log.info("Samples judged:   %s/%s", len(results), len(indices))
    for key, value in averages.items():
        log.info("  %-25s: %.2f/5", key, value)
    log.info("%s", "=" * 60)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    judge_output = {
        "run_name": args.run_name,
        "predictions_file": str(predictions_path),
        "n_judged": len(results),
        "n_total": len(records),
        "judge_model": args.model,
        "seed": args.seed,
        "sample_indices": indices,
        "scores": averages,
        "raw_scores": results,
    }

    output_path = results_dir / f"{args.run_name}_judge.json"
    with output_path.open("w") as handle:
        json.dump(judge_output, handle, indent=2)
    log.info("Judge results saved to %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LLM-as-judge on existing evaluation predictions"
    )
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to predictions JSONL file from evaluate.py",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        required=True,
        help="Name for this judge run (used for output filename)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Number of predictions to judge (default: all)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_JUDGE_MODEL,
        help=f"Ollama model to use as judge (default: {DEFAULT_JUDGE_MODEL})",
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default=DEFAULT_OLLAMA_BASE_URL,
        help=f"Ollama API URL (default: {DEFAULT_OLLAMA_BASE_URL})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for stable judge sample selection",
    )
    args = parser.parse_args()
    run_judge(args)


if __name__ == "__main__":
    main()
