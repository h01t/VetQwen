"""
run_judge.py — Standalone LLM-as-judge evaluation via Ollama

Reads predictions from a previous evaluation run and sends them to
Ollama for clinical quality scoring. Run this AFTER evaluate.py finishes
and the GPU is free for Ollama.

Usage:
    python scripts/run_judge.py --predictions results/baseline_3b_predictions.jsonl --run-name baseline_3b
    python scripts/run_judge.py --predictions results/vetqwen_3b_r16_predictions.jsonl --run-name vetqwen_3b_r16 --sample 50
"""

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Any

import requests

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = "http://localhost:11434"
JUDGE_MODEL = "qwen2.5:7b"

JUDGE_PROMPT_TEMPLATE = """You are an expert veterinary clinician reviewing an AI-generated diagnostic response.

Rate the following veterinary diagnostic response on a 1-5 scale for each criterion.

**Original case prompt:**
{prompt}

**AI response:**
{response}

**Rating criteria:**
- clinical_accuracy: Are the differential diagnoses plausible and appropriate for the species and symptoms?
- completeness: Does the response include reasoning, ranked differentials, triage, and next steps?
- tone: Is the language appropriate for a veterinary diagnostic assistant (clinical, professional)?
- hallucination: Does the response contain invented drug names, impossible symptoms, or fabricated facts? (1=no hallucination, 5=severe hallucination)

Return ONLY valid JSON with no additional text:
{{"clinical_accuracy": <1-5>, "completeness": <1-5>, "tone": <1-5>, "hallucination": <1-5>}}
"""


# ---------------------------------------------------------------------------
# Judge logic
# ---------------------------------------------------------------------------


def judge_response(prompt: str, response: str) -> dict[str, Any] | None:
    """Send a response to Ollama for LLM-as-judge scoring."""
    judge_prompt = JUDGE_PROMPT_TEMPLATE.format(prompt=prompt, response=response)

    payload = {
        "model": JUDGE_MODEL,
        "prompt": judge_prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 100},
    }
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()

        # Extract JSON
        json_str = raw
        if "```" in raw:
            for part in raw.split("```"):
                part = part.strip().lstrip("json").strip()
                if part.startswith("{"):
                    json_str = part
                    break

        scores = json.loads(json_str)
        return scores
    except Exception as e:
        log.warning(f"Judge request failed: {e}")
        return None


def load_predictions(path: str) -> list[dict]:
    """Load predictions JSONL file."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def run_judge(args: argparse.Namespace) -> None:
    preds_path = Path(args.predictions)
    if not preds_path.exists():
        log.error(f"Predictions file not found: {preds_path}")
        return

    records = load_predictions(str(preds_path))
    log.info(f"Loaded {len(records)} predictions from {preds_path}")

    # Verify Ollama is reachable
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        log.info(f"Ollama is running. Using judge model: {JUDGE_MODEL}")
    except Exception as e:
        log.error(f"Cannot reach Ollama at {OLLAMA_BASE_URL}: {e}")
        log.error("Make sure Ollama is running: ollama serve")
        return

    # Sample if needed
    indices = list(range(len(records)))
    if args.sample and args.sample < len(indices):
        indices = random.sample(indices, args.sample)
        log.info(f"Sampling {args.sample} predictions for judging")

    # Run judge
    results: list[dict] = []
    for i, idx in enumerate(indices):
        record = records[idx]
        scores = judge_response(record["user"], record["prediction"])
        if scores:
            results.append(scores)
        else:
            log.warning(f"  Sample {i + 1}: judge returned no scores")

        if (i + 1) % 10 == 0:
            log.info(f"  Progress: {i + 1}/{len(indices)}")
        time.sleep(0.1)

    if not results:
        log.error("No judge scores collected. Check Ollama and model availability.")
        return

    # Compute averages
    keys = ["clinical_accuracy", "completeness", "tone", "hallucination"]
    averages = {
        k: sum(r.get(k, 0) for r in results) / len(results)
        for k in keys
        if any(k in r for r in results)
    }

    # Print results
    log.info("\n" + "=" * 60)
    log.info("LLM-AS-JUDGE RESULTS")
    log.info("=" * 60)
    log.info(f"Predictions file: {preds_path}")
    log.info(f"Samples judged:   {len(results)}/{len(indices)}")
    for k, v in averages.items():
        log.info(f"  {k:25s}: {v:.2f}/5")
    log.info("=" * 60)

    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    judge_output = {
        "run_name": args.run_name,
        "predictions_file": str(preds_path),
        "n_judged": len(results),
        "n_total": len(records),
        "judge_model": JUDGE_MODEL,
        "scores": averages,
        "raw_scores": results,
    }

    output_path = results_dir / f"{args.run_name}_judge.json"
    with output_path.open("w") as f:
        json.dump(judge_output, f, indent=2)
    log.info(f"Judge results saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    global JUDGE_MODEL, OLLAMA_BASE_URL

    parser = argparse.ArgumentParser(
        description="Run LLM-as-judge on existing evaluation predictions"
    )
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to predictions JSONL file from evaluate.py (e.g. results/baseline_3b_predictions.jsonl)",
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
        default=JUDGE_MODEL,
        help=f"Ollama model to use as judge (default: {JUDGE_MODEL})",
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default=OLLAMA_BASE_URL,
        help=f"Ollama API URL (default: {OLLAMA_BASE_URL})",
    )
    args = parser.parse_args()

    JUDGE_MODEL = args.model
    OLLAMA_BASE_URL = args.ollama_url

    run_judge(args)


if __name__ == "__main__":
    main()
