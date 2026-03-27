"""
evaluate.py — VetQwen evaluation script

Evaluates either the base model or the fine-tuned LoRA adapter on
the held-out dataset split using:
  - Primary task metrics:
      - diagnosis hit rate (when canonical condition labels exist)
      - triage section presence
      - parse success rate for the structured template
  - Secondary metrics:
      - ROUGE-L
      - BERTScore F1
      - format compliance rate
      - normalized species-level breakdown
  - Optional LLM-as-judge scores via Ollama

Results are saved to results/<run_name>.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Any

from judge_utils import (
    DEFAULT_JUDGE_MODEL,
    DEFAULT_OLLAMA_BASE_URL,
    judge_response,
    select_sample_indices,
)
from vetqwen_core.constants import EXPECTED_SECTIONS, TRIAGE_LABELS
from vetqwen_core.inference import generate_chat_response, load_inference_model
from vetqwen_core.response_parsing import (
    build_evaluation_rows,
    diagnosis_hit,
    parse_structured_response,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)
def set_random_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        return


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_rouge_l(predictions: list[str], references: list[str]) -> float:
    try:
        from rouge_score import rouge_scorer  # type: ignore
    except ImportError:
        log.warning("rouge-score not installed — returning ROUGE-L of 0.0.")
        return 0.0

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [
        scorer.score(ref, pred)["rougeL"].fmeasure
        for pred, ref in zip(predictions, references)
    ]
    return sum(scores) / len(scores) if scores else 0.0


def compute_bert_score(predictions: list[str], references: list[str]) -> float:
    try:
        from bert_score import score as bert_score_fn  # type: ignore
    except ImportError:
        log.warning("bert-score not installed — returning BERTScore F1 of 0.0.")
        return 0.0

    log.info("Computing BERTScore (this may take a moment) ...")
    _, _, f1 = bert_score_fn(predictions, references, lang="en", verbose=False)
    return f1.mean().item()


def compute_format_compliance(predictions: list[str]) -> float:
    compliant = 0
    for prediction in predictions:
        if all(section in prediction for section in EXPECTED_SECTIONS):
            compliant += 1
    return compliant / len(predictions) if predictions else 0.0


def compute_primary_metrics(
    rows: list[dict[str, Any]]
) -> dict[str, Any]:
    parsed_predictions = [row["parsed_prediction"] for row in rows]

    triage_section_presence = (
        sum(parsed["triage_present"] for parsed in parsed_predictions) / len(parsed_predictions)
        if parsed_predictions
        else 0.0
    )
    parse_success_rate = (
        sum(parsed["parse_success"] for parsed in parsed_predictions) / len(parsed_predictions)
        if parsed_predictions
        else 0.0
    )

    diagnosis_rows = [row for row in rows if row["meta"].get("condition")]
    triage_rows = [row for row in rows if row["reference_triage"]]
    urgent_rows = [row for row in triage_rows if row["reference_triage"] == "Urgent"]
    predicted_urgent_rows = [
        row for row in rows if row["predicted_triage"] == "Urgent"
    ]

    diagnosis_hit_rate = (
        sum(bool(row["diagnosis_hit"]) for row in diagnosis_rows) / len(diagnosis_rows)
        if diagnosis_rows
        else 0.0
    )
    triage_accuracy = (
        sum(bool(row["triage_match"]) for row in triage_rows) / len(triage_rows)
        if triage_rows
        else 0.0
    )
    urgent_recall = (
        sum(row["predicted_triage"] == "Urgent" for row in urgent_rows) / len(urgent_rows)
        if urgent_rows
        else 0.0
    )
    urgent_precision = (
        sum(row["reference_triage"] == "Urgent" for row in predicted_urgent_rows)
        / len(predicted_urgent_rows)
        if predicted_urgent_rows
        else 0.0
    )

    return {
        "diagnosis_hit_rate": diagnosis_hit_rate,
        "triage_section_presence": triage_section_presence,
        "parse_success_rate": parse_success_rate,
        "triage_accuracy": triage_accuracy,
        "urgent_recall": urgent_recall,
        "urgent_precision": urgent_precision,
        "n_condition_labeled": len(diagnosis_rows),
        "n_triage_labeled": len(triage_rows),
        "n_urgent_reference": len(urgent_rows),
        "n_urgent_predicted": len(predicted_urgent_rows),
    }


def compute_triage_confusion(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    confusion = {
        reference: {predicted: 0 for predicted in [*TRIAGE_LABELS, "unknown"]}
        for reference in TRIAGE_LABELS
    }

    for row in rows:
        reference = row["reference_triage"]
        if not reference:
            continue
        predicted = row["predicted_triage"] or "unknown"
        confusion.setdefault(reference, {label: 0 for label in [*TRIAGE_LABELS, "unknown"]})
        confusion[reference].setdefault(predicted, 0)
        confusion[reference][predicted] += 1

    return confusion


def compute_group_breakdown(
    rows: list[dict[str, Any]],
    group_key: str,
) -> dict[str, dict[str, Any]]:
    scorer = None
    try:
        from rouge_score import rouge_scorer  # type: ignore

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    except ImportError:
        log.warning("rouge-score not installed — breakdown ROUGE-L will be 0.0.")
    buckets: dict[str, dict[str, Any]] = {}

    for row in rows:
        group_value = row["meta"].get(group_key) or "unknown"
        score = (
            scorer.score(row["reference"], row["prediction"])["rougeL"].fmeasure
            if scorer
            else 0.0
        )
        bucket = buckets.setdefault(
            group_value,
            {
                "rouge_scores": [],
                "condition_hits": 0,
                "condition_total": 0,
                "triage_hits": 0,
                "triage_total": 0,
                "urgent_hits": 0,
                "urgent_total": 0,
            },
        )
        bucket["rouge_scores"].append(score)

        condition = row["meta"].get("condition")
        if condition:
            bucket["condition_total"] += 1
            if row["diagnosis_hit"]:
                bucket["condition_hits"] += 1
        if row["reference_triage"]:
            bucket["triage_total"] += 1
            if row["triage_match"]:
                bucket["triage_hits"] += 1
        if row["reference_triage"] == "Urgent":
            bucket["urgent_total"] += 1
            if row["predicted_triage"] == "Urgent":
                bucket["urgent_hits"] += 1

    breakdown: dict[str, dict[str, Any]] = {}
    for group_value, stats in buckets.items():
        breakdown[group_value] = {
            "rouge_l": sum(stats["rouge_scores"]) / len(stats["rouge_scores"]),
            "n": len(stats["rouge_scores"]),
            "diagnosis_hit_rate": (
                stats["condition_hits"] / stats["condition_total"]
                if stats["condition_total"]
                else None
            ),
            "triage_accuracy": (
                stats["triage_hits"] / stats["triage_total"]
                if stats["triage_total"]
                else None
            ),
            "urgent_recall": (
                stats["urgent_hits"] / stats["urgent_total"]
                if stats["urgent_total"]
                else None
            ),
        }
    return breakdown


def compute_species_breakdown(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return compute_group_breakdown(rows, "species")


def compute_source_breakdown(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return compute_group_breakdown(rows, "source")


# ---------------------------------------------------------------------------
# Judge evaluation
# ---------------------------------------------------------------------------


def run_llm_judge(
    user_contents: list[str],
    predictions: list[str],
    base_url: str,
    model: str,
    sample_size: int = 100,
    seed: int = 42,
) -> tuple[dict[str, float], list[int]]:
    indices = select_sample_indices(len(predictions), sample_size, seed=seed)

    results: list[dict[str, Any]] = []
    for index, prediction_index in enumerate(indices, start=1):
        scores = judge_response(
            user_contents[prediction_index],
            predictions[prediction_index],
            base_url=base_url,
            model=model,
        )
        if scores:
            results.append(scores)
        if index % 10 == 0:
            log.info("  Judge progress: %s/%s", index, len(indices))
        time.sleep(0.1)

    if not results:
        return {}, indices

    keys = ["clinical_accuracy", "completeness", "tone", "hallucination"]
    return (
        {
            key: sum(result.get(key, 0) for result in results) / len(results)
            for key in keys
            if any(key in result for result in results)
        },
        indices,
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_split_records(path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path) as handle:
        for index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            messages = record["messages"]
            user_msg = next((msg["content"] for msg in messages if msg["role"] == "user"), "")
            assistant_msg = next(
                (msg["content"] for msg in messages if msg["role"] == "assistant"), ""
            )
            records.append(
                {
                    "id": index,
                    "user": user_msg,
                    "reference": assistant_msg,
                    "meta": record.get("_meta", {}),
                }
            )
    return records


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def evaluate(args: argparse.Namespace) -> None:
    split_file = f"data/processed/{args.split}.jsonl"
    log.info("Loading evaluation data from %s ...", split_file)
    records = load_split_records(split_file)

    if args.limit:
        records = records[: args.limit]
    log.info("Evaluating on %s samples", len(records))
    set_random_seed(args.seed)

    model, tokenizer, _ = load_inference_model(
        args.model,
        args.base_model,
        device="cuda",
        enforce_supported_python=True,
        remote_hint=(
            "For the intended workflow on a Mac, run evaluation on the CUDA workstation via:\n"
            "  VETQWEN_REMOTE_HOST=blackbox ./scripts/evaluate_remote.sh ..."
        ),
    )

    log.info("Generating responses ...")
    predictions: list[str] = []
    user_contents = [record["user"] for record in records]
    references = [record["reference"] for record in records]

    for index, user_content in enumerate(user_contents, start=1):
        prediction = generate_chat_response(
            model,
            tokenizer,
            user_content,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
        )
        predictions.append(prediction)
        if index % 10 == 0:
            log.info("  %s/%s", index, len(user_contents))

    evaluation_rows = build_evaluation_rows(predictions, references, records)
    primary_metrics = compute_primary_metrics(evaluation_rows)

    log.info("Computing ROUGE-L ...")
    rouge_l = compute_rouge_l(predictions, references)

    log.info("Computing BERTScore ...")
    bert_f1 = compute_bert_score(predictions, references)

    log.info("Computing format compliance ...")
    format_compliance = compute_format_compliance(predictions)

    log.info("Computing per-species breakdown ...")
    species_breakdown = compute_species_breakdown(evaluation_rows)

    log.info("Computing per-source breakdown ...")
    source_breakdown = compute_source_breakdown(evaluation_rows)

    triage_confusion = compute_triage_confusion(evaluation_rows)

    judge_scores: dict[str, float] = {}
    judge_indices: list[int] = []
    if not args.no_judge:
        log.info("Running LLM-as-judge evaluation (via Ollama) ...")
        judge_scores, judge_indices = run_llm_judge(
            user_contents,
            predictions,
            base_url=args.ollama_url,
            model=args.judge_model,
            sample_size=args.judge_sample,
            seed=args.seed,
        )

    log.info("\n%s", "=" * 60)
    log.info("EVALUATION RESULTS")
    log.info("%s", "=" * 60)
    log.info(
        "Diagnosis hit rate: %s (n=%s)",
        f"{primary_metrics['diagnosis_hit_rate']:.2%}",
        primary_metrics["n_condition_labeled"],
    )
    log.info("Triage section presence: %s", f"{primary_metrics['triage_section_presence']:.2%}")
    log.info("Parse success rate:      %s", f"{primary_metrics['parse_success_rate']:.2%}")
    log.info("Triage accuracy:         %s", f"{primary_metrics['triage_accuracy']:.2%}")
    log.info(
        "Urgent recall:           %s (n=%s)",
        f"{primary_metrics['urgent_recall']:.2%}",
        primary_metrics["n_urgent_reference"],
    )
    log.info("ROUGE-L:                 %.4f", rouge_l)
    log.info("BERTScore F1:            %.4f", bert_f1)
    log.info("Format compliance:       %s", f"{format_compliance:.2%}")
    for species, stats in species_breakdown.items():
        diagnosis_score = stats["diagnosis_hit_rate"]
        diag_str = "n/a" if diagnosis_score is None else f"{diagnosis_score:.2%}"
        triage_score = stats["triage_accuracy"]
        triage_str = "n/a" if triage_score is None else f"{triage_score:.2%}"
        log.info(
            "  [%s] ROUGE-L: %.4f (n=%s, diagnosis_hit=%s, triage=%s)",
            species,
            stats["rouge_l"],
            stats["n"],
            diag_str,
            triage_str,
        )
    for reference_label, predicted_counts in triage_confusion.items():
        log.info("  Triage confusion [%s]: %s", reference_label, predicted_counts)
    if judge_scores:
        log.info(
            "Judge — clinical accuracy: %.2f/5",
            judge_scores.get("clinical_accuracy", 0.0),
        )
        log.info(
            "Judge — completeness:      %.2f/5",
            judge_scores.get("completeness", 0.0),
        )
        log.info("Judge — tone:              %.2f/5", judge_scores.get("tone", 0.0))
        log.info(
            "Judge — hallucination:     %.2f/5",
            judge_scores.get("hallucination", 0.0),
        )
    log.info("%s", "=" * 60)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    results = {
        "run_name": args.run_name,
        "model": args.model,
        "split": args.split,
        "n_samples": len(records),
        "metrics": {
            **primary_metrics,
            "rouge_l": rouge_l,
            "bert_score_f1": bert_f1,
            "format_compliance": format_compliance,
        },
        "evaluation_config": {
            "seed": args.seed,
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
        },
        "triage_confusion": triage_confusion,
        "species_breakdown": species_breakdown,
        "source_breakdown": source_breakdown,
    }
    if judge_scores:
        results["judge_scores"] = judge_scores
        results["judge_sample_indices"] = judge_indices

    results_path = results_dir / f"{args.run_name}.json"
    with results_path.open("w") as handle:
        json.dump(results, handle, indent=2)
    log.info("Results saved to %s", results_path)

    predictions_path = results_dir / f"{args.run_name}_predictions.jsonl"
    with predictions_path.open("w") as handle:
        for record, prediction, evaluation_row in zip(records, predictions, evaluation_rows):
            handle.write(
                json.dumps(
                    {
                        "id": record["id"],
                        "user": record["user"],
                        "reference": record["reference"],
                        "prediction": prediction,
                        "meta": record["meta"],
                        "top_diagnosis": evaluation_row["top_diagnosis"],
                        "predicted_triage": evaluation_row["predicted_triage"],
                        "reference_triage": evaluation_row["reference_triage"],
                        "diagnosis_hit": evaluation_row["diagnosis_hit"],
                        "triage_match": evaluation_row["triage_match"],
                    }
                )
                + "\n"
            )
    log.info("Predictions saved to %s", predictions_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="VetQwen evaluation")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Hugging Face model ID or local path to LoRA adapter",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model Hugging Face ID (required if --model is a local LoRA adapter)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate on",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="evaluation",
        help="Name for this evaluation run",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit evaluation to N samples (useful for quick checks)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic evaluation and judge sampling",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Enable stochastic decoding for exploratory runs (disabled by default)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature when --do-sample is enabled",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling value when --do-sample is enabled",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=350,
        help="Maximum number of tokens to generate per sample",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip LLM-as-judge evaluation (faster, no Ollama required)",
    )
    parser.add_argument(
        "--judge-sample",
        type=int,
        default=100,
        help="Number of samples to send to the LLM judge (default: 100)",
    )
    parser.add_argument(
        "--judge-model",
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
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
