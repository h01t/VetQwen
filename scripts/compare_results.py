"""
compare_results.py — compare VetQwen baseline and candidate evaluation runs.

Usage:
    python scripts/compare_results.py \
        --baseline results/baseline.json \
        --candidate results/candidate.json \
        --baseline-judge results/baseline_judge.json \
        --candidate-judge results/candidate_judge.json \
        --output results/comparisons/candidate_vs_baseline.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SOURCE_GUARDRAIL_SOURCES = (
    "infinite-dataset-hub/VetPetCare",
    "infinite-dataset-hub/VetHealthAssessment",
)
SOURCE_GUARDRAIL_METRICS = ("diagnosis_hit_rate", "triage_accuracy")
MAX_SOURCE_REGRESSION = 0.10


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def metric_value(result: dict[str, Any], metric_name: str) -> float | None:
    value = result.get("metrics", {}).get(metric_name)
    if value is None:
        return None
    return float(value)


def judge_score_value(result: dict[str, Any] | None, metric_name: str) -> float | None:
    if not result:
        return None
    value = result.get("scores", {}).get(metric_name)
    if value is None:
        return None
    return float(value)


def metric_delta(
    baseline_result: dict[str, Any],
    candidate_result: dict[str, Any],
    metric_name: str,
) -> float | None:
    baseline_value = metric_value(baseline_result, metric_name)
    candidate_value = metric_value(candidate_result, metric_name)
    if baseline_value is None or candidate_value is None:
        return None
    return candidate_value - baseline_value


def judge_delta(
    baseline_judge: dict[str, Any] | None,
    candidate_judge: dict[str, Any] | None,
    metric_name: str,
) -> float | None:
    baseline_value = judge_score_value(baseline_judge, metric_name)
    candidate_value = judge_score_value(candidate_judge, metric_name)
    if baseline_value is None or candidate_value is None:
        return None
    return candidate_value - baseline_value


def format_delta(value: float | None, precision: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.{precision}f}"


def format_value(value: float | None, precision: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{precision}f}"


def breakdown_metric_value(
    result: dict[str, Any],
    breakdown_name: str,
    bucket_name: str,
    metric_name: str,
) -> float | None:
    bucket = result.get(breakdown_name, {}).get(bucket_name)
    if not bucket:
        return None
    value = bucket.get(metric_name)
    if value is None:
        return None
    return float(value)


def collect_source_guardrails(
    baseline_result: dict[str, Any],
    candidate_result: dict[str, Any],
) -> list[dict[str, Any]]:
    """Collect guarded source regressions that exceed the allowed tolerance."""
    regressions: list[dict[str, Any]] = []
    for source_name in SOURCE_GUARDRAIL_SOURCES:
        for metric_name in SOURCE_GUARDRAIL_METRICS:
            baseline_value = breakdown_metric_value(
                baseline_result,
                "source_breakdown",
                source_name,
                metric_name,
            )
            candidate_value = breakdown_metric_value(
                candidate_result,
                "source_breakdown",
                source_name,
                metric_name,
            )
            if baseline_value is None or candidate_value is None:
                continue
            delta = candidate_value - baseline_value
            if delta < -MAX_SOURCE_REGRESSION:
                regressions.append(
                    {
                        "source": source_name,
                        "metric": metric_name,
                        "baseline": baseline_value,
                        "candidate": candidate_value,
                        "delta": delta,
                    }
                )
    return regressions


def build_gate(
    name: str,
    passed: bool,
    detail: str,
    skipped: bool = False,
) -> dict[str, Any]:
    return {"name": name, "passed": passed, "detail": detail, "skipped": skipped}


def build_comparison(
    baseline_result: dict[str, Any],
    candidate_result: dict[str, Any],
    baseline_judge: dict[str, Any] | None,
    candidate_judge: dict[str, Any] | None,
) -> dict[str, Any]:
    """Assemble metric deltas, gate checks, and source guardrail outcomes."""
    tracked_metrics = [
        "diagnosis_hit_rate",
        "parse_success_rate",
        "format_compliance",
        "triage_accuracy",
        "urgent_recall",
        "urgent_precision",
        "rouge_l",
        "bert_score_f1",
    ]
    judge_metrics = [
        "clinical_accuracy",
        "completeness",
        "tone",
        "hallucination",
    ]

    metric_deltas = {
        metric_name: {
            "baseline": metric_value(baseline_result, metric_name),
            "candidate": metric_value(candidate_result, metric_name),
            "delta": metric_delta(baseline_result, candidate_result, metric_name),
        }
        for metric_name in tracked_metrics
    }
    judge_deltas = {
        metric_name: {
            "baseline": judge_score_value(baseline_judge, metric_name),
            "candidate": judge_score_value(candidate_judge, metric_name),
            "delta": judge_delta(baseline_judge, candidate_judge, metric_name),
        }
        for metric_name in judge_metrics
    }

    source_regressions = collect_source_guardrails(baseline_result, candidate_result)

    candidate_parse = metric_value(candidate_result, "parse_success_rate") or 0.0
    candidate_format = metric_value(candidate_result, "format_compliance") or 0.0
    candidate_diag = metric_value(candidate_result, "diagnosis_hit_rate") or 0.0
    baseline_diag = metric_value(baseline_result, "diagnosis_hit_rate") or 0.0
    candidate_urgent = metric_value(candidate_result, "urgent_recall") or 0.0

    gates = [
        build_gate(
            "parse_success_rate >= 0.95",
            candidate_parse >= 0.95,
            f"candidate={candidate_parse:.4f}",
        ),
        build_gate(
            "format_compliance >= 0.95",
            candidate_format >= 0.95,
            f"candidate={candidate_format:.4f}",
        ),
        build_gate(
            "diagnosis_hit_rate beats baseline",
            candidate_diag > baseline_diag,
            f"baseline={baseline_diag:.4f}, candidate={candidate_diag:.4f}",
        ),
        build_gate(
            "urgent_recall > 0.0",
            candidate_urgent > 0.0,
            f"candidate={candidate_urgent:.4f}",
        ),
    ]

    baseline_judge_clinical = judge_score_value(baseline_judge, "clinical_accuracy")
    candidate_judge_clinical = judge_score_value(candidate_judge, "clinical_accuracy")
    if baseline_judge_clinical is not None and candidate_judge_clinical is not None:
        gates.append(
            build_gate(
                "judge clinical_accuracy does not regress",
                candidate_judge_clinical >= baseline_judge_clinical,
                (
                    f"baseline={baseline_judge_clinical:.4f}, "
                    f"candidate={candidate_judge_clinical:.4f}"
                ),
            )
        )
    else:
        gates.append(
            build_gate(
                "judge clinical_accuracy does not regress",
                True,
                "Skipped: both baseline and candidate judge files are required.",
                skipped=True,
            )
        )

    gates.append(
        build_gate(
            "source guardrails hold",
            not source_regressions,
            (
                "No VetPetCare/VetHealthAssessment diagnosis or triage metric "
                "regressed by more than 0.10."
                if not source_regressions
                else f"{len(source_regressions)} guarded source regression(s) detected."
            ),
        )
    )

    return {
        "baseline_run": baseline_result.get("run_name"),
        "candidate_run": candidate_result.get("run_name"),
        "baseline_path": baseline_result.get("_path"),
        "candidate_path": candidate_result.get("_path"),
        "metric_deltas": metric_deltas,
        "judge_deltas": judge_deltas,
        "source_regressions": source_regressions,
        "gates": gates,
        "overall_passed": all(gate["passed"] for gate in gates),
    }


def comparison_markdown(
    comparison: dict[str, Any],
    baseline_result: dict[str, Any],
    candidate_result: dict[str, Any],
) -> str:
    """Render a human-readable markdown summary from comparison data."""
    lines = [
        "# VetQwen Comparison Summary",
        "",
        f"- Baseline: `{comparison['baseline_run']}`",
        f"- Candidate: `{comparison['candidate_run']}`",
        f"- Overall gates passed: `{comparison['overall_passed']}`",
        "",
        "## Key Metrics",
        "",
        "| Metric | Baseline | Candidate | Delta |",
        "|---|---:|---:|---:|",
    ]

    for metric_name, payload in comparison["metric_deltas"].items():
        lines.append(
            "| {metric} | {baseline} | {candidate} | {delta} |".format(
                metric=metric_name,
                baseline=format_value(payload["baseline"]),
                candidate=format_value(payload["candidate"]),
                delta=format_delta(payload["delta"]),
            )
        )

    if any(
        payload["baseline"] is not None or payload["candidate"] is not None
        for payload in comparison["judge_deltas"].values()
    ):
        lines.extend(
            [
                "",
                "## Judge Metrics",
                "",
                "| Metric | Baseline | Candidate | Delta |",
                "|---|---:|---:|---:|",
            ]
        )
        for metric_name, payload in comparison["judge_deltas"].items():
            lines.append(
                "| {metric} | {baseline} | {candidate} | {delta} |".format(
                    metric=metric_name,
                    baseline=format_value(payload["baseline"]),
                    candidate=format_value(payload["candidate"]),
                    delta=format_delta(payload["delta"]),
                )
            )

    lines.extend(
        [
            "",
            "## Source Breakdown",
            "",
            "| Source | Metric | Baseline | Candidate | Delta |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for source_name in SOURCE_GUARDRAIL_SOURCES:
        for metric_name in SOURCE_GUARDRAIL_METRICS:
            baseline_value = breakdown_metric_value(
                baseline_result,
                "source_breakdown",
                source_name,
                metric_name,
            )
            candidate_value = breakdown_metric_value(
                candidate_result,
                "source_breakdown",
                source_name,
                metric_name,
            )
            delta = None
            if baseline_value is not None and candidate_value is not None:
                delta = candidate_value - baseline_value
            lines.append(
                "| {source} | {metric} | {baseline} | {candidate} | {delta} |".format(
                    source=source_name,
                    metric=metric_name,
                    baseline=format_value(baseline_value),
                    candidate=format_value(candidate_value),
                    delta=format_delta(delta),
                )
            )

    lines.extend(
        [
            "",
            "## Gates",
            "",
            "| Gate | Passed | Detail |",
            "|---|---|---|",
        ]
    )
    for gate in comparison["gates"]:
        lines.append(
            "| {name} | {status} | {detail} |".format(
                name=gate["name"],
                status="skipped" if gate.get("skipped") else ("yes" if gate["passed"] else "no"),
                detail=gate["detail"],
            )
        )

    if comparison["source_regressions"]:
        lines.extend(["", "## Guardrail Regressions", ""])
        for regression in comparison["source_regressions"]:
            lines.append(
                (
                    f"- {regression['source']} {regression['metric']}: "
                    f"{regression['baseline']:.4f} -> {regression['candidate']:.4f} "
                    f"({regression['delta']:+.4f})"
                )
            )

    return "\n".join(lines) + "\n"


def derive_default_output(
    baseline_path: Path,
    candidate_path: Path,
) -> Path:
    baseline_name = baseline_path.stem
    candidate_name = candidate_path.stem
    return Path("results/comparisons") / f"{candidate_name}_vs_{baseline_name}.md"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare baseline and candidate VetQwen runs."
    )
    parser.add_argument("--baseline", required=True, help="Baseline metrics JSON path")
    parser.add_argument("--candidate", required=True, help="Candidate metrics JSON path")
    parser.add_argument(
        "--baseline-judge",
        default=None,
        help="Optional baseline judge JSON path",
    )
    parser.add_argument(
        "--candidate-judge",
        default=None,
        help="Optional candidate judge JSON path",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Markdown output path (default: "
            "results/comparisons/<candidate>_vs_<baseline>.md)"
        ),
    )
    args = parser.parse_args()

    baseline_path = Path(args.baseline)
    candidate_path = Path(args.candidate)
    output_path = (
        Path(args.output)
        if args.output
        else derive_default_output(baseline_path, candidate_path)
    )

    baseline_result = load_json(baseline_path)
    candidate_result = load_json(candidate_path)
    baseline_result["_path"] = str(baseline_path)
    candidate_result["_path"] = str(candidate_path)
    baseline_judge = load_json(args.baseline_judge) if args.baseline_judge else None
    candidate_judge = load_json(args.candidate_judge) if args.candidate_judge else None

    comparison = build_comparison(
        baseline_result,
        candidate_result,
        baseline_judge,
        candidate_judge,
    )
    markdown = comparison_markdown(comparison, baseline_result, candidate_result)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")

    sidecar_path = output_path.with_suffix(".json")
    sidecar_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    print(f"Comparison summary written to {output_path}")
    print(f"Comparison JSON written to {sidecar_path}")


if __name__ == "__main__":
    main()
