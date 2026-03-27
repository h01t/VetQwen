from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import compare_results


class CompareResultsTests(unittest.TestCase):
    def test_build_comparison_skips_missing_judge_files(self) -> None:
        baseline = {
            "run_name": "baseline",
            "metrics": {
                "diagnosis_hit_rate": 0.2,
                "parse_success_rate": 0.2,
                "format_compliance": 0.1,
                "triage_accuracy": 0.4,
                "urgent_recall": 0.0,
                "urgent_precision": 0.0,
                "rouge_l": 0.2,
                "bert_score_f1": 0.8,
            },
            "source_breakdown": {
                "infinite-dataset-hub/VetPetCare": {
                    "diagnosis_hit_rate": 0.2,
                    "triage_accuracy": 0.5,
                },
                "infinite-dataset-hub/VetHealthAssessment": {
                    "diagnosis_hit_rate": 0.0,
                    "triage_accuracy": 0.0,
                },
            },
        }
        candidate = {
            "run_name": "candidate",
            "metrics": {
                "diagnosis_hit_rate": 0.7,
                "parse_success_rate": 1.0,
                "format_compliance": 1.0,
                "triage_accuracy": 0.9,
                "urgent_recall": 0.5,
                "urgent_precision": 1.0,
                "rouge_l": 0.9,
                "bert_score_f1": 0.99,
            },
            "source_breakdown": {
                "infinite-dataset-hub/VetPetCare": {
                    "diagnosis_hit_rate": 0.6,
                    "triage_accuracy": 0.7,
                },
                "infinite-dataset-hub/VetHealthAssessment": {
                    "diagnosis_hit_rate": 0.0,
                    "triage_accuracy": 0.0,
                },
            },
        }

        comparison = compare_results.build_comparison(baseline, candidate, None, None)
        judge_gate = next(
            gate
            for gate in comparison["gates"]
            if gate["name"] == "judge clinical_accuracy does not regress"
        )

        self.assertTrue(comparison["overall_passed"])
        self.assertTrue(judge_gate["passed"])
        self.assertTrue(judge_gate["skipped"])

    def test_build_comparison_fails_source_guardrail_regression(self) -> None:
        baseline = {
            "run_name": "baseline",
            "metrics": {
                "diagnosis_hit_rate": 0.2,
                "parse_success_rate": 1.0,
                "format_compliance": 1.0,
                "triage_accuracy": 0.8,
                "urgent_recall": 0.2,
                "urgent_precision": 0.5,
                "rouge_l": 0.3,
                "bert_score_f1": 0.8,
            },
            "source_breakdown": {
                "infinite-dataset-hub/VetPetCare": {
                    "diagnosis_hit_rate": 0.8,
                    "triage_accuracy": 0.8,
                },
                "infinite-dataset-hub/VetHealthAssessment": {
                    "diagnosis_hit_rate": 0.3,
                    "triage_accuracy": 0.6,
                },
            },
        }
        candidate = {
            "run_name": "candidate",
            "metrics": {
                "diagnosis_hit_rate": 0.3,
                "parse_success_rate": 1.0,
                "format_compliance": 1.0,
                "triage_accuracy": 0.8,
                "urgent_recall": 0.2,
                "urgent_precision": 0.5,
                "rouge_l": 0.9,
                "bert_score_f1": 0.99,
            },
            "source_breakdown": {
                "infinite-dataset-hub/VetPetCare": {
                    "diagnosis_hit_rate": 0.55,
                    "triage_accuracy": 0.78,
                },
                "infinite-dataset-hub/VetHealthAssessment": {
                    "diagnosis_hit_rate": 0.1,
                    "triage_accuracy": 0.59,
                },
            },
        }

        comparison = compare_results.build_comparison(baseline, candidate, None, None)
        source_gate = next(
            gate for gate in comparison["gates"] if gate["name"] == "source guardrails hold"
        )

        self.assertFalse(comparison["overall_passed"])
        self.assertFalse(source_gate["passed"])
        self.assertGreaterEqual(len(comparison["source_regressions"]), 1)


if __name__ == "__main__":
    unittest.main()
