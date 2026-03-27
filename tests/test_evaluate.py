from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate
import judge_utils


GOOD_RESPONSE = """**Species & Signalment:** Dog, 3 years, Golden Retriever
**Presenting Symptoms:** lethargy, vomiting, bloody diarrhea

**Assessment:**
The symptom pattern is strongly compatible with canine parvovirus.

**Differential Diagnoses (ranked by likelihood):**
1. Canine Parvovirus — classic vomiting and bloody diarrhea pattern.
2. Hemorrhagic gastroenteritis — also possible.

**Triage Recommendation:** Urgent
**Suggested Next Steps:** Perform examination, isolation, CBC, and supportive care.
"""


class EvaluateTests(unittest.TestCase):
    def test_parse_structured_response_success(self) -> None:
        parsed = evaluate.parse_structured_response(GOOD_RESPONSE)

        self.assertTrue(parsed["has_all_sections"])
        self.assertTrue(parsed["triage_present"])
        self.assertTrue(parsed["parse_success"])
        self.assertEqual(parsed["top_diagnosis"], "Canine Parvovirus")

    def test_parse_structured_response_failure(self) -> None:
        parsed = evaluate.parse_structured_response("**Assessment:** incomplete only")

        self.assertFalse(parsed["has_all_sections"])
        self.assertFalse(parsed["parse_success"])
        self.assertFalse(parsed["triage_present"])
        self.assertIsNone(parsed["top_diagnosis"])

    def test_compute_primary_metrics_uses_canonical_condition_labels(self) -> None:
        predictions = [
            GOOD_RESPONSE,
            GOOD_RESPONSE.replace("Canine Parvovirus", "Gastroenteritis"),
            GOOD_RESPONSE,
        ]
        records = [
            {
                "meta": {
                    "condition": "canine parvovirus",
                    "species": "dog",
                    "source": "VetPetCare",
                    "triage": "Urgent",
                },
                "reference": GOOD_RESPONSE,
            },
            {
                "meta": {
                    "condition": "canine parvovirus",
                    "species": "dog",
                    "source": "VetPetCare",
                    "triage": "Urgent",
                },
                "reference": GOOD_RESPONSE,
            },
            {
                "meta": {
                    "condition": None,
                    "species": "cat",
                    "source": "karenwky",
                    "triage": "Urgent",
                },
                "reference": GOOD_RESPONSE,
            },
        ]

        rows = evaluate.build_evaluation_rows(predictions, [GOOD_RESPONSE] * 3, records)
        metrics = evaluate.compute_primary_metrics(rows)

        self.assertEqual(metrics["n_condition_labeled"], 2)
        self.assertAlmostEqual(metrics["diagnosis_hit_rate"], 0.5)
        self.assertAlmostEqual(metrics["triage_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["urgent_recall"], 1.0)
        self.assertAlmostEqual(metrics["triage_section_presence"], 1.0)
        self.assertAlmostEqual(metrics["parse_success_rate"], 1.0)

    def test_species_breakdown_reads_normalized_meta(self) -> None:
        predictions = [GOOD_RESPONSE, GOOD_RESPONSE.replace("Urgent", "Schedule within 48h")]
        references = [GOOD_RESPONSE, GOOD_RESPONSE]
        records = [
            {
                "meta": {
                    "species": "dog",
                    "condition": "canine parvovirus",
                    "source": "VetPetCare",
                    "triage": "Urgent",
                },
                "reference": GOOD_RESPONSE,
            },
            {
                "meta": {
                    "species": "sheep",
                    "condition": "canine parvovirus",
                    "source": "synthetic_ollama",
                    "triage": "Urgent",
                },
                "reference": GOOD_RESPONSE,
            },
        ]

        rows = evaluate.build_evaluation_rows(predictions, references, records)
        breakdown = evaluate.compute_species_breakdown(rows)
        source_breakdown = evaluate.compute_source_breakdown(rows)
        confusion = evaluate.compute_triage_confusion(rows)

        self.assertIn("dog", breakdown)
        self.assertIn("sheep", breakdown)
        self.assertEqual(breakdown["dog"]["n"], 1)
        self.assertEqual(breakdown["sheep"]["n"], 1)
        self.assertIn("VetPetCare", source_breakdown)
        self.assertEqual(confusion["Urgent"]["Schedule within 48h"], 1)

    def test_reference_triage_falls_back_to_reference_text(self) -> None:
        record = {
            "meta": {"species": "dog", "condition": "canine parvovirus", "source": "fixture"},
            "reference": GOOD_RESPONSE,
        }

        rows = evaluate.build_evaluation_rows([GOOD_RESPONSE], [GOOD_RESPONSE], [record])

        self.assertEqual(rows[0]["reference_triage"], "Urgent")
        self.assertTrue(rows[0]["triage_match"])

    def test_select_sample_indices_is_seeded(self) -> None:
        first = judge_utils.select_sample_indices(20, 5, seed=7)
        second = judge_utils.select_sample_indices(20, 5, seed=7)
        third = judge_utils.select_sample_indices(20, 5, seed=8)

        self.assertEqual(first, second)
        self.assertNotEqual(first, third)


if __name__ == "__main__":
    unittest.main()
