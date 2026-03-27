from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_review_subset


class BuildReviewSubsetTests(unittest.TestCase):
    def test_select_review_subset_keeps_all_urgent_cases(self) -> None:
        records = [
            {
                "row_index": 0,
                "species": "dog",
                "condition": "canine parvovirus",
                "urgent": True,
            },
            {
                "row_index": 1,
                "species": "cat",
                "condition": "hyperthyroidism",
                "urgent": False,
            },
            {
                "row_index": 2,
                "species": "dog",
                "condition": "atopy",
                "urgent": False,
            },
        ]

        subset = build_review_subset.select_review_subset(records, target_size=2, seed=42)

        self.assertEqual({record["row_index"] for record in subset}, {0, 1})

    def test_to_review_record_parses_processed_dataset_format(self) -> None:
        row = {
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "Species: Dog\nPresenting complaint: vomiting"},
                {
                    "role": "assistant",
                    "content": (
                        "**Species & Signalment:** Dog, Signalment not provided\n"
                        "**Presenting Symptoms:** vomiting\n\n"
                        "**Assessment:**\nassessment\n\n"
                        "**Differential Diagnoses (ranked by likelihood):**\n"
                        "1. gastritis — likely\n\n"
                        "**Triage Recommendation:** Urgent\n"
                        "**Suggested Next Steps:** exam\n"
                    ),
                },
            ],
            "_meta": {"species": "dog", "condition": "gastritis", "source": "fixture"},
            "_row_index": 3,
        }

        review = build_review_subset.to_review_record(row)

        self.assertEqual(review["reference_triage"], "Urgent")
        self.assertTrue(review["urgent"])
        self.assertEqual(review["source"], "fixture")


if __name__ == "__main__":
    unittest.main()
