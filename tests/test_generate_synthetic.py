from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import generate_synthetic


class GenerateSyntheticTests(unittest.TestCase):
    def test_normalize_generated_case_repairs_imperfect_differentials(self) -> None:
        case = {
            "signalment": "3-year-old ewe",
            "complaint": "off feed for two days",
            "signs": ["depression", "ketone breath"],
            "differentials": [
                {"rank": "1", "diagnosis": "pregnancy toxemia", "reasoning": "late gestation and anorexia"},
                {"rank": "second", "name": "listeriosis"},
                "ruminal acidosis",
                {"rank": 4, "diagnosis": "", "rationale": "missing diagnosis"},
            ],
            "triage": "urgent",
        }

        normalized = generate_synthetic.normalize_generated_case(case)

        self.assertIsNotNone(normalized)
        self.assertEqual(normalized["triage"], "Urgent")
        self.assertEqual(normalized["signs"], "depression, ketone breath")
        self.assertEqual(len(normalized["differentials"]), 3)
        self.assertEqual(
            [item["rank"] for item in normalized["differentials"]],
            [1, 2, 3],
        )
        self.assertEqual(
            normalized["differentials"][1]["diagnosis"],
            "listeriosis",
        )
        self.assertIn(
            "Compatible with the reported signalment",
            normalized["differentials"][1]["rationale"],
        )

    def test_case_to_chatml_handles_fallback_differential(self) -> None:
        record = generate_synthetic.case_to_chatml(
            "Pig",
            {
                "signalment": "2-year-old sow",
                "complaint": "reduced appetite",
                "signs": "fever and lethargy",
                "differentials": [
                    {
                        "rank": 1,
                        "diagnosis": "erysipelas",
                        "rationale": "",
                    }
                ],
                "triage": "Schedule within 48h",
            },
        )

        assistant = record["messages"][2]["content"]
        self.assertIn("1. erysipelas", assistant)
        self.assertIn("Compatible with the reported signalment", assistant)
        self.assertEqual(record["_meta"]["condition"], "erysipelas")
        self.assertEqual(record["_meta"]["triage"], "Schedule within 48h")


if __name__ == "__main__":
    unittest.main()
