from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_dataset


class BuildDatasetTests(unittest.TestCase):
    def test_normalize_pet_health_schema(self) -> None:
        record = {
            "_source": "karenwky/pet-health-symptoms-dataset",
            "text": "My dog has sudden vomiting and bloody diarrhea.",
            "condition": "Digestive Issues",
            "record_type": "Owner Observation",
        }

        normalized = build_dataset.normalize_pet_health_symptoms(record)

        self.assertIsNotNone(normalized)
        self.assertEqual(normalized["_meta"]["source"], record["_source"])
        self.assertEqual(normalized["_meta"]["species"], "dog")
        self.assertIsNone(normalized["_meta"]["condition"])
        self.assertIn("Digestive Issues", normalized["_meta"]["source_labels"])

    def test_normalize_vet_health_assessment_symptom_row(self) -> None:
        record = {
            "_source": "infinite-dataset-hub/VetHealthAssessment",
            "Question": "What are the common symptoms of Canine Parvovirus?",
            "Answer": "Symptoms include lethargy, severe vomiting, loss of appetite, and bloody diarrhea.",
            "Labels": "VetHealthAssessment,CanineDisease,SymptomAssessment",
        }

        normalized = build_dataset.normalize_vet_health_assessment(record)

        self.assertIsNotNone(normalized)
        self.assertEqual(normalized["_meta"]["species"], "dog")
        self.assertEqual(normalized["_meta"]["condition"], "canine parvovirus")
        self.assertIn("SymptomAssessment", normalized["_meta"]["source_labels"])

    def test_filter_target_species_drops_equine(self) -> None:
        record = {
            "_source": "infinite-dataset-hub/VetPetCare",
            "species": "Equine",
            "breed": "Arabian",
            "age": 9,
            "symptoms": "colic",
            "diagnosis": "intestinal blockage",
            "treatment_plan": "surgery",
            "follow_up_result": "recovered",
        }

        normalized = build_dataset.normalize_vet_pet_care(record)
        kept, dropped = build_dataset.filter_target_species([normalized])

        self.assertEqual(kept, [])
        self.assertEqual(dropped["horse"], 1)

    def test_normalize_vet_pet_care_remaps_non_condition_label(self) -> None:
        record = {
            "_source": "infinite-dataset-hub/VetPetCare",
            "species": "Canine",
            "breed": "Mixed",
            "age": 8,
            "symptoms": "diabetes mellitus with weight loss and PU/PD",
            "diagnosis": "insulin therapy",
            "treatment_plan": "insulin and monitoring",
            "follow_up_result": "stable",
        }

        normalized = build_dataset.normalize_vet_pet_care(record)

        self.assertIsNotNone(normalized)
        self.assertEqual(
            normalized["_meta"]["condition"],
            "diabetes mellitus with weight loss and pu/pd",
        )
        self.assertEqual(normalized["_meta"]["condition_source"], "complaint_fallback")
        self.assertIn("diabetes mellitus", normalized["messages"][2]["content"].casefold())
        self.assertNotIn("insulin therapy —", normalized["messages"][2]["content"].casefold())
        self.assertIn("triage", normalized["_meta"])
        self.assertIn("urgent", normalized["_meta"])

    def test_stratified_split_keeps_singleton_labeled_bucket_in_train(self) -> None:
        sample = build_dataset.make_chatml(
            user_content="Species: Dog\nPresenting complaint: diarrhea",
            assistant_content=(
                "**Species & Signalment:** Dog, Signalment not provided\n"
                "**Presenting Symptoms:** diarrhea\n\n"
                "**Assessment:**\npossible parvovirus\n\n"
                "**Differential Diagnoses (ranked by likelihood):**\n"
                "1. canine parvovirus — likely\n\n"
                "**Triage Recommendation:** Urgent\n"
                "**Suggested Next Steps:** exam\n"
            ),
            source="fixture",
            species="dog",
            condition="canine parvovirus",
            triage="Urgent",
            source_labels=[],
            condition_source="fixture",
        )

        train, val, test = build_dataset.stratified_split([sample])

        self.assertEqual(len(train), 1)
        self.assertEqual(len(val), 0)
        self.assertEqual(len(test), 0)

    def test_duplicate_audit_reports_cross_split_duplicates(self) -> None:
        sample = build_dataset.make_chatml(
            user_content="Species: Cat\nPresenting complaint: vomiting and lethargy",
            assistant_content=(
                "**Species & Signalment:** Cat, Signalment not provided\n"
                "**Presenting Symptoms:** vomiting and lethargy\n\n"
                "**Assessment:**\npossible gastritis\n\n"
                "**Differential Diagnoses (ranked by likelihood):**\n"
                "1. gastritis — likely\n\n"
                "**Triage Recommendation:** Schedule within 48h\n"
                "**Suggested Next Steps:** exam\n"
            ),
            source="fixture",
            species="cat",
            condition="gastritis",
            triage="Schedule within 48h",
            source_labels=[],
            condition_source="fixture",
        )

        audit = build_dataset.audit_cross_split_duplicates(
            {"train": [sample], "val": [sample], "test": []},
            near_duplicate_ratio=0.9,
        )

        self.assertEqual(audit["exact_duplicate_count"], 1)
        self.assertGreaterEqual(len(audit["exact_duplicates"]), 1)

    def test_ensure_eval_split_coverage_populates_labeled_and_urgent_cases(self) -> None:
        def sample(species: str, condition: str, triage: str) -> dict:
            return build_dataset.make_chatml(
                user_content=f"Species: {species.title()}\nPresenting complaint: fixture complaint",
                assistant_content=(
                    f"**Species & Signalment:** {species.title()}, Signalment not provided\n"
                    "**Presenting Symptoms:** fixture complaint\n\n"
                    "**Assessment:**\nfixture assessment\n\n"
                    "**Differential Diagnoses (ranked by likelihood):**\n"
                    f"1. {condition} — likely\n\n"
                    f"**Triage Recommendation:** {triage}\n"
                    "**Suggested Next Steps:** exam\n"
                ),
                source="fixture",
                species=species,
                condition=condition,
                triage=triage,
                source_labels=[],
                condition_source="fixture",
            )

        train = [
            sample("dog", "canine parvovirus", "Urgent"),
            sample("cat", "urinary obstruction", "Urgent"),
            sample("dog", "gastroenteritis", "Schedule within 48h"),
            sample("cat", "dermatitis", "Schedule within 48h"),
        ]

        train, val, test = build_dataset.ensure_eval_split_coverage(train, [], [], seed=42)

        self.assertGreaterEqual(
            sum(bool(row["_meta"].get("condition")) for row in val),
            1,
        )
        self.assertGreaterEqual(
            sum(bool(row["_meta"].get("condition")) for row in test),
            1,
        )
        self.assertGreaterEqual(
            sum(bool(row["_meta"].get("urgent")) for row in val),
            1,
        )
        self.assertGreaterEqual(
            sum(bool(row["_meta"].get("urgent")) for row in test),
            1,
        )

    def test_collect_and_write_fixture_dataset(self) -> None:
        raw_sources = {
            "karenwky/pet-health-symptoms-dataset": [
                {
                    "_source": "karenwky/pet-health-symptoms-dataset",
                    "text": "My cat vomited and is hiding.",
                    "condition": "Digestive Issues",
                    "record_type": "Owner Observation",
                }
            ],
            "infinite-dataset-hub/VetHealthAssessment": [
                {
                    "_source": "infinite-dataset-hub/VetHealthAssessment",
                    "Question": "What symptoms might indicate a dog has a UTI?",
                    "Answer": "Symptoms can include frequent urination, straining to urinate, blood in the urine, and discomfort during urination.",
                    "Labels": "VetHealthAssessment,CanineDisease,SymptomAssessment",
                }
            ],
            "infinite-dataset-hub/VetPetCare": [
                {
                    "_source": "infinite-dataset-hub/VetPetCare",
                    "species": "Canine",
                    "breed": "Beagle",
                    "age": 2,
                    "symptoms": "vomiting",
                    "diagnosis": "gastroenteritis",
                    "treatment_plan": "rehydration and antiemetics",
                    "follow_up_result": "recovered",
                }
            ],
        }
        synthetic = [
            {
                "messages": [
                    {"role": "system", "content": "system"},
                    {
                        "role": "user",
                        "content": "Species: Sheep\nPresenting complaint: lamb with head pressing and circling",
                    },
                    {
                        "role": "assistant",
                        "content": "**Differential Diagnoses (ranked by likelihood):**\n1. listeriosis — likely",
                    },
                ],
                "_meta": {"source": "synthetic_ollama", "species": "Sheep"},
            }
        ]

        samples = build_dataset.collect_samples(raw_sources, synthetic)
        filtered, dropped = build_dataset.filter_target_species(samples)
        train, val, test = build_dataset.stratified_split(filtered)

        self.assertEqual(sum(dropped.values()), 0)
        self.assertEqual(len(train) + len(val) + len(test), len(filtered))
        for sample in filtered:
            self.assertIn("species", sample["_meta"])
            self.assertIn("condition", sample["_meta"])
            self.assertIn("triage", sample["_meta"])
            self.assertIn("urgent", sample["_meta"])
            self.assertIn("source_labels", sample["_meta"])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "train.jsonl"
            build_dataset.write_jsonl(filtered, output_path)
            lines = output_path.read_text().strip().splitlines()
            self.assertEqual(len(lines), len(filtered))
            first_record = json.loads(lines[0])
            self.assertIn("_meta", first_record)


if __name__ == "__main__":
    unittest.main()
