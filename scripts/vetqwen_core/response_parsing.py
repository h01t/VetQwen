"""
Structured response parsing and comparison helpers.
"""

from __future__ import annotations

import re
from typing import Any

from vetqwen_core.text import canonicalize_triage, clean_text, normalize_label_text

SECTION_PATTERNS = {
    "species_signalment": re.compile(
        r"\*\*Species & Signalment:\*\*(.*?)(?=\n\*\*|\Z)", re.DOTALL
    ),
    "presenting_symptoms": re.compile(
        r"\*\*Presenting Symptoms:\*\*(.*?)(?=\n\*\*|\Z)", re.DOTALL
    ),
    "assessment": re.compile(r"\*\*Assessment:\*\*(.*?)(?=\n\*\*|\Z)", re.DOTALL),
    "differentials": re.compile(
        r"\*\*Differential Diagnoses[^\n]*\*\*(.*?)(?=\n\*\*|\Z)", re.DOTALL
    ),
    "triage": re.compile(
        r"\*\*Triage Recommendation:\*\*(.*?)(?=\n\*\*|\Z)", re.DOTALL
    ),
    "next_steps": re.compile(
        r"\*\*Suggested Next Steps:\*\*(.*?)(?=\n\*\*|\Z)", re.DOTALL
    ),
}


def extract_first_differential(text: str) -> str | None:
    match = re.search(r"^\s*1\.\s+(.+?)(?:\s+—|\s+-|$)", text, flags=re.MULTILINE)
    if not match:
        return None
    return clean_text(match.group(1))


def parse_structured_response(response: str) -> dict[str, Any]:
    sections: dict[str, str] = {}
    for name, pattern in SECTION_PATTERNS.items():
        match = pattern.search(response)
        sections[name] = clean_text(match.group(1)) if match else ""

    top_diagnosis = extract_first_differential(sections.get("differentials", ""))
    has_all_sections = all(sections.values())
    triage_present = bool(sections.get("triage"))
    triage_label = canonicalize_triage(sections.get("triage"))
    parse_success = has_all_sections and triage_present and bool(top_diagnosis)

    return {
        "sections": sections,
        "has_all_sections": has_all_sections,
        "triage_present": triage_present,
        "triage_label": triage_label,
        "top_diagnosis": top_diagnosis,
        "parse_success": parse_success,
    }


def diagnosis_hit(predicted: str | None, reference: str | None) -> bool:
    if not predicted or not reference:
        return False

    predicted_norm = normalize_label_text(predicted)
    reference_norm = normalize_label_text(reference)
    if not predicted_norm or not reference_norm:
        return False

    return (
        predicted_norm == reference_norm
        or predicted_norm.startswith(reference_norm)
        or reference_norm.startswith(predicted_norm)
        or reference_norm in predicted_norm
        or predicted_norm in reference_norm
    )


def get_reference_triage(record: dict[str, Any]) -> str | None:
    triage = canonicalize_triage(record.get("meta", {}).get("triage"))
    if triage:
        return triage
    return parse_structured_response(record.get("reference", "")).get("triage_label")


def build_evaluation_rows(
    predictions: list[str],
    references: list[str],
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prediction, reference, record in zip(predictions, references, records):
        parsed = parse_structured_response(prediction)
        condition = record.get("meta", {}).get("condition")
        reference_triage = get_reference_triage(record)
        rows.append(
            {
                "prediction": prediction,
                "reference": reference,
                "meta": record.get("meta", {}),
                "parsed_prediction": parsed,
                "top_diagnosis": parsed["top_diagnosis"],
                "predicted_triage": parsed["triage_label"],
                "reference_triage": reference_triage,
                "diagnosis_hit": diagnosis_hit(parsed["top_diagnosis"], condition)
                if condition
                else None,
                "triage_match": (
                    parsed["triage_label"] == reference_triage
                    if reference_triage and parsed["triage_label"]
                    else None
                ),
            }
        )
    return rows

