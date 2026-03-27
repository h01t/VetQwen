"""
Prompt and record-format helpers shared across data prep and demo paths.
"""

from __future__ import annotations

from typing import Any

from vetqwen_core.constants import SYSTEM_PROMPT
from vetqwen_core.text import canonicalize_triage, clean_text


def display_species(species: str) -> str:
    labels = {
        "dog": "Dog",
        "cat": "Cat",
        "cattle": "Cattle",
        "pig": "Pig",
        "sheep": "Sheep",
        "horse": "Horse",
        "unknown": "Unknown species",
    }
    return labels.get(species, species.title())


def get_message_content(messages: list[dict[str, Any]], role: str) -> str:
    return next((message.get("content", "") for message in messages if message.get("role") == role), "")


def build_patient_prompt(
    species: str,
    complaint: str,
    age: str | None = None,
    sex: str | None = None,
    breed: str | None = None,
    note_type: str | None = None,
) -> str:
    parts = [f"Species: {display_species(species)}"]
    if age:
        parts.append(f"Age: {clean_text(age)}")
    if sex:
        parts.append(f"Sex: {clean_text(sex)}")
    if breed:
        parts.append(f"Breed: {clean_text(breed)}")
    if note_type:
        parts.append(f"Source note type: {clean_text(note_type)}")
    parts.append(f"Presenting complaint: {clean_text(complaint)}")
    return "\n".join(parts)


def build_structured_response(
    species: str,
    presenting_symptoms: str,
    differential_lines: list[str],
    assessment: str,
    triage: str,
    next_steps: str,
    signalment: str | None = None,
) -> str:
    signalment_text = signalment or "Signalment not provided"
    symptom_text = clean_text(presenting_symptoms) or "Not provided"
    differential_block = (
        "\n".join(differential_lines)
        if differential_lines
        else "1. Requires further evaluation — Insufficient information provided."
    )

    return (
        f"**Species & Signalment:** {display_species(species)}, {signalment_text}\n"
        f"**Presenting Symptoms:** {symptom_text}\n\n"
        f"**Assessment:**\n{clean_text(assessment)}\n\n"
        f"**Differential Diagnoses (ranked by likelihood):**\n{differential_block}\n\n"
        f"**Triage Recommendation:** {clean_text(triage)}\n"
        f"**Suggested Next Steps:** {clean_text(next_steps)}\n\n"
        "*Note: This output is not a substitute for professional veterinary examination.*"
    )


def make_chatml_record(
    user_content: str,
    assistant_content: str,
    source: str,
    species: str,
    condition: str | None,
    triage: str | None,
    source_labels: list[str] | None = None,
    condition_source: str | None = None,
) -> dict[str, Any]:
    normalized_triage = canonicalize_triage(triage)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "_meta": {
            "source": source,
            "species": species,
            "condition": condition,
            "triage": normalized_triage,
            "urgent": bool(normalized_triage == "Urgent"),
            "source_labels": source_labels or [],
            "condition_source": condition_source,
        },
    }

