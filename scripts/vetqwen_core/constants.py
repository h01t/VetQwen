"""
Shared constants used across VetQwen scripts and the demo.
"""

from __future__ import annotations

SYSTEM_PROMPT = (
    "You are VetQwen, an expert veterinary diagnostic assistant. "
    "Given a patient signalment and clinical symptoms, provide a structured "
    "differential diagnosis with clinical reasoning, ranked differentials, and "
    "a triage recommendation. Always remind the user that your output is not a "
    "substitute for professional veterinary examination."
)

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_ADAPTER = "./adapter"

EXPECTED_SECTIONS = [
    "**Species & Signalment:**",
    "**Presenting Symptoms:**",
    "**Assessment:**",
    "**Differential Diagnoses",
    "**Triage Recommendation:**",
    "**Suggested Next Steps:**",
]

TRIAGE_LABELS = ["Urgent", "Schedule within 48h", "Monitor at home"]

