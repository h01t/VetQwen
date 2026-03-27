"""
Shared text normalization helpers.
"""

from __future__ import annotations

import re
from typing import Any


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_label_text(value: Any) -> str:
    text = clean_text(value).casefold()
    text = re.sub(r"\*+", "", text)
    text = re.sub(r"[^a-z0-9\s/&-]", "", text)
    text = re.sub(r"\s+", " ", text).strip(" -")
    return text


def canonicalize_triage(value: Any) -> str | None:
    text = clean_text(value).casefold()
    if not text:
        return None
    if "urgent" in text:
        return "Urgent"
    if "monitor" in text:
        return "Monitor at home"
    if "48h" in text or "48 h" in text or "schedule" in text:
        return "Schedule within 48h"
    return clean_text(value) or None


def extract_triage_from_response(text: str) -> str | None:
    match = re.search(
        r"\*\*Triage Recommendation:\*\*\s*(.*?)(?:\n\*\*|$)",
        text,
        flags=re.DOTALL,
    )
    if not match:
        return None
    return canonicalize_triage(match.group(1))

